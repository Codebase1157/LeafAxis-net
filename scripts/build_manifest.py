"""Build the frozen partition manifest from a local copy of the three datasets.

Writes one row per image with:
    path, dataset, class_name, label, sha256, phash, embedding_path

SHA-256 is computed from the file bytes. The 64-bit perceptual hash is from
`imagehash.phash` on a 32x32 greyscale downscale. The embedding is a 512-d
feature from a frozen ResNet-50 (ImageNet-pretrained, penultimate layer), written
to `./embeddings/<dataset>.npy` with row index matching the manifest row.

The `component` and `partition` columns are added by `make_splits.py`.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing.dataset import iter_images, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def phash64(img: Image.Image) -> str:
    """Return the 64-bit pHash as a 16-char hex string."""
    return str(imagehash.phash(img, hash_size=8))


def build_embedder(device: torch.device) -> tuple[nn.Module, transforms.Compose]:
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    resnet.fc = nn.Identity()
    resnet.eval().to(device)
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return resnet, tfm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="Folder with SLD/, DK/, BD/ subfolders of class folders of images.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Where to write the manifest CSV.")
    ap.add_argument("--embeddings-dir", default=Path("embeddings"), type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.embeddings_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    embedder, tfm = build_embedder(device)

    rows: list[dict] = []
    embeddings_by_dataset: dict[str, list[np.ndarray]] = {}
    label_by_dataset: dict[str, dict[str, int]] = {}

    for dataset, class_name, img_path in tqdm(list(iter_images(args.root)), desc="hashing"):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[skip] {img_path}: {e}")
            continue

        label_by_dataset.setdefault(dataset, {})
        label_map = label_by_dataset[dataset]
        if class_name not in label_map:
            label_map[class_name] = len(label_map)
        label = label_map[class_name]

        with torch.no_grad():
            emb = embedder(tfm(img).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        embeddings_by_dataset.setdefault(dataset, []).append(emb)

        rows.append({
            "path": str(img_path),
            "dataset": dataset,
            "class_name": class_name,
            "label": label,
            "sha256": sha256_of(img_path),
            "phash": phash64(img),
        })

    for dataset, embs in embeddings_by_dataset.items():
        np.save(args.embeddings_dir / f"{dataset}.npy", np.vstack(embs).astype(np.float32))

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    for dataset, sub in df.groupby("dataset"):
        print(f"{dataset}: {len(sub)} images, {sub['class_name'].nunique()} classes")


if __name__ == "__main__":
    main()

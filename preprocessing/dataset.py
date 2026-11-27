"""Image loading and preprocessing for the three sugarcane leaf datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ImageNet statistics; matches what every baseline in this repo was pretrained with.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(image_size: int = 224) -> Callable:
    """Training-time augmentations: resize, random resized crop, horizontal flip, colour jitter.

    These augmentations are applied only to the training split; validation and test transforms
    do not include randomness so that every re-evaluation produces the same number.
    """
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(image_size: int = 224) -> Callable:
    """Evaluation-time transform: deterministic resize + centre crop + normalise."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ManifestDataset(Dataset):
    """A dataset backed by a frozen manifest CSV.

    The manifest has one row per retained image with at least these columns:
        path          absolute or project-relative path to the image file
        label         integer class id, unique within its `dataset`
        dataset       one of {"SLD", "DK", "BD"}
        component     integer connected-component id (for leakage control)
        partition     one of {"train", "val", "test"} or the name of a re-split
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        dataset: str,
        partition: str,
        transform: Callable | None = None,
        resplit_column: str = "partition",
    ):
        df = pd.read_csv(manifest_csv)
        df = df[df["dataset"] == dataset]
        df = df[df[resplit_column] == partition].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(
                f"Manifest {manifest_csv!s} has no rows for dataset={dataset!r} "
                f"and {resplit_column}={partition!r}."
            )
        self.df = df
        self.transform = transform or eval_transform()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        img = self.transform(img)
        return img, int(row["label"]), int(row["component"])


def iter_images(root: str | Path) -> Iterable[tuple[str, str, Path]]:
    """Yield (dataset, class_name, image_path) for every image under `root`.

    Expects `root/<dataset>/<class_name>/<file>.{jpg,png,jpeg}`.
    """
    root = Path(root)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        dataset = ds_dir.name
        for cls_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            cls_name = cls_dir.name
            for img in sorted(cls_dir.iterdir()):
                if img.suffix.lower() in exts:
                    yield dataset, cls_name, img

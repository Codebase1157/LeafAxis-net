"""Compute the SHA-256 of every artifact in the repo and record it in REPRODUCE.json.

Hashes what actually exists on disk. Does NOT invent hashes for files that are not
present. If a file is missing, its `sha256` field in REPRODUCE.json is set to null
and the file is listed under `missing_artifacts` at the end so the next step is
obvious.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reproduce-json", default=Path("REPRODUCE.json"), type=Path)
    args = ap.parse_args()

    repro = json.loads(args.reproduce_json.read_text())
    missing: list[str] = []

    for entry in repro["artifacts"]:
        p = Path(entry["path"])
        if p.exists() and p.is_file():
            entry["sha256"] = sha256_of(p)
            entry["bytes"] = p.stat().st_size
        else:
            entry["sha256"] = None
            entry["bytes"] = None
            missing.append(entry["path"])

    repro["missing_artifacts"] = missing
    args.reproduce_json.write_text(json.dumps(repro, indent=2))
    print(f"updated {args.reproduce_json} ({len(missing)} missing of {len(repro['artifacts'])})")
    if missing:
        print("missing (run the scripts that produce these before claiming reproducibility):")
        for m in missing:
            print(f"  - {m}")


if __name__ == "__main__":
    main()

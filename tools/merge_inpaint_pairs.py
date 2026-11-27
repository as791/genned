#!/usr/bin/env python3
"""Merge inpainting pair artifacts (inpaint-data.yml) into one training folder.

Each artifact folder holds real/<name>.png and ai/<name>.png, the same photo before and
after an AI model filled part of it. Older batches reused file names across jobs, so the
folders can't simply be copied on top of each other. This copies each pair as
<artifact folder name>-<name>, keeps the two halves together, and skips a pair whose real
photo is byte-identical to one already kept (photo samples of two jobs can overlap).

Usage:
    python tools/merge_inpaint_pairs.py --out data/train/inpaint inpaint-src/*/*
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path


def merge(folders: list[Path], out: Path) -> dict[str, int]:
    """Copy every complete pair; returns pairs kept per folder name."""
    (out / "real").mkdir(parents=True, exist_ok=True)
    (out / "ai").mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    kept: dict[str, int] = {}
    for folder in sorted(folders):
        real_dir, ai_dir = folder / "real", folder / "ai"
        if not real_dir.is_dir() or not ai_dir.is_dir():
            continue
        n = 0
        for real in sorted(real_dir.iterdir()):
            edit = ai_dir / real.name
            if not real.is_file() or not edit.is_file():
                continue
            digest = hashlib.sha256(real.read_bytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            name = f"{folder.name}-{real.name}"
            shutil.copyfile(real, out / "real" / name)
            shutil.copyfile(edit, out / "ai" / name)
            n += 1
        kept[folder.name] = n
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("folders", type=Path, nargs="+")
    args = parser.parse_args()
    kept = merge(args.folders, args.out)
    for name, n in kept.items():
        print(f"{name}: {n} pairs")
    total = sum(kept.values())
    print(f"Inpainting pairs kept: {total} from {len(kept)} artifacts")
    if total == 0:
        sys.exit("No inpainting pairs found")


if __name__ == "__main__":
    main()

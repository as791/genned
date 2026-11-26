#!/usr/bin/env python3
"""Fetch AI-edited real photos from MagicBrush for the edited-photo benchmark.

MagicBrush (Zhang et al., NeurIPS 2023; CC-BY-4.0) pairs real MS-COCO photos with
versions edited by an AI tool (DALL-E 2 inpainting) from a text instruction such as
"put a cat on the seat". The test split is used, first turn only: `source_img` is the
real photo and `target_img` the same photo after one AI edit. Writes:

    <out>/real/<n>.png   the real photo
    <out>/ai/<n>.png     the same photo after the AI edit (same <n>: the pairs line up)

Both are decoded and saved as lossless PNG at the edit's size, so the file format and
size can't tell the two apart: only the edit can. Evaluation only (DALL-E 2 output is
never used for training); the photos stay on the CI runner.

Usage:
    python tools/fetch_magicbrush.py --out eval-data/magicbrush --count 250
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image

DATASET = "osunlp/MagicBrush"


def first_turn_pairs(rows: Iterable[dict], count: int) -> Iterator[tuple[bytes, bytes]]:
    """(real bytes, edited bytes) for the first edit of each distinct photo, up to count."""
    seen: set[str] = set()
    for row in rows:
        if len(seen) >= count:
            return
        if int(row["turn_index"]) != 1 or row["img_id"] in seen:
            continue
        source, target = row["source_img"].get("bytes"), row["target_img"].get("bytes")
        if not source or not target:
            continue
        seen.add(row["img_id"])
        yield source, target


def as_matching_png(source: bytes, target: bytes) -> tuple[bytes, bytes]:
    """Both images as RGB PNG at the edited image's size (the editor's output size)."""
    edited = Image.open(io.BytesIO(target)).convert("RGB")
    real = Image.open(io.BytesIO(source)).convert("RGB")
    if real.size != edited.size:
        real = real.resize(edited.size, Image.LANCZOS)
    out = []
    for image in (real, edited):
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        out.append(buffer.getvalue())
    return out[0], out[1]


def write_pairs(pairs: Iterable[tuple[bytes, bytes]], out: Path) -> int:
    (out / "real").mkdir(parents=True, exist_ok=True)
    (out / "ai").mkdir(parents=True, exist_ok=True)
    n = 0
    for source, target in pairs:
        try:
            real, edited = as_matching_png(source, target)
        except OSError:  # an undecodable image: skip the pair, keep the classes balanced
            continue
        (out / "real" / f"{n:05d}.png").write_bytes(real)
        (out / "ai" / f"{n:05d}.png").write_bytes(edited)
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=250, help="Pairs (real + edited)")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    from datasets import Image as ImageFeature, load_dataset

    ds = load_dataset(DATASET, split=args.split, streaming=True)
    for column in ("source_img", "target_img", "mask_img"):
        ds = ds.cast_column(column, ImageFeature(decode=False))
    n = write_pairs(first_turn_pairs(ds, args.count), args.out)
    print(f"Kept {n} MagicBrush pairs (real photo + its AI edit)")
    if n < args.count // 2:
        sys.exit("Too few pairs; check the dataset layout with the Dataset probe workflow.")


if __name__ == "__main__":
    main()

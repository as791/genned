#!/usr/bin/env python3
"""Fetch real face photos from UTKFace by age, as a test-only check for young children.

The app wrongly called photos of young children AI more often than adults' (FairFace,
internal-docs/MODEL.md). UTKFace (in-the-wild faces aged 0-116; non-commercial research
use) is an independent check of that: it is never used for training. The photos are
balanced over age buckets so children and adults can be compared. Writes:

    <out>/real/<n>.jpg     original bytes
    <out>/groups.csv       file,age,gender,ethnicity   (age is the bucket)

tools/group_report.py reads groups.csv. The dataset's files are sorted by age, so the
whole set (about 23,000 rows) is scanned and each bucket filled to its quota. The photos
stay on the CI runner.

Usage:
    python tools/fetch_utkface.py --out eval-data/utkface --count 600
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

DATASET = "nu-delta/utkface"
BUCKETS = ((0, 2, "0-2"), (3, 9, "3-9"), (10, 19, "10-19"), (20, 39, "20-39"), (40, 59, "40-59"), (60, 200, "60+"))


def bucket(age: int) -> str | None:
    for low, high, name in BUCKETS:
        if low <= age <= high:
            return name
    return None


def select(rows: Iterable[dict], count: int) -> Iterator[tuple[bytes, dict]]:
    """(image bytes, labels), up to count // len(BUCKETS) per age bucket."""
    quota = max(1, count // len(BUCKETS))
    kept: Counter = Counter()
    for row in rows:
        if all(kept[name] >= quota for _, _, name in BUCKETS):
            return
        try:
            name = bucket(int(row["age"]))
        except (TypeError, ValueError):
            continue
        data = (row.get("image") or {}).get("bytes")
        if name is None or kept[name] >= quota or not data:
            continue
        kept[name] += 1
        yield data, {"age": name, "gender": row.get("gender") or "", "ethnicity": row.get("ethnicity") or ""}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=600, help="Photos in total, split evenly over age buckets")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    from datasets import Image as ImageFeature, load_dataset

    ds = load_dataset(DATASET, split="train", streaming=True)
    ds = ds.cast_column("image", ImageFeature(decode=False)).shuffle(seed=args.seed, buffer_size=5000)
    real_dir = args.out / "real"
    real_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for data, labels in select(ds, args.count):
        name = f"{len(rows):05d}.jpg"
        (real_dir / name).write_bytes(data)
        rows.append({"file": f"real/{name}", **labels})
    with (args.out / "groups.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "age", "gender", "ethnicity"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Kept {len(rows)} UTKFace photos: {dict(Counter(r['age'] for r in rows))}")
    if len(rows) < args.count // 2:
        sys.exit("Too few photos; check the dataset layout with the Dataset probe workflow.")


if __name__ == "__main__":
    main()

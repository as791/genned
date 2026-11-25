#!/usr/bin/env python3
"""Fetch real face photos from FairFace for the fairness benchmark (real images only).

FairFace (Kärkkäinen & Joo, 2021; CC-BY-4.0) labels each face with an age group, gender
and ethnicity. The sample is balanced over ethnicity x gender, so every group has the same
number of photos. Writes:

    <out>/real/<n>.jpg     original bytes, never re-encoded
    <out>/groups.csv       file,ethnicity,gender,age   (one row per photo)

tools/group_report.py joins groups.csv with the app's scores to report false "Likely AI"
answers per group. Evaluation only: the photos stay on the CI runner.

Usage:
    python tools/fetch_fairface.py --out eval-data/fairface --count 700
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

DATASET = "HuggingFaceM4/FairFace"
CONFIG = "1.25"  # faces cropped with 25% margin: closer to how faces sit in a photo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=700, help="Photos in total, split evenly over the groups")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-scan", type=int, default=20000)
    args = parser.parse_args()

    from datasets import Image as ImageFeature, load_dataset

    ds = load_dataset(DATASET, CONFIG, split=args.split, streaming=True)
    names = {k: ds.features[k].names for k in ("race", "gender", "age")}
    ds = ds.cast_column("image", ImageFeature(decode=False)).shuffle(seed=args.seed, buffer_size=2000)

    groups = [(r, g) for r in names["race"] for g in names["gender"]]
    quota = max(1, args.count // len(groups))
    kept: Counter = Counter()
    real_dir = args.out / "real"
    real_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scanned, row in enumerate(ds):
        if scanned >= args.max_scan or all(kept[g] >= quota for g in groups):
            break
        race, gender, age = names["race"][row["race"]], names["gender"][row["gender"]], names["age"][row["age"]]
        if kept[(race, gender)] >= quota:
            continue
        data = row["image"].get("bytes")
        if not data:
            continue
        name = f"{len(rows):05d}.jpg"
        (real_dir / name).write_bytes(data)
        rows.append({"file": f"real/{name}", "ethnicity": race, "gender": gender, "age": age})
        kept[(race, gender)] += 1

    with (args.out / "groups.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "ethnicity", "gender", "age"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Kept {len(rows)} FairFace photos ({quota} per ethnicity x gender); "
          f"short groups: {[g for g in groups if kept[g] < quota] or 'none'}")
    if len(rows) < args.count // 2:
        sys.exit("Too few photos; check the dataset layout with the Dataset probe workflow.")
    random.seed(args.seed)  # (kept for parity with the other fetchers' --seed)


if __name__ == "__main__":
    main()

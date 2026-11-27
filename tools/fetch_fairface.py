#!/usr/bin/env python3
"""Fetch real face photos from FairFace (real images only), for the fairness benchmark or
as real training examples.

FairFace (Kärkkäinen & Joo, 2021; CC-BY-4.0) labels each face with an age group, gender
and ethnicity. The sample is balanced over ethnicity x gender, so every group has the same
number of photos. With --young-share, that share of the photos comes from ages 0-19
(still balanced over ethnicity x gender), for training on young faces, which the app
wrongly called AI more often. Writes:

    <out>/real/<n>.jpg     original bytes, never re-encoded
    <out>/groups.csv       file,ethnicity,gender,age   (one row per photo)

The benchmark uses the validation split (tools/group_report.py joins groups.csv with the
app's scores); training uses the train split, so the two never share a photo. The photos
stay on the CI runner.

Usage:
    python tools/fetch_fairface.py --out eval-data/fairface --count 700
    python tools/fetch_fairface.py --out data/train/faces --split train --count 1500 --young-share 0.5
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

DATASET = "HuggingFaceM4/FairFace"
CONFIG = "1.25"  # faces cropped with 25% margin: closer to how faces sit in a photo
YOUNG_AGES = frozenset({"0-2", "3-9", "10-19"})


def quotas(races: list[str], genders: list[str], count: int, young_share: float | None) -> dict[tuple, int]:
    """Photos wanted per (ethnicity, gender[, young]) group."""
    pairs = [(r, g) for r in races for g in genders]
    if young_share is None:
        return {p: max(1, count // len(pairs)) for p in pairs}
    young = max(1, round(count * young_share) // len(pairs))
    rest = max(1, (count - young * len(pairs)) // len(pairs))
    return {**{(*p, True): young for p in pairs}, **{(*p, False): rest for p in pairs}}


def select(rows: Iterable[dict], names: dict[str, list[str]], count: int, young_share: float | None,
           max_scan: int) -> Iterator[tuple[bytes, dict]]:
    """(image bytes, labels) for each kept photo, filling every group's quota."""
    wanted = quotas(names["race"], names["gender"], count, young_share)
    kept: Counter = Counter()
    for scanned, row in enumerate(rows):
        if scanned >= max_scan or all(kept[g] >= q for g, q in wanted.items()):
            return
        race, gender, age = names["race"][row["race"]], names["gender"][row["gender"]], names["age"][row["age"]]
        key = (race, gender) if young_share is None else (race, gender, age in YOUNG_AGES)
        if kept[key] >= wanted[key]:
            continue
        data = (row.get("image") or {}).get("bytes")
        if not data:
            continue
        kept[key] += 1
        yield data, {"ethnicity": race, "gender": gender, "age": age}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=700, help="Photos in total, split evenly over the groups")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--young-share", type=float, default=None,
                        help="Share of photos from ages 0-19 (default: no age weighting)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-scan", type=int, default=40000)
    args = parser.parse_args()

    from datasets import Image as ImageFeature, load_dataset

    ds = load_dataset(DATASET, CONFIG, split=args.split, streaming=True)
    names = {k: ds.features[k].names for k in ("race", "gender", "age")}
    ds = ds.cast_column("image", ImageFeature(decode=False)).shuffle(seed=args.seed, buffer_size=2000)

    real_dir = args.out / "real"
    real_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for data, labels in select(ds, names, args.count, args.young_share, args.max_scan):
        name = f"{len(rows):05d}.jpg"
        (real_dir / name).write_bytes(data)
        rows.append({"file": f"real/{name}", **labels})

    with (args.out / "groups.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "ethnicity", "gender", "age"])
        writer.writeheader()
        writer.writerows(rows)
    young = sum(r["age"] in YOUNG_AGES for r in rows)
    print(f"Kept {len(rows)} FairFace {args.split} photos ({young} aged 0-19)")
    if len(rows) < args.count // 2:
        sys.exit("Too few photos; check the dataset layout with the Dataset probe workflow.")


if __name__ == "__main__":
    main()

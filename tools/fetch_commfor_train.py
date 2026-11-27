#!/usr/bin/env python3
"""Sample training images from the Community Forensics training set, for generator variety.

Community Forensics (OwensLab/CommunityForensics, CC-BY-4.0) holds images from thousands of
open-weights generators downloaded from Hugging Face, with real photos (LAION and others).
The app is weakest on unfamiliar generators (Community Forensics-Eval, AUC 0.868), so a
few images from each of many generators help more than many from a few.

Only the "Systematic" and "Manual" subsets are read, and only rows whose own `split` is
"train": "Commercial" holds proprietary generators (never trained on) and "PublicEval" is
kept apart from the evaluation. Any model name matching --exclude-generators is skipped as
well. The files are huge and ordered by generator, so instead of streaming them this reads
a few small batches from row groups spread through each file (parquet allows reading one
row group at a time), keeping at most --max-per-generator images per generator. A file
holds about one generator, so all files are read by default (about 4 s each). Writes:

    <out>/real/<n>.<ext>                 original bytes
    <out>/ai/<generator>/<n>.<ext>       original bytes, generator = model name with / -> _

Images stay on the CI runner. Credit: Park & Owens, "Community Forensics" (CVPR 2025).

Usage:
    python tools/fetch_commfor_train.py --out data/train/cftrain --per-class 2000
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

DATASET = "OwensLab/CommunityForensics"
SUBSETS = ("systematic", "manual")
COLUMNS = ["image_data", "model_name", "label", "split"]


def sniff(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def select(rows: Iterable[dict], per_class: int, max_per_generator: int, exclude: re.Pattern | None,
           counts: Counter, seen: set[str]) -> Iterator[tuple[str, bytes, str]]:
    """(folder key, bytes, extension) for rows to keep: train split only, excluded
    generators skipped, at most max_per_generator per generator, per_class per class.
    `counts` and `seen` carry over between calls (one call per row group)."""
    for row in rows:
        if str(row.get("split", "")).lower() != "train":
            continue
        data = row.get("image_data")
        if not data:
            continue
        is_ai = int(row.get("label", -1)) == 1
        if not is_ai and int(row.get("label", -1)) != 0:
            continue
        model = str(row.get("model_name") or "unknown")
        if is_ai:
            if exclude and exclude.search(model):
                continue
            key = model.replace("/", "_")
            ai_total = sum(n for k, n in counts.items() if k != "real")
            if ai_total >= per_class or counts[key] >= max_per_generator:
                continue
        else:
            key = "real"
            if counts["real"] >= per_class:
                continue
        extension = sniff(data)
        digest = hashlib.sha256(data).hexdigest()
        if extension is None or digest in seen:
            continue
        seen.add(digest)
        counts[key] += 1
        yield key, data, extension


def spread(n: int, k: int) -> list[int]:
    """k indexes spread evenly over range(n)."""
    if n <= 0 or k <= 0:
        return []
    k = min(k, n)
    return sorted({round(i * (n - 1) / max(1, k - 1)) for i in range(k)}) if k > 1 else [n // 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=2000)
    parser.add_argument("--max-per-generator", type=int, default=8)
    parser.add_argument("--files", type=int, default=0,
                        help="Files to read, spread over both subsets (0 = all; each holds about one generator)")
    parser.add_argument("--groups-per-file", type=int, default=20,
                        help="Row groups read per file (each tends to hold one generator)")
    parser.add_argument("--rows-per-group", type=int, default=48,
                        help="Rows read from the start of each chosen row group")
    parser.add_argument("--exclude-generators", default=None)
    args = parser.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = sorted(p for s in SUBSETS for p in fs.glob(f"datasets/{DATASET}/data/{s}/*.parquet"))
    if not files:
        sys.exit("No Systematic/Manual parquet files found; check the layout with the Dataset probe.")
    chosen = files if args.files <= 0 else [files[i] for i in spread(len(files), args.files)]
    exclude = re.compile(args.exclude_generators, re.IGNORECASE) if args.exclude_generators else None
    counts: Counter = Counter()
    seen: set[str] = set()
    for path in chosen:
        with fs.open(path, "rb", block_size=1 << 20) as f:
            # Row groups here can be gigabytes: read a few small batches from the start of
            # each chosen group (no pre-buffering), never a whole group.
            parquet = pq.ParquetFile(f, pre_buffer=False, buffer_size=1 << 20)
            for group in spread(parquet.num_row_groups, args.groups_per_file):
                rows: list[dict] = []
                for batch in parquet.iter_batches(batch_size=16, row_groups=[group], columns=COLUMNS):
                    rows += batch.to_pylist()
                    if len(rows) >= args.rows_per_group:
                        break
                for key, data, ext in select(rows, args.per_class, args.max_per_generator, exclude, counts, seen):
                    folder = args.out / "real" if key == "real" else args.out / "ai" / key
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / f"{counts[key] - 1:05d}.{ext}").write_bytes(data)
        ai = sum(n for k, n in counts.items() if k != "real")
        print(f"{path.rsplit('/', 2)[-2:]}: real {counts['real']}, AI {ai} from {len(counts) - ('real' in counts)} generators",
              flush=True)
        if ai >= args.per_class:
            break
    ai = sum(n for k, n in counts.items() if k != "real")
    print(f"Kept {counts['real']} real and {ai} AI images from {len(counts) - ('real' in counts)} generators "
          "(real photos come from the other training sets)")
    if ai < args.per_class // 4:
        sys.exit("Too few images; check the layout with the Dataset probe workflow.")


if __name__ == "__main__":
    main()

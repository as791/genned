#!/usr/bin/env python3
"""Print a Hugging Face dataset's layout so a fetch preset can be written for it.

Prints splits, features, the repository's files (largest first), the non-image fields of
the first few rows, and value counts of the short text/int columns over the first
--rows rows. Never prints or saves image bytes. Used by .github/workflows/dataset-probe.yml,
because Hugging Face isn't reachable from every development environment.

Usage:
    python tools/probe_dataset.py ComplexDataLab/OpenFake [--config NAME] [--rows 2000]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset")
    parser.add_argument("--config", default=None)
    parser.add_argument("--rows", type=int, default=2000, help="Rows to scan per split for value counts")
    args = parser.parse_args()

    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(args.dataset, files_metadata=True)
    print(f"# {args.dataset}")
    print(f"license/tags: {[t for t in (info.tags or []) if t.startswith(('license:', 'size_categories:'))]}")
    print(f"gated: {getattr(info, 'gated', None)}")
    files = sorted(info.siblings or [], key=lambda f: -(f.size or 0))
    by_ext: dict[str, list[int]] = defaultdict(list)
    for f in files:
        by_ext[os.path.splitext(f.rfilename)[1] or "(none)"].append(f.size or 0)
    print(f"files: {len(files)}; by extension: "
          + ", ".join(f"{ext}={len(s)} ({sum(s) / 1e9:.1f} GB)" for ext, s in sorted(by_ext.items())))
    for f in files[:15]:
        print(f"  {(f.size or 0) / 1e6:10.1f} MB  {f.rfilename}")
    top_dirs = Counter(f.rfilename.split("/")[0] for f in files)
    print(f"top-level entries: {dict(top_dirs.most_common(30))}")

    try:
        print(f"configs: {get_dataset_config_names(args.dataset)}")
    except Exception as e:  # noqa: BLE001 - informational only
        print(f"configs: unavailable ({e})")
    splits = get_dataset_split_names(args.dataset, args.config)
    print(f"splits: {splits}")

    for split in splits:
        ds = load_dataset(args.dataset, args.config, split=split, streaming=True)
        print(f"\n## split {split}\nfeatures: {ds.features}")
        counts: dict[str, Counter] = defaultdict(Counter)
        for i, row in enumerate(ds):
            if i >= args.rows:
                break
            for key, value in row.items():
                if isinstance(value, (str, int, bool)) and len(str(value)) <= 64:
                    counts[key][value] += 1
            if i < 3:
                print(f"row {i}: " + ", ".join(
                    f"{k}={str(v)[:80]!r}" for k, v in row.items() if not isinstance(v, (bytes, dict))
                    and type(v).__name__ not in ("JpegImageFile", "PngImageFile", "Image", "WebPImageFile")))
        print(f"value counts over the first {min(i + 1, args.rows)} rows (columns with <= 60 distinct values):")
        for key, counter in counts.items():
            if len(counter) <= 60:
                print(f"  {key}: {dict(counter.most_common())}")
            else:
                print(f"  {key}: {len(counter)} distinct values")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # same streaming-thread shutdown issue as fetch_eval_data.py

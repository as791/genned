#!/usr/bin/env python3
"""Print a Hugging Face dataset's layout so a fetch preset can be written for it.

Prints splits, features, the repository's files (largest first), the non-image fields of
the first few rows, and value counts of the short text/int columns over the first
--rows rows. Never prints or saves image bytes. Used by .github/workflows/dataset-probe.yml,
because Hugging Face isn't reachable from every development environment.

Usage:
    python tools/probe_dataset.py ComplexDataLab/OpenFake [--config NAME] [--rows 2000]
    python tools/probe_dataset.py "search:laion mobile"   # list matches, then probe the top one
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict


def search(api, query: str) -> str | None:
    """Lists datasets matching every word of `query` (most downloaded first); returns the top id."""
    words = query.split()
    found = {d.id: d for d in api.list_datasets(search=words[0], sort="downloads", direction=-1, limit=200)
             if all(w.lower() in d.id.lower() for w in words)}
    hits = sorted(found.values(), key=lambda d: -(getattr(d, "downloads", 0) or 0))
    print(f"# datasets matching {query!r}: {len(hits)}")
    for d in hits[:20]:
        license_tags = [t for t in (d.tags or []) if t.startswith("license:")]
        print(f"  {d.id}  downloads={getattr(d, 'downloads', None)}  {license_tags}")
    print()
    return hits[0].id if hits else None


SMALL_FILE_BYTES = 300_000_000


def summarize_file(dataset: str, filename: str, rows: int) -> None:
    """Prints the schema, first rows and short-column value counts of one small data file."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(dataset, filename, repo_type="dataset")
    print(f"\n## file {filename}")
    if filename.endswith(".json"):
        text = open(path, encoding="utf-8").read()
        print(text[:3000] + (" ..." if len(text) > 3000 else ""))
        return
    import pandas as pd

    if filename.endswith(".parquet"):
        import pyarrow.parquet as pq

        meta = pq.ParquetFile(path)
        print(f"rows: {meta.metadata.num_rows}\nschema:\n{meta.schema_arrow}")
        df = meta.read_row_group(0).to_pandas().head(rows)
    elif filename.endswith(".jsonl"):
        df = pd.read_json(path, lines=True, nrows=rows)
    else:
        df = pd.read_csv(path, nrows=rows)
    for i in range(min(3, len(df))):
        print(f"row {i}: " + ", ".join(f"{k}={str(v)[:80]!r}" for k, v in df.iloc[i].items()
                                       if not isinstance(v, (bytes, bytearray))))
    print(f"value counts over the first {len(df)} rows (columns with <= 60 distinct values):")
    for col in df.columns:
        try:
            counts = df[col].astype(str).str.slice(0, 64).value_counts()
        except Exception:  # noqa: BLE001
            continue
        if len(counts) <= 60:
            print(f"  {col}: { {k: int(v) for k, v in counts.items()} }")
        else:
            print(f"  {col}: {len(counts)} distinct values; top: { {k: int(v) for k, v in counts.head(15).items()} }")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset")
    parser.add_argument("--config", default=None)
    parser.add_argument("--rows", type=int, default=2000, help="Rows to scan per split for value counts")
    args = parser.parse_args()

    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
    from huggingface_hub import HfApi

    if args.dataset.startswith("search:"):
        args.dataset = search(HfApi(), args.dataset[len("search:"):].strip())
        if args.dataset is None:
            sys.exit("No dataset matched the search.")

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
    # Shard families, e.g. core/train-000NN-of-00032-000MM.parquet -> core/train-*.parquet.
    families: dict[str, list[int]] = defaultdict(list)
    for f in files:
        families[re.sub(r"\d{3,}", "N", f.rfilename)].append(f.size or 0)
    print("file families:")
    for name, sizes in sorted(families.items(), key=lambda kv: -sum(kv[1]))[:40]:
        print(f"  {len(sizes):5d} files {sum(sizes) / 1e9:9.1f} GB  {name}")

    try:
        print(f"configs: {get_dataset_config_names(args.dataset)}")
    except Exception as e:  # noqa: BLE001 - informational only
        print(f"configs: unavailable ({e})")
    try:
        splits = get_dataset_split_names(args.dataset, args.config)
    except Exception as e:  # noqa: BLE001 - e.g. a streaming (MDS) layout `datasets` can't parse
        print(f"splits: unavailable ({type(e).__name__}: {e}); summarizing small data files instead")
        splits = []
        for f in files:
            if f.rfilename.endswith((".parquet", ".json", ".jsonl", ".csv")) and (f.size or 0) <= SMALL_FILE_BYTES:
                summarize_file(args.dataset, f.rfilename, args.rows)
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

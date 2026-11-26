#!/usr/bin/env python3
"""How many files in a benchmark set carry Content Credentials (C2PA) at all.

The Content Credentials check can only help on files that carry a manifest. This counts,
per set and class, the files with an embedded C2PA manifest store: a JUMBF box labelled
"c2pa", the container every format uses (JPEG APP11, PNG caBX, ISO BMFF uuid box, WebP
C2PA chunk, ...). It looks for the container only and doesn't validate it: a count of
"how often could the check run", not of valid credentials. Files are read on the runner
only; only the counts are written.

Usage:
    python tools/credential_coverage.py eval-data/openfake --name openfake --json results/coverage-openfake.json
    python tools/credential_coverage.py --report results/     # table of every coverage-*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".tif", ".tiff", ".heic", ".heif", ".avif", ".mp4", ".mov"}


def has_manifest_store(data: bytes) -> bool:
    """A JUMBF description box ('jumd') whose label is 'c2pa' (the manifest store)."""
    start = 0
    while True:
        i = data.find(b"jumd", start)
        if i < 0:
            return False
        # jumd: 16-byte type UUID, 1 toggle byte, then the NUL-terminated label.
        label = data[i + 4 + 16 + 1: i + 4 + 16 + 1 + 5]
        if label == b"c2pa\x00":
            return True
        start = i + 4


def count(folder: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for cls in ("real", "ai"):
        files = [p for p in sorted((folder / cls).rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS] \
            if (folder / cls).is_dir() else []
        out[cls] = {"files": len(files), "with_credentials": sum(has_manifest_store(p.read_bytes()) for p in files)}
    return out


def report(results: Path) -> str:
    rows = ["## Content Credentials coverage (files carrying a C2PA manifest at all)\n",
            "| Set | Real files | with credentials | AI files | with credentials |", "|---|---|---|---|---|"]
    for path in sorted(results.glob("coverage-*.json")):
        c = json.loads(path.read_text())
        def cell(cls: str) -> str:
            n, k = c["counts"][cls]["files"], c["counts"][cls]["with_credentials"]
            return f"{n} | {k} ({k / n:.1%})" if n else "0 | –"
        rows.append(f"| {c['name']} | {cell('real')} | {cell('ai')} |")
    if len(rows) == 3:
        return "No coverage files."
    rows.append("\nSets whose fetcher re-encodes images (magicbrush) lose any credentials, so 0 there says nothing.")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, nargs="?")
    parser.add_argument("--name", default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None, help="Print the table for a results folder")
    args = parser.parse_args()
    if args.report:
        print(report(args.report))
        return
    if args.folder is None:
        parser.error("give a dataset folder or --report")
    result = {"name": args.name or args.folder.name, "counts": count(args.folder)}
    print(json.dumps(result))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result))


if __name__ == "__main__":
    main()

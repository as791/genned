#!/usr/bin/env python3
"""Fetch AI-inpainted real photos from OpenSDI's test set, per editor (test only).

OpenSDI (nebula/OpenSDI_test, CC-BY-SA-4.0) holds real photos with part of the picture
filled in by an open-weights model ("partial" keys) and whole AI pictures ("entire"). This
fetches partial edits only, one folder per editor, from editors the training data never
uses (SD 2, SD 3, SDXL, Flux): a test of whether the app catches AI edits of part of a
photo from tools it has not seen. The set has no real photos, so it is AI-only: the
report gives the share of edits shown "Likely not AI-generated" per editor.

Only one row group of one partial-only file per editor is read (the files run
partial-only from the indexes below). Writes <out>/ai/<editor>/<n>.<ext> with the
original bytes. Images stay on the CI runner.

Usage:
    python tools/fetch_opensdi.py --out eval-data/opensdi-partial --per-editor 75
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Iterator

DATASET = "nebula/OpenSDI_test"
# A file that holds only partial edits, per editor (Dataset probe, prefix counts of "key").
PARTIAL_FILES = {
    "sd2": "data/sd2-00005-of-00007.parquet",
    "sd3": "data/sd3-00006-of-00010.parquet",
    "sdxl": "data/sdxl-00006-of-00009.parquet",
    "flux": "data/flux-00005-of-00007.parquet",
}


def partial_rows(rows: Iterable[dict], editor: str, count: int) -> Iterator[tuple[bytes, str]]:
    """(image bytes, file extension) for up to `count` partial edits by `editor`."""
    kept = 0
    for row in rows:
        if kept >= count:
            return
        key = str(row.get("key") or "")
        if not key.startswith(f"partial/{editor}/"):
            continue
        image = row.get("image") or {}
        data = image.get("bytes") if isinstance(image, dict) else None
        if not data:
            continue
        kept += 1
        yield data, (Path(key).suffix.lower() or ".png")


def read_rows(path: str, count: int) -> list[dict]:
    """The first row groups of one parquet file, enough for `count` rows."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open(f"datasets/{DATASET}/{path}", "rb") as f:
        parquet = pq.ParquetFile(f)
        rows: list[dict] = []
        for group in range(parquet.num_row_groups):
            rows += parquet.read_row_group(group, columns=["key", "image"]).to_pylist()
            if len(rows) >= count * 2:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-editor", type=int, default=75)
    args = parser.parse_args()

    total = 0
    for editor, path in PARTIAL_FILES.items():
        folder = args.out / "ai" / editor
        folder.mkdir(parents=True, exist_ok=True)
        n = 0
        for data, ext in partial_rows(read_rows(path, args.per_editor), editor, args.per_editor):
            (folder / f"{n:05d}{ext}").write_bytes(data)
            n += 1
        print(f"{editor}: {n} partial edits")
        total += n
    if total < len(PARTIAL_FILES) * args.per_editor // 2:
        sys.exit("Too few edits; check the dataset layout with the Dataset probe workflow.")


if __name__ == "__main__":
    main()

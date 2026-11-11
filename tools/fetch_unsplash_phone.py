#!/usr/bin/env python3
"""Fetches real smartphone photos listed in the Unsplash Lite dataset, for evaluation only.

A second source of real phone photos, independent of LAION-Mobile (which the detector is
also trained on): if a fine-tune only learned "the LAION look", it shows here. The Lite
dataset (25,000 photos) is a table of metadata with a camera make per photo and a link to
the original file on Unsplash's image server; this script keeps the rows whose camera make
is a phone brand, picks a seeded sample, and downloads those originals into <out>/real/.

Terms (github.com/unsplash/datasets, TERMS.md): the Lite dataset may be downloaded and used
internally, including commercially; the data must not be redistributed or published. So the
photos stay on the machine that runs this (a CI runner) and only numbers leave it. They are
never used for training.

Usage:
    python tools/fetch_unsplash_phone.py --out eval-data/unsplash-phone --count 300
"""

from __future__ import annotations

import argparse
import io
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

DATASET = "1aurent/unsplash-lite"
# Brands that only make phones (or whose cameras are not on Unsplash in numbers): Sony and
# Leica are left out because nearly all their photos there come from dedicated cameras.
PHONE_MAKES = (r"apple|samsung|google|xiaomi|redmi|huawei|honor|oneplus|oppo|vivo|realme|motorola|"
               r"lge|lg electronics|nokia|hmd global|nothing|fairphone")
IMAGE_HOST = "https://images.unsplash.com/"
MAX_BYTES = 40_000_000
MIN_SIDE = 512


def phone_rows(rows, make_regex: str = PHONE_MAKES) -> list[tuple[str, str, str]]:
    """(photo id, image url, make in lower case) for rows whose whole camera make is a phone brand."""
    pattern = re.compile(rf"(?:{make_regex})", re.IGNORECASE)
    kept = {}
    for row in rows:
        photo, exif = row.get("photo") or {}, row.get("exif") or {}
        make = str(exif.get("camera_make") or "").strip()
        photo_id, url = str(photo.get("id") or ""), str(photo.get("image_url") or "")
        if photo_id and url.startswith(IMAGE_HOST) and pattern.fullmatch(make):
            kept[photo_id] = (photo_id, url, make.lower())
    return sorted(kept.values())


def sample_order(candidates: list[tuple[str, str, str]], seed: int) -> list[tuple[str, str, str]]:
    """The candidates in a seeded order (the same on every machine)."""
    ordered = sorted(candidates)
    random.Random(seed).shuffle(ordered)
    return ordered


def usable_photo(data: bytes) -> bool:
    """A decodable picture of a sensible size (not an error page or a thumbnail)."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return min(image.size) >= MIN_SIDE
    except Exception:  # noqa: BLE001 - anything undecodable is simply not usable
        return False


def download(url: str, timeout: float = 60.0) -> bytes | None:
    request = urllib.request.Request(url, headers={"User-Agent": "dataset-fetch/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read(MAX_BYTES + 1)
            return data if len(data) <= MAX_BYTES else None
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 410):
                return None  # removed from Unsplash
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(2 * (attempt + 1))
    return None


def fetch(rows, out: Path, count: int, seed: int, make_regex: str = PHONE_MAKES, get=download) -> Counter:
    candidates = phone_rows(rows, make_regex)
    print(f"{len(candidates)} photos with a phone camera make; sampling {count} (seed {seed})")
    real_dir = out / "real"
    real_dir.mkdir(parents=True, exist_ok=True)
    makes: Counter = Counter()
    skipped: Counter = Counter()
    for photo_id, url, make in sample_order(candidates, seed):
        if sum(makes.values()) >= count:
            break
        data = get(url)
        if data is None:
            skipped["not downloadable"] += 1
            continue
        if not usable_photo(data):
            skipped["not a usable picture"] += 1
            continue
        (real_dir / f"{re.sub(r'[^A-Za-z0-9_-]', '_', photo_id)}.jpg").write_bytes(data)
        makes[make] += 1
        if sum(makes.values()) % 50 == 0:
            print(f"  saved {sum(makes.values())}/{count}", flush=True)
    print(f"Kept {sum(makes.values())} phone photos; by make: {dict(makes.most_common())}")
    print(f"Skipped: {dict(skipped)}")
    return makes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--make-regex", default=PHONE_MAKES, help="Camera makes to keep (whole value, any case)")
    args = parser.parse_args()

    from datasets import load_dataset

    rows = load_dataset(args.dataset, split="train", streaming=True).select_columns(["photo", "exif"])
    makes = fetch(rows, args.out, args.count, args.seed, args.make_regex)
    if sum(makes.values()) < min(args.count, 50):
        sys.exit("Too few phone photos could be fetched.")


if __name__ == "__main__":
    main()

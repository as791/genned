#!/usr/bin/env python3
"""Fetches a random sample of real phone photos from a MosaicML-streaming (MDS) dataset on
Hugging Face, e.g. LAION-Mobile, into <out>/real/ for a real-only benchmark.

`datasets` can't read MDS shards, and they are 2.6 GB each, so this reads single samples
with HTTP range requests: a shard starts with its sample count and an offset table, so one
small read finds a sample and one more fetches it. The image bytes are saved untouched
(EXIF included), as a phone would share them.

Only photos whose EXIF camera make is a phone brand are kept (--make-regex), so the set is
what the app sees from a phone's gallery. The counts per make are printed; no image or file
name leaves the runner.

Usage:
    python tools/fetch_mds_photos.py --dataset sumathiselvan/LAION-Mobile-streaming \
        --out eval-data/phone-photos --count 300
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import struct
import sys
from collections import Counter
from pathlib import Path

PHONE_MAKES = (r"apple|samsung|google|xiaomi|redmi|huawei|honor|oneplus|oppo|vivo|realme|motorola|"
               r"lge|lg electronics|sony|nokia|hmd|asus|zte|nothing|fairphone|tecno|infinix|meizu")
EXIF_MAKE = 271


def decode_sample(data: bytes, names: list[str], encodings: list[str], sizes: list[int | None]) -> dict:
    """Mirrors streaming's MDSReader.decode_sample: the variable-size columns' lengths come
    first (uint32 each, in column order), then every column's bytes in column order."""
    lengths, index = [], 0
    for size in sizes:
        if size:
            lengths.append(size)
        else:
            lengths.append(struct.unpack_from("<I", data, index)[0])
            index += 4
    sample = {}
    for name, encoding, length in zip(names, encodings, lengths):
        value = data[index:index + length]
        index += length
        if encoding == "str":
            sample[name] = value.decode("utf-8")
        elif encoding == "int":
            sample[name] = struct.unpack("<q", value)[0]
        else:
            sample[name] = value
    return sample


def sample_range(header: bytes, idx: int) -> tuple[int, int]:
    """(begin, end) byte range of sample `idx` from a shard's offset table (uint32s after the count)."""
    return struct.unpack_from("<II", header, 4 + 4 * idx)


def camera_make(image_bytes: bytes) -> str | None:
    from PIL import Image

    try:
        make = Image.open(io.BytesIO(image_bytes)).getexif().get(EXIF_MAKE)
    except Exception:  # noqa: BLE001 - unreadable image or EXIF: not a usable phone photo
        return None
    return str(make).strip("\x00 ").strip() or None if make else None


class HubShards:
    """Range reads from a dataset repo's files."""

    def __init__(self, dataset: str):
        from huggingface_hub import HfApi, HfFileSystem

        self.fs = HfFileSystem()
        self.root = f"datasets/{dataset}"
        self.files = set(HfApi().list_repo_files(dataset, repo_type="dataset"))

    def read(self, path: str, start: int | None = None, end: int | None = None) -> bytes:
        return self.fs.cat_file(f"{self.root}/{path}", start=start, end=end)


def fetch(source, out: Path, count: int, make_regex: str, seed: int, max_tries: int) -> Counter:
    index = json.loads(source.read("streaming/index.json"))
    listed = [s for s in index["shards"] if s.get("format") == "mds" and not s.get("compression")]
    # A partial mirror can list shards it never uploaded.
    shards = [s for s in listed if "streaming/" + s["raw_data"]["basename"] in source.files]
    print(f"index lists {len(listed)} uncompressed MDS shards; {len(shards)} are in the repository")
    if not shards:
        sys.exit("No uncompressed MDS shards from streaming/index.json are in the repository.")
    population = [(shard_no, i) for shard_no, s in enumerate(shards) for i in range(s["samples"])]
    print(f"{len(shards)} shards, {len(population)} samples; sampling {count} phone photos (seed {seed})")
    rng = random.Random(seed)
    picks = rng.sample(population, min(len(population), max_tries))

    real_dir = out / "real"
    real_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(make_regex, re.IGNORECASE)
    makes, skipped = Counter(), Counter()
    headers: dict[int, bytes] = {}
    for tried, (shard_no, idx) in enumerate(picks, 1):
        if sum(makes.values()) >= count:
            break
        shard = shards[shard_no]
        path = "streaming/" + shard["raw_data"]["basename"]
        if shard_no not in headers:  # the offset table, a few KB per shard
            headers[shard_no] = source.read(path, 0, 4 * (shard["samples"] + 2))
        begin, end = sample_range(headers[shard_no], idx)
        try:
            sample = decode_sample(source.read(path, begin, end), shard["column_names"],
                                   shard["column_encodings"], shard["column_sizes"])
        except Exception as e:  # noqa: BLE001 - a network or format problem with one sample
            skipped[f"read error ({type(e).__name__})"] += 1
            continue
        image = sample.get("image")
        if not isinstance(image, bytes) or not image:
            skipped["no image bytes"] += 1
            continue
        if sample.get("sha256") and hashlib.sha256(image).hexdigest() != sample["sha256"]:
            skipped["sha256 mismatch"] += 1
            continue
        if str(sample.get("label", "Real")).lower() != "real":
            skipped["not labelled real"] += 1
            continue
        make = camera_make(image)
        if make is None:
            skipped["no EXIF camera make"] += 1
            continue
        if not pattern.search(make):
            skipped["camera, not a phone brand"] += 1
            continue
        extension = sample.get("extension") or ".jpg"
        (real_dir / f"{sum(makes.values()):05d}{extension}").write_bytes(image)
        makes[make.split()[0].lower()] += 1
        if tried % 50 == 0:
            print(f"  tried {tried}: kept {sum(makes.values())}/{count}")
    print(f"Kept {sum(makes.values())} phone photos (tried {tried}); by make: {dict(makes.most_common())}")
    print(f"Skipped: {dict(skipped)}")
    return makes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--make-regex", default=PHONE_MAKES, help="EXIF Make values to keep")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-tries", type=int, default=None, help="Samples to try (default: 4x --count)")
    args = parser.parse_args()
    makes = fetch(HubShards(args.dataset), args.out, args.count, args.make_regex, args.seed,
                  args.max_tries or 4 * args.count)
    if sum(makes.values()) < args.count // 2:
        sys.exit(f"Only {sum(makes.values())} phone photos found; expected at least {args.count // 2}.")


if __name__ == "__main__":
    main()

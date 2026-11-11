"""Unit tests for tools/fetch_mds_photos.py (no downloads: the shards are built in memory).

Run: python -m unittest discover -s tools/tests
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch_mds_photos as mds  # noqa: E402

COLUMNS = ["image", "label", "sha256", "extension"]
ENCODINGS = ["bytes", "str", "str", "str"]


def phone_jpeg(shard_no: int, make: str = "Apple") -> bytes:
    """A tiny JPEG whose width records which shard it came from, with an EXIF camera make."""
    image = Image.new("RGB", (10 + shard_no, 8), (shard_no * 7 % 255, 90, 160))
    exif = Image.Exif()
    exif[mds.EXIF_MAKE] = make
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", exif=exif)
    return buffer.getvalue()


def encode_sample(image: bytes) -> bytes:
    values = [image, b"Real", hashlib.sha256(image).hexdigest().encode(), b".jpg"]
    return b"".join(struct.pack("<I", len(v)) for v in values) + b"".join(values)


def build_shard(samples: list[bytes]) -> bytes:
    """count, then the offset table (count + 1 absolute offsets), then the samples."""
    header_size = 4 + 4 * (len(samples) + 1)
    offsets, position = [], header_size
    for sample in samples:
        offsets.append(position)
        position += len(sample)
    offsets.append(position)
    return struct.pack("<I", len(samples)) + b"".join(struct.pack("<I", o) for o in offsets) + b"".join(samples)


class FakeSource:
    """Stands in for HubShards: files held in memory, read by byte range."""

    def __init__(self, shard_numbers: list[int], samples_per_shard: int = 6, make: str = "Apple"):
        self.blobs: dict[str, bytes] = {}
        shards = []
        for number in shard_numbers:
            name = f"shard.{number:05d}.mds"
            self.blobs["streaming/" + name] = build_shard(
                [encode_sample(phone_jpeg(number, make)) for _ in range(samples_per_shard)],
            )
            shards.append({
                "format": "mds", "compression": None, "raw_data": {"basename": name},
                "samples": samples_per_shard, "column_names": COLUMNS, "column_encodings": ENCODINGS,
                "column_sizes": [None] * len(COLUMNS),
            })
        self.blobs["streaming/index.json"] = json.dumps({"shards": shards}).encode()
        self.files = set(self.blobs)

    def read(self, path: str, start: int | None = None, end: int | None = None) -> bytes:
        return self.blobs[path][start:end]


def shards_fetched(source: FakeSource, split: str, count: int = 1000) -> set[int]:
    with tempfile.TemporaryDirectory() as out:
        mds.fetch(source, Path(out), count, mds.PHONE_MAKES, seed=1, max_tries=10_000, split=split)
        widths = set()
        for path in (Path(out) / "real").iterdir():
            with Image.open(path) as image:
                widths.add(image.width - 10)
        return widths


class SplitTest(unittest.TestCase):
    def test_shard_number_reads_the_file_name(self):
        self.assertEqual(mds.shard_number("shard.00012.mds"), 12)
        self.assertEqual(mds.shard_number("shard.00000.mds"), 0)
        # No number in the name: still a stable assignment.
        self.assertEqual(mds.shard_number("odd-name"), mds.shard_number("odd-name"))

    def test_train_and_test_shards_never_overlap_and_cover_everything(self):
        names = [f"shard.{n:05d}.mds" for n in range(40)]
        train = {n for n in names if mds.in_split(n, "train")}
        test = {n for n in names if mds.in_split(n, "test")}
        self.assertFalse(train & test)
        self.assertEqual(train | test, set(names))
        self.assertEqual(len(test), 40 // mds.SPLIT_MOD)
        self.assertTrue(all(mds.in_split(n, "all") for n in names))

    def test_fetch_draws_only_from_its_split(self):
        source = FakeSource(list(range(12)))
        test = shards_fetched(source, "test")
        train = shards_fetched(source, "train")
        self.assertEqual(test, {0, 5, 10})
        self.assertEqual(train, set(range(12)) - {0, 5, 10})
        self.assertEqual(shards_fetched(source, "all"), set(range(12)))

    def test_split_survives_shards_added_to_the_mirror(self):
        before = shards_fetched(FakeSource([0, 1, 2, 5]), "test")
        after = shards_fetched(FakeSource([0, 1, 2, 3, 4, 5, 6]), "test")
        self.assertEqual(before, {0, 5})
        self.assertEqual(after, {0, 5})

    def test_non_phone_cameras_are_still_skipped(self):
        with tempfile.TemporaryDirectory() as out:
            makes = mds.fetch(FakeSource([0, 5], make="Canon"), Path(out), 10, mds.PHONE_MAKES, 1, 1000, "test")
        self.assertEqual(sum(makes.values()), 0)


if __name__ == "__main__":
    unittest.main()

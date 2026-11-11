"""Tests for the dataset probe's label reading (no network: tars are built in memory)."""
from __future__ import annotations

import io
import json
import sys
import tarfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe_dataset  # noqa: E402


def make_tar(records: list[dict], media_bytes: int = 2000) -> io.BytesIO:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for i, record in enumerate(records):
            for name, data in ((f"{i:05d}.mp4", b"\x00" * media_bytes), (f"{i:05d}.json", json.dumps(record).encode())):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    buffer.seek(0)
    return buffer


class NonSeekable(io.RawIOBase):
    """A read-only stream that refuses to seek, like a network download."""

    def __init__(self, data: bytes):
        self._inner = io.BytesIO(data)

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        return self._inner.readinto(b)

    def seekable(self) -> bool:
        return False


class ProbeDatasetTest(unittest.TestCase):
    def test_labels_are_read_from_a_stream_without_touching_the_media(self):
        records = [{"filename": f"v{i}.mp4", "generator": "wan" if i % 2 else "real", "is_ai_generated": bool(i % 2)}
                   for i in range(10)]
        stream = NonSeekable(make_tar(records).getvalue())
        read, extensions, sizes = probe_dataset.scan_tar_labels(stream, rows=6)
        self.assertEqual(read, records[:6])
        self.assertEqual(extensions[".json"], 6)
        self.assertEqual(sizes[".mp4"], [2000] * 6)

    def test_a_broken_json_member_is_skipped(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            for name, data in (("a.json", b"\xff\xfe not json"), ("b.json", b'{"label": "x"}'), ("c.json", b"[1, 2]")):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        buffer.seek(0)
        read, _, _ = probe_dataset.scan_tar_labels(buffer, rows=10)
        self.assertEqual(read, [{"label": "x"}])

    def test_nested_fields_are_flattened_one_level(self):
        self.assertEqual(
            probe_dataset.flat_fields({"json": {"label": "ai", "generator": "wan"}, "__key__": "0"}),
            {"json.label": "ai", "json.generator": "wan", "__key__": "0"},
        )

    def test_spread_picks_first_and_last_and_never_repeats(self):
        names = [f"shard-{i:03d}.tar" for i in range(84)]
        chosen = probe_dataset.spread_files(names, 8)
        self.assertEqual(len(chosen), 8)
        self.assertEqual(len(set(chosen)), 8)
        self.assertEqual((chosen[0], chosen[-1]), (names[0], names[-1]))
        self.assertEqual(probe_dataset.spread_files(names[:3], 8), names[:3])
        self.assertEqual(probe_dataset.spread_files(names, 1), names[:1])


if __name__ == "__main__":
    unittest.main()

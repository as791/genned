"""Unit tests for tools/fetch_video_frames.py (in-memory tars, no downloads, no video decoding).

Run: python -m unittest discover -s tools/tests
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch_video_frames as fvf  # noqa: E402


def make_tar(clips: list[tuple[str, dict | bytes, bytes | None]], label_first: bool = True) -> bytes:
    """A tar of clips: (name, label record or raw label bytes, video bytes or None for no video)."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, record, video in clips:
            label = record if isinstance(record, bytes) else json.dumps(record).encode()
            members = [(f"{name}.json", label)] + ([(f"{name}.mp4", video)] if video is not None else [])
            for member_name, data in members if label_first else reversed(members):
                info = tarfile.TarInfo(member_name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class NonSeekable(io.RawIOBase):
    """A stream like a network response: read-only, no seeking."""

    def __init__(self, data: bytes):
        self._inner = io.BytesIO(data)

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        return self._inner.readinto(b)


def fake(generator: str) -> dict:
    return {"label": "fake", "is_ai_generated": "1", "generator": generator}


REAL = {"label": "real", "is_ai_generated": "0", "generator": "real"}


def five_frames(video: bytes) -> list[bytes]:
    return [video + bytes([i]) for i in range(5)]


class SplitTest(unittest.TestCase):
    def test_shards_split_by_number(self):
        self.assertEqual(fvf.shard_number("data/dataset_shard_000012.tar"), 12)
        self.assertTrue(fvf.in_split("dataset_shard_000010.tar", "test"))
        self.assertTrue(fvf.in_split("dataset_shard_000011.tar", "val"))
        for n in (12, 13, 14):
            self.assertTrue(fvf.in_split(f"dataset_shard_{n:06d}.tar", "train"))
        self.assertFalse(fvf.in_split("dataset_shard_000010.tar", "train"))
        self.assertTrue(fvf.in_split("dataset_shard_000010.tar", "all"))
        with self.assertRaises(ValueError):
            fvf.shard_number("no-number.tar")


class WantedTest(unittest.TestCase):
    def test_real_and_open_generators_are_kept(self):
        self.assertEqual(fvf.wanted(REAL, "pika"), ("real", "real"))
        self.assertEqual(fvf.wanted(fake("SVD"), "pika"), ("ai", "svd"))
        self.assertEqual(fvf.wanted(fake("Open Sora/v1"), "pika"), ("ai", "open_sora_v1"))

    def test_excluded_and_unlabelled_clips_are_skipped(self):
        self.assertIsNone(fvf.wanted(fake("Pika"), "pika"))
        self.assertIsNone(fvf.wanted(fake("PIKA-1.0"), "pika"))
        self.assertIsNone(fvf.wanted({"label": "fake", "generator": ""}, "pika"))
        self.assertIsNone(fvf.wanted({"label": "fake", "generator": "real"}, "pika"))
        self.assertIsNone(fvf.wanted({"generator": "SVD"}, "pika"))

    def test_only_listed_generators_pass_the_allow_list(self):
        for name in ("SVD", "SEINE", "OpenSora", "ST2V", "VideoCrafter", "Latte", "COG", "DynamicCrafter", "ZeroScope"):
            self.assertEqual(fvf.wanted(fake(name), "pika", fvf.OPEN_GENERATORS), ("ai", name.lower()))
        for name in ("Pika", "Sora", "Gen2", "SVD-XT-new"):
            self.assertIsNone(fvf.wanted(fake(name), "pika", fvf.OPEN_GENERATORS))
        self.assertEqual(fvf.wanted(REAL, "pika", fvf.OPEN_GENERATORS), ("real", "real"))


class IterClipsTest(unittest.TestCase):
    def test_label_and_video_pair_up_in_either_order(self):
        clips = [("./Real/real_1", REAL, b"v1"), ("x/svd_2", fake("SVD"), b"v2")]
        for label_first in (True, False):
            found = list(fvf.iter_clips(NonSeekable(make_tar(clips, label_first))))
            self.assertEqual([(k, r["label"], v) for k, r, v in found],
                             [("./Real/real_1", "real", b"v1"), ("x/svd_2", "fake", b"v2")])

    def test_broken_labels_and_lone_members_are_skipped(self):
        clips = [("a", b"not json", b"va"), ("b", REAL, None), ("c", b"[1, 2]", b"vc"), ("d", fake("SVD"), b"vd")]
        found = list(fvf.iter_clips(io.BytesIO(make_tar(clips))))
        self.assertEqual([k for k, _, _ in found], ["d"])


class FetchTest(unittest.TestCase):
    def shards(self, *tars: bytes):
        return [(f"dataset_shard_{i + 2:06d}.tar", lambda data=data: io.BytesIO(data)) for i, data in enumerate(tars)]

    def test_frames_land_in_the_training_layout(self):
        tar = make_tar([("r1", REAL, b"R1"), ("s1", fake("SVD"), b"S1"), ("p1", fake("Pika"), b"P1"),
                        ("c1", fake("COG"), b"C1")])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            per_class, per_generator = fvf.fetch(self.shards(tar), out, frames_per_class=100, max_per_generator=100,
                                                 frames_per_clip=3, exclude_regex="pika", seed=1, sampler=five_frames)
            self.assertEqual(dict(per_class), {"real": 3, "ai": 6})
            self.assertEqual(dict(per_generator), {"real": 3, "svd": 3, "cog": 3})
            self.assertEqual(sorted(d.name for d in (out / "ai").iterdir()), ["cog", "svd"])
            self.assertEqual(len(list((out / "real").glob("*.jpg"))), 3)
            self.assertEqual(fvf.excluded_folders(out, "pika"), [])
            # Each file holds one of that clip's own sampled frames.
            for path in (out / "ai" / "svd").iterdir():
                self.assertEqual(path.read_bytes(), b"S1" + bytes([int(path.stem.rsplit("_", 1)[1])]))

    def test_caps_stop_a_class_and_a_generator(self):
        first = make_tar([(f"s{i}", fake("SVD"), b"S") for i in range(4)] + [("r1", REAL, b"R")])
        second = make_tar([("c1", fake("COG"), b"C"), ("r2", REAL, b"R"), ("r3", REAL, b"R")])
        with tempfile.TemporaryDirectory() as tmp:
            per_class, per_generator = fvf.fetch(self.shards(first, second), Path(tmp), frames_per_class=4,
                                                 max_per_generator=2, frames_per_clip=2, exclude_regex="pika",
                                                 seed=1, sampler=five_frames)
            self.assertEqual(per_generator["svd"], 2)  # one clip of two frames, then the cap
            self.assertEqual(per_generator["cog"], 2)
            self.assertEqual(per_class["real"], 4)  # r1 and r2; r3 is over the class size

    def test_undecodable_clips_and_broken_shards_do_not_end_the_run(self):
        good = make_tar([("r1", REAL, b"R"), ("s1", fake("SVD"), b"S")])
        broken = make_tar([("r2", REAL, b"R" * 4000)])[:1500]
        sampler = lambda video: [] if video == b"R" else five_frames(video)  # noqa: E731
        with tempfile.TemporaryDirectory() as tmp:
            per_class, _ = fvf.fetch(self.shards(broken, good), Path(tmp), frames_per_class=10, max_per_generator=10,
                                     frames_per_clip=5, exclude_regex="pika", seed=1, sampler=sampler)
            self.assertEqual(dict(per_class), {"ai": 5})

    def test_an_excluded_folder_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "ai" / "pika").mkdir(parents=True)
            (out / "ai" / "svd").mkdir()
            self.assertEqual(fvf.excluded_folders(out, "pika"), ["pika"])
            self.assertEqual(fvf.excluded_folders(out, ""), [])

    def test_the_same_seed_picks_the_same_frames(self):
        tar = make_tar([("r1", REAL, b"R1"), ("s1", fake("SVD"), b"S1")])
        names = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                fvf.fetch(self.shards(tar), Path(tmp), 10, 10, 2, "pika", seed=7, sampler=five_frames)
                names.append(sorted(str(p.relative_to(tmp)) for p in Path(tmp).rglob("*.jpg")))
        self.assertEqual(names[0], names[1])
        self.assertEqual(len(names[0]), 4)


if __name__ == "__main__":
    unittest.main()

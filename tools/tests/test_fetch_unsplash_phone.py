"""Tests for the Unsplash phone-photo fetcher (no network: rows and downloads are faked)."""
from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch_unsplash_phone as fup  # noqa: E402


def row(photo_id: str, make, url: str | None = None) -> dict:
    return {"photo": {"id": photo_id, "image_url": url or f"{fup.IMAGE_HOST}photo-{photo_id}"},
            "exif": {"camera_make": make}}


def jpeg(side: int = 800) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (side, side), (120, 90, 60)).save(buffer, "JPEG")
    return buffer.getvalue()


class UnsplashPhoneTest(unittest.TestCase):
    def test_only_phone_makes_are_kept(self):
        rows = [row("a", "Apple"), row("b", "samsung"), row("c", "SAMSUNG"), row("d", "Canon"),
                row("e", "SONY"), row("f", None), row("g", "nan"), row("h", "Google"),
                row("i", "DJI Mavic Pro"), row("j", "Apple Computer"), row("k", " HUAWEI ")]
        kept = fup.phone_rows(rows)
        self.assertEqual([photo_id for photo_id, _, _ in kept], ["a", "b", "c", "h", "k"])
        self.assertEqual({make for _, _, make in kept}, {"apple", "samsung", "google", "huawei"})

    def test_links_to_other_hosts_and_repeated_ids_are_dropped(self):
        rows = [row("a", "Apple", "https://example.com/a.jpg"), row("b", "Apple"), row("b", "Apple")]
        self.assertEqual([photo_id for photo_id, _, _ in fup.phone_rows(rows)], ["b"])

    def test_the_sample_is_the_same_whatever_order_the_rows_arrive_in(self):
        rows = [row(f"p{i:03d}", "Apple") for i in range(40)]
        first = fup.sample_order(fup.phone_rows(rows), seed=7)
        second = fup.sample_order(fup.phone_rows(list(reversed(rows))), seed=7)
        self.assertEqual(first, second)
        self.assertNotEqual(first, fup.sample_order(fup.phone_rows(rows), seed=8))

    def test_fetch_skips_what_cannot_be_used_and_stops_at_the_count(self):
        rows = [row(f"p{i:03d}", "Apple") for i in range(30)]
        order = [photo_id for photo_id, _, _ in fup.sample_order(fup.phone_rows(rows), seed=1)]
        gone, tiny, page = order[0], order[1], order[2]

        def fake_get(url: str):
            photo_id = url.rsplit("photo-", 1)[1]
            if photo_id == gone:
                return None
            if photo_id == tiny:
                return jpeg(100)
            if photo_id == page:
                return b"<html>not found</html>"
            return jpeg()

        with tempfile.TemporaryDirectory() as tmp:
            makes = fup.fetch(rows, Path(tmp), count=5, seed=1, get=fake_get)
            saved = sorted(p.stem for p in (Path(tmp) / "real").iterdir())
        self.assertEqual(makes, {"apple": 5})
        self.assertEqual(saved, sorted(order[3:8]))


if __name__ == "__main__":
    unittest.main()

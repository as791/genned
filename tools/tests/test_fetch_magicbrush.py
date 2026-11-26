"""tools/fetch_magicbrush.py on fake rows (no downloads)."""
import io
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch_magicbrush as mb  # noqa: E402


def encoded(size, colour, fmt):
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, fmt)
    return buffer.getvalue()


def row(img_id, turn, source=None, target=None):
    return {"img_id": img_id, "turn_index": turn,
            "source_img": {"bytes": source or encoded((640, 480), (10, 20, 30), "JPEG")},
            "target_img": {"bytes": target or encoded((512, 512), (200, 20, 30), "PNG")}}


class FirstTurnPairsTest(unittest.TestCase):
    def test_keeps_one_first_turn_per_photo_up_to_count(self):
        rows = [row("a", 1), row("a", 2), row("b", 2), row("b", 1), row("a", 1), row("c", 1), row("d", 1)]
        pairs = list(mb.first_turn_pairs(rows, count=2))
        self.assertEqual(len(pairs), 2)  # a and b; the repeated a and the second turns are skipped

    def test_rows_without_bytes_are_skipped(self):
        rows = [{"img_id": "a", "turn_index": 1, "source_img": {"bytes": None}, "target_img": {"bytes": b"x"}}]
        self.assertEqual(list(mb.first_turn_pairs(rows, 5)), [])


class MatchingFormatTest(unittest.TestCase):
    def test_real_and_edited_share_format_and_size(self):
        # The real photo arrives as a 640x480 JPEG and the edit as a 512x512 PNG: written
        # out, both must be PNG at 512x512, so format and size can't give the answer.
        with tempfile.TemporaryDirectory() as tmp:
            n = mb.write_pairs(mb.first_turn_pairs([row("a", 1), row("b", 1)], 5), Path(tmp))
            self.assertEqual(n, 2)
            for name in ("00000.png", "00001.png"):
                real, edited = Image.open(Path(tmp, "real", name)), Image.open(Path(tmp, "ai", name))
                self.assertEqual((real.format, edited.format), ("PNG", "PNG"))
                self.assertEqual(real.size, edited.size)
                self.assertEqual(real.size, (512, 512))

    def test_an_undecodable_pair_is_dropped_from_both_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            n = mb.write_pairs([(b"not an image", encoded((8, 8), (0, 0, 0), "PNG"))], Path(tmp))
            self.assertEqual(n, 0)
            self.assertEqual(list(Path(tmp, "real").iterdir()), [])
            self.assertEqual(list(Path(tmp, "ai").iterdir()), [])


if __name__ == "__main__":
    unittest.main()

"""tools/fetch_opensdi.py row selection (no downloads)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch_opensdi as fo  # noqa: E402


class PartialRowsTest(unittest.TestCase):
    def test_keeps_only_this_editors_partial_edits(self):
        rows = [
            {"key": "entire/sdxl/fake/1.jpg", "image": {"bytes": b"a"}},
            {"key": "partial/sdxl/fake/2.png", "image": {"bytes": b"b"}},
            {"key": "partial/flux/fake/3.png", "image": {"bytes": b"c"}},
            {"key": "partial/sdxl/fake/4.jpg", "image": {"bytes": None}},
            {"key": "partial/sdxl/fake/5.JPG", "image": {"bytes": b"e"}},
        ]
        kept = list(fo.partial_rows(rows, "sdxl", 10))
        self.assertEqual(kept, [(b"b", ".png"), (b"e", ".jpg")])

    def test_count_is_a_cap(self):
        rows = [{"key": f"partial/sd3/fake/{i}.png", "image": {"bytes": b"x"}} for i in range(10)]
        self.assertEqual(len(list(fo.partial_rows(rows, "sd3", 4))), 4)

    def test_every_editor_has_a_file(self):
        self.assertEqual(set(fo.PARTIAL_FILES), {"sd2", "sd3", "sdxl", "flux"})


if __name__ == "__main__":
    unittest.main()

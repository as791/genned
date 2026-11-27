"""tools/fetch_commfor_train.py selection (no downloads)."""
import re
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch_commfor_train as fc  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n"


def row(model, label, split="train", data=None, n=0):
    return {"model_name": model, "label": label, "split": split, "image_data": data or PNG + bytes([n % 256, n // 256])}


class SelectTest(unittest.TestCase):
    def run_select(self, rows, per_class=100, max_per_generator=3, exclude=None):
        counts, seen = Counter(), set()
        return list(fc.select(rows, per_class, max_per_generator, exclude, counts, seen)), counts

    def test_caps_each_generator_and_keeps_only_train_rows(self):
        rows = [row("a/m1", 1, n=i) for i in range(10)] + [row("b/m2", 1, n=100 + i) for i in range(10)]
        rows += [row("a/m1", 1, split="eval", n=500)]
        kept, counts = self.run_select(rows)
        self.assertEqual(counts["a_m1"], 3)
        self.assertEqual(counts["b_m2"], 3)
        self.assertEqual(len(kept), 6)

    def test_real_rows_and_per_class_cap(self):
        rows = [row("real-src", 0, n=i) for i in range(10)]
        kept, counts = self.run_select(rows, per_class=4)
        self.assertEqual(counts["real"], 4)
        self.assertTrue(all(k == "real" for k, _, _ in kept))

    def test_excluded_generators_duplicates_and_bad_bytes_are_skipped(self):
        rows = [row("acme/midjourney-like", 1, n=1), row("ok/m", 1, n=2), row("ok/m", 1, n=2),
                row("ok/m", 1, data=b"not an image"), row("ok/m", 7, n=3)]
        kept, counts = self.run_select(rows, exclude=re.compile("midjourney"))
        self.assertEqual([k for k, _, _ in kept], ["ok_m"])


class SpreadTest(unittest.TestCase):
    def test_spread(self):
        self.assertEqual(fc.spread(10, 3), [0, 4, 9])
        self.assertEqual(fc.spread(2, 5), [0, 1])
        self.assertEqual(fc.spread(0, 3), [])
        self.assertEqual(fc.spread(7, 1), [3])


if __name__ == "__main__":
    unittest.main()

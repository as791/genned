"""tools/fetch_utkface.py selection, on fake rows (no downloads)."""
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch_utkface as fu  # noqa: E402


class BucketTest(unittest.TestCase):
    def test_buckets(self):
        self.assertEqual(fu.bucket(1), "0-2")
        self.assertEqual(fu.bucket(9), "3-9")
        self.assertEqual(fu.bucket(10), "10-19")
        self.assertEqual(fu.bucket(45), "40-59")
        self.assertEqual(fu.bucket(116), "60+")


class SelectTest(unittest.TestCase):
    def test_even_over_buckets_even_when_rows_are_sorted_by_age(self):
        # Sorted like the dataset: all the young ones first, so a quota is needed per bucket.
        rows = [{"age": a, "gender": "Male", "ethnicity": "White", "image": {"bytes": b"x"}}
                for a in sorted(list(range(1, 90)) * 20)]
        kept = list(fu.select(rows, count=60))
        self.assertEqual(Counter(l["age"] for _, l in kept), {name: 10 for _, _, name in fu.BUCKETS})

    def test_bad_rows_are_skipped(self):
        rows = [{"age": "x", "image": {"bytes": b"x"}}, {"age": 5, "image": {"bytes": None}}, {"age": 5, "image": None}]
        self.assertEqual(list(fu.select(rows, 12)), [])


if __name__ == "__main__":
    unittest.main()

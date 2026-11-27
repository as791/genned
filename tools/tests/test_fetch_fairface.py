"""tools/fetch_fairface.py selection, on fake rows (no downloads)."""
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch_fairface as ff  # noqa: E402

NAMES = {"race": ["A", "B"], "gender": ["F", "M"], "age": ["0-2", "3-9", "10-19", "20-29", "30-39"]}


def rows(n=4000):
    for i in range(n):
        yield {"race": i % 2, "gender": (i // 2) % 2, "age": (i // 4) % 5, "image": {"bytes": b"x%d" % i}}


class SelectTest(unittest.TestCase):
    def test_balanced_over_ethnicity_and_gender(self):
        kept = list(ff.select(rows(), NAMES, count=40, young_share=None, max_scan=10_000))
        self.assertEqual(len(kept), 40)
        groups = Counter((l["ethnicity"], l["gender"]) for _, l in kept)
        self.assertEqual(set(groups.values()), {10})

    def test_young_share_puts_half_in_ages_0_to_19(self):
        kept = list(ff.select(rows(), NAMES, count=40, young_share=0.5, max_scan=10_000))
        self.assertEqual(len(kept), 40)
        young = sum(l["age"] in ff.YOUNG_AGES for _, l in kept)
        self.assertEqual(young, 20)
        # Still balanced over ethnicity x gender within each age half.
        groups = Counter((l["ethnicity"], l["gender"], l["age"] in ff.YOUNG_AGES) for _, l in kept)
        self.assertEqual(set(groups.values()), {5})

    def test_rows_without_bytes_are_skipped_and_scan_is_capped(self):
        empty = ({**r, "image": {"bytes": None}} for r in rows())
        self.assertEqual(list(ff.select(empty, NAMES, 8, None, max_scan=10_000)), [])
        self.assertLessEqual(len(list(ff.select(rows(), NAMES, 400, None, max_scan=12))), 12)


if __name__ == "__main__":
    unittest.main()

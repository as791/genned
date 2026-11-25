"""tools/group_report.py: per-group false alarms from the app's scores."""
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import group_report  # noqa: E402
from ensemble_calibrate import BUNDLED_MODEL_NAME, COMMFOR_MODEL_NAME  # noqa: E402


class WilsonTest(unittest.TestCase):
    def test_known_values(self):
        lo, hi = group_report.wilson(0, 100)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.037, places=3)
        lo, hi = group_report.wilson(10, 100)
        self.assertAlmostEqual(lo, 0.0552, places=3)
        self.assertAlmostEqual(hi, 0.1744, places=3)


class GroupTableTest(unittest.TestCase):
    def test_flags_only_a_group_clearly_worse_than_the_rest(self):
        # Group A: 2 of 100 shown HIGH; group B: 20 of 100.
        scores = ([({"g": "A", "is_ai": 0}, 0.95)] * 2 + [({"g": "A", "is_ai": 0}, 0.1)] * 98
                  + [({"g": "B", "is_ai": 0}, 0.95)] * 20 + [({"g": "B", "is_ai": 0}, 0.1)] * 80)
        _, summary = group_report.group_table(scores, "g", 0.05, 2.0)
        by = {s["group"]: s for s in summary}
        self.assertFalse(by["A"]["flagged"])
        self.assertTrue(by["B"]["flagged"])
        self.assertAlmostEqual(by["B"]["rate"], 0.2)


class CliTest(unittest.TestCase):
    def test_joins_groups_and_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with (tmp / "groups-faces.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["file", "ethnicity"])
                for i in range(40):
                    w.writerow([f"real/{i:05d}.jpg", "X" if i < 20 else "Y"])
            rng = np.random.default_rng(0)
            with (tmp / "scores-faces.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["model", "dataset", "condition", "preprocess", "image", "generator", "is_ai", "logit_diff"])
                for i in range(40):
                    d, c = (8.0, 8.0) if i >= 30 else (rng.normal(-3, 1), rng.normal(-6, 1))
                    w.writerow([BUNDLED_MODEL_NAME, "faces", "original", "avg", f"real/{i:05d}.jpg", "real", 0, d])
                    w.writerow([COMMFOR_MODEL_NAME, "faces", "original", "native", f"real/{i:05d}.jpg", "real", 0, c])
            tool = Path(group_report.__file__)
            out = subprocess.run([sys.executable, str(tool), str(tmp), "--conditions", "original"],
                                 capture_output=True, text=True, check=True).stdout
            self.assertIn("## faces · original · by ethnicity", out)
            self.assertIn("| Y | 20 | 10 (50.0%)", out)
            self.assertIn("faces original ethnicity=Y", out)


if __name__ == "__main__":
    unittest.main()

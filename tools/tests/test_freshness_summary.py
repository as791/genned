"""tools/freshness_summary.py on hand-made result files."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import freshness_summary as fs  # noqa: E402


def payload(dataset, auc, model="app as shipped", condition="original"):
    return {"model": model, "dataset": dataset, "condition": condition, "metrics": {
        "n": 500, "auc": auc, "ci95": {"auc": [auc - 0.02, auc + 0.02]}, "catch_at_5": 0.6,
        "bands_real": {"high": 0.017}, "bands_ai": {"low": 0.07},
        "per_generator": {"real": {"n": 250, "ap": None}, "flux2": {"n": 20, "ap": 0.55},
                          "mj7": {"n": 20, "ap": 0.81}, "sd15": {"n": 20, "ap": 0.97},
                          "rare": {"n": 2, "ap": 0.10}, "gpt-image": {"n": 20, "ap": 0.70}}}}


def write(directory, items):
    for i, item in enumerate(items):
        Path(directory, f"{i}.json").write_text(json.dumps(item))


class FreshnessSummaryTest(unittest.TestCase):
    def test_reads_only_the_shipped_app_on_unedited_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, [payload("openfake", 0.9), payload("openfake", 0.5, condition="jpeg75"),
                        payload("openfake", 0.4, model="commfor")])
            (Path(tmp) / "broken.json").write_text("{")
            loaded = fs.load(Path(tmp))
            self.assertEqual(list(loaded), ["openfake"])
            self.assertEqual(loaded["openfake"]["metrics"]["auc"], 0.9)

    def test_weakest_generators_skip_real_and_tiny_groups(self):
        names = [n for n, _ in fs.weakest(payload("x", 0.9)["metrics"])]
        self.assertEqual(names, ["flux2", "gpt-image", "mj7"])

    def test_a_drop_beyond_the_alert_is_marked(self):
        text = fs.summary({"openfake": payload("openfake", 0.85)}, {"openfake": payload("openfake", 0.90)}, 0.03)
        self.assertIn("-0.050 vs last month, **dropped**", text)
        text = fs.summary({"openfake": payload("openfake", 0.89)}, {"openfake": payload("openfake", 0.90)}, 0.03)
        self.assertIn("-0.010 vs last month)", text)
        self.assertNotIn("dropped", text)

    def test_line_content(self):
        text = fs.summary({"cf-eval": payload("cf-eval", 0.758)}, {}, 0.03)
        self.assertIn("**cf-eval** (n=500)", text)
        self.assertIn("AUC 0.758 [0.738–0.778]", text)
        self.assertIn("real shown Likely AI 1.7%", text)
        self.assertNotIn("last month", text)

    def test_no_results_says_so(self):
        self.assertIn("No results", fs.summary({}, {}, 0.03))


if __name__ == "__main__":
    unittest.main()

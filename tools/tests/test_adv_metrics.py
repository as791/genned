"""Attack visibility (SSIM) in tools/adv_eval.py and the curve/visibility tables in
tools/adv_report.py."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
except ImportError:  # the report tests below need no torch
    torch = None

import adv_report  # noqa: E402


@unittest.skipIf(torch is None, "needs torch")
class SsimTest(unittest.TestCase):
    def setUp(self):
        import adv_eval

        self.ssim = adv_eval.ssim
        g = torch.Generator().manual_seed(0)
        self.x = torch.rand(2, 3, 64, 64, generator=g)
        self.noise = torch.rand(2, 3, 64, 64, generator=g) * 2 - 1

    def test_identical_is_one(self):
        self.assertTrue(torch.allclose(self.ssim(self.x, self.x), torch.ones(2), atol=1e-5))

    def test_bigger_perturbation_is_less_similar(self):
        small = self.ssim(self.x, (self.x + 2 / 255 * self.noise.sign()).clamp(0, 1))
        big = self.ssim(self.x, (self.x + 8 / 255 * self.noise.sign()).clamp(0, 1))
        self.assertTrue(bool((big < small).all()))
        self.assertTrue(bool((small < 1).all()))

    def test_unrelated_images_score_low(self):
        other = torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(9))
        self.assertTrue(bool((self.ssim(self.x, other) < 0.2).all()))


class ReportTablesTest(unittest.TestCase):
    RESULT = {
        "model": "ensemble", "defense": "none",
        "attacks": [
            {"attack": "pgd (eot)", "eps": e, "goal": "evasion", "laundered": laundered, "n": 120,
             "success": s, "flagged": 0.0, "mean_prob": 0.5, "success_ci95": [s / 2, min(1.0, s * 2)],
             "ssim_mean": 0.99 - e / 100, "ssim_min": 0.9}
            for e, s in ((2.0, 0.1), (4.0, 0.5), (8.0, 0.9)) for laundered in (False, True)
        ],
    }

    def test_curve_has_one_column_per_budget(self):
        lines = adv_report.curve_table([self.RESULT])
        self.assertIn("2/255 | 4/255 | 8/255", lines[1])
        row = next(line for line in lines if "pgd (eot)" in line)
        self.assertIn("10.0% [5%–20%]", row)
        self.assertIn("90.0% [45%–100%]", row)

    def test_visibility_skips_laundered_rows_and_missing_lpips(self):
        lines = adv_report.visibility_table([self.RESULT])
        data = [line for line in lines if line.startswith("| ensemble")]
        self.assertEqual(len(data), 3)
        self.assertTrue(all("n/a | n/a" in line for line in data))

    def test_budgets_from_parallel_jobs_are_merged(self):
        def part(eps):
            return {"model": "ensemble", "defense": "none", "clean": {"auc": 0.9},
                    "attacks": [a for a in self.RESULT["attacks"] if a["eps"] == eps]}

        merged = adv_report.merge_budgets([part(8.0), part(2.0), part(4.0)])
        self.assertEqual(len(merged), 1)
        self.assertEqual([a["eps"] for a in merged[0]["attacks"]], [2.0, 2.0, 4.0, 4.0, 8.0, 8.0])

    def test_old_results_without_intervals_still_render(self):
        old = {"attack": "fgsm", "eps": 8.0, "success": 0.25}
        self.assertEqual(adv_report.pct_ci(old), "25.0%")


if __name__ == "__main__":
    unittest.main()

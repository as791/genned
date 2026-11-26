"""tools/metrics.py: each metric against a hand-computed answer."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import metrics  # noqa: E402


class AucTest(unittest.TestCase):
    def test_matches_the_rank_definition_with_ties(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        s = np.array([0.1, 0.4, 0.4, 0.8, 0.2, 0.9])
        # Pairs (AI, real) won: 0.4 vs {0.1, 0.2} = 2, tie with 0.4 = 0.5; 0.8 and 0.9 beat all 3.
        self.assertAlmostEqual(metrics.fast_auc(y, s), (2.5 + 3 + 3) / 9)

    def test_needs_both_classes(self):
        self.assertTrue(np.isnan(metrics.fast_auc(np.array([1, 1]), np.array([0.2, 0.3]))))

    def test_agrees_with_evaluate_roc_auc(self):
        import evaluate

        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 400)
        s = np.round(rng.random(400) + 0.3 * y, 2)  # rounding makes ties
        self.assertAlmostEqual(metrics.fast_auc(y, s), evaluate.roc_auc(y, s), places=12)


class AveragePrecisionTest(unittest.TestCase):
    def test_hand_computed(self):
        # Ranked high to low: AI, real, AI, real. Precision at each AI: 1/1, 2/3.
        y = np.array([1, 0, 1, 0])
        s = np.array([0.9, 0.8, 0.7, 0.1])
        self.assertAlmostEqual(metrics.average_precision(y, s), (1 + 2 / 3) / 2)

    def test_perfect_ranking_is_one(self):
        self.assertAlmostEqual(metrics.average_precision(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])), 1.0)

    def test_ties_form_one_threshold(self):
        # All tied: one threshold with precision 1/2 and recall 1.
        self.assertAlmostEqual(metrics.average_precision(np.array([0, 1]), np.array([0.5, 0.5])), 0.5)


class CatchRateTest(unittest.TestCase):
    def test_threshold_lets_through_at_most_the_fpr(self):
        real = np.linspace(0, 0.99, 100)           # 100 real scores
        ai = np.array([0.5, 0.95, 0.985, 0.995])   # 4 AI scores
        y = np.r_[np.zeros(100), np.ones(4)]
        s = np.r_[real, ai]
        # At 5% FPR the threshold is the 95th real score (0.94): 0.95, 0.985 and 0.995 are above it.
        self.assertAlmostEqual(metrics.catch_rate_at_fpr(y, s, 0.05), 0.75)
        # At 1% FPR the threshold is 0.98: only 0.985 and 0.995 are above it.
        self.assertAlmostEqual(metrics.catch_rate_at_fpr(y, s, 0.01), 0.5)


class AnswerQualityTest(unittest.TestCase):
    def test_counts_each_answer(self):
        y = np.array([1, 1, 0, 0, 1, 0])
        s = np.array([0.95, 0.5, 0.92, 0.05, 0.1, 0.5])
        q = metrics.answer_quality(y, s, low=0.15, high=0.90)
        self.assertAlmostEqual(q["high_right"], 1 / 2)    # HIGH: 0.95 (AI), 0.92 (real)
        self.assertAlmostEqual(q["low_right"], 1 / 2)     # LOW: 0.05 (real), 0.1 (AI)
        self.assertAlmostEqual(q["answered"], 4 / 6)
        self.assertAlmostEqual(q["answered_right"], 2 / 4)

    def test_no_answers_gives_nan(self):
        q = metrics.answer_quality(np.array([0, 1]), np.array([0.5, 0.5]), 0.15, 0.9)
        self.assertTrue(np.isnan(q["high_right"]))
        self.assertEqual(q["answered"], 0.0)


class WilsonTest(unittest.TestCase):
    def test_known_values(self):
        # 0 of 100: [0, 3.7%]; 10 of 100: [5.5%, 17.4%] (standard Wilson tables).
        lo, hi = metrics.wilson(0, 100)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.0370, places=3)
        lo, hi = metrics.wilson(10, 100)
        self.assertAlmostEqual(lo, 0.0552, places=3)
        self.assertAlmostEqual(hi, 0.1744, places=3)

    def test_empty_is_nan(self):
        self.assertTrue(all(np.isnan(v) for v in metrics.wilson(0, 0)))


class BootstrapTest(unittest.TestCase):
    def test_interval_contains_the_value_and_narrows_with_more_data(self):
        rng = np.random.default_rng(3)

        def sample(n):
            y = np.r_[np.zeros(n), np.ones(n)]
            return y, np.r_[rng.normal(0, 1, n), rng.normal(1.5, 1, n)]

        stats = {"auc": metrics.fast_auc}
        y, s = sample(100)
        small = metrics.bootstrap_ci(y, s, stats, resamples=400)["auc"]
        self.assertLessEqual(small[0], metrics.fast_auc(y, s))
        self.assertGreaterEqual(small[1], metrics.fast_auc(y, s))
        y, s = sample(2000)
        big = metrics.bootstrap_ci(y, s, stats, resamples=400)["auc"]
        self.assertLess(big[1] - big[0], small[1] - small[0])

    def test_resamples_keep_class_sizes(self):
        seen = []
        y = np.r_[np.zeros(7), np.ones(3)]
        metrics.bootstrap_ci(y, np.arange(10.0), {"n_ai": lambda yy, ss: seen.append(yy.sum()) or 0.0}, resamples=20)
        self.assertTrue(all(n == 3 for n in seen))

    def test_is_reproducible(self):
        y = np.r_[np.zeros(50), np.ones(50)]
        s = np.random.default_rng(0).random(100)
        stats = metrics.standard_statistics(0.15, 0.9)
        self.assertEqual(metrics.bootstrap_ci(y, s, stats, resamples=50), metrics.bootstrap_ci(y, s, stats, resamples=50))


class ComputeMetricsTest(unittest.TestCase):
    def test_new_fields_are_reported(self):
        import evaluate

        preds = [evaluate.Prediction(Path(f"{i}.jpg"), i % 2 == 1, p, "gen" if i % 2 else "real", 0.0)
                 for i, p in enumerate([0.05, 0.97, 0.1, 0.95, 0.5, 0.92, 0.02, 0.3])]
        m = evaluate.compute_metrics(preds, 0.5, resamples=50)
        for key in ("ap", "catch_at_1", "catch_at_5", "answers", "ci95", "mean_generator_ap"):
            self.assertIn(key, m)
        self.assertIn("auc", m["ci95"])
        self.assertIn("ap", m["per_generator"]["gen"])
        self.assertEqual(evaluate.compute_metrics(preds, 0.5, resamples=0)["ci95"], {})


if __name__ == "__main__":
    unittest.main()

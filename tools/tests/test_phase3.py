"""Unit tests for the Phase 3 tooling (no downloads, CPU only).

Run: python -m unittest discover -s tools/tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate  # noqa: E402


def sample_image(seed: int = 0, size=(320, 240)) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)
    return Image.fromarray(array)


class EditConditionsTest(unittest.TestCase):
    def test_every_condition_runs_and_is_repeatable(self):
        image = sample_image()
        for condition in evaluate.ALL_CONDITIONS:
            seed = evaluate.condition_seed(Path("img.png"), condition)
            a = np.asarray(evaluate.degrade(image, condition, seed))
            b = np.asarray(evaluate.degrade(image, condition, seed))
            self.assertTrue(np.array_equal(a, b), condition)
            self.assertEqual(a.shape[2], 3, condition)

    def test_standard_conditions_unchanged(self):
        # Other tools iterate over CONDITIONS; the edits must stay opt-in.
        self.assertEqual(evaluate.CONDITIONS, ("original", "jpeg75", "social"))

    def test_noise_strength_orders(self):
        image = sample_image(1)
        base = np.asarray(image, dtype=np.float32)
        diffs = [np.abs(np.asarray(evaluate.degrade(image, c, 7), dtype=np.float32) - base).mean()
                 for c in ("noise2", "noise4", "noise8")]
        self.assertLess(diffs[0], diffs[1])
        self.assertLess(diffs[1], diffs[2])

    def test_rotate_crop_has_no_empty_corners(self):
        image = Image.new("RGB", (300, 200), (200, 100, 50))
        out = np.asarray(evaluate._rotate_crop(image, 3.0))
        corners = out[[0, 0, -1, -1], [0, -1, 0, -1]]
        self.assertTrue((corners.sum(axis=1) > 0).all())

    def test_seed_is_stable_across_processes(self):
        # zlib.crc32, not Python's randomized hash().
        self.assertEqual(evaluate.condition_seed(Path("a/b.png"), "noise4"),
                         evaluate.condition_seed(Path("c/b.png"), "noise4"))


class UniversalTrainingTest(unittest.TestCase):
    def setUp(self):
        import torch

        import adv_finetune

        self.torch, self.ft = torch, adv_finetune
        torch.manual_seed(0)

    def tiny_model(self):
        torch = self.torch

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 4, 4, 4)
                self.head = torch.nn.Linear(4 * 8 * 8, 1)

            def forward(self, x):
                return self.head(self.conv(x).flatten(1)).reshape(-1)

        return Tiny()

    def test_fit_universal_stays_in_budget_and_hurts_the_model(self):
        torch = self.torch
        model = self.tiny_model().eval()
        x = torch.rand(16, 3, 32, 32)
        with torch.no_grad():
            y = (model(x) > 0).float()  # labels the model gets right
        eps = 8 / 255
        deltas = self.ft.fit_universal(model, x, y, eps, epochs=8, batch=4, device=torch.device("cpu"), amp=False)
        self.assertEqual(tuple(deltas.shape), (2, 3, 32, 32))
        self.assertLessEqual(float(deltas.abs().max()), eps + 1e-6)
        loss = torch.nn.functional.binary_cross_entropy_with_logits
        with torch.no_grad():
            clean = loss(model(x), y)
            attacked = loss(model((x + deltas[y.long()]).clamp(0, 1)), y)
        self.assertGreater(float(attacked), float(clean))

    def test_prepared_views_are_exact(self):
        import tempfile

        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sub, seed in (("ai/x", 1), ("real", 2)):
                (root / sub).mkdir(parents=True)
                for i in range(3):
                    sample_image(seed * 10 + i, (300 + 10 * i, 260)).save(root / sub / f"{i}.png")
            dataset = self.ft.Images([root], 3, "commfor", train=False, seed=0, condition="noise4")
            x_u8, y = self.ft.prepare_views(dataset, workers=0, batch=2, label="test")
            direct = torch.stack([dataset[i][0] for i in range(len(dataset))])
            self.assertEqual(x_u8.dtype, torch.uint8)
            self.assertTrue(torch.equal(x_u8.float() / 255, direct))
            self.assertEqual(y.tolist(), [dataset[i][1].item() for i in range(len(dataset))])

    def test_split_halves_is_stratified(self):
        torch = self.torch
        y = torch.tensor([1.0] * 6 + [0.0] * 6)  # all AI first, as the validation set is listed
        fit, test = self.ft.split_halves(y)
        self.assertEqual(set(fit.tolist()) & set(test.tolist()), set())
        self.assertEqual(len(fit) + len(test), len(y))
        for half in (fit, test):
            self.assertEqual(sorted(set(y[half].tolist())), [0.0, 1.0])

    def test_jitter_keeps_shape(self):
        torch = self.torch
        delta = torch.rand(1, 3, 40, 40)
        self.assertEqual(tuple(self.ft.jitter(delta).shape), (1, 3, 40, 40))

    def test_selection_score(self):
        metrics = {"edit_auc_mean": 0.8, "uap_robust_acc_8": 0.4}
        self.assertAlmostEqual(self.ft.selection_score("aug", metrics, 8.0), 0.8)
        self.assertAlmostEqual(self.ft.selection_score("uat", metrics, 8.0), 0.4)
        self.assertAlmostEqual(self.ft.selection_score("aug+uat", metrics, 8.0), 0.6)


class FetchResumeTest(unittest.TestCase):
    def test_existing_state_counts_and_hashes(self):
        import hashlib
        import tempfile

        import fetch_eval_data

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "real").mkdir()
            (out / "ai" / "flux.2-dev").mkdir(parents=True)
            (out / "real" / "00000.jpg").write_bytes(b"real-a")
            (out / "real" / "00001.jpg").write_bytes(b"real-b")
            (out / "ai" / "flux.2-dev" / "00000.png").write_bytes(b"ai-a")
            counts, hashes = fetch_eval_data.existing_state(out)
            self.assertEqual(counts["real"], 2)
            self.assertEqual(counts["flux.2-dev"], 1)
            self.assertIn(hashlib.sha256(b"ai-a").hexdigest(), hashes)
            self.assertEqual(len(hashes), 3)

    def test_existing_state_empty_dir(self):
        import tempfile

        import fetch_eval_data

        with tempfile.TemporaryDirectory() as tmp:
            counts, hashes = fetch_eval_data.existing_state(Path(tmp))
            self.assertEqual(sum(counts.values()), 0)
            self.assertEqual(hashes, set())


if __name__ == "__main__":
    unittest.main()

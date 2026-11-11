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


class ScreenshotCropTest(unittest.TestCase):
    def test_crop_recovers_the_picture_in_app_screenshots(self):
        import screenshot_crop

        for seed, size in ((0, (1080, 1350)), (1, (1000, 1000)), (2, (1600, 900)), (3, (900, 1600))):
            image = sample_image(seed, size)
            rng = np.random.default_rng(seed)  # seed parity picks light (even draws) or dark themes
            shot = evaluate._app_screenshot(image, rng)
            shown_h = min(1350, round(size[1] * 1080 / size[0]))
            left, top, right, bottom = screenshot_crop.classifier_region(shot)
            self.assertLessEqual(abs(top - 400), 72, (seed, size, top))
            self.assertLessEqual(abs(bottom - (400 + shown_h)), 72, (seed, size, bottom))
            self.assertEqual((left, right), (0, 1080))

    def test_plain_photo_falls_back_to_the_centred_square(self):
        import screenshot_crop

        left, top, right, bottom = screenshot_crop.classifier_region(sample_image(4, (1080, 2400)))
        self.assertEqual((right - left, bottom - top), (1080, 1080))
        self.assertEqual(top, 96 + (2400 - 96 - 120 - 1080) // 2)


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

    def test_edit_weights_parse_and_reject_bad_input(self):
        self.assertEqual(self.ft.parse_edit_weights(""), {})
        self.assertEqual(self.ft.parse_edit_weights("grain=3, sharpen=2.5"), {"grain": 3.0, "sharpen": 2.5})
        for bad in ("glitter=2", "grain=0", "grain=-1", "grain"):
            with self.assertRaises(ValueError):
                self.ft.parse_edit_weights(bad)

    def test_balanced_weights_give_each_class_the_same_total(self):
        labels = [1] * 3 + [0] * 9  # e.g. extra real phone photos
        weights = self.ft.balanced_sample_weights(labels)
        ai = sum(w for w, label in zip(weights, labels) if label == 1)
        real = sum(w for w, label in zip(weights, labels) if label == 0)
        self.assertAlmostEqual(ai, real)
        with self.assertRaises(ValueError):
            self.ft.balanced_sample_weights([0, 0, 0])

    def training_folder(self, tmp: str, with_ai: bool = True) -> Path:
        root = Path(tmp)
        for sub, seed in ((("ai/x", 1),) if with_ai else ()) + (("real", 2),):
            (root / sub).mkdir(parents=True)
            for i in range(3):
                sample_image(seed * 10 + i, (300 + 10 * i, 260)).save(root / sub / f"{i}.png")
        return root

    def test_weighted_edits_are_drawn_more_often(self):
        import tempfile

        drawn = []
        original = self.ft.degrade
        self.ft.degrade = lambda image, condition, seed=0: drawn.append(condition) or image
        try:
            with tempfile.TemporaryDirectory() as tmp:
                dataset = self.ft.Images([self.training_folder(tmp)], 3, "commfor", train=True, seed=0,
                                         edit_share=1.0, edit_weights={"grain": 1000.0})
                for _ in range(20):
                    for i in range(len(dataset)):
                        dataset[i]
        finally:
            self.ft.degrade = original
        self.assertEqual(len(drawn), 120)
        self.assertGreater(drawn.count("grain"), 100)

    def test_clean_partner_is_the_same_view_without_the_edit(self):
        import tempfile

        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self.ft.Images([self.training_folder(tmp)], 3, "commfor", train=True, seed=0,
                                     edit_share=1.0, edit_weights={"noise8": 1000.0}, with_clean=True)
            x, x_plain, y = dataset[0]
            self.assertEqual(x.shape, x_plain.shape)
            self.assertIn(float(y), (0.0, 1.0))
            # Same picture, flip and view: noise of sigma 8/255 moves pixels a little, not a lot.
            difference = (x - x_plain).abs().mean().item()
            self.assertGreater(difference, 0.0)
            self.assertLess(difference, 0.06)
            # Evaluation items stay (x, y).
            evaluation = self.ft.Images([Path(tmp)], 3, "commfor", train=False, seed=0, with_clean=True)
            self.assertEqual(len(evaluation[0]), 2)

    def test_a_truncated_image_still_loads(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            folder = self.training_folder(tmp)
            victim = sorted((folder / "real").iterdir())[0]
            data = victim.read_bytes()
            victim.write_bytes(data[: len(data) * 2 // 3])  # a download cut short
            with self.assertRaises(OSError):
                Image.open(victim).convert("RGB")
            dataset = self.ft.Images([folder], 3, "commfor", train=True, seed=0, with_clean=True)
            for i in range(len(dataset)):
                x, x_plain, y = dataset[i]
                self.assertEqual(x.shape, x_plain.shape)
            self.assertEqual(dataset.unreadable, set())

    def test_an_unreadable_image_is_replaced_by_one_of_the_same_class(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            folder = self.training_folder(tmp)
            victim = sorted((folder / "ai").iterdir())[0]
            victim.write_bytes(b"not an image at all")
            dataset = self.ft.Images([folder], 3, "commfor", train=False, seed=0)
            index = [path for path, _ in dataset.items].index(victim)
            path, image, label = dataset.load(index)
            self.assertNotEqual(path, victim)
            self.assertEqual(label, 1)
            self.assertIn("ai", path.parts)
            self.assertEqual(dataset.unreadable, {victim})
            x, y = dataset[index]
            self.assertEqual(float(y), 1.0)

    def test_a_real_only_folder_adds_real_images(self):
        import tempfile

        with tempfile.TemporaryDirectory() as both, tempfile.TemporaryDirectory() as phone:
            dataset = self.ft.Images([self.training_folder(both), self.training_folder(phone, with_ai=False)],
                                     3, "commfor", train=True, seed=0)
            self.assertEqual(sorted(dataset.labels()), [0] * 6 + [1] * 3)

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


class CalibrationRefitTest(unittest.TestCase):
    def test_weighted_platt_with_unit_weights_matches_fit_platt(self):
        from calibrate import fit_platt
        from calibration_refit import fit_platt_weighted

        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 400)
        d = rng.normal(0, 1, 400) + 1.5 * (2 * y - 1)
        np.testing.assert_allclose(fit_platt_weighted(d, y, np.ones(400)), fit_platt(d, y), atol=1e-6)

    def test_weighted_platt_recovers_known_constants(self):
        from calibration_refit import fit_platt_weighted

        rng = np.random.default_rng(1)
        d = rng.normal(0, 1.5, 20000)
        y = (rng.random(20000) < 1 / (1 + np.exp(-(2.0 * d - 0.5)))).astype(int)
        slope, intercept = fit_platt_weighted(d, y, rng.uniform(0.5, 2.0, 20000))
        self.assertAlmostEqual(slope, 2.0, delta=0.15)
        self.assertAlmostEqual(intercept, -0.5, delta=0.15)

    def test_fold_split_is_deterministic_and_balanced(self):
        from calibration_refit import fold_of

        names = [f"real/{i:05d}.jpg" for i in range(1000)]
        folds = [fold_of(n) for n in names]
        self.assertEqual(folds, [fold_of(n) for n in names])
        self.assertTrue(400 < sum(folds) < 600)
        self.assertEqual(fold_of("real/00001.jpg"), fold_of("other/dir/00001.jpg"))


class BandSweepTest(unittest.TestCase):
    def test_bands_match_evaluate_at_the_shipped_cut_off(self):
        from band_sweep import bands

        p = np.random.default_rng(2).random(500)
        b = evaluate._bands(p)
        np.testing.assert_allclose(bands(p, evaluate.LOW_THRESHOLD), (b["low"], b["uncertain"], b["high"]), atol=1e-12)

    def test_raising_low_only_moves_uncertain_to_low(self):
        from band_sweep import bands

        p = np.random.default_rng(3).random(500)
        lo1, unc1, hi1 = bands(p, 0.15)
        lo2, unc2, hi2 = bands(p, 0.20)
        self.assertEqual(hi1, hi2)
        self.assertGreaterEqual(lo2, lo1)
        self.assertAlmostEqual(lo1 + unc1, lo2 + unc2)


if __name__ == "__main__":
    unittest.main()

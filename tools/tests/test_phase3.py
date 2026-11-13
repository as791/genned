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

    def test_universal_fit_in_slices_matches_the_whole_minibatch(self):
        import random

        torch = self.torch
        model = self.tiny_model().eval()
        x = torch.rand(10, 3, 32, 32)
        y = torch.tensor([1.0, 0.0] * 5)
        fits = []
        for chunks in (1, 3):
            torch.manual_seed(3)  # the order of the minibatches
            random.seed(3)  # the jitter of the perturbation
            sizes = []
            hook = model.register_forward_hook(lambda module, inputs, output: sizes.append(len(inputs[0])))
            fits.append(self.ft.fit_universal(model, x, y, 8 / 255, epochs=2, batch=5, device=torch.device("cpu"),
                                              amp=False, chunks=chunks))
            hook.remove()
            self.assertLessEqual(max(sizes), 5 if chunks == 1 else 2)
        # Sign steps: equal except where a gradient is within rounding of zero.
        same = (fits[0] == fits[1]).float().mean()
        self.assertGreater(float(same), 0.98)

    def test_memory_helpers_run(self):
        self.assertGreaterEqual(self.ft.current_memory_gb(), 0.0)
        self.ft.release_memory()

    def test_a_batch_in_slices_gives_the_same_gradients(self):
        import random

        torch = self.torch
        x, x_plain = torch.rand(6, 3, 32, 32), torch.rand(6, 3, 32, 32)
        y = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        runs = []
        for chunks in (1, 3, 4, 99):
            torch.manual_seed(0)
            model = self.tiny_model().train()
            sizes = []
            model.register_forward_hook(lambda module, inputs, output: sizes.append(len(inputs[0])))
            d = (torch.rand(2, 3, 32, 32) * 0.02).requires_grad_(True)
            random.seed(1)  # the same random crops of the perturbation (jitter) for every run
            calls = []
            out = self.ft.phase3_backward(model, x, y, x_plain, d, 1.0, 1.0, scaler, False, chunks,
                                          after_slice=lambda: calls.append(1))
            self.assertEqual(len(calls), len(sizes))  # once per slice
            runs.append(([p.grad.clone() for p in model.parameters()], d.grad.clone(), out, sizes))
        self.assertEqual(runs[0][3], [18])  # one forward: 6 images x 3 views
        self.assertEqual(runs[1][3], [6, 6, 6])
        self.assertEqual(runs[3][3], [3] * 6)  # never more slices than images
        grads, d_grad, out, _ = runs[0]
        self.assertGreater(float(d_grad.abs().sum()), 0.0)
        for other_grads, other_d_grad, other_out, _ in runs[1:]:
            for a, b in zip(grads, other_grads):
                self.assertTrue(torch.allclose(a, b, atol=1e-6))
            self.assertTrue(torch.allclose(d_grad, other_d_grad, atol=1e-7))
            for a, b in zip(out, other_out):
                self.assertTrue(torch.allclose(a, b, atol=1e-5))

    def test_a_batch_without_attack_or_unedited_view(self):
        torch = self.torch
        model = self.tiny_model().train()
        x, y = torch.rand(5, 3, 32, 32), torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        loss, clean, robust, consistency, adv_logit = self.ft.phase3_backward(
            model, x, y, None, None, 1.0, 0.0, scaler, False, chunks=2)
        self.assertAlmostEqual(float(loss), float(clean), places=6)
        self.assertAlmostEqual(float(robust), float(clean), places=6)
        self.assertEqual(float(consistency), 0.0)
        with torch.no_grad():
            self.assertTrue(torch.allclose(adv_logit, model(x), atol=1e-6))

    def test_available_memory_is_a_number_or_unknown(self):
        free = self.ft.available_memory_gb()
        self.assertTrue(free is None or free > 0)

    def test_validation_never_exceeds_the_batch_size(self):
        torch = self.torch
        model = self.tiny_model().eval()
        sizes = []
        model.register_forward_hook(lambda module, inputs, output: sizes.append(len(inputs[0])))
        x_u8 = (torch.rand(22, 3, 32, 32) * 255).round().to(torch.uint8)
        y = torch.tensor([1.0] * 11 + [0.0] * 11)
        views = {"clean": (x_u8, y), "noise4": (x_u8, y)}
        metrics = self.ft.evaluate_phase3(model, views, torch.device("cpu"), 8 / 255, amp=False, batch=4, uap_epochs=1)
        self.assertLessEqual(max(sizes), 4)
        self.assertEqual(metrics["n"], 22)
        for key in ("uap_robust_acc_8", "uap_robust_acc_ai_8", "uap_robust_acc_real_8"):
            self.assertTrue(0.0 <= metrics[key] <= 1.0)
        self.assertGreater(self.ft.peak_memory_gb(), 0.0)

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
            victim = sorted((folder / "ai" / "x").iterdir())[0]
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

    @staticmethod
    def _rows():
        rng = np.random.default_rng(4)
        rows = []
        for dataset, classes in (("bench", (0, 1)), ("phone-photos", (0,))):
            for condition in ("original", "grain"):
                for is_ai in classes:
                    for i in range(60):
                        rows.append({"dataset": dataset, "condition": condition, "image": f"{is_ai}/{i}.jpg",
                                     "is_ai": is_ai, "d": float(rng.normal(4 * is_ai - 2, 3)),
                                     "c": float(rng.normal(6 * is_ai - 3, 4))})
        return rows

    def test_sweep_uses_the_candidate_constants_when_given(self):
        from band_sweep import app_probability, sweep

        rows = self._rows()
        params = {"mean_d": 0.5, "std_d": 3.0, "mean_c": -1.0, "std_c": 5.0, "photo": (4.0, 0.3)}
        np.testing.assert_allclose(
            app_probability(rows[:5], params),
            [1 / (1 + np.exp(-(4.0 * evaluate.ensemble_score(r["d"], r["c"], params) + 0.3))) for r in rows[:5]])
        np.testing.assert_allclose(
            app_probability(rows[:5]),
            [1 / (1 + np.exp(-(evaluate.APP_ENSEMBLE["photo"][0] * evaluate.ensemble_score(r["d"], r["c"])
                               + evaluate.APP_ENSEMBLE["photo"][1]))) for r in rows[:5]])
        out, _ = sweep(self._rows(), [0.20], params)
        self.assertIn("candidate set's own constants", "\n".join(out))
        out, _ = sweep(self._rows(), [0.20])
        self.assertIn("the shipped constants", "\n".join(out))

    def test_sweep_summary_has_a_row_per_low_and_high_and_a_column_per_real_only_set(self):
        from band_sweep import sweep

        out, summary = sweep(self._rows(), [0.15, 0.20], None, [0.85])
        self.assertEqual(set(summary), {(0.15, 0.90), (0.20, 0.90), (0.15, 0.85), (0.20, 0.85)})
        for (lo, hi), s in summary.items():
            self.assertEqual(list(s["real_only"]), ["phone-photos"])
            self.assertEqual(s["real_only"]["phone-photos"], s["phone"])
            self.assertAlmostEqual(sum(s["phone"]), 1.0)
        # A lower HIGH cut-off can only show more images HIGH; LOW is untouched.
        self.assertGreaterEqual(summary[(0.20, 0.85)]["worst_real_high"], summary[(0.20, 0.90)]["worst_real_high"])
        self.assertEqual(summary[(0.20, 0.85)]["worst_ai_low_edits"], summary[(0.20, 0.90)]["worst_ai_low_edits"])
        text = "\n".join(out)
        self.assertIn("phone-photos (original) LOW / UNCERTAIN / HIGH", text)
        header = next(l for l in out if l.startswith("| LOW cut-off"))
        rows = [l for l in out if l.startswith("| 0.")]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(l.count("|") == header.count("|") for l in rows))


class CalibrationRefitParamsTest(unittest.TestCase):
    def test_refit_reads_the_candidate_constants(self):
        import csv
        import json
        import subprocess
        import sys
        import tempfile

        from ensemble_calibrate import BUNDLED_MODEL_NAME, COMMFOR_MODEL_NAME

        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for dataset, classes in (("defactify", (0, 1)), ("phone-photos", (0,)), ("unsplash-phone", (0,))):
                with (tmp / f"scores-{dataset}.csv").open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["dataset", "condition", "image", "is_ai", "model", "preprocess", "logit_diff"])
                    for condition in ("original", "jpeg75", "social"):
                        for is_ai in classes:
                            for i in range(80):
                                image = f"{'ai' if is_ai else 'real'}/{i}.jpg"
                                w.writerow([dataset, condition, image, is_ai, BUNDLED_MODEL_NAME, "avg",
                                            rng.normal(4 * is_ai - 2, 2)])
                                w.writerow([dataset, condition, image, is_ai, COMMFOR_MODEL_NAME, "native",
                                            rng.normal(8 * is_ai - 4, 3)])
            params = tmp / "ensemble-params.json"
            params.write_text(json.dumps({
                "standardization": {"mean_d": 0.0, "std_d": 3.0, "mean_c": 0.0, "std_c": 5.0},
                "photo": {"slope": 4.5, "intercept": 0.25}, "video": {"slope": 1.0, "intercept": 0.0}}))
            tool = Path(evaluate.__file__).with_name("calibration_refit.py")
            run = lambda *extra: subprocess.run(  # noqa: E731
                [sys.executable, str(tool), str(tmp), "--check", "unsplash-phone", "--json", str(tmp / "out.json"), *extra],
                capture_output=True, text=True, check=True).stdout
            built = run("--ensemble-params", str(params))
            self.assertIn("| as built | 4.50000 | 0.25000 |", built)
            self.assertIn("candidate set's own standardization", built)
            self.assertIn("| unsplash-phone | original | 80 |", built)
            self.assertEqual(json.loads((tmp / "out.json").read_text())["as built"]["slope"], 4.5)
            shipped = run()
            self.assertIn(f"| shipped | {evaluate.APP_ENSEMBLE['photo'][0]:.5f} |", shipped)
            self.assertIn("the shipped standardization", shipped)


if __name__ == "__main__":
    unittest.main()

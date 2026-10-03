#!/usr/bin/env python3
"""Phase 2 (#15): adversarial robustness benchmark for Genned's detector and defenses.

Threat model
------------
Genned is open source and ships its model in the APK, so an attacker can compute
gradients against the exact on-device detector: white-box attacks are realistic.
The attacker controls the image the app analyzes. To keep attacks exact and cheap,
every image is first center-cropped and resized to the model's 380x380 input, so both
of the app's views (squash and center crop) see the same pixels. The app's own JPEG
q92 re-encode is part of the forward pass (straight-through / BPDA for gradients).

* Evasion (the main threat): push an AI image to be shown LOW (calibrated P(ai) < LOW
  band, and not flagged by a detect-and-abstain defense).
* Framing: push a real image into HIGH.
* Real world: every adversarial image is also re-scored after Instagram-like laundering
  (JPEG q75) - perturbations that don't survive sharing matter less.

Attacks (--attacks; budgets are L-inf eps in /255):
  fgsm          one-step white box
  pgd           PGD-Linf with random start and EOT over the defense's randomness (plus an
                adaptive PGD that also evades the consistency check, for that defense)
  transfer      PGD on one of the app's own models only (the attacker knows one of two)
  framing       PGD pushing real images to HIGH
  square        Square Attack: score-based, query-only black box
  uap           universal perturbation (Phase 3): ONE pattern per budget and goal, fit by
                minibatch PGD on images disjoint from the test images (with random crops
                and rescales so it survives reposting), then added unchanged to every
                test image. Scored direct, laundered, and after a random crop/rescale.
                This is the "publish one filter that beats Genned" threat.
  transfer-ext  MI-DI-FGSM on an ensemble of open detectors that are NOT in the app
                (--surrogates; tools/eval_candidates.py): what generic "beat AI
                detectors" tools do, with no access to Genned at all

Every defense wraps the detector chosen with --model: `ensemble` (default; the app as
shipped: bundled + Community Forensics 224 with the shipped EnsembleConfig constants) or
`bundled` (the primary model alone, as before the ensemble).

Defenses (all phone-feasible; cost = model runs per check):
  none          the app as shipped
  transform     random resize+pad+blur, averaged over K draws (Xie et al. 2018 style)
  smoothing     Gaussian noise (sigma), averaged over N draws (randomized smoothing, soft)
  consistency   feature-squeezing check: raw vs (JPEG q75 + 3x3 median); disagreement
                above the clean 95th percentile -> abstain (shown UNCERTAIN)
  ensemble      bundled + CF 224 with standardization refit on the clean split
                (only meaningful with --model bundled; the shipped ensemble is --model ensemble)

Every randomized defense is attacked with EOT; JPEG is handled with BPDA; the
consistency check gets an adaptive loss. Without these, robustness numbers are
meaningless (Athalye et al. 2018).

Needs torch, torchvision, timm, onnx2torch, onnxruntime, pillow. Writes one JSON per
defense (tools/adv_report.py merges them). Nothing but numbers leaves the runner.

Usage:
    python tools/adv_eval.py --model ensemble --defense consistency \\
        --datasets eval-data/defactify eval-data/mj-dalle-sd-nbp --out adv-results [--smoke]
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate import fit_platt  # noqa: E402
from evaluate import (  # noqa: E402
    APP_CALIBRATION,
    APP_ENSEMBLE,
    HIGH_THRESHOLD,
    INPUT_NAME,
    INPUT_SIZE,
    LOW_THRESHOLD,
    MEAN,
    STD,
    collect_images,
    logit_difference,
    roc_auc,
)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(max(1, torch.get_num_threads()))
SIZE = INPUT_SIZE
MEAN_T = torch.tensor(MEAN).view(1, 3, 1, 1)
STD_T = torch.tensor(STD).view(1, 3, 1, 1)


ATTACKS = ("fgsm", "pgd", "transfer", "framing", "square", "uap", "transfer-ext")


# --------------------------------------------------------------------------- data

def square_380(image: Image.Image) -> torch.Tensor:
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side)).resize((SIZE, SIZE), Image.BILINEAR)
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)


def select_paths(dirs: list[Path], per_class: int, seed: int,
                 exclude: frozenset[Path] = frozenset()) -> list[tuple[Path, int, str]]:
    """per_class AI + per_class real image paths, split evenly across datasets, skipping
    `exclude` (so splits can be made disjoint)."""
    rng = np.random.default_rng(seed)
    items = []
    for d in dirs:
        for label, sub in ((1, "ai"), (0, "real")):
            paths = [p for p in collect_images(d / sub) if p not in exclude]
            rng.shuffle(paths)
            items += [(p, label, d.name) for p in paths[: max(1, per_class // len(dirs))]]
    return items


def load_split(dirs: list[Path], per_class: int, seed: int,
               exclude: frozenset[Path] = frozenset()) -> list[tuple[torch.Tensor, int, str]]:
    """per_class AI + per_class real images (380x380 tensors), split evenly across datasets."""
    return [(square_380(Image.open(p).convert("RGB")), label, name)
            for p, label, name in select_paths(dirs, per_class, seed, exclude)]


def jpeg(x: torch.Tensor, quality: int) -> torch.Tensor:
    """JPEG round trip of a (B,3,H,W) batch in [0,1] - not differentiable."""
    out = []
    for img in x.detach().clamp(0, 1):
        arr = (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        out.append(torch.from_numpy(np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1))
    return torch.stack(out)


def jpeg_ste(x: torch.Tensor, quality: int) -> torch.Tensor:
    """JPEG in the forward pass, identity in the backward pass (BPDA)."""
    return x + (jpeg(x, quality) - x).detach()


def median3(x: torch.Tensor) -> torch.Tensor:
    """3x3 per-channel median, edge pixels replicated (as the app and consistency_calibrate.py)."""
    padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
    patches = padded.unfold(2, 3, 1).unfold(3, 3, 1)  # B,C,H,W,3,3
    return patches.reshape(*patches.shape[:4], 9).median(dim=-1).values


# --------------------------------------------------------------------------- models

class Bundled(torch.nn.Module):
    """The app's primary detector as a differentiable torch module (onnx2torch).
    forward(x in [0,1]) -> logit gap (ai - human), after the app's JPEG q92."""

    runs = 2  # the app runs two views (identical here, since inputs are pre-squared)
    models = 1

    def __init__(self, onnx_path: Path):
        super().__init__()
        import onnx
        from onnx2torch import convert

        from fp16_weights import restore_fp32

        # The shipped file stores weights as fp16 + Cast; onnx2torch needs them as
        # initializers. restore_fp32 is exact (the same upcast the runtime Cast does).
        self.net = convert(restore_fp32(onnx.load(str(onnx_path)))).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor, app_jpeg: bool = True) -> torch.Tensor:
        if app_jpeg:
            x = jpeg_ste(x, 92)
        z = (x - MEAN_T) / STD_T
        # The export has a fixed batch of 1; run images one at a time.
        logits = torch.cat([self.net(z[i : i + 1]) for i in range(z.shape[0])])
        return logits[:, 0] - logits[:, 1]


class CommunityForensics(torch.nn.Module):
    """The shipped Community Forensics ViT-S 224 file (onnx2torch) with the app's exact
    preprocessing. The resize is straight-through: the forward pass is the app's own
    geometry (uint8 -> PIL bilinear -> short side 256 -> center crop 224, as
    evaluate.commfor_view), the backward pass a torch antialiased bilinear resize. They
    differ by < 1/255 per pixel, but this ViT's logit moves by up to ~1.3 on that, so the
    forward pass must be the app's pixels exactly."""

    runs = 1
    models = 1

    def __init__(self, onnx_path: Path):
        super().__init__()
        import onnx
        from onnx2torch import convert

        from fp16_weights import restore_fp32

        self.net = convert(restore_fp32(onnx.load(str(onnx_path)))).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _app_view(x: torch.Tensor) -> torch.Tensor:
        from evaluate import commfor_view

        out = []
        for img in x.detach().clamp(0, 1):
            arr = (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            view = commfor_view(Image.fromarray(arr))
            out.append(torch.from_numpy(np.asarray(view, dtype=np.float32) / 255.0).permute(2, 0, 1))
        return torch.stack(out)

    def forward(self, x: torch.Tensor, app_jpeg: bool = True) -> torch.Tensor:
        if app_jpeg:
            x = jpeg_ste(x, 92)
        smooth = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
        smooth = smooth[:, :, 16:240, 16:240]
        view = smooth + (self._app_view(x) - smooth).detach()
        z = (view - MEAN_T) / STD_T
        return torch.cat([self.net(z[i : i + 1]).reshape(-1) for i in range(z.shape[0])])


class Ensemble(torch.nn.Module):
    """Mean of standardized logits. shipped=True uses the app's EnsembleConfig constants
    (tools/evaluate.py APP_ENSEMBLE); otherwise fit() estimates them on the clean split."""

    runs = 3
    models = 2

    def __init__(self, bundled: Bundled, cf: CommunityForensics, shipped: bool = False, params: dict | None = None):
        super().__init__()
        self.bundled, self.cf = bundled, cf
        e = params or APP_ENSEMBLE
        self.stats = (e["mean_d"], e["std_d"], e["mean_c"], e["std_c"]) if shipped else (0.0, 1.0, 0.0, 1.0)

    def fit(self, clean: torch.Tensor) -> None:
        with torch.no_grad():
            d = torch.cat([self.bundled(clean[i : i + 8]) for i in range(0, len(clean), 8)])
            c = torch.cat([self.cf(clean[i : i + 8]) for i in range(0, len(clean), 8)])
        self.stats = (d.mean().item(), d.std().item(), c.mean().item(), c.std().item())

    def forward(self, x: torch.Tensor, app_jpeg: bool = True) -> torch.Tensor:
        md, sd, mc, sc = self.stats
        return ((self.bundled(x, app_jpeg) - md) / sd + (self.cf(x, app_jpeg) - mc) / sc) / 2


# --------------------------------------------------------------------------- defenses

@dataclass
class Prediction:
    score: torch.Tensor    # defended raw score (higher = more AI)
    flagged: torch.Tensor  # bool: the defense abstained (shown UNCERTAIN)


class Defense:
    name = "none"
    randomized = False
    # The detector's own shipped calibration; set by main() from --model.
    shipped_calibration = APP_CALIBRATION

    def __init__(self, detector: torch.nn.Module):
        self.detector = detector
        self.cost = detector.runs  # model runs per check in the app

    def raw(self, x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """One differentiable draw of the defended score (what EOT averages)."""
        return self.detector(x)

    def predict(self, x: torch.Tensor, gen: torch.Generator) -> Prediction:
        with torch.no_grad():
            return Prediction(self.detector(x), torch.zeros(x.shape[0], dtype=torch.bool))

    def calibrate(self, clean: torch.Tensor, labels: torch.Tensor, gen: torch.Generator) -> None:
        """Probability mapping for the defended score; the app's shipped calibration by default."""
        self.slope, self.intercept = self.shipped_calibration

    def probability(self, score: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.slope * score + self.intercept)


class RandomTransform(Defense):
    name, randomized = "transform", True

    def __init__(self, detector, draws: int = 4):
        super().__init__(detector)
        self.draws = draws
        # Each draw replaces the app's views with one transformed view per model.
        self.cost = draws * detector.models

    def _transform(self, x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        scale = 0.8 + 0.2 * torch.rand(1, generator=gen).item()
        size = int(SIZE * scale)
        y = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        pad = SIZE - size
        left = int(torch.randint(0, pad + 1, (1,), generator=gen))
        top = int(torch.randint(0, pad + 1, (1,), generator=gen))
        y = F.pad(y, (left, pad - left, top, pad - top), mode="reflect")
        sigma = torch.rand(1, generator=gen).item()
        if sigma > 0.1:
            k = torch.arange(-2, 3, dtype=torch.float32)
            k = torch.exp(-(k ** 2) / (2 * sigma ** 2))
            k = (k / k.sum()).view(1, 1, 5)
            y = F.conv2d(F.pad(y, (2, 2, 0, 0), mode="reflect"), k.view(1, 1, 1, 5).repeat(3, 1, 1, 1), groups=3)
            y = F.conv2d(F.pad(y, (0, 0, 2, 2), mode="reflect"), k.view(1, 1, 5, 1).repeat(3, 1, 1, 1), groups=3)
        return y

    def raw(self, x, gen):
        return self.detector(self._transform(x, gen))

    def predict(self, x, gen):
        with torch.no_grad():
            score = torch.stack([self.raw(x, gen) for _ in range(self.draws)]).mean(0)
        return Prediction(score, torch.zeros(x.shape[0], dtype=torch.bool))

    def calibrate(self, clean, labels, gen):
        self.slope, self.intercept = fit_platt(self.predict(clean, gen).score.numpy().astype(np.float64),
                                               labels.numpy())


class Smoothing(RandomTransform):
    name = "smoothing"

    def __init__(self, detector, draws: int = 8, sigma: float = 0.03):
        super().__init__(detector, draws)
        self.sigma = sigma

    def _transform(self, x, gen):
        return (x + self.sigma * torch.randn(x.shape, generator=gen)).clamp(0, 1)


class Consistency(Defense):
    name = "consistency"

    def __init__(self, detector):
        super().__init__(detector)
        self.cost = 2 * detector.runs  # the detector on the image and on its squeezed copy

    def squeezed(self, x: torch.Tensor) -> torch.Tensor:
        return median3(jpeg_ste(x, 75))

    def disagreement(self, x: torch.Tensor) -> torch.Tensor:
        return (self.detector(x) - self.detector(self.squeezed(x))).abs()

    def calibrate(self, clean, labels, gen):
        super().calibrate(clean, labels, gen)
        with torch.no_grad():
            gaps = torch.cat([self.disagreement(clean[i : i + 8]) for i in range(0, len(clean), 8)])
        self.tau = float(torch.quantile(gaps, 0.95))  # 5% of clean images abstain

    def predict(self, x, gen):
        with torch.no_grad():
            raw, squeezed = self.detector(x), self.detector(self.squeezed(x))
        return Prediction(raw, (raw - squeezed).abs() > self.tau)


class EnsembleDefense(Defense):
    name = "ensemble"

    def calibrate(self, clean, labels, gen):
        self.detector.fit(clean)
        self.slope, self.intercept = fit_platt(self.predict(clean, gen).score.numpy().astype(np.float64),
                                               labels.numpy())


# --------------------------------------------------------------------------- attacks

def eot_score(defense: Defense, x: torch.Tensor, gen: torch.Generator, eot: int) -> torch.Tensor:
    draws = eot if defense.randomized else 1
    return torch.stack([defense.raw(x, gen) for _ in range(draws)]).mean(0)


def attack_loss(defense: Defense, x: torch.Tensor, gen: torch.Generator, eot: int, direction: float,
                adaptive: bool) -> torch.Tensor:
    """Per-image loss to MINIMIZE. direction=+1 evasion (lower P(ai)), -1 framing.

    The score is scaled by the defense's calibration slope, so the attack always moves
    the *shown probability* the intended way (and uses the same gradient scale for the
    adaptive term as the app sees)."""
    loss = direction * defense.slope * eot_score(defense, x, gen, eot)
    if adaptive and isinstance(defense, Consistency):
        # Also keep the raw/squeezed disagreement under the abstain threshold.
        loss = loss + 4.0 * F.relu(defense.disagreement(x) - 0.5 * defense.tau)
    return loss


def fgsm(defense, x, eps, gen, eot, direction, adaptive=False):
    x = x.clone().requires_grad_(True)
    attack_loss(defense, x, gen, eot, direction, adaptive).sum().backward()
    return (x - eps * x.grad.sign()).clamp(0, 1).detach()


def pgd(defense, x, eps, steps, gen, eot, direction, adaptive=False):
    alpha = eps / 4
    adv = (x + eps * (2 * torch.rand(x.shape, generator=gen) - 1)).clamp(0, 1)
    for _ in range(steps):
        adv = adv.clone().requires_grad_(True)
        attack_loss(defense, adv, gen, eot, direction, adaptive).sum().backward()
        adv = adv - alpha * adv.grad.sign()
        adv = (x + (adv - x).clamp(-eps, eps)).clamp(0, 1).detach()
    return adv


def square_attack(defense, x, eps, queries, gen, direction):
    """Score-based L-inf Square Attack (Andriushchenko et al. 2020), one image at a time.
    The attacker only sees the app's output (defended score + whether it abstained)."""
    out = []
    for img in x:
        img = img.unsqueeze(0)
        c, h, w = img.shape[1:]

        def loss(candidate):
            p = defense.predict(candidate, gen)
            return direction * defense.slope * p.score.item() + (1e3 if p.flagged.item() else 0.0)

        stripes = eps * (2 * torch.randint(0, 2, (1, c, 1, w), generator=gen).float() - 1)
        adv = (img + stripes).clamp(0, 1)
        best = loss(adv)
        for i in range(queries - 1):
            frac = i / max(1, queries)
            p = 0.05 * 0.5 ** sum(frac > t for t in (0.001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8))
            s = max(1, int(round(math.sqrt(p * h * w))))
            vh = int(torch.randint(0, h - s + 1, (1,), generator=gen))
            vw = int(torch.randint(0, w - s + 1, (1,), generator=gen))
            delta = adv - img
            delta[:, :, vh : vh + s, vw : vw + s] = eps * (2 * torch.randint(0, 2, (1, c, 1, 1), generator=gen).float() - 1)
            candidate = (img + delta).clamp(0, 1)
            value = loss(candidate)
            if value < best:
                adv, best = candidate, value
        out.append(adv)
    return torch.cat(out)


def random_geometry(x: torch.Tensor, gen: torch.Generator, min_scale: float = 0.85) -> torch.Tensor:
    """Random crop of 85-100% of the side, resized back to SIZE: what a repost, a crop or a
    screenshot does to where a fixed pattern lands on the model's input."""
    scale = min_scale + (1 - min_scale) * torch.rand(1, generator=gen).item()
    size = max(8, int(SIZE * scale))
    top = int(torch.randint(0, SIZE - size + 1, (1,), generator=gen))
    left = int(torch.randint(0, SIZE - size + 1, (1,), generator=gen))
    crop = x[:, :, top : top + size, left : left + size]
    return F.interpolate(crop, size=(SIZE, SIZE), mode="bilinear", align_corners=False, antialias=True)


def band_margin_loss(defense: Defense, x: torch.Tensor, gen: torch.Generator, direction: float) -> torch.Tensor:
    """Per-image hinge on the shown probability's logit: evasion pushes it 1 below the LOW
    band, framing 1 above the HIGH band, and an image already there stops contributing, so
    a universal pattern spends its budget on the images it hasn't fooled yet."""
    z = defense.slope * eot_score(defense, x, gen, 1) + defense.intercept
    low, high = math.log(LOW_THRESHOLD / (1 - LOW_THRESHOLD)), math.log(HIGH_THRESHOLD / (1 - HIGH_THRESHOLD))
    loss = F.relu(z - low + 1.0) if direction > 0 else F.relu(high - z + 1.0)
    if isinstance(defense, Consistency):
        loss = loss + 4.0 * F.relu(defense.disagreement(x) - 0.5 * defense.tau)
    return loss


def fit_uap(defense: Defense, fit_x: torch.Tensor, eps: float, epochs: int, gen: torch.Generator,
            direction: float, batch: int = 4) -> torch.Tensor:
    """Universal perturbation (stochastic PGD on one shared delta; Shafahi et al. 2020),
    fit through random crops/rescales so it is the pattern a real attacker would publish."""
    delta = torch.zeros(1, 3, SIZE, SIZE)
    alpha = eps / 8
    for epoch in range(epochs):
        order = torch.randperm(len(fit_x), generator=gen)
        fooled = []
        for i in range(0, len(fit_x), batch):
            xb = fit_x[order[i : i + batch]]
            delta = delta.clone().requires_grad_(True)
            adv = random_geometry((xb + delta).clamp(0, 1), gen)
            loss = band_margin_loss(defense, adv, gen, direction)
            loss.mean().backward()
            fooled.append((loss.detach() == 0).float())
            delta = (delta - alpha * delta.grad.sign()).clamp(-eps, eps).detach()
        print(f"  uap eps={eps * 255:.0f}/255 {'evasion' if direction > 0 else 'framing'} epoch {epoch + 1}/{epochs}: "
              f"fit images past the band {torch.cat(fooled).mean():.1%}", flush=True)
    return delta


class Surrogate(torch.nn.Module):
    """An open detector that is not in the app (tools/eval_candidates.py), as a
    differentiable logit gap (higher = more AI) on [0,1] images of any size. Its native
    resize + normalization are approximated in torch; exactness doesn't matter for a
    surrogate, only the target is scored exactly."""

    def __init__(self, candidate):
        super().__init__()
        self.name = candidate.name
        if hasattr(candidate, "input_size"):  # Community Forensics: Resize(short) + CenterCrop
            self.net = candidate.model
            self.resize = 440 if candidate.input_size == 384 else 256
            self.crop = candidate.input_size
            self.mean, self.std = MEAN_T, STD_T
            self.gap = lambda z: self.net(z).reshape(-1)
        else:  # transformers image classifier
            proc, model = candidate.processor, candidate.model
            size = proc.size
            self.resize = size.get("shortest_edge") or size.get("height")
            crop = getattr(proc, "crop_size", None) if getattr(proc, "do_center_crop", False) else None
            self.crop = (crop or {}).get("height") or self.resize
            self.mean = torch.tensor(proc.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
            self.std = torch.tensor(proc.image_std, dtype=torch.float32).view(1, 3, 1, 1)
            ai, real = candidate.ai_index, candidate.real_index
            self.gap = lambda z: (lambda lg: lg[:, ai] - lg[:, real])(model(pixel_values=z).logits)
            self.net = model
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.scale = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.interpolate(x, size=(self.resize, self.resize), mode="bilinear", align_corners=False, antialias=True)
        off = (self.resize - self.crop) // 2
        y = y[:, :, off : off + self.crop, off : off + self.crop]
        return self.gap((y - self.mean) / self.std) / self.scale


def load_surrogates(names: list[str], calib_x: torch.Tensor) -> list[Surrogate]:
    from eval_candidates import CANDIDATES

    out = []
    for name in names:
        try:
            candidate = CANDIDATES[name]
            candidate.load()
            surrogate = Surrogate(candidate).eval()
            with torch.no_grad():
                # Equal say for every surrogate: divide by its logit spread on clean images.
                surrogate.scale = max(1e-3, float(surrogate(calib_x).std()))
            out.append(surrogate)
            print(f"surrogate {name}: loaded (clean logit std {surrogate.scale:.2f})")
        except (Exception, SystemExit) as e:  # noqa: BLE001 - a missing surrogate must not sink the run
            print(f"::warning::surrogate {name} failed to load: {e!r}")
    return out


def diverse_input(x: torch.Tensor, gen: torch.Generator, p: float = 0.7) -> torch.Tensor:
    """DI-FGSM input diversity (Xie et al. 2019): random resize + pad, with probability p."""
    if torch.rand(1, generator=gen).item() >= p:
        return x
    big = int(SIZE * 1.1)
    size = int(torch.randint(SIZE, big, (1,), generator=gen))
    y = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    left = int(torch.randint(0, big - size + 1, (1,), generator=gen))
    top = int(torch.randint(0, big - size + 1, (1,), generator=gen))
    return F.pad(y, (left, big - size - left, top, big - size - top), value=0.0)


def mi_di_fgsm(surrogates: list[Surrogate], x: torch.Tensor, eps: float, steps: int, gen: torch.Generator,
               direction: float, momentum: float = 1.0) -> torch.Tensor:
    """Momentum + input-diversity iterative FGSM on the surrogate ensemble (Dong et al. 2018;
    Xie et al. 2019) - the standard recipe for perturbations that transfer to unseen models."""
    alpha = eps / steps
    adv, g = x.clone(), torch.zeros_like(x)
    for _ in range(steps):
        adv = adv.clone().requires_grad_(True)
        z = diverse_input(adv, gen)
        loss = direction * torch.stack([s(z) for s in surrogates]).mean(0)
        grad, = torch.autograd.grad(loss.sum(), adv)
        g = momentum * g + grad / grad.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        adv = (adv - alpha * g.sign()).detach()
        adv = (x + (adv - x).clamp(-eps, eps)).clamp(0, 1)
    return adv


# --------------------------------------------------------------------------- evaluation

def outcomes(defense: Defense, x: torch.Tensor, gen: torch.Generator) -> dict[str, np.ndarray]:
    p = defense.predict(x, gen)
    prob = defense.probability(p.score).numpy()
    flagged = p.flagged.numpy()
    return {"prob": prob, "flagged": flagged,
            "low": (prob < LOW_THRESHOLD) & ~flagged, "high": (prob >= HIGH_THRESHOLD) & ~flagged}


def batched(fn, x: torch.Tensor, size: int = 4):
    return torch.cat([fn(x[i : i + size]) for i in range(0, len(x), size)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="ensemble", choices=["ensemble", "bundled"],
                        help="Detector the defenses wrap: the shipped ensemble or the bundled model alone")
    parser.add_argument("--defense", required=True, choices=["none", "transform", "smoothing", "consistency", "ensemble"])
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument("--bundled", type=Path, default=Path("app/src/main/assets/models/ai-image-detector.onnx"))
    parser.add_argument("--commfor-onnx", type=Path, default=Path("app/src/main/assets/models/commfor-224.onnx"))
    parser.add_argument("--ensemble-params", type=Path, default=None,
                        help="ensemble-params.json for a candidate model set (default: the app's APP_ENSEMBLE)")
    parser.add_argument("--clean-per-class", type=int, default=60)
    parser.add_argument("--attack-per-class", type=int, default=24)
    parser.add_argument("--eps", default="4,8", help="L-inf budgets in /255")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--eot", type=int, default=4)
    parser.add_argument("--square-queries", type=int, default=200)
    parser.add_argument("--square-images", type=int, default=12)
    parser.add_argument("--attacks", default="fgsm,pgd,transfer,framing,square",
                        help="Comma-separated subset of " + ",".join(ATTACKS))
    parser.add_argument("--uap-fit-per-class", type=int, default=40,
                        help="Images per class the universal perturbation is fit on (disjoint from test)")
    parser.add_argument("--uap-epochs", type=int, default=8)
    parser.add_argument("--surrogates", default="commfor-384,ateeqq-siglip,dima806-vit",
                        help="Open detectors not in the app, for transfer-ext (tools/eval_candidates.py names)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--smoke", action="store_true", help="Tiny run to check the pipeline end to end")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    attacks = {a.strip() for a in args.attacks.split(",") if a.strip()}
    unknown = attacks - set(ATTACKS)
    if unknown:
        parser.error(f"unknown attacks {sorted(unknown)}; choose from {ATTACKS}")
    if args.smoke:
        args.clean_per_class, args.attack_per_class, args.steps, args.eot = 4, 2, 2, 2
        args.square_queries, args.square_images, args.eps = 6, 1, "8"
        args.uap_fit_per_class, args.uap_epochs = 2, 1

    started = time.time()
    gen = torch.Generator().manual_seed(args.seed)
    bundled = Bundled(args.bundled)

    # Parity: the differentiable copy must match the shipped ONNX before any number counts.
    import onnxruntime as ort

    session = ort.InferenceSession(str(args.bundled), providers=["CPUExecutionProvider"])
    probe = load_split(args.datasets, 4, args.seed + 1)
    probe_x = torch.stack([t for t, _, _ in probe])
    with torch.no_grad():
        torch_gaps = bundled(probe_x, app_jpeg=False)
    ort_gaps = [logit_difference(session.run(None, {INPUT_NAME: ((x.unsqueeze(0) - MEAN_T) / STD_T).numpy()})[0])
                for x in probe_x]
    parity = float(np.max(np.abs(torch_gaps.numpy() - np.array(ort_gaps))))
    print(f"onnx2torch parity (max |gap diff|): {parity:.5f}")
    if parity > 1e-2:
        sys.exit(f"Differentiable copy doesn't match the shipped model (max diff {parity}); refusing to report.")

    # Community Forensics: the ensemble's partner, and the surrogate for transfer attacks
    # on single-model defenses (no gradients from the target).
    cf = CommunityForensics(args.commfor_onnx)
    from evaluate import CommforOnnx

    shipped_cf = CommforOnnx(args.commfor_onnx)
    with torch.no_grad():
        torch_cf = cf(probe_x, app_jpeg=False)
    onnx_cf = [shipped_cf.gap(Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)))
               for x in probe_x]
    cf_parity = float(np.max(np.abs(torch_cf.numpy() - np.array(onnx_cf))))
    print(f"Community Forensics torch vs shipped ONNX parity (max |logit diff|): {cf_parity:.5f}")
    if cf_parity > 0.1:
        sys.exit(f"Community Forensics copy doesn't match the shipped model (max diff {cf_parity}); refusing to report.")

    if args.model == "ensemble":
        if args.defense == "ensemble":
            sys.exit("--defense ensemble refits the ensemble; with --model ensemble use --defense none.")
        from evaluate import load_ensemble_params

        params = load_ensemble_params(args.ensemble_params) if args.ensemble_params else APP_ENSEMBLE
        detector = Ensemble(bundled, cf, shipped=True, params=params)
        Defense.shipped_calibration = params["photo"]
    else:
        detector = bundled
    defense: Defense = {
        "none": lambda: Defense(detector),
        "transform": lambda: RandomTransform(detector),
        "smoothing": lambda: Smoothing(detector),
        "consistency": lambda: Consistency(detector),
        "ensemble": lambda: EnsembleDefense(Ensemble(bundled, cf)),
    }[args.defense]()

    clean = load_split(args.datasets, args.clean_per_class, args.seed)
    clean_x = torch.stack([t for t, _, _ in clean])
    clean_y = torch.tensor([y for _, y, _ in clean])
    defense.calibrate(clean_x, clean_y, gen)
    clean_out = {k: np.concatenate(v) for k, v in zip(
        ("prob", "flagged", "low", "high"),
        zip(*[outcomes(defense, clean_x[i : i + 8], gen).values() for i in range(0, len(clean_x), 8)]))}
    y = clean_y.numpy()
    result = {
        "model": args.model, "defense": args.defense, "cost": defense.cost, "parity": parity,
        "cf_parity": cf_parity,
        "clean": {
            "n": int(len(y)), "auc": roc_auc(y, clean_out["prob"]),
            "real_shown_high": float(clean_out["high"][y == 0].mean()),
            "ai_shown_low": float(clean_out["low"][y == 1].mean()),
            "flagged_real": float(clean_out["flagged"][y == 0].mean()),
            "flagged_ai": float(clean_out["flagged"][y == 1].mean()),
        },
        "attacks": [],
    }
    print(json.dumps(result["clean"], indent=2))

    attack_paths = select_paths(args.datasets, args.attack_per_class, args.seed + 2)
    attack_set = [(square_380(Image.open(p).convert("RGB")), label, name) for p, label, name in attack_paths]
    ai_x = torch.stack([t for t, lab, _ in attack_set if lab == 1])
    real_x = torch.stack([t for t, lab, _ in attack_set if lab == 0])
    # Transfer: crafted with gradients from a model that isn't (all of) the target. Against
    # an ensemble target the surrogate is the bundled model alone (the attacker knows one of
    # the two models); against the bundled model alone it is Community Forensics.
    ensemble_target = args.model == "ensemble" or args.defense == "ensemble"
    surrogate = Defense(bundled) if ensemble_target else Defense(cf)
    surrogate.slope, surrogate.intercept = 1.0, 0.0  # both surrogates: higher logit = more AI
    result["surrogate"] = "bundled" if ensemble_target else "community-forensics"

    def record(name: str, eps: float, adv: torch.Tensor, goal: str) -> None:
        for laundered in (False, True):
            x = jpeg(adv, 75) if laundered else adv
            o = {k: np.concatenate(v) for k, v in zip(
                ("prob", "flagged", "low", "high"),
                zip(*[outcomes(defense, x[i : i + 8], gen).values() for i in range(0, len(x), 8)]))}
            success = o["low"] if goal == "evasion" else o["high"]
            result["attacks"].append({
                "attack": name, "eps": eps, "goal": goal, "laundered": laundered, "n": int(len(x)),
                "success": float(success.mean()), "flagged": float(o["flagged"].mean()),
                "mean_prob": float(o["prob"].mean()),
            })
            print(f"{name:>20} eps={eps:.0f}/255 {goal:<8} {'laundered' if laundered else 'direct':<9} "
                  f"success={success.mean():.1%} flagged={o['flagged'].mean():.1%} mean P={o['prob'].mean():.3f}",
                  flush=True)

    if "uap" in attacks:
        # Fit images are disjoint from the clean and test images: universality is measured on
        # images the pattern has never seen.
        clean_paths = select_paths(args.datasets, args.clean_per_class, args.seed)
        used = frozenset(p for p, _, _ in clean_paths + attack_paths)
        fit = load_split(args.datasets, args.uap_fit_per_class, args.seed + 3, exclude=used)
        fit_ai = torch.stack([t for t, lab, _ in fit if lab == 1])
        fit_real = torch.stack([t for t, lab, _ in fit if lab == 0])
        result["uap_fit"] = {"ai": int(len(fit_ai)), "real": int(len(fit_real)), "epochs": args.uap_epochs}
    surrogates = []
    if "transfer-ext" in attacks:
        surrogates = load_surrogates([n for n in args.surrogates.split(",") if n], clean_x[:16])
        result["external_surrogates"] = [s.name for s in surrogates]

    for eps_255 in [float(e) for e in args.eps.split(",")]:
        eps = eps_255 / 255
        if "fgsm" in attacks:
            record("fgsm", eps_255, batched(lambda b: fgsm(defense, b, eps, gen, args.eot, 1.0), ai_x), "evasion")
        if "pgd" in attacks:
            record("pgd (eot)", eps_255,
                   batched(lambda b: pgd(defense, b, eps, args.steps, gen, args.eot, 1.0), ai_x), "evasion")
            if isinstance(defense, Consistency):
                record("pgd adaptive", eps_255,
                       batched(lambda b: pgd(defense, b, eps, args.steps, gen, args.eot, 1.0, adaptive=True), ai_x),
                       "evasion")
        if "transfer" in attacks:
            record("transfer (surrogate)", eps_255,
                   batched(lambda b: pgd(surrogate, b, eps, args.steps, gen, 1, 1.0), ai_x), "evasion")
        if "framing" in attacks:
            record("pgd framing", eps_255,
                   batched(lambda b: pgd(defense, b, eps, args.steps, gen, args.eot, -1.0,
                                         adaptive=isinstance(defense, Consistency)), real_x), "framing")
        if "uap" in attacks:
            for goal, direction, fit_x, test_x in (("evasion", 1.0, fit_ai, ai_x), ("framing", -1.0, fit_real, real_x)):
                delta = fit_uap(defense, fit_x, eps, args.uap_epochs, gen, direction)
                adv = (test_x + delta).clamp(0, 1)
                record("uap", eps_255, adv, goal)
                geometry_gen = torch.Generator().manual_seed(args.seed + 4)
                record("uap (cropped/rescaled)", eps_255,
                       torch.cat([random_geometry(a.unsqueeze(0), geometry_gen) for a in adv]), goal)
        if surrogates:
            record("transfer-ext (mi-di)", eps_255,
                   batched(lambda b: mi_di_fgsm(surrogates, b, eps, args.steps, gen, 1.0), ai_x), "evasion")
            record("transfer-ext framing", eps_255,
                   batched(lambda b: mi_di_fgsm(surrogates, b, eps, args.steps, gen, -1.0), real_x), "framing")
    if "square" in attacks:
        square_x = ai_x[: args.square_images]
        record("square (black box)", 8.0, square_attack(defense, square_x, 8 / 255, args.square_queries, gen, 1.0),
               "evasion")

    result["seconds"] = round(time.time() - started)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"adv__{args.model}__{args.defense}.json").write_text(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()

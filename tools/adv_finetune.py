#!/usr/bin/env python3
"""Fine-tuning of the app's two detectors: Phase 3 (accuracy, everyday edits, universal
attacks) and Phase 2b (#15, per-image adversarial training).

Phase 3 objectives (internal-docs/MODEL.md "Phase 3"):
  aug       clean BCE on images that get a random everyday edit (tools/evaluate.py
            EDIT_CONDITIONS: noise, filters, rescaling, crops, screenshots, ...) 60% of
            the time, else original / jpeg75 / social. The accuracy lever.
  uat       universal adversarial training (Shafahi et al. 2020): two perturbations
            shared by the whole dataset (one pushes AI images toward "real", one real
            images toward "AI"), randomly cropped/rescaled each step. One backward pass
            trains the model on clean + perturbed images and gives the perturbations'
            gradient, which they ascend. About the cost of training on 2x the images.
  aug+uat   both (the Phase 3 default in notebooks/robust_finetune_kaggle.ipynb)
Validation for these: clean AUC/accuracy, AUC under a fixed set of edits, and robust
accuracy against a FRESH universal perturbation fit on one half of the validation images
against the current model and scored on the other half (--eps). Selection: the eligible
checkpoint with the best edited AUC (aug), universal robust accuracy (uat), or their
mean (aug+uat).

Phase 2b objectives, kept for reference (they gained no robustness, see MODEL.md):

Fine-tunes one of the shipped models so small, invisible perturbations can't flip it.
Objective: TRADES (Zhang et al. 2019), loss = BCE(clean) + beta * KL(clean || adv), where
every image gets an adversarial copy crafted with PGD-5 to maximize that KL, in [0,1]
pixel space before ImageNet normalization (exactly where tools/adv_eval.py attacks).
--objective pgd-at trains on attacked images instead (Madry et al.).

History (internal-docs/MODEL.md): attempt 1 (PGD-AT, lr 1e-5) learned no robustness;
attempt 2 (TRADES beta 6, lr 1e-4) collapsed to a constant 50/50 output. The defaults
are therefore conservative: lr 2e-5 (commfor) / 1e-5 (bundled); beta ramps 0 -> 3 over
2 epochs and the training eps ramps 0 -> 4/255 over 1 epoch; validation every half
epoch; training stops if clean AUC falls more than 0.05 below the starting model; and
a checkpoint is only eligible if clean AUC stays within 0.01 and clean accuracy within
5 points of the start. If nothing is eligible, nothing is saved or uploaded.

  --model commfor   Community Forensics ViT-S 224 (OwensLab/commfor-model-224, MIT)
  --model bundled   Dafilab/ai-image-detector EfficientNet-B4 (Apache-2.0)

Training data comes from tools/fetch_eval_data.py --split train --strict-split. The
benchmarks only ever read the datasets' *test* splits, so there is no train/test
leakage. Train on Defactify only: fine-tuning on a dataset's train split makes its test
split in-distribution, so MJ/DALL-E/SD/NBP and the two video sets stay held out and are
what the clean-accuracy gates use (internal-docs/MODEL.md). Images get the app's own
preprocessing (tools/evaluate.py: a random original / jpeg75 / social condition, then
app_normalize, then the model's view).

Validation (every half epoch): clean AUC/accuracy, and robust accuracy under PGD-10 at
each --eval-eps (default 2 and 4 /255). The saved checkpoint is the eligible validation
point with the best robust accuracy at --eps.

Outputs in --out: <model>-robust[-tag].pt (state_dict) and <model>-train-log[-tag].json
(numbers only). --push-to-hub uploads both to a *private* Hugging Face model repo. Nothing
else leaves the machine.

Needs a GPU for real runs (Colab / Kaggle T4 at the defaults with 3,000 images per class:
about 45-60 min for commfor, 2-3 h for bundled). --dry-run uses randomly initialised
architectures and a few synthetic images, to test the whole loop on a CPU without downloads.

Usage:
    python tools/adv_finetune.py --model commfor \\
        --train-data data/train/defactify --val-data data/val/defactify \\
        --tag v3 --out runs --push-to-hub you/genned-robust
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (  # noqa: E402
    CONDITIONS,
    EDIT_CONDITIONS,
    MEAN,
    STD,
    app_normalize,
    collect_images,
    commfor_view,
    condition_seed,
    degrade,
    roc_auc,
    to_model_input,
)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MEAN_T = torch.tensor(MEAN).view(1, 3, 1, 1)
STD_T = torch.tensor(STD).view(1, 3, 1, 1)
RECIPE_VERSION = "phase3-v3"  # checked by notebooks/robust_finetune_kaggle.ipynb
PHASE2B_RECIPE = "adv-finetune-v3"  # checked by notebooks/adversarial_finetune.ipynb (Phase 2b)
PHASE3_OBJECTIVES = ("aug", "uat", "aug+uat")
# Edits the validation AUC is measured under (a spread of the everyday-edit families).
VAL_EDITS = ("noise4", "filter", "rescale", "webp50", "screenshot", "chain_shot")
CHECKPOINT_NAME = "{model}-robust{suffix}.pt"
LOG_NAME = "{model}-train-log{suffix}.json"
STATE_NAME = "{model}-state{suffix}.pt"
FINAL_NAME = "{model}-final{suffix}.pt"


# --------------------------------------------------------------------------- models

class Detector(nn.Module):
    """Wraps a backbone so forward(x in [0,1]) returns the AI logit (higher = more AI):
    the bundled model's ai - human gap, or Community Forensics' single logit."""

    def __init__(self, net: nn.Module, kind: str):
        super().__init__()
        self.net, self.kind = net, kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - MEAN_T.to(x.device)) / STD_T.to(x.device)
        out = self.net(z)
        return out[:, 0] - out[:, 1] if self.kind == "bundled" else out.reshape(-1)


def build_commfor(dry_run: bool) -> nn.Module:
    if not dry_run:
        from eval_candidates import CommunityForensics

        candidate = CommunityForensics(224)
        candidate.load()
        return candidate.model
    import timm

    class ViTClassifier(nn.Module):  # same module structure (and state-dict keys) as the real one
        def __init__(self):
            super().__init__()
            self.vit = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=False)
            self.vit.head = nn.Linear(384, 1)

        def forward(self, x):
            return self.vit(x)

    return ViTClassifier()


def build_bundled(dry_run: bool) -> nn.Module:
    import timm

    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=2)
    if not dry_run:
        from huggingface_hub import hf_hub_download

        from convert_model import CHECKPOINT_FILENAME, MODEL_ID, _load_state_dict

        path = hf_hub_download(repo_id=MODEL_ID, filename=CHECKPOINT_FILENAME)
        missing, unexpected = model.load_state_dict(_load_state_dict(Path(path)), strict=False)
        if missing or unexpected:
            sys.exit(f"Dafilab checkpoint didn't match efficientnet_b4: missing={missing} unexpected={unexpected}")
    return model


def freeze_batchnorm(module: nn.Module) -> None:
    """Keep BatchNorm running statistics fixed: small fine-tuning batches would corrupt them."""
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


# --------------------------------------------------------------------------- data

class Images(torch.utils.data.Dataset):
    """(image tensor in [0,1] at the model's view size, label 1 = AI) from fetched folders."""

    def __init__(self, dirs: list[Path], per_class: int, kind: str, train: bool, seed: int,
                 edit_share: float = 0.0, condition: str = "original", items: list | None = None):
        """train: random condition per image (an everyday edit with probability edit_share,
        else original / jpeg75 / social) and random flips/views. Not train: the fixed
        `condition`, seeded per image so every evaluation sees the same pixels."""
        self.items: list[tuple[Path, int]] = []
        if items is not None:
            self.items = list(items)
        else:
            rng = random.Random(seed)
            for d in dirs:
                for label, sub in ((1, "ai"), (0, "real")):
                    paths = collect_images(d / sub)
                    rng.shuffle(paths)
                    self.items += [(p, label) for p in paths[:per_class]]
        if not self.items:
            sys.exit(f"No images under {dirs}")
        self.kind, self.train, self.edit_share, self.condition = kind, train, edit_share, condition

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        image = Image.open(path).convert("RGB")
        if self.train:
            pool = EDIT_CONDITIONS if random.random() < self.edit_share else CONDITIONS
            image = degrade(image, random.choice(pool), random.getrandbits(32))
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif self.condition != "original":
            image = degrade(image, self.condition, condition_seed(path, self.condition))
        image = app_normalize(image)
        if self.kind == "commfor":
            view = commfor_view(image)
        else:
            view = to_model_input(image, random.choice(("squash", "center_crop")) if self.train else "squash")
        x = torch.from_numpy(np.asarray(view, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return x, torch.tensor(float(label))


# --------------------------------------------------------------------------- training

def binary_kl(p_logit: torch.Tensor, q_logit: torch.Tensor) -> torch.Tensor:
    """Per-sample KL(Bernoulli(sigmoid p) || Bernoulli(sigmoid q)), from logits (stable)."""
    p = torch.sigmoid(p_logit)
    return (p * (F.logsigmoid(p_logit) - F.logsigmoid(q_logit))
            + (1 - p) * (F.logsigmoid(-p_logit) - F.logsigmoid(-q_logit)))


def attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float, alpha: float, steps: int, amp: bool,
           clean_logit: torch.Tensor | None = None) -> torch.Tensor:
    """L-inf PGD in [0,1] pixel space. With clean_logit (TRADES) it maximizes the KL between the
    clean and adversarial predictions, starting next to x; otherwise it maximizes the loss on
    the true label from a random start (PGD-AT, and the evaluation attack)."""
    if clean_logit is None:
        adv = (x + eps * (2 * torch.rand_like(x) - 1)).clamp(0, 1)
    else:
        adv = (x + 0.001 * torch.randn_like(x)).clamp(0, 1)
    for _ in range(steps):
        adv.requires_grad_(True)
        with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=amp):
            logit = model(adv).float()
        loss = (binary_kl(clean_logit, logit).sum() if clean_logit is not None
                else F.binary_cross_entropy_with_logits(logit, y))
        grad, = torch.autograd.grad(loss, adv)
        adv = (x + (adv.detach() + alpha * grad.sign() - x).clamp(-eps, eps)).clamp(0, 1)
    return adv.detach()


def evaluate(model: nn.Module, loader, device, eps_list: list[float], amp: bool) -> dict:
    """Clean AUC/accuracy, and accuracy under PGD-10 (random start, step eps/4) at each budget."""
    model.eval()
    scores, labels = [], []
    robust = {e: [] for e in eps_list}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            s = model(x).float()
        scores += s.cpu().tolist()
        labels += y.cpu().tolist()
        for e in eps_list:
            adv = attack(model, x, y, e / 255, e / 255 / 4, 10, amp)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                s_adv = model(adv).float()
            robust[e] += ((s_adv > 0).float() == y).cpu().tolist()
    labels_a, scores_a = np.array(labels), np.array(scores)
    out = {"clean_auc": roc_auc(labels_a, scores_a), "clean_acc": float(((scores_a > 0) == labels_a).mean()),
           "n": len(labels)}
    for e in eps_list:
        out[f"robust_acc_{e:g}"] = float(np.mean(robust[e]))
    return out


def jitter(delta: torch.Tensor, min_scale: float = 0.85) -> torch.Tensor:
    """Random crop of 85-100% of a perturbation, resized back: where a fixed pattern lands
    after a repost, crop or screenshot (as tools/adv_eval.py random_geometry)."""
    size = delta.shape[-1]
    crop = max(8, int(size * (min_scale + (1 - min_scale) * random.random())))
    top, left = random.randint(0, size - crop), random.randint(0, size - crop)
    return F.interpolate(delta[:, :, top : top + crop, left : left + crop], size=delta.shape[-2:],
                         mode="bilinear", align_corners=False)


def fit_universal(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, eps: float, epochs: int,
                  batch: int, device, amp: bool) -> torch.Tensor:
    """Fresh universal perturbations against the current model (validation attacker): one for
    the AI images (push toward real) and one for the real images (push toward AI), each by
    minibatch sign-ascent on BCE through random crops/rescales. Returns (2, 3, H, W):
    index 1 for AI images, 0 for real."""
    deltas = torch.zeros(2, 3, *images.shape[-2:], device=device)
    alpha = eps / 8
    model.eval()
    for _ in range(epochs):
        order = torch.randperm(len(images))
        for i in range(0, len(images), batch):
            idx = order[i : i + batch]
            x, y = images[idx].to(device), labels[idx].to(device)
            d = deltas.clone().requires_grad_(True)
            per_image = d[y.long()]
            adv = (x + torch.cat([jitter(per_image[j : j + 1]) for j in range(len(x))])).clamp(0, 1)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logit = model(adv).float()
            grad, = torch.autograd.grad(F.binary_cross_entropy_with_logits(logit, y), d)
            deltas = (deltas + alpha * grad.sign()).clamp(-eps, eps).detach()
    return deltas


def prepare_views(dataset, workers: int, batch: int, label: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads a validation set once, in parallel, as uint8 tensors (exact: the views are uint8
    pixels / 255), so every later validation is GPU-only. Decoding and editing full-size
    images costs 0.1-0.7 s each, which made each validation take ~20 silent minutes."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, num_workers=workers)
    xs, ys, done, started = [], [], 0, time.time()
    for x, y in loader:
        xs.append((x * 255).round().to(torch.uint8))
        ys.append(y)
        done += len(x)
        if done % (batch * 10) < batch or done == len(dataset):
            print(f"  preparing validation images ({label}): {done}/{len(dataset)} "
                  f"({time.time() - started:.0f}s)", flush=True)
    return torch.cat(xs), torch.cat(ys)


def split_halves(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed, stratified split for the validation universal attack: images of each class
    alternate between the fit half and the test half, so both halves hold AI and real
    images. (The validation set lists all AI images first, so a plain first/second half
    split fits the patterns on AI images only and tests on real ones only.)"""
    fit, test = [], []
    for label in (0.0, 1.0):
        idx = (y == label).nonzero().flatten().tolist()
        fit += idx[0::2]
        test += idx[1::2]
    return torch.tensor(sorted(fit), dtype=torch.long), torch.tensor(sorted(test), dtype=torch.long)


def evaluate_phase3(model: nn.Module, views: dict[str, tuple[torch.Tensor, torch.Tensor]], device, eps: float,
                    amp: bool, batch: int, uap_epochs: int) -> dict:
    """Clean AUC/accuracy, AUC under each validation edit, and accuracy of the other half of the
    validation images under a fresh universal perturbation fit on the first half. `views` maps
    "clean" and each edit name to prepare_views() output."""
    model.eval()

    def scores_of(x_u8: torch.Tensor, y: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        s = []
        for i in range(0, len(x_u8), batch):
            xb = x_u8[i : i + batch].to(device).float() / 255
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                s += model(xb).float().cpu().tolist()
        return y.numpy(), np.array(s)

    labels, scores = scores_of(*views["clean"])
    out = {"clean_auc": roc_auc(labels, scores), "clean_acc": float(((scores > 0) == labels).mean()), "n": len(labels)}
    edit_aucs = {}
    for name, (x_u8, y) in views.items():
        if name != "clean":
            l, s = scores_of(x_u8, y)
            edit_aucs[name] = roc_auc(l, s)
    out["edit_auc"] = edit_aucs
    valid = [v for v in edit_aucs.values() if v is not None]
    out["edit_auc_mean"] = float(np.mean(valid)) if valid else None
    out["edit_auc_min"] = float(np.min(valid)) if valid else None

    x_u8, y = views["clean"]
    x = x_u8.float() / 255
    fit_idx, test_idx = split_halves(y)
    deltas = fit_universal(model, x[fit_idx], y[fit_idx], eps, uap_epochs, batch, device, amp)
    test_x, test_y = x[test_idx].to(device), y[test_idx].to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        clean_logit = model(test_x).float()
        adv_logit = model((test_x + deltas[test_y.long()]).clamp(0, 1)).float()
    # Robust = right on the clean image AND under the pattern, so a model that gets an image
    # wrong either way (or a degenerate constant model) can't score as robust.
    robust = ((clean_logit > 0).float() == test_y) & ((adv_logit > 0).float() == test_y)
    out[f"uap_robust_acc_{eps * 255:g}"] = float(robust.float().mean())
    # Per class: AI images that stay AI under the evasion pattern, real ones that stay real
    # under the framing pattern.
    out[f"uap_robust_acc_ai_{eps * 255:g}"] = float(robust[test_y == 1].float().mean())
    out[f"uap_robust_acc_real_{eps * 255:g}"] = float(robust[test_y == 0].float().mean())
    return out


def selection_score(objective: str, metrics: dict, eps_255: float) -> float:
    """What the Phase 3 objectives maximize among eligible checkpoints."""
    edited = metrics.get("edit_auc_mean") or 0.0
    universal = metrics[f"uap_robust_acc_{eps_255:g}"]
    return {"aug": edited, "uat": universal, "aug+uat": (edited + universal) / 2}[objective]


MODEL_DEFAULTS = {  # lr, epochs, batch
    "commfor": (2e-5, 6, 32),
    "bundled": (1e-5, 4, 16),
}


def assess(metrics: dict, base: dict, auc_gate: float, acc_drop: float, collapse_drop: float) -> tuple[bool, bool]:
    """(eligible, collapsed) for a validation result against the starting model.

    eligible: clean AUC within auc_gate of the start and clean accuracy within acc_drop, so a
      checkpoint can't be chosen for "robustness" that is really a degenerate constant output
      (attempt 2: a 50/50 model scored 31.7% "robust").
    collapsed: clean AUC fell more than collapse_drop below the start - stop training."""
    base_auc, auc = base["clean_auc"], metrics["clean_auc"]
    if base_auc is None or auc is None:
        return True, False
    eligible = auc >= base_auc - auc_gate and metrics["clean_acc"] >= base["clean_acc"] - acc_drop
    return eligible, auc < base_auc - collapse_drop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["commfor", "bundled"])
    parser.add_argument("--train-data", type=Path, nargs="+")
    parser.add_argument("--val-data", type=Path, nargs="+")
    parser.add_argument("--per-class", type=int, default=3000, help="Training images per class per dataset")
    parser.add_argument("--val-per-class", type=int, default=150)
    parser.add_argument("--objective", choices=["trades", "pgd-at", *PHASE3_OBJECTIVES], default="trades",
                        help="Phase 3: aug, uat, aug+uat. Phase 2b: TRADES (clean CE + beta * KL(clean || adv)) "
                             "or PGD-AT (CE on attacked images)")
    parser.add_argument("--edit-share", type=float, default=0.6,
                        help="aug objectives: share of training images given a random everyday edit")
    parser.add_argument("--uat-weight", type=float, default=1.0, help="uat objectives: weight of the perturbed loss")
    parser.add_argument("--uat-step", type=float, default=0.25,
                        help="uat objectives: universal perturbation step, as a fraction of the current eps")
    parser.add_argument("--val-uap-epochs", type=int, default=3,
                        help="Phase 3 validation: epochs to fit the fresh universal perturbation")
    parser.add_argument("--beta", type=float, default=3.0, help="TRADES robustness weight (after the ramp)")
    parser.add_argument("--beta-ramp-epochs", type=float, default=2.0, help="Linear ramp of beta from 0")
    parser.add_argument("--eps-ramp-epochs", type=float, default=1.0, help="Linear ramp of the training eps from 0")
    parser.add_argument("--adv-fraction", type=float, default=1.0,
                        help="PGD-AT only: share of each batch that is attacked")
    parser.add_argument("--epochs", type=int, default=None, help="Default: 6 commfor, 4 bundled")
    parser.add_argument("--batch-size", type=int, default=None, help="Default: 32 commfor, 16 bundled")
    parser.add_argument("--lr", type=float, default=None, help="Default: 2e-5 commfor, 1e-5 bundled")
    parser.add_argument("--warmup", type=float, default=0.05, help="Share of steps with linear LR warmup")
    parser.add_argument("--eps", type=float, default=None,
                        help="L-inf budget in /255 (training and selection). Default 8 for Phase 3 "
                             "(universal), 4 for Phase 2b (per image)")
    parser.add_argument("--train-steps", type=int, default=5,
                        help="PGD steps when crafting training examples (step size = current eps / 4)")
    parser.add_argument("--eval-eps", default="2,4", help="Budgets (/255) for validation robust accuracy")
    parser.add_argument("--evals-per-epoch", type=int, default=2)
    parser.add_argument("--auc-gate", type=float, default=0.01, help="Eligible: clean AUC >= start - this")
    parser.add_argument("--acc-drop", type=float, default=0.05, help="Eligible: clean accuracy >= start - this")
    parser.add_argument("--collapse-drop", type=float, default=0.05, help="Stop: clean AUC < start - this")
    parser.add_argument("--tag", default="", help="Suffix for the output names, e.g. v3 -> commfor-robust-v3.pt")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="Data-loading processes (image decoding and edits are the bottleneck, not the GPU)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--push-to-hub", default=None, help="Private HF model repo to upload to, e.g. you/genned-robust")
    parser.add_argument("--dry-run", action="store_true", help="Random-init models + synthetic images, CPU-friendly")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="Stop training after this long (e.g. under a CI job limit): validate once more, save the "
                             "best checkpoint and, with --save-state, everything needed to continue")
    parser.add_argument("--save-state", action="store_true",
                        help="Write <model>-state[-tag].pt (model, optimizer, schedule, perturbations, step, log) so a "
                             "later run can continue with --resume-state")
    parser.add_argument("--resume-state", type=Path, default=None,
                        help="Continue a run saved with --save-state (same data and settings)")
    parser.add_argument("--torch-threads", type=int, default=None,
                        help="CPU compute threads (leave cores for the data workers on CPU-only machines)")
    args = parser.parse_args()
    phase3 = args.objective in PHASE3_OBJECTIVES
    uses_aug = args.objective in ("aug", "aug+uat")
    uses_uat = args.objective in ("uat", "aug+uat")
    if args.eps is None:
        args.eps = 8.0 if phase3 else 4.0

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    lr_default, epochs_default, batch_default = MODEL_DEFAULTS[args.model]
    lr = args.lr or lr_default
    epochs = args.epochs or epochs_default
    batch = args.batch_size or batch_default
    eval_eps = sorted({float(e) for e in args.eval_eps.split(",")} | {args.eps})
    key_name = f"uap_robust_acc_{args.eps:g}" if phase3 else f"robust_acc_{args.eps:g}"
    suffix = f"-{args.tag}" if args.tag else ""
    args.out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        args.train_data = [synthetic_dataset(args.out / "dry-run-data" / "train", 4, args.seed)]
        args.val_data = [synthetic_dataset(args.out / "dry-run-data" / "val", 2, args.seed + 1)]
        args.per_class, args.val_per_class, batch = 4, 2, 2
    if not args.train_data or not args.val_data:
        parser.error("--train-data and --val-data are required (or use --dry-run)")

    net = build_commfor(args.dry_run) if args.model == "commfor" else build_bundled(args.dry_run)
    model = Detector(net, args.model).to(device)
    train_set = Images(args.train_data, args.per_class, args.model, train=True, seed=args.seed,
                       edit_share=args.edit_share if uses_aug else 0.0)
    val_set = Images(args.val_data, args.val_per_class, args.model, train=False, seed=args.seed + 1)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch, shuffle=True, drop_last=True, num_workers=args.workers, pin_memory=amp,
        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers > 0 else None)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch, num_workers=args.workers)
    steps_per_epoch = len(train_loader)
    total_steps = max(1, epochs * steps_per_epoch)
    eval_every = max(1, steps_per_epoch // max(1, args.evals_per_epoch))
    print(f"{args.model}: {len(train_set)} training / {len(val_set)} validation images, device {device}, "
          f"{args.objective}, eps {args.eps}/255 (ramp {args.eps_ramp_epochs} ep), beta {args.beta} "
          f"(ramp {args.beta_ramp_epochs} ep), lr {lr}, {epochs} epochs, batch {batch}", flush=True)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=0.01)
    warmup_steps = max(1, int(args.warmup * total_steps))

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))

    def ramp(step: int, ramp_epochs: float) -> float:
        return 1.0 if ramp_epochs <= 0 else min(1.0, step / (ramp_epochs * steps_per_epoch))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    views: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    if phase3:
        # The same validation images clean and under each edit (fixed per-image seeds), prepared once.
        views["clean"] = prepare_views(val_set, args.workers, batch, "clean")
        for e in VAL_EDITS:
            views[e] = prepare_views(Images([], 0, args.model, train=False, seed=0, condition=e, items=val_set.items),
                                     args.workers, batch, e)

    def validate() -> dict:
        started_v = time.time()
        print("validating...", flush=True)
        if phase3:
            metrics = evaluate_phase3(model, views, device, args.eps / 255, amp, batch, args.val_uap_epochs)
        else:
            metrics = evaluate(model, val_loader, device, eval_eps, amp)
        print(f"  validation took {time.time() - started_v:.0f}s", flush=True)
        return metrics

    resumed = torch.load(args.resume_state, map_location="cpu", weights_only=False) if args.resume_state else None
    if resumed is not None:
        if resumed["recipe"] != (RECIPE_VERSION if phase3 else PHASE2B_RECIPE):
            sys.exit(f"--resume-state {args.resume_state} is from recipe {resumed['recipe']}; refusing to continue.")
        if resumed["total_steps"] != total_steps:
            # A re-download can differ by a few images; keep the original schedule (the LR
            # schedule and ramps read total_steps / warmup_steps when called).
            print(f"note: this data gives {total_steps} steps, the saved run planned {resumed['total_steps']}; "
                  f"keeping the saved schedule", flush=True)
            total_steps = resumed["total_steps"]
            warmup_steps = max(1, int(args.warmup * total_steps))
        model.net.load_state_dict(resumed["net"])
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        scaler.load_state_dict(resumed["scaler"])
        random.setstate(resumed["rng"]["python"])
        np.random.set_state(resumed["rng"]["numpy"])
        torch.set_rng_state(resumed["rng"]["torch"])
        base = resumed["log"]["base"]  # the gates always compare with the model as shipped
        print(f"resuming at step {resumed['step']}/{total_steps} (start as shipped: {base})", flush=True)
    else:
        base = validate()
        print(f"start (as shipped): {base}", flush=True)
    view = views["clean"][0].shape[-2:] if phase3 else val_set[0][0].shape[-2:]
    deltas = torch.zeros(2, 3, *view, device=device)  # uat: index 1 for AI images, 0 for real
    if resumed is not None:
        deltas = resumed["deltas"].to(device)
    log = {"recipe": RECIPE_VERSION if phase3 else PHASE2B_RECIPE, "model": args.model,
           "objective": args.objective, "beta": args.beta, "edit_share": args.edit_share if uses_aug else 0.0,
           "uat_weight": args.uat_weight if uses_uat else None, "uat_step": args.uat_step if uses_uat else None,
           "eps_255": args.eps,
           "beta_ramp_epochs": args.beta_ramp_epochs, "eps_ramp_epochs": args.eps_ramp_epochs,
           "train_steps": args.train_steps, "lr": lr, "epochs": epochs, "batch": batch,
           "train_images": len(train_set), "val_images": len(val_set), "per_class": args.per_class,
           "train_data": [p.name for p in args.train_data], "dry_run": args.dry_run,
           "base": base, "evaluations": [], "stopped": None, "selected": None, "meets_clean_gate": False}
    best_robust, best_state = -1.0, None
    step = 0
    if resumed is not None:
        log = resumed["log"]
        best_robust, best_state, step = resumed["best_score"], resumed["best_state"], resumed["step"]
    log["jobs"] = log.get("jobs", 0) + 1
    started = time.time()
    validation_seconds = 0.0
    first_step = step
    timed_out = False
    sums = {"loss": 0.0, "clean": 0.0, "robust": 0.0, "batch_robust_acc": 0.0}
    since = 0

    while step < total_steps and not log["stopped"] and not timed_out:
        for x, y in train_loader:
            if step >= total_steps:
                break
            if args.max_hours and time.time() - started > args.max_hours * 3600:
                timed_out = True
                print(f"time limit ({args.max_hours} h) reached at step {step}/{total_steps}", flush=True)
                break
            epoch = step // steps_per_epoch + 1
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            cur_eps = args.eps / 255 * ramp(step, args.eps_ramp_epochs)
            cur_beta = args.beta * ramp(step, args.beta_ramp_epochs)
            if phase3:
                model.train()
                freeze_batchnorm(model)
                d = deltas.clone().requires_grad_(True)
                if uses_uat:
                    per_image = d[y.long()]
                    adv = (x + torch.cat([jitter(per_image[j : j + 1]) for j in range(len(x))])).clamp(0, 1)
                    batch_in = torch.cat([x, adv])
                else:
                    batch_in = x
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    logits = model(batch_in).float()
                clean_logit = logits[: len(x)]
                adv_logit = logits[len(x):] if uses_uat else clean_logit
                clean_loss = F.binary_cross_entropy_with_logits(clean_logit, y)
                robust_loss = F.binary_cross_entropy_with_logits(adv_logit, y) if uses_uat else clean_loss
                loss = clean_loss + (args.uat_weight * robust_loss if uses_uat else 0.0)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if uses_uat and d.grad is not None and torch.isfinite(d.grad).all() and cur_eps > 0:
                    # The perturbations ascend the same loss the model descends.
                    deltas = (deltas + args.uat_step * cur_eps * d.grad.sign()).clamp(-cur_eps, cur_eps).detach()
                scaler.step(optimizer)
                scale_before = scaler.get_scale()
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
            else:
                model.eval()  # craft the attack with frozen layers
                if cur_eps <= 0:
                    adv = x
                elif args.objective == "trades":
                    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                        clean_for_attack = model(x).float()
                    adv = attack(model, x, y, cur_eps, cur_eps / 4, args.train_steps, amp, clean_logit=clean_for_attack)
                else:
                    n_adv = int(round(args.adv_fraction * len(x)))
                    adv = torch.cat([attack(model, x[:n_adv], y[:n_adv], cur_eps, cur_eps / 4, args.train_steps, amp),
                                     x[n_adv:]])
                model.train()
                freeze_batchnorm(model)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    logits = model(torch.cat([x, adv])).float()
                clean_logit, adv_logit = logits[: len(x)], logits[len(x):]
                clean_loss = F.binary_cross_entropy_with_logits(clean_logit, y)
                if args.objective == "trades":
                    robust_loss = binary_kl(clean_logit, adv_logit).mean()
                    loss = clean_loss + cur_beta * robust_loss
                else:
                    robust_loss = F.binary_cross_entropy_with_logits(adv_logit, y)
                    loss = robust_loss
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scale_before = scaler.get_scale()
                scaler.update()
                if scaler.get_scale() >= scale_before:  # AMP took the step: advance the schedule
                    scheduler.step()
            step += 1
            since += 1
            sums["loss"] += loss.item()
            sums["clean"] += clean_loss.item()
            sums["robust"] += robust_loss.item()
            sums["batch_robust_acc"] += ((adv_logit.detach() > 0).float() == y).float().mean().item()
            if step == first_step + 1:
                print(f"first training step done after {time.time() - started:.0f}s "
                      f"({steps_per_epoch} steps per epoch)", flush=True)
            if step % 50 == 0 or (step - first_step <= 100 and step % 10 == 0):
                avg = {k: v / since for k, v in sums.items()}
                elapsed = time.time() - started - validation_seconds
                rate = (step - first_step) * batch / max(1e-6, elapsed)
                eta = (total_steps - step) * batch / max(1e-6, rate)
                print(f"epoch {epoch} step {step}/{total_steps} loss {avg['loss']:.4f} (clean {avg['clean']:.4f}, "
                      f"robust {avg['robust']:.4f}, in-batch robust acc {avg['batch_robust_acc']:.1%}) "
                      f"eps {cur_eps * 255:.2f}/255 beta {cur_beta:.2f} lr {scheduler.get_last_lr()[0]:.2e} "
                      f"- {rate:.1f} img/s, training ETA {eta / 60:.0f} min ({time.time() - started:.0f}s)",
                      flush=True)
            if step % eval_every != 0 and step != total_steps:
                continue

            started_v = time.time()
            metrics = validate()
            validation_seconds += time.time() - started_v
            train_avg = {f"train_{k}": v / max(1, since) for k, v in sums.items()}
            sums = {k: 0.0 for k in sums}
            since = 0
            eligible, collapsed = assess(metrics, base, args.auc_gate, args.acc_drop, args.collapse_drop)
            record = {"step": step, "epoch": round(step / steps_per_epoch, 2), "eps_255": round(cur_eps * 255, 3),
                      "beta": round(cur_beta, 3), "eligible": eligible, **metrics, **train_avg}
            log["evaluations"].append(record)
            if phase3:
                print(f"validation @ epoch {record['epoch']}: clean AUC {metrics['clean_auc']:.3f} "
                      f"(start {base['clean_auc']:.3f}), clean acc {metrics['clean_acc']:.1%}, edited AUC mean "
                      f"{metrics['edit_auc_mean']:.3f} / min {metrics['edit_auc_min']:.3f} (start "
                      f"{base['edit_auc_mean']:.3f} / {base['edit_auc_min']:.3f}), universal robust acc "
                      f"@{args.eps:g}/255 {metrics[key_name]:.1%} (start {base[key_name]:.1%}) - "
                      f"{'eligible' if eligible else 'NOT eligible'}", flush=True)
            else:
                print(f"validation @ epoch {record['epoch']}: clean AUC {metrics['clean_auc']:.3f} "
                      f"(start {base['clean_auc']:.3f}), clean acc {metrics['clean_acc']:.1%}, "
                      f"robust @2/255 {metrics.get('robust_acc_2', float('nan')):.1%}, "
                      f"@{args.eps:g}/255 {metrics[key_name]:.1%} - {'eligible' if eligible else 'NOT eligible'}",
                      flush=True)
            score = selection_score(args.objective, metrics, args.eps) if phase3 else metrics[key_name]
            if eligible and score > best_robust:
                best_robust = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.net.state_dict().items()}
                log["selected"], log["meets_clean_gate"] = record, True
            if collapsed:
                log["stopped"] = (f"clean AUC {metrics['clean_auc']:.3f} < {base['clean_auc'] - args.collapse_drop:.3f}"
                                  f" (start {base['clean_auc']:.3f} - {args.collapse_drop}) at epoch {record['epoch']}")
                print(f"STOPPED: {log['stopped']}", flush=True)
                break
            model.train()

    last_validated = log["evaluations"][-1]["step"] if log["evaluations"] else -1
    if timed_out and step > last_validated and step > first_step:
        # Don't lose the training since the last validation: score it once before stopping.
        metrics = validate()
        eligible, _ = assess(metrics, base, args.auc_gate, args.acc_drop, args.collapse_drop)
        record = {"step": step, "epoch": round(step / steps_per_epoch, 2), "eligible": eligible, **metrics}
        log["evaluations"].append(record)
        score = selection_score(args.objective, metrics, args.eps) if phase3 else metrics[key_name]
        print(f"validation @ epoch {record['epoch']} (time limit): clean AUC {metrics['clean_auc']:.3f}, "
              f"{key_name} {metrics[key_name]:.1%} - {'eligible' if eligible else 'NOT eligible'}", flush=True)
        if eligible and score > best_robust:
            best_robust = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.net.state_dict().items()}
            log["selected"], log["meets_clean_gate"] = record, True
    log["finished"] = step >= total_steps or bool(log["stopped"])
    log["step"] = step
    if args.save_state:
        state_path = args.out / STATE_NAME.format(model=args.model, suffix=suffix)
        torch.save({"recipe": RECIPE_VERSION if phase3 else PHASE2B_RECIPE, "total_steps": total_steps,
                    "step": step, "net": model.net.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "deltas": deltas.cpu(),
                    "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                            "torch": torch.get_rng_state()},
                    "log": log, "best_score": best_robust, "best_state": best_state}, state_path)
        print(f"Saved training state {state_path} (step {step}/{total_steps}"
              f"{'' if log['finished'] else '; continue with --resume-state'})", flush=True)

    log["seconds"] = round(time.time() - started)
    if log["finished"]:
        # The last weights too, whatever validation selected (benchmarks can compare both).
        final_path = args.out / FINAL_NAME.format(model=args.model, suffix=suffix)
        torch.save({k: v.detach().cpu() for k, v in model.net.state_dict().items()}, final_path)
        print(f"Saved final weights {final_path}", flush=True)
    checkpoint = args.out / CHECKPOINT_NAME.format(model=args.model, suffix=suffix)
    log_path = args.out / LOG_NAME.format(model=args.model, suffix=suffix)
    log_path.write_text(json.dumps(log, indent=2))
    if best_state is None:
        print(f"No usable checkpoint: no validation kept clean accuracy within the gate. Nothing saved or "
              f"uploaded. Log: {log_path}")
        return
    torch.save(best_state, checkpoint)
    sel = log["selected"]
    print(f"Saved {checkpoint} (epoch {sel['epoch']}): clean AUC {sel['clean_auc']:.3f} vs {base['clean_auc']:.3f}, "
          f"{key_name} {sel[key_name]:.1%} vs {base[key_name]:.1%} as shipped")

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to_hub, private=True, exist_ok=True)
        for path in (checkpoint, log_path):
            api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name, repo_id=args.push_to_hub)
        print(f"Uploaded to https://huggingface.co/{args.push_to_hub} (private)")


def synthetic_dataset(root: Path, per_class: int, seed: int) -> Path:
    """A few blurred-noise images in the fetch_eval_data layout, for --dry-run only."""
    from PIL import ImageFilter

    rng = np.random.default_rng(seed)
    for sub in ("ai/synthetic", "real"):
        (root / sub).mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            a = (rng.random((300 + 20 * i, 420, 3)) * 255).astype(np.uint8)
            Image.fromarray(a).filter(ImageFilter.GaussianBlur(1 + i % 3)).save(root / sub / f"{i}.png")
    return root


if __name__ == "__main__":
    main()

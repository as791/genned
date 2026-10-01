#!/usr/bin/env python3
"""Phase 2b (#15): adversarial fine-tuning of the app's two detectors.

Fine-tunes one of the shipped models so small, invisible perturbations can't flip it.
Default objective: TRADES (Zhang et al. 2019), loss = BCE(clean) + beta * KL(clean || adv),
where every image gets an adversarial copy crafted with PGD-5 at --eps (default 4/255) to
maximize that KL, in [0,1] pixel space before ImageNet normalization (exactly where
tools/adv_eval.py attacks). --objective pgd-at trains on attacked images (Madry et al.).

Attempt 1 (PGD-AT, half the batch attacked, lr 1e-5, 2 epochs) left Community Forensics
at 1% robust accuracy, so the defaults are now a much stronger budget: lr 1e-4 (commfor) /
5e-5 (bundled), 8 / 4 epochs, 5% warmup then cosine. The log shows clean loss, robust
loss and in-batch robust accuracy every 50 steps, so a run that isn't learning
robustness is visible early.

  --model commfor   Community Forensics ViT-S 224 (OwensLab/commfor-model-224, MIT)
  --model bundled   Dafilab/ai-image-detector EfficientNet-B4 (Apache-2.0)

Training data comes from tools/fetch_eval_data.py --split train --strict-split. The
benchmarks only ever read the datasets' *test* splits, so there is no train/test
leakage. Train on Defactify only: fine-tuning on a dataset's train split makes its test
split in-distribution, so MJ/DALL-E/SD/NBP and the two video sets stay held out and are
what the clean-accuracy gates use (internal-docs/MODEL.md). Images get the app's own
preprocessing (tools/evaluate.py: a random original / jpeg75 / social condition, then
app_normalize, then the model's view).

Every epoch is scored on a validation folder: clean AUC, and robust accuracy under
PGD-10 at each --eval-eps (default 2 and 4 /255). The saved checkpoint is the epoch with
the best robust accuracy at --eps whose clean AUC is within 0.01 of the starting model's. If no epoch qualifies, the last one is
saved and the log says so.

Outputs in --out: <model>-robust[-tag].pt (state_dict) and <model>-train-log[-tag].json
(numbers only). --push-to-hub uploads both to a *private* Hugging Face model repo. Nothing
else leaves the machine.

Needs a GPU for real runs (Colab / Kaggle T4 at the defaults with 3,000 images per class:
about 45 min for commfor, 2-3 h for bundled). --dry-run uses randomly initialised
architectures and a few synthetic images, to test the whole loop on a CPU without downloads.

Usage:
    python tools/adv_finetune.py --model commfor \\
        --train-data data/train/defactify --val-data data/val/defactify \\
        --tag v2 --out runs --push-to-hub you/genned-robust
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
    MEAN,
    STD,
    app_normalize,
    collect_images,
    commfor_view,
    degrade,
    roc_auc,
    to_model_input,
)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MEAN_T = torch.tensor(MEAN).view(1, 3, 1, 1)
STD_T = torch.tensor(STD).view(1, 3, 1, 1)
CHECKPOINT_NAME = "{model}-robust{suffix}.pt"
LOG_NAME = "{model}-train-log{suffix}.json"


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

    def __init__(self, dirs: list[Path], per_class: int, kind: str, train: bool, seed: int):
        rng = random.Random(seed)
        self.items: list[tuple[Path, int]] = []
        for d in dirs:
            for label, sub in ((1, "ai"), (0, "real")):
                paths = collect_images(d / sub)
                rng.shuffle(paths)
                self.items += [(p, label) for p in paths[:per_class]]
        if not self.items:
            sys.exit(f"No images under {dirs}")
        self.kind, self.train = kind, train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        image = Image.open(path).convert("RGB")
        if self.train:
            image = degrade(image, random.choice(CONDITIONS))
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
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


MODEL_DEFAULTS = {  # lr, epochs, batch
    "commfor": (1e-4, 8, 32),
    "bundled": (5e-5, 4, 16),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["commfor", "bundled"])
    parser.add_argument("--train-data", type=Path, nargs="+")
    parser.add_argument("--val-data", type=Path, nargs="+")
    parser.add_argument("--per-class", type=int, default=3000, help="Training images per class per dataset")
    parser.add_argument("--val-per-class", type=int, default=150)
    parser.add_argument("--objective", choices=["trades", "pgd-at"], default="trades",
                        help="TRADES (clean CE + beta * KL(clean || adv)) or PGD-AT (CE on attacked images)")
    parser.add_argument("--beta", type=float, default=6.0, help="TRADES robustness weight")
    parser.add_argument("--adv-fraction", type=float, default=1.0,
                        help="PGD-AT only: share of each batch that is attacked")
    parser.add_argument("--epochs", type=int, default=None, help="Default: 8 commfor, 4 bundled")
    parser.add_argument("--batch-size", type=int, default=None, help="Default: 32 commfor, 16 bundled")
    parser.add_argument("--lr", type=float, default=None, help="Default: 1e-4 commfor, 5e-5 bundled")
    parser.add_argument("--warmup", type=float, default=0.05, help="Share of steps with linear LR warmup")
    parser.add_argument("--eps", type=float, default=4.0, help="L-inf budget in /255 (training and selection)")
    parser.add_argument("--alpha", type=float, default=1.0, help="PGD step in /255 when crafting training examples")
    parser.add_argument("--train-steps", type=int, default=5, help="PGD steps when crafting training examples")
    parser.add_argument("--eval-eps", default="2,4", help="Budgets (/255) for validation robust accuracy")
    parser.add_argument("--tag", default="", help="Suffix for the output names, e.g. v2 -> commfor-robust-v2.pt")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--push-to-hub", default=None, help="Private HF model repo to upload to, e.g. you/genned-robust")
    parser.add_argument("--dry-run", action="store_true", help="Random-init models + synthetic images, CPU-friendly")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    lr_default, epochs_default, batch_default = MODEL_DEFAULTS[args.model]
    lr = args.lr or lr_default
    epochs = args.epochs or epochs_default
    batch = args.batch_size or batch_default
    eps, alpha = args.eps / 255, args.alpha / 255
    eval_eps = sorted({float(e) for e in args.eval_eps.split(",")} | {args.eps})
    key_name = f"robust_acc_{args.eps:g}"
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
    train_set = Images(args.train_data, args.per_class, args.model, train=True, seed=args.seed)
    val_set = Images(args.val_data, args.val_per_class, args.model, train=False, seed=args.seed + 1)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True, drop_last=True,
                                               num_workers=args.workers, pin_memory=amp)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch, num_workers=args.workers)
    print(f"{args.model}: {len(train_set)} training / {len(val_set)} validation images, device {device}, "
          f"{args.objective}, eps {args.eps}/255, lr {lr}, {epochs} epochs, batch {batch}", flush=True)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=0.01)
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = max(1, int(args.warmup * total_steps))

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    base = evaluate(model, val_loader, device, eval_eps, amp)
    print(f"epoch 0 (as shipped): {base}", flush=True)
    log = {"model": args.model, "objective": args.objective, "beta": args.beta, "eps_255": args.eps,
           "alpha_255": args.alpha, "train_steps": args.train_steps, "lr": lr, "epochs": epochs, "batch": batch,
           "train_images": len(train_set), "val_images": len(val_set), "per_class": args.per_class,
           "train_data": [p.name for p in args.train_data], "dry_run": args.dry_run,
           "history": [{"epoch": 0, **base}]}
    best_key, best_state = None, None
    started = time.time()

    for epoch in range(1, epochs + 1):
        sums = {"loss": 0.0, "clean": 0.0, "robust": 0.0, "batch_robust_acc": 0.0}
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            model.eval()  # craft the attack with frozen layers
            if args.objective == "trades":
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    clean_for_attack = model(x).float()
                adv = attack(model, x, y, eps, alpha, args.train_steps, amp, clean_logit=clean_for_attack)
            else:
                n_adv = int(round(args.adv_fraction * len(x)))
                adv = torch.cat([attack(model, x[:n_adv], y[:n_adv], eps, alpha, args.train_steps, amp), x[n_adv:]])
            model.train()
            freeze_batchnorm(model)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(torch.cat([x, adv])).float()
            clean_logit, adv_logit = logits[: len(x)], logits[len(x):]
            clean_loss = F.binary_cross_entropy_with_logits(clean_logit, y)
            if args.objective == "trades":
                robust_loss = binary_kl(clean_logit, adv_logit).mean()
                loss = clean_loss + args.beta * robust_loss
            else:
                robust_loss = F.binary_cross_entropy_with_logits(adv_logit, y)
                loss = robust_loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scale_before = scaler.get_scale()
            scaler.update()
            if scaler.get_scale() >= scale_before:  # AMP skipped no step: advance the schedule
                scheduler.step()
            sums["loss"] += loss.item()
            sums["clean"] += clean_loss.item()
            sums["robust"] += robust_loss.item()
            sums["batch_robust_acc"] += ((adv_logit.detach() > 0).float() == y).float().mean().item()
            if step % 50 == 0 or step == len(train_loader):
                avg = {k: v / step for k, v in sums.items()}
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {avg['loss']:.4f} "
                      f"(clean {avg['clean']:.4f}, robust {avg['robust']:.4f}, "
                      f"in-batch robust acc {avg['batch_robust_acc']:.1%}) "
                      f"lr {scheduler.get_last_lr()[0]:.2e} ({time.time() - started:.0f}s)", flush=True)
        metrics = evaluate(model, val_loader, device, eval_eps, amp)
        metrics.update({f"train_{k}": v / max(1, len(train_loader)) for k, v in sums.items()})
        print(f"epoch {epoch}: {metrics}", flush=True)
        log["history"].append({"epoch": epoch, **metrics})
        clean_ok = base["clean_auc"] is None or (metrics["clean_auc"] or 0) >= base["clean_auc"] - 0.01
        key = (clean_ok, metrics[key_name])
        if best_key is None or key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.net.state_dict().items()}
            log["selected_epoch"], log["meets_clean_gate"] = epoch, clean_ok
            log["selected"] = metrics

    log["seconds"] = round(time.time() - started)
    checkpoint = args.out / CHECKPOINT_NAME.format(model=args.model, suffix=suffix)
    log_path = args.out / LOG_NAME.format(model=args.model, suffix=suffix)
    torch.save(best_state, checkpoint)
    log_path.write_text(json.dumps(log, indent=2))
    print(f"Saved {checkpoint} (epoch {log['selected_epoch']}, clean gate met: {log['meets_clean_gate']}, "
          f"{key_name}: {log['selected'][key_name]:.1%} vs {base[key_name]:.1%} as shipped)")

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

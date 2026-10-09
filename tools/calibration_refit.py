#!/usr/bin/env python3
"""Test refitting the app's photo calibration with real phone photos counted as real.

The photo path scores an image as P(ai) = sigmoid(slope * s + intercept), where s is the
ensemble score with the shipped standardization (evaluate.APP_ENSEMBLE). Only slope and
intercept are refit here; the models and the standardization stay as shipped.

Inputs are the per-image scores CSVs Robustness eval writes (scores-<dataset>.csv, both
models). Candidates:
- shipped:  the app's current constants;
- pooled:   Platt fit on the fit datasets, every image weighted equally (the shipped method);
- balanced: the same, but each dataset carries equal total weight.

The fit datasets are the AI benchmarks the shipped fit used plus the real-only phone set;
the check datasets (e.g. OpenFake) are never fit on. Phone photos are reported held-out:
2-fold by image, each fold scored with constants fit without it.

A candidate qualifies when, on every dataset x condition, real shown HIGH <= 5% and AI shown
LOW <= 10% at the app's bands, and fewer unedited phone photos read UNCERTAIN than shipped.

Usage:
    python tools/calibration_refit.py results/ [--fit defactify,mj-dalle-sd-nbp,phone-photos]
        [--check openfake] [--holdout phone-photos] [--json refit.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate import HIGH_GRID, LOW_GRID, sigmoid, worst_case  # noqa: E402
from ensemble_calibrate import load_pairs  # noqa: E402
from evaluate import APP_ENSEMBLE, HIGH_THRESHOLD, LOW_THRESHOLD  # noqa: E402

CONDITIONS = ("original", "jpeg75", "social")
MAX_REAL_HIGH, MAX_AI_LOW, MIN_GAP = 0.05, 0.10, 0.20


def fit_platt_weighted(d: np.ndarray, y: np.ndarray, w: np.ndarray, iterations: int = 100) -> tuple[float, float]:
    """calibrate.fit_platt with per-sample weights (normalized to mean 1)."""
    w = w / w.mean()
    positives = float(y.sum())
    negatives = float(len(y) - positives)
    targets = np.where(y == 1, (positives + 1) / (positives + 2), 1 / (negatives + 2))
    x = np.stack([d, np.ones_like(d)], axis=1)

    def loss(theta: np.ndarray) -> float:
        z = x @ theta
        return float(np.mean(w * (np.logaddexp(0.0, z) - targets * z)))

    theta = np.array([1.0 / max(1.0, float(np.std(d))), 0.0])
    for _ in range(iterations):
        p = sigmoid(x @ theta)
        gradient = x.T @ (w * (p - targets)) / len(d)
        hessian = (x * (w * p * (1 - p))[:, None]).T @ x / len(d) + 1e-9 * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        current = loss(theta)
        scale = 1.0
        while scale > 1e-8 and loss(theta - scale * step) > current - 1e-4 * scale * float(gradient @ step):
            scale *= 0.5
        theta = theta - scale * step
        if np.abs(scale * step).max() < 1e-10:
            break
    return float(theta[0]), float(theta[1])


def fold_of(image: str) -> int:
    """Deterministic 2-fold split by image id (all conditions of one photo share a fold)."""
    return zlib.crc32(Path(image).name.encode()) % 2


def ensemble_score(d: float, c: float) -> float:
    e = APP_ENSEMBLE
    return ((d - e["mean_d"]) / e["std_d"] + (c - e["mean_c"]) / e["std_c"]) / 2.0


def fit(rows: list[dict], balanced: bool) -> tuple[float, float]:
    s = np.array([r["s"] for r in rows])
    y = np.array([r["is_ai"] for r in rows])
    if not balanced:
        return fit_platt_weighted(s, y, np.ones(len(rows)))
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["dataset"]] = counts.get(r["dataset"], 0) + 1
    return fit_platt_weighted(s, y, np.array([1.0 / counts[r["dataset"]] for r in rows]))


def bands(p: np.ndarray) -> tuple[float, float, float]:
    if len(p) == 0:
        return float("nan"), float("nan"), float("nan")
    low = float((p < LOW_THRESHOLD).mean())
    high = float((p >= HIGH_THRESHOLD).mean())
    return low, 1.0 - low - high, high


def rule_bands(rows: list[dict], key: str) -> tuple[float, float]:
    """calibrate.py's rule (HIGH: real HIGH <= 5%; LOW: AI LOW <= 10%) on the given probabilities."""
    # worst_case applies sigmoid(slope * x + intercept); feed it logit(p) with slope 1.
    p = np.clip(np.array([r[key] for r in rows]), 1e-9, 1 - 1e-9)
    calib = [{"dataset": r["dataset"], "condition": r["condition"], "is_ai": r["is_ai"], "logit_diff": float(x)}
             for r, x in zip(rows, np.log(p / (1 - p)))]
    high = next((t for t in HIGH_GRID if worst_case(calib, 1.0, 0.0, t, 0)[0] <= MAX_REAL_HIGH), HIGH_GRID[-1])
    lows = [t for t in LOW_GRID if t <= high - MIN_GAP]
    low = next((t for t in reversed(lows) if worst_case(calib, 1.0, 0.0, 1.1, t)[1] <= MAX_AI_LOW),
               lows[0] if lows else LOW_GRID[0])
    return high, low


def pct(v: float) -> str:
    return "–" if v != v else f"{v:.1%}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="Folder with scores-<dataset>.csv files")
    parser.add_argument("--fit", default="defactify,mj-dalle-sd-nbp,phone-photos")
    parser.add_argument("--check", default="openfake")
    parser.add_argument("--holdout", default="phone-photos", help="Dataset reported held-out (2-fold)")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    fit_sets = [d for d in args.fit.split(",") if d]
    check_sets = [d for d in args.check.split(",") if d]
    rows: list[dict] = []
    for name in fit_sets + check_sets:
        path = args.results / f"scores-{name}.csv"
        if not path.exists():
            print(f"(skipping {name}: {path} not found)", file=sys.stderr)
            continue
        rows += [r for r in load_pairs(path, "avg", "native") if r["condition"] in CONDITIONS]
    for r in rows:
        r["s"] = ensemble_score(r["d"], r["c"])
    fit_rows = [r for r in rows if r["dataset"] in fit_sets]
    if not fit_rows or not any(r["is_ai"] for r in fit_rows):
        sys.exit("No fit rows with AI images - check --fit and the results folder.")

    shipped = APP_ENSEMBLE["photo"]
    candidates: dict[str, tuple[float, float]] = {"shipped": shipped}
    for name, balanced in (("pooled", False), ("balanced", True)):
        candidates[name] = fit(fit_rows, balanced)
        for r in rows:
            r[name] = float(sigmoid(candidates[name][0] * r["s"] + candidates[name][1]))
        # The holdout dataset is scored with constants fit without each of its folds.
        for k in (0, 1):
            train = [r for r in fit_rows if not (r["dataset"] == args.holdout and fold_of(r["image"]) == k)]
            slope, intercept = fit(train, balanced)
            for r in rows:
                if r["dataset"] == args.holdout and fold_of(r["image"]) == k:
                    r[name] = float(sigmoid(slope * r["s"] + intercept))
    for r in rows:
        r["shipped"] = float(sigmoid(shipped[0] * r["s"] + shipped[1]))

    names = list(candidates)
    datasets = [d for d in fit_sets + check_sets if any(r["dataset"] == d for r in rows)]
    out = ["# Photo calibration refit: phone photos counted as real\n",
           f"Ensemble score with the shipped standardization; only slope/intercept change. Fit on "
           f"{', '.join(fit_sets)}; {', '.join(check_sets) or 'nothing'} is check-only (never fit on). "
           f"`{args.holdout}` is reported held-out (2-fold by image). Bands as in the app: LOW < "
           f"{LOW_THRESHOLD:.0%}, HIGH ≥ {HIGH_THRESHOLD:.0%}.\n",
           "| Candidate | Slope | Intercept |", "|---|---|---|"]
    out += [f"| {n} | {candidates[n][0]:.5f} | {candidates[n][1]:.5f} |" for n in names]

    out += ["\n## Real images: LOW / UNCERTAIN / HIGH\n",
            "| Dataset | Condition | n | " + " | ".join(names) + " |", "|---|---|---|" + "---|" * len(names)]
    for d in datasets:
        for c in CONDITIONS:
            sub = [r for r in rows if r["dataset"] == d and r["condition"] == c and r["is_ai"] == 0]
            if sub:
                cells = [" / ".join(pct(v) for v in bands(np.array([r[n] for r in sub]))) for n in names]
                out.append(f"| {d} | {c} | {len(sub)} | " + " | ".join(cells) + " |")

    out += ["\n## AI images: LOW / HIGH\n",
            "| Dataset | Condition | n | " + " | ".join(names) + " |", "|---|---|---|" + "---|" * len(names)]
    for d in datasets:
        for c in CONDITIONS:
            sub = [r for r in rows if r["dataset"] == d and r["condition"] == c and r["is_ai"] == 1]
            if sub:
                cells = []
                for n in names:
                    low, _, high = bands(np.array([r[n] for r in sub]))
                    cells.append(f"{pct(low)} / {pct(high)}")
                out.append(f"| {d} | {c} | {len(sub)} | " + " | ".join(cells) + " |")

    summary = {}
    phone_unc = {}
    for n in names:
        cells = {}
        for d in datasets:
            for c in CONDITIONS:
                sub = [r for r in rows if r["dataset"] == d and r["condition"] == c]
                real = np.array([r[n] for r in sub if r["is_ai"] == 0])
                ai = np.array([r[n] for r in sub if r["is_ai"] == 1])
                cells[(d, c)] = (bands(real), bands(ai))
        worst_real_high = max(v[0][2] for v in cells.values() if v[0][2] == v[0][2])
        worst_ai_low = max(v[1][0] for v in cells.values() if v[1][0] == v[1][0])
        min_ai_high = min(v[1][2] for v in cells.values() if v[1][2] == v[1][2])
        held = cells.get((args.holdout, "original"))
        phone_unc[n] = held[0][1] if held else float("nan")
        high, low = rule_bands(rows, n)
        summary[n] = {"slope": candidates[n][0], "intercept": candidates[n][1],
                      "worst_real_high": worst_real_high, "worst_ai_low": worst_ai_low,
                      "min_ai_high": min_ai_high, "holdout_uncertain_original": phone_unc[n],
                      "holdout_low_original": held[0][0] if held else float("nan"),
                      "rule_high": high, "rule_low": low}
    for n in names:
        s = summary[n]
        s["qualifies"] = bool(n != "shipped" and s["worst_real_high"] <= MAX_REAL_HIGH
                              and s["worst_ai_low"] <= MAX_AI_LOW
                              and s["holdout_uncertain_original"] < phone_unc["shipped"])

    out += ["\n## Summary (worst case over every dataset × condition)\n",
            f"| Candidate | Real shown HIGH (≤ {MAX_REAL_HIGH:.0%}) | AI shown LOW (≤ {MAX_AI_LOW:.0%}) | "
            f"Lowest AI shown HIGH | {args.holdout} LOW / UNCERTAIN (original) | Bands the rule picks | Qualifies |",
            "|---|---|---|---|---|---|---|"]
    for n in names:
        s = summary[n]
        out.append(f"| {n} | {pct(s['worst_real_high'])} | {pct(s['worst_ai_low'])} | {pct(s['min_ai_high'])} | "
                   f"{pct(s['holdout_low_original'])} / {pct(s['holdout_uncertain_original'])} | "
                   f"HIGH ≥ {s['rule_high']:.2f}, LOW < {s['rule_low']:.2f} | "
                   f"{'–' if n == 'shipped' else ('**yes**' if s['qualifies'] else 'no')} |")
    print("\n".join(out))
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

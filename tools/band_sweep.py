#!/usr/bin/env python3
"""What moving the photo LOW cut-off does, on the app's exact photo scores.

Reads the per-image scores CSVs Robustness eval writes (scores-<dataset>.csv, both
models), rebuilds the app's photo probability (shipped standardization and calibration,
evaluate.APP_ENSEMBLE, exactly as calibration_refit.py does; or a candidate model set's own
constants with --ensemble-params), and for each candidate LOW cut-off (HIGH at the app's
value) reports per dataset x condition:
- real images: LOW / UNCERTAIN / HIGH;
- AI images: LOW / UNCERTAIN / HIGH;
and a summary per LOW x HIGH pair (--highs adds HIGH cut-offs to the summary only): worst
real shown HIGH, worst AI shown LOW on clean conditions (the calibration rule) and under
everyday edits (the edit gate), cells failing AI LOW <= 10%, how much of UNCERTAIN each
cut-off removes, and LOW / UNCERTAIN / HIGH for every real-only dataset (phone photos).

Usage:
    python tools/band_sweep.py results/ [--lows 0.15,0.20] [--highs 0.85]
        [--ensemble-params ensemble-params.json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate import sigmoid  # noqa: E402
from calibration_refit import ensemble_score  # noqa: E402
from ensemble_calibrate import load_pairs  # noqa: E402
from evaluate import ALL_CONDITIONS, APP_ENSEMBLE, CONDITIONS, HIGH_THRESHOLD, load_ensemble_params  # noqa: E402

MAX_AI_LOW = 0.10


def bands(p: np.ndarray, low: float, high: float = HIGH_THRESHOLD) -> tuple[float, float, float]:
    """(LOW, UNCERTAIN, HIGH) shares, as EvidenceEngine.classify: LOW is p < low, HIGH is p >= high."""
    if len(p) == 0:
        return float("nan"), float("nan"), float("nan")
    lo = float((p < low).mean())
    hi = float((p >= high).mean())
    return lo, 1.0 - lo - hi, hi


def app_probability(rows: list[dict], params: dict | None = None) -> np.ndarray:
    params = params or APP_ENSEMBLE
    slope, intercept = params["photo"]
    return sigmoid(slope * np.array([ensemble_score(r["d"], r["c"], params) for r in rows]) + intercept)


def pct(v: float) -> str:
    return "–" if v != v else f"{v:.1%}"


def sweep(rows: list[dict], lows: list[float], params: dict | None = None,
          highs: list[float] | None = None) -> tuple[list[str], dict]:
    """The report lines and the summary, keyed by (LOW, HIGH). `highs` are extra HIGH
    cut-offs for the summary; the per-cell tables stay at the app's HIGH."""
    p_all = app_probability(rows, params)
    for r, p in zip(rows, p_all):
        r["p"] = float(p)
    order = {c: i for i, c in enumerate(ALL_CONDITIONS)}
    datasets = sorted({r["dataset"] for r in rows})
    conditions = sorted({r["condition"] for r in rows}, key=lambda c: order.get(c, 99))
    cells = {}
    for d in datasets:
        for c in conditions:
            sub = [r for r in rows if r["dataset"] == d and r["condition"] == c]
            if sub:
                cells[(d, c)] = (np.array([r["p"] for r in sub if r["is_ai"] == 0]),
                                 np.array([r["p"] for r in sub if r["is_ai"] == 1]))

    head = " | ".join(f"LOW < {lo:.2f}" for lo in lows)
    whose = "the candidate set's own" if params else "the shipped"
    out = [f"# Photo LOW cut-off sweep (HIGH ≥ {HIGH_THRESHOLD:.2f} fixed)\n",
           f"App photo scores rebuilt from both models' logits with {whose} constants. "
           "Cells: LOW / UNCERTAIN / HIGH.\n",
           "## Real images\n", f"| Dataset | Condition | n | {head} |", "|---|---|---|" + "---|" * len(lows)]
    for (d, c), (real, _) in cells.items():
        if len(real):
            out.append(f"| {d} | {c} | {len(real)} | "
                       + " | ".join(" / ".join(pct(v) for v in bands(real, lo)) for lo in lows) + " |")
    out += ["\n## AI images\n", f"| Dataset | Condition | n | {head} |", "|---|---|---|" + "---|" * len(lows)]
    for (d, c), (_, ai) in cells.items():
        if len(ai):
            out.append(f"| {d} | {c} | {len(ai)} | "
                       + " | ".join(" / ".join(pct(v) for v in bands(ai, lo)) for lo in lows) + " |")

    real_only = [d for d in datasets if not any(len(ai) for (dd, _), (_, ai) in cells.items() if dd == d)]
    summary = {}
    for hi in sorted({HIGH_THRESHOLD, *(highs or [])}, reverse=True):
        for lo in lows:
            clean = [bands(ai, lo, hi)[0] for (d, c), (_, ai) in cells.items() if len(ai) and c in CONDITIONS]
            edits = [bands(ai, lo, hi)[0] for (d, c), (_, ai) in cells.items() if len(ai) and c not in CONDITIONS]
            failing = sorted(f"{d} {c} ({bands(ai, lo, hi)[0]:.1%})" for (d, c), (_, ai) in cells.items()
                             if len(ai) and bands(ai, lo, hi)[0] > MAX_AI_LOW)
            real_high = [bands(real, lo, hi)[2] for (real, _) in cells.values() if len(real)]
            real_unc = [bands(real, lo, hi)[1] for (_, c), (real, _) in cells.items() if len(real) and c == "original"]
            ai_unc = [bands(ai, lo, hi)[1] for (_, c), (_, ai) in cells.items() if len(ai) and c == "original"]
            ai_high = [bands(ai, lo, hi)[2] for (_, c), (_, ai) in cells.items() if len(ai) and c == "original"]
            phone = cells.get(("phone-photos", "original"), (np.array([]), None))[0]
            summary[(lo, hi)] = {
                "worst_real_high": max(real_high) if real_high else float("nan"),
                "worst_ai_low_clean": max(clean) if clean else float("nan"),
                "worst_ai_low_edits": max(edits) if edits else float("nan"),
                "failing": failing,
                "mean_real_uncertain": float(np.mean(real_unc)) if real_unc else float("nan"),
                "mean_ai_uncertain": float(np.mean(ai_unc)) if ai_unc else float("nan"),
                "min_ai_high": min(ai_high) if ai_high else float("nan"),
                "phone": bands(phone, lo, hi),
                "real_only": {d: bands(cells[(d, "original")][0], lo, hi) for d in real_only
                              if (d, "original") in cells},
            }
    shown = [d for d in real_only if any(d in s["real_only"] for s in summary.values())]
    out += ["\n## Summary\n",
            "| LOW cut-off | HIGH cut-off | Worst real shown HIGH, any cell (rule ≤ 5%) | Worst AI shown LOW, clean (rule ≤ 10%) "
            "| Worst AI shown LOW, edits | Cells with AI LOW > 10% | Real UNCERTAIN, mean (original) "
            "| AI UNCERTAIN, mean (original) | Lowest AI shown HIGH (original) | "
            + " | ".join(f"{d} (original) LOW / UNCERTAIN / HIGH" for d in shown) + (" |" if shown else ""),
            "|---|---|---|---|---|---|---|---|---|" + "---|" * len(shown)]
    for (lo, hi), s in summary.items():
        out.append(f"| {lo:.2f} | {hi:.2f} | {pct(s['worst_real_high'])} | {pct(s['worst_ai_low_clean'])} "
                   f"| {pct(s['worst_ai_low_edits'])} | {len(s['failing'])} "
                   f"| {pct(s['mean_real_uncertain'])} | {pct(s['mean_ai_uncertain'])} | {pct(s['min_ai_high'])} | "
                   + " | ".join(" / ".join(pct(v) for v in s["real_only"].get(d, (float("nan"),) * 3)) for d in shown)
                   + (" |" if shown else ""))
    for (lo, hi), s in summary.items():
        if s["failing"] and hi == HIGH_THRESHOLD:
            out.append(f"\nAI LOW > 10% at LOW < {lo:.2f}: " + "; ".join(s["failing"]))
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="Folder with scores-<dataset>.csv files")
    parser.add_argument("--lows", default="0.15,0.20")
    parser.add_argument("--highs", default="", help="Extra HIGH cut-offs for the summary, e.g. 0.85")
    parser.add_argument("--ensemble-params", type=Path, default=None,
                        help="ensemble-params.json for a candidate model set (default: the app's APP_ENSEMBLE)")
    args = parser.parse_args()
    rows: list[dict] = []
    for path in sorted(args.results.glob("scores-*.csv")):
        rows += load_pairs(path, "avg", "native")
    if not rows:
        sys.exit(f"No paired photo scores under {args.results}.")
    out, _ = sweep(rows, [float(v) for v in args.lows.split(",") if v],
                   load_ensemble_params(args.ensemble_params) if args.ensemble_params else None,
                   [float(v) for v in args.highs.split(",") if v])
    print("\n".join(out))


if __name__ == "__main__":
    main()

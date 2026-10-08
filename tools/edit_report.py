#!/usr/bin/env python3
"""Everyday-edit robustness report (Phase 3) from tools/evaluate.py --json-dir results.

For one model (default: the app's shipped ensemble), one row per dataset x condition with
the app's own bands, the change from the unedited image, and the Phase 3 gates:
  - real shown HIGH <= 5% and AI shown LOW <= 10% under every condition;
  - AI shown HIGH at most 10 points below the unedited ("original") images.
A final table gives the worst case over datasets for each condition.

Usage:
    python tools/edit_report.py results/ [--model "app ensemble"] > edits.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import ALL_CONDITIONS, HIGH_THRESHOLD, LOW_THRESHOLD  # noqa: E402

REAL_HIGH_MAX, AI_LOW_MAX, AI_HIGH_DROP_MAX = 0.05, 0.10, 0.10


def pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def row_values(metrics: dict) -> dict:
    """AI-side numbers are None for a real-only dataset (e.g. phone-photos)."""
    has_ai = metrics.get("n_ai", 1) > 0
    return {"auc": metrics["auc"],
            "ai_high": metrics["bands_ai"]["high"] if has_ai else None,
            "ai_low": metrics["bands_ai"]["low"] if has_ai else None,
            "real_high": metrics["bands_real"]["high"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--model", default="app ensemble")
    args = parser.parse_args()

    rows: dict[tuple[str, str], dict] = {}
    for path in sorted(args.results.glob("*.json")):
        r = json.loads(path.read_text())
        if r.get("model") == args.model and "metrics" in r:
            rows[(r["dataset"], r["condition"])] = row_values(r["metrics"])
    if not rows:
        print(f"No results for model {args.model!r}.")
        return
    datasets = sorted({d for d, _ in rows})
    order = {c: i for i, c in enumerate(ALL_CONDITIONS)}
    conditions = sorted({c for _, c in rows}, key=lambda c: order.get(c, 99))

    out = [f"# Everyday edits: `{args.model}`\n",
           f"Bands as in the app: LOW < {LOW_THRESHOLD:.0%}, HIGH >= {HIGH_THRESHOLD:.0%}. Gates: real shown HIGH "
           f"<= {REAL_HIGH_MAX:.0%}, AI shown LOW <= {AI_LOW_MAX:.0%}, AI shown HIGH at most "
           f"{AI_HIGH_DROP_MAX * 100:.0f} points below the unedited images.\n",
           "| Dataset | Condition | AUC | AI shown HIGH | Change vs original | AI shown LOW | Real shown HIGH | Gates |",
           "|---|---|---|---|---|---|---|---|"]
    worst: dict[str, dict] = {}
    failures = 0
    for dataset in datasets:
        base = rows.get((dataset, "original"))
        for condition in conditions:
            v = rows.get((dataset, condition))
            if v is None:
                continue
            drop = (None if base is None or v["ai_high"] is None or base["ai_high"] is None
                    else base["ai_high"] - v["ai_high"])
            ok = (v["real_high"] <= REAL_HIGH_MAX and (v["ai_low"] is None or v["ai_low"] <= AI_LOW_MAX)
                  and (drop is None or drop <= AI_HIGH_DROP_MAX))
            failures += not ok
            change = ("–" if drop is None or condition == "original"
                      else f"{(v['ai_high'] - base['ai_high']) * 100:+.1f} pts")
            auc = "n/a" if v["auc"] is None else f"{v['auc']:.3f}"
            out.append(f"| {dataset} | {condition} | {auc} | {pct(v['ai_high'])} | {change} | "
                       f"{pct(v['ai_low'])} | {pct(v['real_high'])} | {'pass' if ok else '**fail**'} |")
            w = worst.setdefault(condition, {"auc": None, "ai_high": None, "ai_low": None, "real_high": 0.0,
                                             "drop": 0.0})
            if v["auc"] is not None:
                w["auc"] = v["auc"] if w["auc"] is None else min(w["auc"], v["auc"])
            if v["ai_high"] is not None:
                w["ai_high"] = v["ai_high"] if w["ai_high"] is None else min(w["ai_high"], v["ai_high"])
                w["ai_low"] = v["ai_low"] if w["ai_low"] is None else max(w["ai_low"], v["ai_low"])
            w["real_high"] = max(w["real_high"], v["real_high"])
            w["drop"] = max(w["drop"], drop or 0.0)

    out += ["\n## Worst case over datasets\n",
            "| Condition | Lowest AUC | Lowest AI shown HIGH | Largest drop | Highest AI shown LOW | Highest real shown HIGH |",
            "|---|---|---|---|---|---|"]
    for condition in conditions:
        w = worst[condition]
        auc = "n/a" if w["auc"] is None else f"{w['auc']:.3f}"
        out.append(f"| {condition} | {auc} | {pct(w['ai_high'])} | {w['drop'] * 100:.1f} pts | "
                   f"{pct(w['ai_low'])} | {pct(w['real_high'])} |")
    out.append(f"\n{failures} dataset x condition cell(s) fail a gate.")
    print("\n".join(out))


if __name__ == "__main__":
    main()

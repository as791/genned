#!/usr/bin/env python3
"""Fairness report: how the app's answers differ between groups of images.

For each dataset with a groups file (results/groups-<dataset>.csv: `file` plus one column
per grouping, e.g. FairFace's ethnicity/gender/age or the phone make of real phone
photos), joins the app's per-image scores (results/scores-<dataset>.csv, both models) and
reports per group: how many real photos are wrongly shown "Likely AI-generated" (with a
95% Wilson interval), and how many are shown "Uncertain" and "Likely not".

A group is flagged when its false-alarm rate is above --max-rate (5%, the app's rule for
any dataset), or when it is more than --max-ratio (2x) the dataset's overall rate and its
interval lies wholly above the overall rate (so a single unlucky photo in a small group
doesn't trigger it).

Usage:
    python tools/group_report.py results/ [--conditions original,social]
        [--ensemble-params ensemble-params.json] [--json results/groups.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensemble_calibrate import load_pairs  # noqa: E402
from evaluate import APP_ENSEMBLE, HIGH_THRESHOLD, LOW_THRESHOLD, ensemble_score, load_ensemble_params  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def app_probability(d: float, c: float, params: dict) -> float:
    slope, intercept = params["photo"]
    return float(1 / (1 + np.exp(-(slope * ensemble_score(d, c, params) + intercept))))


def group_table(scores: list[tuple[dict, float]], column: str, max_rate: float, max_ratio: float) -> tuple[list[str], list[dict]]:
    """Rows of (group value, n, false alarms...) for one grouping column; real images only."""
    real = [(g, p) for g, p in scores if g["is_ai"] == 0]
    overall_high = sum(p >= HIGH_THRESHOLD for _, p in real)
    overall = overall_high / len(real) if real else float("nan")
    by_value: dict[str, list[float]] = defaultdict(list)
    for g, p in real:
        by_value[g[column]].append(p)
    lines, summary = [], []
    for value in sorted(by_value, key=lambda v: (-len(by_value[v]), v)):
        ps = np.array(by_value[value])
        n, k = len(ps), int((ps >= HIGH_THRESHOLD).sum())
        rate = k / n
        lo, hi = wilson(k, n)
        flagged = rate > max_rate or (rate > max_ratio * overall and lo > overall)
        summary.append({"group": value, "n": n, "false_alarms": k, "rate": rate, "ci95": [lo, hi],
                        "uncertain": float(((ps >= LOW_THRESHOLD) & (ps < HIGH_THRESHOLD)).mean()),
                        "low": float((ps < LOW_THRESHOLD).mean()), "flagged": flagged})
        lines.append(f"| {value} | {n} | {k} ({rate:.1%}) [{lo:.1%}–{hi:.1%}] | "
                     f"{summary[-1]['uncertain']:.1%} | {summary[-1]['low']:.1%} | {'**yes**' if flagged else ''} |")
    lo, hi = wilson(overall_high, len(real))
    lines.append(f"| **all** | {len(real)} | {overall_high} ({overall:.1%}) [{lo:.1%}–{hi:.1%}] | "
                 f"{np.mean([(p >= LOW_THRESHOLD) and (p < HIGH_THRESHOLD) for _, p in real]):.1%} | "
                 f"{np.mean([p < LOW_THRESHOLD for _, p in real]):.1%} | |")
    return lines, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--conditions", default="original,social")
    parser.add_argument("--ensemble-params", type=Path, default=None)
    parser.add_argument("--max-rate", type=float, default=0.05)
    parser.add_argument("--max-ratio", type=float, default=2.0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    params = load_ensemble_params(args.ensemble_params) if args.ensemble_params else APP_ENSEMBLE

    out = ["# Fairness: false alarms by group\n",
           f"Real photos wrongly shown *Likely AI-generated* (score ≥ {HIGH_THRESHOLD:.0%}), with 95% Wilson "
           f"intervals, and the share shown *Uncertain* and *Likely not*. Flagged: above {args.max_rate:.0%}, or more "
           f"than {args.max_ratio:g}× the overall rate with the interval wholly above it.\n"]
    report: dict = {}
    group_files = sorted(args.results.glob("groups-*.csv"))
    if not group_files:
        print("No groups-<dataset>.csv files in the results; nothing to report.")
        return
    for group_file in group_files:
        dataset = group_file.stem.removeprefix("groups-")
        scores_file = args.results / f"scores-{dataset}.csv"
        if not scores_file.exists():
            out.append(f"_{dataset}: no scores file, skipped._\n")
            continue
        with group_file.open() as f:
            groups = {row["file"]: row for row in csv.DictReader(f)}
        columns = [c for c in next(iter(groups.values())) if c != "file"] if groups else []
        pairs = load_pairs(scores_file, "avg", "native")
        for condition in [c for c in args.conditions.split(",") if c]:
            scored = [({**groups[r["image"]], "is_ai": r["is_ai"]}, app_probability(r["d"], r["c"], params))
                      for r in pairs if r["condition"] == condition and r["image"] in groups]
            if not scored:
                continue
            for column in columns:
                lines, summary = group_table(scored, column, args.max_rate, args.max_ratio)
                report.setdefault(dataset, {}).setdefault(condition, {})[column] = summary
                out += [f"## {dataset} · {condition} · by {column}\n",
                        "| Group | n | Shown Likely AI | Uncertain | Likely not | Flagged |",
                        "|---|---|---|---|---|---|", *lines, ""]
    flagged = [(d, c, col, g["group"]) for d, conds in report.items() for c, cols in conds.items()
               for col, rows in cols.items() for g in rows if g["flagged"]]
    out.append("**Flagged groups:** " + (", ".join(f"{d} {c} {col}={g}" for d, c, col, g in flagged) or "none") + "\n")
    print("\n".join(out))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One line per benchmark set for the monthly freshness re-test (freshness.yml).

Reads tools/evaluate.py's JSON results for the app as shipped on unedited images, and
prints one Markdown line per set: AUC with its 95% interval, the share of AI caught at 5%
false alarms, how often real photos are shown "Likely AI-generated" and AI images
"Likely not", and the three generators the app does worst on. With --previous (last
month's results), each AUC also shows the change, and a drop beyond --alert is marked,
so a new generator slipping through shows up month to month.

Usage:
    python tools/freshness_summary.py results/ [--previous last-month/] [--alert 0.03]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL = "app as shipped"


def load(results: Path) -> dict[str, dict]:
    """{dataset: payload} for the shipped app on unedited images."""
    out = {}
    for path in sorted(results.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if payload.get("model") == MODEL and payload.get("condition") == "original" and "metrics" in payload:
            out[payload["dataset"]] = payload
    return out


def weakest(metrics: dict, k: int = 3, min_n: int = 5) -> list[tuple[str, float]]:
    gens = [(name, g["ap"]) for name, g in metrics.get("per_generator", {}).items()
            if name != "real" and g.get("n", 0) >= min_n and g.get("ap") == g.get("ap") and g.get("ap") is not None]
    return sorted(gens, key=lambda x: x[1])[:k]


def line(dataset: str, payload: dict, previous: dict | None, alert: float) -> str:
    m = payload["metrics"]
    ci = m.get("ci95", {}).get("auc")
    auc = f"AUC {m['auc']:.3f}" + (f" [{ci[0]:.3f}–{ci[1]:.3f}]" if ci else "")
    if previous is not None:
        delta = m["auc"] - previous["metrics"]["auc"]
        auc += f" ({delta:+.3f} vs last month{', **dropped**' if delta < -alert else ''})"
    parts = [f"**{dataset}** (n={m['n']})", auc,
             f"caught at 5% false alarms {m['catch_at_5']:.0%}",
             f"real shown Likely AI {m['bands_real']['high']:.1%}",
             f"AI shown Likely not {m['bands_ai']['low']:.1%}"]
    worst = weakest(m)
    if worst:
        parts.append("weakest generators (AP): " + ", ".join(f"{name} {ap:.2f}" for name, ap in worst))
    return "- " + " · ".join(parts)


def summary(current: dict[str, dict], previous: dict[str, dict], alert: float) -> str:
    if not current:
        return "- No results: every set failed to fetch or score (see the run log)."
    return "\n".join(line(d, p, previous.get(d), alert) for d, p in sorted(current.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--previous", type=Path, default=None)
    parser.add_argument("--alert", type=float, default=0.03, help="Mark an AUC drop larger than this")
    args = parser.parse_args()
    previous = load(args.previous) if args.previous and args.previous.is_dir() else {}
    print(summary(load(args.results), previous, args.alert))


if __name__ == "__main__":
    main()

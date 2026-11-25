"""Detection metrics beyond AUC, shared by the evaluation tools.

- average_precision: area under the precision-recall curve (the step-wise definition
  scikit-learn uses), the usual "AP" / "mAP" in detection papers.
- catch_rate_at_fpr: share of AI images caught when at most `max_fpr` of real images are
  flagged (TPR at a fixed false-positive rate); 1% and 5% are reported.
- answer_quality: how good the app's three answers are: how often "Likely AI-generated"
  is right, how often "Likely not" is right, and how often it gives a definite answer.
- bootstrap_ci: 95% confidence intervals by stratified bootstrap (real and AI resampled
  separately, so every resample keeps the class sizes).

Everything is plain numpy, so the tools need no scikit-learn.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def fast_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC (Mann-Whitney, ties count half), vectorized; nan without both classes."""
    pos = scores[labels == 1]
    neg = np.sort(scores[labels == 0])
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    below = np.searchsorted(neg, pos, side="left")
    tied = np.searchsorted(neg, pos, side="right") - below
    return float((below + 0.5 * tied).sum() / (len(pos) * len(neg)))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve, AI as the positive class.

    AP = sum over distinct score thresholds (high to low) of (R_n - R_{n-1}) * P_n,
    the same step-wise definition as sklearn.metrics.average_precision_score.
    """
    labels = np.asarray(labels)
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    # Last index of each run of equal scores: a threshold includes all tied images.
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[last]
    predicted = last + 1
    precision = tp / predicted
    recall = tp / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def catch_rate_at_fpr(labels: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    """Share of AI images scored above the threshold that flags at most `max_fpr` of real
    images (same rule as tools/ensemble.py tpr_at_fpr)."""
    real = np.sort(scores[labels == 0])
    if len(real) == 0 or (labels == 1).sum() == 0:
        return float("nan")
    index = int(np.ceil((1 - max_fpr) * len(real))) - 1
    threshold = real[min(max(index, 0), len(real) - 1)]
    return float((scores[labels == 1] > threshold).mean())


def answer_quality(labels: np.ndarray, scores: np.ndarray, low: float, high: float) -> dict[str, float]:
    """How the app's three answers hold up, at the dataset's own real/AI mix.

    - high_right: share of "Likely AI-generated" answers that are AI (precision);
    - low_right: share of "Likely not AI-generated" answers that are real;
    - answered: share of images that get a definite answer (not "Uncertain");
    - answered_right: share of definite answers that are right.
    A share with nothing to count is nan.
    """
    is_high, is_low = scores >= high, scores < low
    answered = is_high | is_low

    def share(num: np.ndarray, den: np.ndarray) -> float:
        return float(num.sum() / den.sum()) if den.sum() else float("nan")

    right = (is_high & (labels == 1)) | (is_low & (labels == 0))
    return {
        "high_right": share(is_high & (labels == 1), is_high),
        "low_right": share(is_low & (labels == 0), is_low),
        "answered": float(answered.mean()) if len(scores) else float("nan"),
        "answered_right": share(right, answered),
    }


def bootstrap_ci(labels: np.ndarray, scores: np.ndarray,
                 statistics: dict[str, Callable[[np.ndarray, np.ndarray], float]],
                 resamples: int = 1000, seed: int = 0, level: float = 0.95) -> dict[str, tuple[float, float]]:
    """Percentile confidence interval of each statistic, resampling real and AI images
    separately with replacement. A statistic that is nan in a resample is skipped there;
    one that is nan in every resample gets (nan, nan)."""
    rng = np.random.default_rng(seed)
    real_idx, ai_idx = np.nonzero(labels == 0)[0], np.nonzero(labels == 1)[0]
    values: dict[str, list[float]] = {name: [] for name in statistics}
    for _ in range(resamples):
        idx = np.concatenate([rng.choice(real_idx, len(real_idx)) if len(real_idx) else real_idx,
                              rng.choice(ai_idx, len(ai_idx)) if len(ai_idx) else ai_idx])
        y, s = labels[idx], scores[idx]
        for name, fn in statistics.items():
            v = fn(y, s)
            if v == v:  # skip nan
                values[name].append(v)
    tail = (1 - level) / 2 * 100
    return {name: ((float(np.percentile(v, tail)), float(np.percentile(v, 100 - tail))) if v else
                   (float("nan"), float("nan"))) for name, v in values.items()}


def standard_statistics(low: float, high: float) -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    """The statistics every report gives an interval for."""
    def band(cls: int, which: str) -> Callable[[np.ndarray, np.ndarray], float]:
        def fn(y: np.ndarray, s: np.ndarray) -> float:
            sub = s[y == cls]
            if len(sub) == 0:
                return float("nan")
            return float((sub >= high).mean() if which == "high" else (sub < low).mean())
        return fn

    return {
        "auc": fast_auc,
        "ap": average_precision,
        "catch_at_1": lambda y, s: catch_rate_at_fpr(y, s, 0.01),
        "catch_at_5": lambda y, s: catch_rate_at_fpr(y, s, 0.05),
        "real_high": band(0, "high"),
        "ai_low": band(1, "low"),
        "ai_high": band(1, "high"),
        "high_right": lambda y, s: answer_quality(y, s, low, high)["high_right"],
        "answered": lambda y, s: answer_quality(y, s, low, high)["answered"],
    }

"""Shared metric helpers (calibration + classification)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Equal-width reliability bins: mean predicted prob vs observed frequency."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if not mask.any():
            continue
        out.append(
            {
                "bin_lower": round(float(lo), 3),
                "bin_upper": round(float(hi), 3),
                "n": int(mask.sum()),
                "mean_predicted": round(float(y_prob[mask].mean()), 5),
                "observed_frequency": round(float(y_true[mask].mean()), 5),
            }
        )
    return out


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Weighted mean gap between confidence and accuracy across equal-width bins."""
    curve = reliability_curve(y_true, y_prob, n_bins)
    n = len(y_true)
    return round(
        float(sum(b["n"] / n * abs(b["mean_predicted"] - b["observed_frequency"]) for b in curve)),
        6,
    )


def calibration_report(y_true: np.ndarray, y_raw: np.ndarray, y_cal: np.ndarray, n_bins: int = 10) -> dict:
    return {
        "n": int(len(y_true)),
        "n_bins": n_bins,
        "brier_raw": round(float(brier_score_loss(y_true, y_raw)), 6),
        "brier_calibrated": round(float(brier_score_loss(y_true, y_cal)), 6),
        "ece_raw": expected_calibration_error(y_true, y_raw, n_bins),
        "ece_calibrated": expected_calibration_error(y_true, y_cal, n_bins),
        "reliability_raw": reliability_curve(y_true, y_raw, n_bins),
        "reliability_calibrated": reliability_curve(y_true, y_cal, n_bins),
    }


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresholds=(0.3, 0.5, 0.7)) -> dict:
    y_true = np.asarray(y_true)
    rep: dict = {
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 5),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 5),
        "prevalence": round(float(y_true.mean()), 5),
        "at_threshold": {},
    }
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        rep["at_threshold"][str(t)] = {
            "precision": round(float(precision_score(y_true, pred, zero_division=0)), 5),
            "recall": round(float(recall_score(y_true, pred, zero_division=0)), 5),
            "n_flagged": int(pred.sum()),
        }
    return rep

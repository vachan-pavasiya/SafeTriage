"""Phase 3 — model-disagreement uncertainty proxy.

``uncertainty`` = the normalised absolute gap between the calibrated XGBoost
probability and the Random-Forest probability for the same transaction:

    raw_gap   = |p_xgb_calibrated - p_rf|
    uncertainty = clip( (raw_gap - p05) / (p95 - p05), 0, 1 )

where p05/p95 come from the gap distribution on the validation split
(``uncertainty_norm_<dataset>.json``, written by training).

This is a **disagreement proxy only** — it is NOT epistemic/Bayesian
uncertainty. Two models agreeing does not mean the prediction is correct; it
means they are not a useful place to ask for a second opinion.
``confidence_score = 1 - uncertainty``.
"""

from __future__ import annotations

import json

import numpy as np

from safetriage import ARTIFACTS_DIR

LABEL = "model-disagreement uncertainty proxy (not epistemic uncertainty)"


def _norm(dataset: str) -> dict:
    p = ARTIFACTS_DIR / f"uncertainty_norm_{dataset}.json"
    if not p.exists():
        return {"p05": 0.0, "p95": 1.0}
    return json.loads(p.read_text())


def disagreement_uncertainty(p_xgb_calibrated, p_rf, dataset: str) -> np.ndarray:
    """Vectorised normalised disagreement in [0, 1]."""
    n = _norm(dataset)
    gap = np.abs(np.asarray(p_xgb_calibrated, dtype=float) - np.asarray(p_rf, dtype=float))
    lo, hi = float(n["p05"]), float(n["p95"])
    scaled = (gap - lo) / max(hi - lo, 1e-9)
    return np.clip(scaled, 0.0, 1.0)


def uncertainty_row(p_xgb_calibrated: float, p_rf: float, dataset: str) -> dict:
    u = float(disagreement_uncertainty([p_xgb_calibrated], [p_rf], dataset)[0])
    return {
        "uncertainty": round(u, 5),
        "confidence_score": round(1.0 - u, 5),
        "raw_gap": round(abs(p_xgb_calibrated - p_rf), 5),
        "label": LABEL,
    }

"""Phase 5 — explainability (SHAP) + deterministic reason codes.

Two *separate* layers:

* **Model-level** (this file, SHAP): which features moved THIS prediction, and by
  how much, in the XGBoost margin space. ``+`` pushes toward fraud, ``-`` away.
* **Decision-level** (safetriage/policy.py): why the policy engine chose the
  action. Kept distinct on purpose — a feature can push the score up while the
  policy still routes to review rather than block.

``reason_code`` is a small deterministic rule set over raw fields + scores, not a
model output.
"""

from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import shap

from safetriage import ARTIFACTS_DIR
from safetriage.features import load_feature_columns

REASON_CODES = (
    "ATO_NEW_DEVICE_HIGH_VELOCITY",
    "UNUSUAL_AMOUNT_NEW_BENEFICIARY",
    "KNOWN_DEVICE_NORMAL_BEHAVIOR",
    "HIGH_RISK_BUT_BEHAVIORALLY_CONSISTENT",
    "MODEL_DISAGREEMENT_REVIEW",
    "MULE_NETWORK_PROXIMITY",
    "REVIEW_MIXED_SIGNALS",
)

_FRIENDLY = {
    "TransactionAmt": "transaction amount",
    "amount_deviation_ratio": "amount vs the customer's usual",
    "is_new_device": "new device",
    "device_age_days": "device age",
    "is_new_beneficiary": "new beneficiary",
    "beneficiary_age_days": "beneficiary age",
    "beneficiary_transaction_count": "beneficiary history depth",
    "beneficiary_added_minutes_ago": "beneficiary added very recently",
    "velocity_1h_count": "transactions in the last hour",
    "velocity_24h_count": "transactions in the last 24h",
    "velocity_24h_amt_sum": "amount moved in the last 24h",
    "recent_password_reset": "recent password reset",
    "typing_cadence_score": "typing-cadence match",
    "shared_device_with_flagged_account": "device shared with a flagged account",
    "hop_distance_to_known_mule": "graph distance to a known mule",
    "account_age_days": "account age",
}


@lru_cache(maxsize=4)
def _explainer(dataset: str):
    from safetriage.scoring import _artifact_suffix

    sfx = _artifact_suffix()
    p = ARTIFACTS_DIR / f"xgb_{dataset}{sfx}.joblib"
    if not p.exists():
        p = ARTIFACTS_DIR / f"xgb_{dataset}.joblib"
    model = joblib.load(p)
    # tree_path_dependent -> no background data needed, deterministic, fast.
    return shap.TreeExplainer(model), load_feature_columns(dataset)


def _row_top(cols, shap_row, value_row, k) -> list[dict]:
    order = np.argsort(np.abs(shap_row))[::-1][:k]
    return [{
        "feature": cols[i],
        "friendly": _FRIENDLY.get(cols[i], cols[i].replace("_", " ")),
        "shap_value": round(float(shap_row[i]), 5),
        "value": round(float(value_row[i]), 5),
        "direction": "increases_fraud_risk" if shap_row[i] > 0 else "decreases_fraud_risk",
    } for i in order]


def shap_top_features(dataset: str, X_row: pd.DataFrame, k: int = 5) -> list[dict]:
    explainer, cols = _explainer(dataset)
    X_row = X_row.reindex(columns=cols, fill_value=0).astype(float)
    vals = np.asarray(explainer.shap_values(X_row))
    vals = vals[0] if vals.ndim == 2 else vals
    return _row_top(cols, vals, X_row.to_numpy()[0], k)


def shap_top_features_batch(dataset: str, X: pd.DataFrame, k: int = 5) -> list[list[dict]]:
    """Compute SHAP for the whole matrix once, return per-row top-k lists."""
    explainer, cols = _explainer(dataset)
    X = X.reindex(columns=cols, fill_value=0).astype(float)
    vals = np.asarray(explainer.shap_values(X))
    if vals.ndim == 3:  # (n, features, classes) -> positive class
        vals = vals[..., -1]
    arr = X.to_numpy()
    return [_row_top(cols, vals[r], arr[r], k) for r in range(len(X))]


def evidence_split(top_features: list[dict]) -> tuple[list[str], list[str]]:
    support, counter = [], []
    for f in top_features:
        phrase = f"{f['friendly']} (SHAP {f['shap_value']:+.2f})"
        (support if f["shap_value"] > 0 else counter).append(phrase)
    return support, counter


def reason_code(raw: dict, *, calibrated_probability: float, behavioural_consistency: float,
                uncertainty: float, top_features: list[dict], th=None, dataset: str | None = None) -> str:
    from safetriage.policy import PolicyThresholds

    th = th or PolicyThresholds.from_config(dataset=dataset)
    new_device = float(raw.get("is_new_device", 0)) >= 0.5
    new_benef = float(raw.get("is_new_beneficiary", 0)) >= 0.5
    v1 = float(raw.get("velocity_1h_count", 0))
    adr = float(raw.get("amount_deviation_ratio", 1.0))
    benef_recent = float(raw.get("beneficiary_added_minutes_ago", 1e9)) < 120
    hop = float(raw.get("hop_distance_to_known_mule", 99))
    shared = float(raw.get("shared_device_with_flagged_account", 0)) >= 0.5

    if uncertainty >= th.uncertainty_high:
        return "MODEL_DISAGREEMENT_REVIEW"
    if new_device and v1 >= 4:
        return "ATO_NEW_DEVICE_HIGH_VELOCITY"
    if (new_benef or benef_recent) and adr >= 2.0:
        return "UNUSUAL_AMOUNT_NEW_BENEFICIARY"
    if shared or hop <= 2:
        return "MULE_NETWORK_PROXIMITY"
    if calibrated_probability >= th.p_high and behavioural_consistency >= th.consistency_high:
        return "HIGH_RISK_BUT_BEHAVIORALLY_CONSISTENT"
    if calibrated_probability < th.p_low and behavioural_consistency >= th.consistency_high:
        return "KNOWN_DEVICE_NORMAL_BEHAVIOR"
    return "REVIEW_MIXED_SIGNALS"

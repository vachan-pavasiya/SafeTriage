"""Phase 3 — behavioural consistency.

``behavioural_consistency_score`` is 0..1 where **1.0 = this transaction looks
like the customer's own history** and **0.0 = strongly out of pattern**. It is
computed only from *observable* fields (amount vs the customer's history, hour vs
their normal hour, device / beneficiary provenance, velocity, location) and is
kept deliberately **separate from the fraud probability**:

    high probability + high consistency  -> de-escalate (model may be wrong /
                                            unusual-but-legitimate purchase)
    high probability + low  consistency  -> escalate

Every component is a documented bounded transform (no learned weights); the six
components are combined with the weights in ``config/policy_config.yaml``
(``behaviour_weights``). ``behavioural_anomaly_score = 1 - consistency``.

For IEEE-CIS (no customer-history fields) :func:`consistency_ieee_proxy` returns a
coarse score from the C*/D* counting/recency columns, always ``is_proxy=True``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from safetriage import load_policy_config

COMPONENTS = [
    "amount_consistency", "time_consistency", "device_consistency",
    "beneficiary_consistency", "location_consistency", "velocity_consistency",
]


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Individual components — each returns a Series/array in [0, 1].
# --------------------------------------------------------------------------- #
def _amount_consistency(df: pd.DataFrame) -> np.ndarray:
    # ratio of this amount to the customer's historical median. 1.0 == typical.
    r = df["TransactionAmt"].to_numpy(float) / np.maximum(df["historical_median_amount"].to_numpy(float), 1.0)
    excess = np.maximum(0.0, r - 1.0)
    return _clip01(1.0 / (1.0 + (excess / 1.5) ** 1.3))


def _time_consistency(df: pd.DataFrame) -> np.ndarray:
    h = df["transaction_hour"].to_numpy(float)
    nh = df["normal_transaction_hour"].to_numpy(float)
    gap = np.minimum((h - nh) % 24, (nh - h) % 24)   # circular hour distance
    return _clip01(1.0 - gap / 8.0)


def _device_consistency(df: pd.DataFrame) -> np.ndarray:
    new = df["is_new_device"].to_numpy(float)
    age = df["device_age_days"].to_numpy(float)
    known = _clip01(0.65 + np.minimum(age, 720.0) / 720.0 * 0.35)
    fresh = _clip01(0.10 + np.minimum(age, 3.0) / 3.0 * 0.10)
    return np.where(new >= 0.5, fresh, known)


def _beneficiary_consistency(df: pd.DataFrame) -> np.ndarray:
    new = df["is_new_beneficiary"].to_numpy(float)
    age = df["beneficiary_age_days"].to_numpy(float)
    cnt = df["beneficiary_transaction_count"].to_numpy(float)
    added_min = df["beneficiary_added_minutes_ago"].to_numpy(float)
    base = 0.5 * np.minimum(cnt, 10.0) / 10.0 + 0.5 * np.minimum(age, 180.0) / 180.0
    base = np.where(new >= 0.5, base * 0.4, base)
    base = np.where(added_min < 60.0, base * 0.5, base)          # added minutes ago
    return _clip01(base)


def _location_consistency(df: pd.DataFrame) -> np.ndarray:
    # the only coarse location signal we have is published directly upstream.
    return _clip01(df["location_consistency"].to_numpy(float))


def _velocity_consistency(df: pd.DataFrame) -> np.ndarray:
    v1 = df["velocity_1h_count"].to_numpy(float)
    v24 = df["velocity_24h_count"].to_numpy(float)
    p1 = 1.0 / (1.0 + (np.maximum(0.0, v1 - 1.0) / 3.0) ** 1.2)
    p24 = 1.0 / (1.0 + (np.maximum(0.0, v24 - 5.0) / 8.0) ** 1.2)
    return _clip01(np.minimum(p1, p24))


_COMPONENT_FN = {
    "amount_consistency": _amount_consistency,
    "time_consistency": _time_consistency,
    "device_consistency": _device_consistency,
    "beneficiary_consistency": _beneficiary_consistency,
    "location_consistency": _location_consistency,
    "velocity_consistency": _velocity_consistency,
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def behavioural_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with the six components + consistency + anomaly scores."""
    weights = load_policy_config()["behaviour_weights"]
    out = pd.DataFrame(index=df.index)
    for name, fn in _COMPONENT_FN.items():
        out[name] = np.round(fn(df), 5)
    total = sum(out[name] * weights[name] for name in COMPONENTS)
    out["behavioural_consistency_score"] = np.round(_clip01(total), 5)
    out["behavioural_anomaly_score"] = np.round(1.0 - out["behavioural_consistency_score"], 5)
    out["is_proxy"] = False
    return out


def score_row(row: dict) -> dict:
    """Single-transaction convenience wrapper around :func:`behavioural_frame`."""
    res = behavioural_frame(pd.DataFrame([row])).iloc[0].to_dict()
    res["is_proxy"] = bool(res["is_proxy"])
    return res


def consistency_ieee_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Coarse behavioural-consistency PROXY for IEEE-CIS rows.

    IEEE-CIS has no customer-history fields. We use the C* columns (transaction
    counts on shared attributes -> velocity proxy) and D* columns (timedeltas ->
    recency proxy): many recent counts / very small recency -> lower consistency.
    Flagged ``is_proxy=True`` so nothing downstream mistakes it for the real
    behavioural score.
    """
    c_cols = [c for c in df.columns if c[:1] == "C" and c[1:].isdigit()]
    d_cols = [c for c in df.columns if c[:1] == "D" and c[1:].isdigit()]
    out = pd.DataFrame(index=df.index)
    def _minmax(a: np.ndarray) -> np.ndarray:
        lo = np.nanmin(a, axis=0)
        rng = np.nanmax(a, axis=0) - lo
        return (a - lo) / (rng + 1e-9)

    if c_cols:
        c_rank = _minmax(df[c_cols].to_numpy(float))
        velocity_proxy = _clip01(1.0 - np.nan_to_num(np.nanmean(c_rank, axis=1), nan=0.5))
    else:
        velocity_proxy = np.full(len(df), 0.5)
    if d_cols:
        d_rank = _minmax(np.abs(df[d_cols].to_numpy(float)))
        recency_proxy = _clip01(np.nan_to_num(np.nanmean(d_rank, axis=1), nan=0.5))
    else:
        recency_proxy = np.full(len(df), 0.5)
    score = _clip01(0.6 * velocity_proxy + 0.4 * recency_proxy)
    out["behavioural_consistency_score"] = np.round(score, 5)
    out["behavioural_anomaly_score"] = np.round(1.0 - score, 5)
    out["velocity_proxy"] = np.round(velocity_proxy, 5)
    out["recency_proxy"] = np.round(recency_proxy, 5)
    out["is_proxy"] = True
    return out

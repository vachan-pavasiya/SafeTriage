"""Feature engineering + matrix construction, shared by training and scoring.

Two rules keep this honest:

1. **Everything is fit on the TRAIN split only.** Frequency encodings and
   per-key aggregates are learned from ``df.iloc[train_idx]`` and then *applied*
   to validation / test / inference rows. Unseen keys fall back to a global
   default. The fitted statistics are persisted to ``fe_<dataset>.joblib`` so a
   single live transaction is engineered identically.
2. **The exact training-time column layout** is persisted to
   ``feature_spec_<dataset>.json``; ``transform_for_model`` reindexes onto it so
   inference can never silently gain or lose a column.

The four upstream ``*_consistency`` columns and ``location_consistency`` are
still withheld from the classifier (see ``RESERVED_FOR_POLICY`` in the synthetic
generator) so the probability-only baseline cannot use signals SafeTriage
reserves for its interpretation layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from safetriage import ARTIFACTS_DIR
from safetriage.data.generate_synthetic_dataset import CATEGORICAL_FEATURES, MODEL_FEATURES

# raw high-cardinality keys retained (un-encoded) in processed_ieee.parquet so
# per-key aggregates can be fit on train only. They are NOT model features.
IEEE_KEY_COLS = ["k_card1", "k_card2", "k_addr1", "k_pemail", "k_remail"]


def _spec_path(dataset: str) -> Path:
    return ARTIFACTS_DIR / f"feature_spec_{dataset}.json"


def _stats_path(dataset: str) -> Path:
    return ARTIFACTS_DIR / f"fe_{dataset}.joblib"


def _safe_div(a, b):
    return np.asarray(a, dtype=float) / (np.asarray(b, dtype=float) + 1.0)


def _circ_hour_gap(h, ref):
    d = np.abs(np.asarray(h, float) - np.asarray(ref, float)) % 24.0
    return np.minimum(d, 24.0 - d)


# ===================================================================== #
# Synthetic (Dataset B)
# ===================================================================== #
def _base_synthetic_matrix(df: pd.DataFrame) -> pd.DataFrame:
    base = df[MODEL_FEATURES].copy()
    return pd.get_dummies(base, columns=CATEGORICAL_FEATURES, dtype=np.int8)


# kept for backwards compatibility (older callers / tests)
def build_synthetic_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return _base_synthetic_matrix(df)


def _fit_stats_synthetic(df_train: pd.DataFrame) -> dict:
    freq = df_train["card_id"].value_counts(normalize=True)
    return {
        "card_id_freq": freq.to_dict(),
        "card_id_freq_default": float(freq.min()) if len(freq) else 0.0,
    }


def _engineer_synthetic(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    X = _base_synthetic_matrix(df)
    amt = df["TransactionAmt"].to_numpy(float)
    eng = pd.DataFrame(index=df.index)

    eng["amt_log"] = np.log1p(amt)
    eng["amt_vs_hist_avg"] = _safe_div(amt, df["historical_avg_amount"])
    eng["amt_vs_hist_median"] = _safe_div(amt, df["historical_median_amount"])
    eng["amt_abs_dev"] = np.abs(amt - df["historical_avg_amount"].to_numpy(float))
    eng["acct_age_log"] = np.log1p(df["account_age_days"].to_numpy(float))
    eng["dev_age_log"] = np.log1p(df["device_age_days"].to_numpy(float))
    eng["benef_age_log"] = np.log1p(df["beneficiary_age_days"].to_numpy(float))

    hg = _circ_hour_gap(df["transaction_hour"], df["normal_transaction_hour"])
    eng["hour_gap_to_normal"] = hg
    eng["hour_sin"] = np.sin(2 * np.pi * df["transaction_hour"].to_numpy(float) / 24.0)
    eng["hour_cos"] = np.cos(2 * np.pi * df["transaction_hour"].to_numpy(float) / 24.0)
    eng["is_night_txn"] = (df["transaction_hour"].to_numpy(float) < 6).astype(np.int8)

    v1 = df["velocity_1h_count"].to_numpy(float)
    v24 = df["velocity_24h_count"].to_numpy(float)
    eng["vel_1h_share_24h"] = _safe_div(v1, v24)
    eng["amt_per_txn_24h"] = _safe_div(df["velocity_24h_amt_sum"], v24)
    eng["amt_share_24h"] = _safe_div(amt, df["velocity_24h_amt_sum"])
    eng["benef_txn_per_age"] = _safe_div(df["beneficiary_transaction_count"], df["beneficiary_age_days"])
    eng["benef_added_hours_ago"] = df["beneficiary_added_minutes_ago"].to_numpy(float) / 60.0
    eng["benef_added_within_24h"] = (df["beneficiary_added_minutes_ago"].to_numpy(float) < 1440).astype(np.int8)

    nd = df["is_new_device"].to_numpy(float)
    nb = df["is_new_beneficiary"].to_numpy(float)
    eng["new_device_x_new_benef"] = (nd * nb).astype(np.int8)
    eng["new_device_x_pwd_reset"] = (nd * df["recent_password_reset"].to_numpy(float)).astype(np.int8)
    eng["low_typing_x_new_device"] = ((df["typing_cadence_score"].to_numpy(float) < 0.6) & (nd > 0.5)).astype(np.int8)
    eng["mule_proximity"] = (
        (df["hop_distance_to_known_mule"].to_numpy(float) <= 2)
        | (df["shared_device_with_flagged_account"].to_numpy(float) > 0.5)
    ).astype(np.int8)

    cf = stats.get("card_id_freq", {})
    dflt = stats.get("card_id_freq_default", 0.0)
    if "card_id" in df.columns:
        eng["card_id_freq"] = df["card_id"].map(cf).fillna(dflt).astype(np.float32)
    else:  # live payload without the raw key
        eng["card_id_freq"] = np.float32(dflt)

    return pd.concat([X, eng.astype(np.float32)], axis=1)


# ===================================================================== #
# IEEE-CIS (Dataset A)  — engineered on top of processed_ieee.parquet
# ===================================================================== #
def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    return (df[col].to_numpy(float) if col in df.columns
            else np.full(len(df), default, dtype=float))


def _fit_stats_ieee(df_train: pd.DataFrame) -> dict:
    amt = df_train["TransactionAmt"].astype(float)
    stats: dict = {"amt_global_mean": float(amt.mean()), "amt_global_std": float(amt.std() + 1.0),
                   "keys": {}}
    for k in IEEE_KEY_COLS:
        if k not in df_train.columns:
            continue
        g = amt.groupby(df_train[k].astype("string").fillna("__nan"))
        stats["keys"][k] = {
            "count": g.count().to_dict(),
            "mean": g.mean().to_dict(),
            "std": (g.std().fillna(0.0) + 1.0).to_dict(),
            "count_default": 1.0,
        }
    return stats


def _engineer_ieee(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    drop = [c for c in (["TransactionID", "isFraud"] + IEEE_KEY_COLS) if c in df.columns]
    X = df.drop(columns=drop, errors="ignore").astype(np.float32)
    eng = pd.DataFrame(index=df.index)

    amt = _num(df, "TransactionAmt")
    eng["amt_log"] = np.log1p(np.clip(amt, 0, None))
    cents = np.round((amt - np.floor(amt)) * 100).astype(int)
    eng["amt_cents"] = cents.astype(np.float32)
    eng["amt_is_round"] = (cents == 0).astype(np.int8)

    dt = _num(df, "TransactionDT")
    eng["dt_hour"] = (dt / 3600.0) % 24.0
    eng["dt_dow"] = (dt / 86400.0) % 7.0
    eng["dt_hour_sin"] = np.sin(2 * np.pi * eng["dt_hour"] / 24.0)
    eng["dt_hour_cos"] = np.cos(2 * np.pi * eng["dt_hour"] / 24.0)

    c_cols = [f"C{i}" for i in range(1, 15) if f"C{i}" in df.columns]
    if c_cols:
        C = df[c_cols].to_numpy(float)
        eng["C_sum"] = C.sum(axis=1)
        eng["C_mean"] = C.mean(axis=1)
        eng["C_nonzero"] = (C > 0).sum(axis=1)
        eng["C_max"] = C.max(axis=1)
    d_cols = [f"D{i}" for i in range(1, 16) if f"D{i}" in df.columns]
    if d_cols:
        D = np.abs(df[d_cols].to_numpy(float))
        eng["D_min"] = D.min(axis=1)
        eng["D_mean"] = D.mean(axis=1)

    gmean, gstd = stats.get("amt_global_mean", 0.0), stats.get("amt_global_std", 1.0)
    for k, kst in stats.get("keys", {}).items():
        if k not in df.columns:
            continue
        key = df[k].astype("string").fillna("__nan")
        cnt = key.map(kst["count"]).fillna(kst.get("count_default", 1.0)).to_numpy(float)
        kmean = key.map(kst["mean"]).fillna(gmean).to_numpy(float)
        kstd = key.map(kst["std"]).fillna(gstd).to_numpy(float)
        short = k.replace("k_", "")
        eng[f"{short}_count_enc"] = np.log1p(cnt)
        eng[f"amt_vs_{short}_mean"] = _safe_div(amt, kmean)
        eng[f"amt_{short}_z"] = (amt - kmean) / kstd

    return pd.concat([X, eng.astype(np.float32)], axis=1)


# ===================================================================== #
# Public fit / transform
# ===================================================================== #
def _engineer(dataset: str, df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    if dataset == "synthetic":
        return _engineer_synthetic(df, stats)
    return _engineer_ieee(df, stats)


def fit_and_save_features(dataset: str, df: pd.DataFrame, train_idx: np.ndarray) -> pd.DataFrame:
    """Fit FE stats on ``df.iloc[train_idx]``, transform the whole frame, and
    persist both the stats and the resulting column layout. Returns full X."""
    df_train = df.iloc[np.asarray(train_idx)]
    stats = _fit_stats_synthetic(df_train) if dataset == "synthetic" else _fit_stats_ieee(df_train)
    X = _engineer(dataset, df, stats)
    joblib.dump(stats, _stats_path(dataset))
    _spec_path(dataset).write_text(json.dumps(
        {"dataset": dataset, "n_features": X.shape[1], "columns": X.columns.tolist()}, indent=2))
    return X


def load_feature_columns(dataset: str) -> list[str]:
    return json.loads(_spec_path(dataset).read_text())["columns"]


def save_feature_spec(dataset: str, columns: list[str]) -> None:  # kept for compatibility
    _spec_path(dataset).write_text(json.dumps({"dataset": dataset, "columns": list(columns)}, indent=2))


def transform_for_model(dataset: str, row: dict | pd.DataFrame) -> pd.DataFrame:
    """Engineer + reindex a raw input onto the exact training-time layout."""
    df = pd.DataFrame([row]) if isinstance(row, dict) else row
    cols = load_feature_columns(dataset)
    stats_p = _stats_path(dataset)
    stats = joblib.load(stats_p) if stats_p.exists() else {}
    X = _engineer(dataset, df, stats)
    return X.reindex(columns=cols, fill_value=0).astype(np.float32)

"""Phase 1 / Dataset A — IEEE-CIS Fraud Detection preprocessing.

IEEE-CIS is the real-world benchmark. It is used as-is (never turned into a
synthetic set). The Kaggle ``test_*.csv`` files have **no isFraud column**, so
they are unusable for evaluation; our held-out test split is carved out of
``train_transaction.csv`` (see safetriage.data.splits).

IMPORTANT: IEEE-CIS contains **no customer-history / behavioural-consistency
features**. The C* (counting) and D* (timedelta) columns are used downstream only
as *coarse velocity / recency proxies*, always flagged ``is_proxy=True``. Do not
read behavioural-consistency claims into this dataset.

Run: ``python -m safetriage.data.preprocess_ieee``
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config.seed import SEED
from safetriage import DATA_DIR, IEEE_RAW_DIR

# --------------------------------------------------------------------------- #
# Column selection. Kept explicit and deterministic so the feature set is
# reproducible and documented (DATASET_README.md).
# --------------------------------------------------------------------------- #
TXN_BASE = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD"]
CARD_COLS = ["card1", "card2", "card3", "card4", "card5", "card6"]
ADDR_DIST = ["addr1", "addr2", "dist1", "dist2"]
EMAIL_COLS = ["P_emaildomain", "R_emaildomain"]
C_COLS = [f"C{i}" for i in range(1, 15)]          # counting features (proxy velocity)
D_COLS = [f"D{i}" for i in range(1, 16)]          # timedelta features (proxy recency)
M_COLS = [f"M{i}" for i in range(1, 10)]          # match flags (mostly T/F)
ID_NUM = [f"id_0{i}" if i < 10 else f"id_{i}" for i in range(1, 12)]  # id_01..id_11
ID_CAT = ["id_12", "id_15", "id_16", "id_28", "id_29", "id_30", "id_31",
          "id_33", "id_35", "id_36", "id_37", "id_38", "DeviceType", "DeviceInfo"]

LOW_CARD_CAT = ["ProductCD", "card4", "card6", *M_COLS, "DeviceType",
                "id_12", "id_15", "id_16", "id_28", "id_29", "id_35", "id_36", "id_37", "id_38"]
HIGH_CARD_CAT = ["card1", "card2", "card3", "card5", "addr1", "addr2",
                 "P_emaildomain", "R_emaildomain", "DeviceInfo", "id_30", "id_31", "id_33"]

V_MAX_MISSING = 0.25     # drop V columns missing more than this fraction
V_KEEP_TOP = 60          # then keep this many by variance
MISSING_FLAG_MIN = 0.05  # add an explicit missing-indicator when > this fraction missing
PROXY_COLS = set(C_COLS + D_COLS)  # flagged is_proxy downstream


def _read_raw() -> pd.DataFrame:
    tx_path = IEEE_RAW_DIR / "train_transaction.csv"
    id_path = IEEE_RAW_DIR / "train_identity.csv"
    if not tx_path.exists():
        raise FileNotFoundError(
            f"{tx_path} not found. Place the IEEE-CIS CSVs under {IEEE_RAW_DIR}/ "
            "(see DATASET_README.md / EVAL_PROTOCOL.md fallback note)."
        )
    header = pd.read_csv(tx_path, nrows=0).columns.tolist()
    v_cols = [c for c in header if c.startswith("V")]
    usecols = [c for c in (TXN_BASE + CARD_COLS + ADDR_DIST + EMAIL_COLS
                           + C_COLS + D_COLS + M_COLS) if c in header] + v_cols
    tx = pd.read_csv(tx_path, usecols=usecols)

    idh = pd.read_csv(id_path, nrows=0).columns.tolist()
    id_use = [c for c in (["TransactionID"] + ID_NUM + ID_CAT) if c in idh]
    idf = pd.read_csv(id_path, usecols=id_use)

    df = tx.merge(idf, on="TransactionID", how="left")
    return df


def _select_v_columns(df: pd.DataFrame) -> list[str]:
    v_cols = [c for c in df.columns if c.startswith("V")]
    miss = df[v_cols].isna().mean()
    keep = miss[miss <= V_MAX_MISSING].index.tolist()
    var = df[keep].var(numeric_only=True).sort_values(ascending=False)
    return var.index[:V_KEEP_TOP].tolist()


def preprocess() -> tuple[pd.DataFrame, dict]:
    raw = _read_raw()
    v_keep = _select_v_columns(raw)

    numeric_cols = (["TransactionAmt", "TransactionDT"] + ADDR_DIST + C_COLS + D_COLS
                    + [c for c in ID_NUM if c in raw.columns] + v_keep)
    numeric_cols = [c for c in numeric_cols if c in raw.columns]

    parts: list[pd.DataFrame] = [
        pd.DataFrame(
            {"TransactionID": raw["TransactionID"].astype(int),
             "isFraud": raw["isFraud"].astype(int)},
            index=raw.index,
        )
    ]

    # numeric: median-impute + explicit missing indicator where informative
    missing_flags: list[str] = []
    num_block: dict[str, pd.Series] = {}
    for c in numeric_cols:
        s = pd.to_numeric(raw[c], errors="coerce")
        if float(s.isna().mean()) > MISSING_FLAG_MIN:
            num_block[f"{c}__isna"] = s.isna().astype(np.int8)
            missing_flags.append(f"{c}__isna")
        num_block[c] = s.fillna(s.median()).astype(np.float32)
    parts.append(pd.DataFrame(num_block, index=raw.index))

    # low-cardinality -> one-hot (top levels only, rest -> __other)
    onehot_cols: list[str] = []
    for c in [c for c in LOW_CARD_CAT if c in raw.columns]:
        s = raw[c].astype("string").fillna("__nan")
        top = s.value_counts().index[:12]
        s = s.where(s.isin(top), "__other")
        dummies = pd.get_dummies(s, prefix=c, dtype=np.int8)
        parts.append(dummies)
        onehot_cols.extend(dummies.columns.tolist())

    # high-cardinality -> frequency encoding
    freq_cols: list[str] = []
    freq_block: dict[str, pd.Series] = {}
    for c in [c for c in HIGH_CARD_CAT if c in raw.columns]:
        s = raw[c].astype("string").fillna("__nan")
        freq_block[f"{c}__freq"] = s.map(s.value_counts(normalize=True)).astype(np.float32).fillna(0.0)
        freq_cols.append(f"{c}__freq")
    parts.append(pd.DataFrame(freq_block, index=raw.index))

    # Raw high-cardinality keys, retained un-encoded so safetriage/features.py can
    # fit per-key aggregates on the TRAIN split only. NOT model features.
    key_src = {"k_card1": "card1", "k_card2": "card2", "k_addr1": "addr1",
               "k_pemail": "P_emaildomain", "k_remail": "R_emaildomain"}
    key_block = {kk: raw[src].astype("string").fillna("__nan")
                 for kk, src in key_src.items() if src in raw.columns}
    parts.append(pd.DataFrame(key_block, index=raw.index))

    out = pd.concat(parts, axis=1)
    key_cols = list(key_block.keys())
    feature_cols = [c for c in out.columns if c not in ("TransactionID", "isFraud", *key_cols)]

    manifest = {
        "dataset": "ieee_cis",
        "seed": SEED,
        "source_files": ["train_transaction.csv", "train_identity.csv"],
        "n_rows": int(len(out)),
        "fraud_prevalence": round(float(out["isFraud"].mean()), 5),
        "n_features": len(feature_cols),
        "numeric_cols": numeric_cols,
        "v_columns_kept": v_keep,
        "onehot_columns": onehot_cols,
        "frequency_encoded_columns": freq_cols,
        "missing_indicator_columns": missing_flags,
        "raw_key_columns_retained": key_cols,
        "engineered_in_features_py": [
            "amt_log", "amt_cents", "amt_is_round", "dt_hour", "dt_dow",
            "dt_hour_sin", "dt_hour_cos", "C_sum", "C_mean", "C_nonzero", "C_max",
            "D_min", "D_mean",
            "card1_count_enc", "amt_vs_card1_mean", "amt_card1_z",
            "card2_count_enc", "amt_vs_card2_mean", "amt_card2_z",
            "addr1_count_enc", "amt_vs_addr1_mean", "amt_addr1_z",
            "pemail_count_enc", "amt_vs_pemail_mean", "amt_pemail_z",
            "remail_count_enc", "amt_vs_remail_mean", "amt_remail_z",
        ],
        "proxy_velocity_recency_columns": sorted(PROXY_COLS & set(numeric_cols)),
        "note": (
            "IEEE-CIS has NO customer-history / behavioural-consistency fields. "
            "C*/D* columns are coarse velocity/recency proxies only."
        ),
        "feature_cols": feature_cols,
    }
    return out, manifest


def main() -> None:
    df, manifest = preprocess()
    out_path = DATA_DIR / "processed_ieee.parquet"
    df.to_parquet(out_path, index=False)
    (DATA_DIR / "feature_manifest_ieee.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_path}  ({manifest['n_rows']} rows, {manifest['n_features']} features, "
          f"prevalence={manifest['fraud_prevalence']})")
    print(f"V columns kept: {len(manifest['v_columns_kept'])}, "
          f"one-hot: {len(manifest['onehot_columns'])}, freq-encoded: {len(manifest['frequency_encoded_columns'])}, "
          f"missing-flags: {len(manifest['missing_indicator_columns'])}")


if __name__ == "__main__":
    main()

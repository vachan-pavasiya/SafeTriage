"""Phase 1 / Dataset B — controlled synthetic *behavioural* dataset.

The IEEE-CIS data (Dataset A) has no customer-history / behavioural-consistency
fields. SafeTriage's interpretation and decision layers need them, so we generate
a small, fully-controlled dataset with:

* explicit customer history (avg/median amount, normal hour, account age),
* device / beneficiary provenance,
* velocity, security and graph-proxy signals,
* four fraud typologies plus NONE, at ~3-4% prevalence,
* **deliberate class overlap** — some legitimate transactions look suspicious and
  some fraud looks perfectly normal — because the whole research question only
  exists when the score is ambiguous.

Nothing here is claimed to reflect production fraud rates or behaviour. It is a
controlled experimental instrument. Run:  ``python -m safetriage.data.generate_synthetic_dataset``
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config.seed import SEED, set_all_seeds
from safetriage import DATA_DIR

N_RECORDS = 3000
FRAUD_PREVALENCE = 0.042  # -> ~125 fraud rows, within the spec's 2-4% band
TYPOLOGIES = ("ACCOUNT_TAKEOVER", "SYNTHETIC_IDENTITY", "PUSH_PAYMENT", "MONEY_MULE")

# Fraction of fraud rows that are made to "look normal" (perturbations suppressed),
# and fraction of legit rows given 1-2 suspicious values. These create the overlap.
FRAUD_LOOKS_NORMAL_FRAC = 0.18
LEGIT_LOOKS_SUSPICIOUS_FRAC = 0.22
# Probability a typology actually expresses its signature signal on a given
# non-camouflaged fraud row. < 1.0 keeps the classes genuinely overlapping so
# the full multivariate model lands well short of AUC 1.0 (target ~0.85-0.90,
# with no single feature exceeding ~0.75).
SIGNAL_EXPRESSION_P = 0.92

DEVICE_TYPES = ("desktop", "mobile")
DEVICE_INFOS = ("Windows", "iOS Device", "MacOS", "Android 10", "Android 11", "Chrome", "Samsung", "Trident/7.0")
PRODUCTS = ("W", "C", "R", "H", "S")


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate() -> pd.DataFrame:
    set_all_seeds(SEED)
    rng = np.random.default_rng(SEED)
    n = N_RECORDS

    # ---- assign typology / label ----------------------------------------- #
    is_fraud = rng.random(n) < FRAUD_PREVALENCE
    typology = np.array(["NONE"] * n, dtype=object)
    fraud_idx = np.where(is_fraud)[0]
    typ_choice = rng.integers(0, len(TYPOLOGIES), size=fraud_idx.size)
    for k, ti in enumerate(fraud_idx):
        typology[ti] = TYPOLOGIES[typ_choice[k]]

    # which fraud rows are camouflaged, which legit rows are noisy
    looks_normal = np.zeros(n, dtype=bool)
    looks_normal[fraud_idx] = rng.random(fraud_idx.size) < FRAUD_LOOKS_NORMAL_FRAC
    legit_idx = np.where(~is_fraud)[0]
    looks_susp = np.zeros(n, dtype=bool)
    looks_susp[legit_idx] = rng.random(legit_idx.size) < LEGIT_LOOKS_SUSPICIOUS_FRAC

    # A non-camouflaged fraud row expresses its typology signature only with
    # probability SIGNAL_EXPRESSION_P; otherwise its signals stay near-baseline.
    express = rng.random(n) < SIGNAL_EXPRESSION_P
    sig = ~looks_normal & express  # "show the typology's signature on this row"

    # ---- customer history baseline ------------------------------------------ #
    account_age_days = rng.integers(20, 3200, size=n).astype(float)
    # synthetic-identity accounts are young
    si = typology == "SYNTHETIC_IDENTITY"
    account_age_days[si] = rng.integers(5, 90, size=si.sum())

    hist_avg = np.round(rng.lognormal(mean=4.2, sigma=0.5, size=n), 2)          # ~ $30-300
    hist_median = np.round(hist_avg * rng.uniform(0.7, 0.98, size=n), 2)
    normal_hour = rng.integers(7, 23, size=n)

    # ---- transaction amount ------------------------------------------------- #
    # legit: near the customer's own history; fraud: pushed higher (unless camo)
    dev_mult = rng.lognormal(mean=0.0, sigma=0.35, size=n)  # ~1.0 centred
    amt = hist_avg * dev_mult

    high_amt_typ = np.isin(typology, ["ACCOUNT_TAKEOVER", "PUSH_PAYMENT", "SYNTHETIC_IDENTITY"])
    boost = rng.uniform(1.9, 5.0, size=n)
    amt = np.where(high_amt_typ & sig, hist_avg * boost, amt)
    # a few legit rows get a big-ticket purchase
    amt = np.where(looks_susp & (rng.random(n) < 0.45), hist_avg * rng.uniform(2.2, 4.5, size=n), amt)
    amt = np.round(np.maximum(amt, 1.0), 2)
    amount_deviation_ratio = np.round(amt / np.maximum(hist_median, 1.0), 3)

    # ---- time / transaction clock ---------------------------------------- #
    txn_hour = rng.integers(0, 24, size=n)
    # legit usually transact near their normal hour
    near_normal = (normal_hour + rng.integers(-2, 3, size=n)) % 24
    keep_normal = (~is_fraud & ~looks_susp) | looks_normal
    txn_hour = np.where(keep_normal, near_normal, txn_hour)
    transaction_dt = (rng.integers(0, 183, size=n) * 86400 + txn_hour * 3600 + rng.integers(0, 3600, size=n)).astype(int)

    # ---- device ---------------------------------------------------------- #
    is_new_device = (rng.random(n) < 0.05).astype(int)
    ato = typology == "ACCOUNT_TAKEOVER"
    ato_sig = ato & sig
    si_sig = si & sig
    is_new_device[ato_sig] = 1                       # ATO almost always from a new device
    is_new_device[si_sig & (rng.random(n) < 0.6)] = 1
    is_new_device[looks_susp & (rng.random(n) < 0.5)] = 1
    device_age_days = np.where(is_new_device == 1, rng.integers(0, 3, size=n), rng.integers(10, 900, size=n)).astype(float)
    device_type = rng.choice(DEVICE_TYPES, size=n, p=[0.45, 0.55])
    device_info = rng.choice(DEVICE_INFOS, size=n)

    # ---- beneficiary --------------------------------------------------------- #
    is_new_beneficiary = (rng.random(n) < 0.07).astype(int)
    pp = typology == "PUSH_PAYMENT"
    is_new_beneficiary[pp & sig] = 1
    is_new_beneficiary[si_sig] = 1
    is_new_beneficiary[looks_susp & (rng.random(n) < 0.4)] = 1
    beneficiary_age_days = np.where(
        is_new_beneficiary == 1, rng.integers(0, 2, size=n), rng.integers(20, 1200, size=n)
    ).astype(float)
    beneficiary_transaction_count = np.where(
        is_new_beneficiary == 1, rng.integers(0, 2, size=n), rng.integers(2, 60, size=n)
    ).astype(float)
    mule = typology == "MONEY_MULE"
    mule_sig = mule & sig
    # mule beneficiaries: young account but many recent transactions (fan-in)
    beneficiary_transaction_count[mule_sig] = rng.integers(12, 70, size=int(mule_sig.sum()))
    beneficiary_age_days[mule_sig] = rng.integers(1, 40, size=int(mule_sig.sum()))

    beneficiary_added_minutes_ago = np.where(
        is_new_beneficiary == 1, rng.integers(1, 90, size=n), rng.integers(1440, 200000, size=n)
    ).astype(float)
    beneficiary_added_minutes_ago[pp & sig] = rng.integers(1, 30, size=int((pp & sig).sum()))

    # ---- velocity ------------------------------------------------------------ #
    base_v1 = rng.poisson(0.4, size=n)
    base_v24 = rng.poisson(3.0, size=n) + 1
    v1 = base_v1.copy()
    v24 = base_v24.copy()
    ato_open = ato & sig
    mule_open = mule & sig
    susp_burst = looks_susp & (rng.random(n) < 0.32)
    v1[ato_open] += rng.integers(3, 8, size=int(ato_open.sum()))
    v24[mule_open] += rng.integers(7, 20, size=int(mule_open.sum()))
    v1[susp_burst] += rng.integers(2, 5, size=int(susp_burst.sum()))
    v24 = np.maximum(v24, v1)
    velocity_1h_count = v1.astype(int)
    velocity_24h_count = v24.astype(int)
    velocity_24h_amt_sum = np.round(amt + hist_avg * rng.uniform(0.5, 4.0, size=n) * (velocity_24h_count - 1).clip(0), 2)

    # ---- security ------------------------------------------------------------ #
    recent_password_reset = (rng.random(n) < 0.025).astype(int)
    recent_password_reset[ato_sig] = (rng.random(int(ato_sig.sum())) < 0.75).astype(int)
    typing_cadence_score = _clip01(rng.normal(0.77, 0.12, size=n))  # 1 = matches enrolled profile
    typing_cadence_score[ato_sig] = _clip01(rng.normal(0.45, 0.15, size=int(ato_sig.sum())))
    typing_cadence_score[si_sig] = _clip01(rng.normal(0.58, 0.15, size=int(si_sig.sum())))
    typing_cadence_score = np.round(typing_cadence_score, 3)

    # ---- graph proxy ------------------------------------------------------- #
    shared_device_with_flagged_account = (rng.random(n) < 0.02).astype(int)
    shared_device_with_flagged_account[mule_sig] = 1
    shared_device_with_flagged_account[looks_susp & (rng.random(n) < 0.06)] = 1
    hop_distance_to_known_mule = rng.choice([1, 2, 3, 4, 5, 99], size=n, p=[0.01, 0.02, 0.05, 0.10, 0.12, 0.70])
    hop_distance_to_known_mule = np.where(mule_sig, rng.choice([1, 2, 3], size=n), hop_distance_to_known_mule)

    # ---- location consistency (only coarse signal we have) --------------- #
    location_consistency = np.round(_clip01(rng.normal(0.88, 0.08, size=n)), 3)
    location_consistency = np.where(
        (is_fraud & sig), np.round(_clip01(rng.normal(0.60, 0.16, size=n)), 3), location_consistency
    )
    location_consistency = np.where(
        looks_susp & (rng.random(n) < 0.3), np.round(_clip01(rng.normal(0.6, 0.15, size=n)), 3), location_consistency
    )

    # ---- reference upstream consistency signals (Phase 1 spec columns) --- #
    # These mirror what an upstream feature store might publish. behavior.py
    # RE-DERIVES its own components from raw fields; these are kept for the
    # dashboard / cross-checking and are NOT fed to the classifier.
    hour_gap = np.minimum((txn_hour - normal_hour) % 24, (normal_hour - txn_hour) % 24)
    time_consistency = np.round(_clip01(1.0 - hour_gap / 8.0), 3)
    device_consistency = np.round(
        _clip01(np.where(is_new_device == 1, rng.normal(0.2, 0.1, size=n), rng.normal(0.9, 0.07, size=n))), 3
    )
    beneficiary_consistency = np.round(
        _clip01(np.where(is_new_beneficiary == 1, rng.normal(0.25, 0.12, size=n), rng.normal(0.88, 0.08, size=n))), 3
    )

    df = pd.DataFrame(
        {
            "TransactionID": np.arange(1, n + 1),
            "TransactionDT": transaction_dt,
            "TransactionAmt": amt,
            "ProductCD": rng.choice(PRODUCTS, size=n, p=[0.55, 0.2, 0.12, 0.08, 0.05]),
            "card_id": [f"C{ci:05d}" for ci in rng.integers(1, 1500, size=n)],
            "account_age_days": account_age_days,
            "historical_avg_amount": hist_avg,
            "historical_median_amount": hist_median,
            "normal_transaction_hour": normal_hour,
            "transaction_hour": txn_hour,
            "DeviceType": device_type,
            "DeviceInfo": device_info,
            "is_new_device": is_new_device,
            "device_age_days": device_age_days,
            "beneficiary_id": [f"B{bi:05d}" for bi in rng.integers(1, 2200, size=n)],
            "is_new_beneficiary": is_new_beneficiary,
            "beneficiary_age_days": beneficiary_age_days,
            "beneficiary_transaction_count": beneficiary_transaction_count,
            "amount_deviation_ratio": amount_deviation_ratio,
            "time_consistency": time_consistency,
            "location_consistency": location_consistency,
            "device_consistency": device_consistency,
            "beneficiary_consistency": beneficiary_consistency,
            "velocity_1h_count": velocity_1h_count,
            "velocity_24h_count": velocity_24h_count,
            "velocity_24h_amt_sum": velocity_24h_amt_sum,
            "recent_password_reset": recent_password_reset,
            "beneficiary_added_minutes_ago": beneficiary_added_minutes_ago,
            "typing_cadence_score": typing_cadence_score,
            "shared_device_with_flagged_account": shared_device_with_flagged_account,
            "hop_distance_to_known_mule": hop_distance_to_known_mule,
            "isFraud": is_fraud.astype(int),
            "fraud_typology": typology,
            "_looks_normal_fraud": looks_normal.astype(int),
            "_looks_suspicious_legit": looks_susp.astype(int),
        }
    )
    return df


# Columns the fraud classifier is allowed to use. The *_consistency and
# location_consistency columns are deliberately withheld so the probability-only
# baseline cannot "cheat" with signals SafeTriage reserves for its policy layer.
MODEL_FEATURES = [
    "TransactionAmt", "ProductCD", "account_age_days", "historical_avg_amount",
    "historical_median_amount", "normal_transaction_hour", "transaction_hour",
    "DeviceType", "DeviceInfo", "is_new_device", "device_age_days",
    "is_new_beneficiary", "beneficiary_age_days", "beneficiary_transaction_count",
    "amount_deviation_ratio", "velocity_1h_count", "velocity_24h_count",
    "velocity_24h_amt_sum", "recent_password_reset", "beneficiary_added_minutes_ago",
    "typing_cadence_score", "shared_device_with_flagged_account",
    "hop_distance_to_known_mule",
]
CATEGORICAL_FEATURES = ["ProductCD", "DeviceType", "DeviceInfo"]
RESERVED_FOR_POLICY = [
    "time_consistency", "location_consistency", "device_consistency",
    "beneficiary_consistency",
]


def separability_report(df: pd.DataFrame) -> dict:
    """Confirm the data is NOT trivially separable by any single numeric feature."""
    y = df["isFraud"].to_numpy()
    scores = {}
    for col in MODEL_FEATURES:
        if col in CATEGORICAL_FEATURES:
            continue
        x = df[col].to_numpy(dtype=float)
        try:
            auc = roc_auc_score(y, x)
            scores[col] = round(float(max(auc, 1 - auc)), 4)
        except ValueError:
            continue
    top = dict(sorted(scores.items(), key=lambda kv: -kv[1])[:5])
    return {"max_single_feature_auc": max(scores.values()), "top5": top}


def main() -> None:
    df = generate()
    out = DATA_DIR / "synthetic.csv"
    df.to_csv(out, index=False)

    rep = separability_report(df)
    manifest = {
        "seed": SEED,
        "n_records": int(len(df)),
        "fraud_prevalence": round(float(df["isFraud"].mean()), 5),
        "typology_counts": df["fraud_typology"].value_counts().to_dict(),
        "looks_normal_fraud": int(df["_looks_normal_fraud"].sum()),
        "looks_suspicious_legit": int(df["_looks_suspicious_legit"].sum()),
        "model_features": MODEL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "reserved_for_policy_layer": RESERVED_FOR_POLICY,
        "separability_check": rep,
    }
    (DATA_DIR / "feature_manifest_synthetic.json").write_text(json.dumps(manifest, indent=2))

    print(f"wrote {out}  ({len(df)} rows, prevalence={manifest['fraud_prevalence']})")
    print(f"typologies: {manifest['typology_counts']}")
    print(f"separability: max single-feature AUC = {rep['max_single_feature_auc']:.3f}  {rep['top5']}")
    if rep["max_single_feature_auc"] > 0.92:
        raise SystemExit(
            "ABORT: a single feature separates the classes (AUC > 0.92). "
            "The dataset must stay ambiguous — adjust the generator."
        )


if __name__ == "__main__":
    main()

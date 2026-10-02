# SafeTriage — Datasets (Phase 1)

SafeTriage uses **two complementary datasets**. They answer different questions
and are never mixed or relabelled.

---

## Dataset A — IEEE-CIS Fraud Detection (real-world benchmark)

**Location:** `ieee-fraud-detection/` (provided).
**Access note:** the data is present locally, so the Phase-0 "no IEEE access"
fallback is not used. The Kaggle competition `test_*.csv` files carry **no
`isFraud` column**, so they cannot be scored for evaluation — our held-out test
split is carved out of `train_transaction.csv` instead.

**Preprocessing** (`safetriage/data/preprocess_ieee.py` → `data/processed_ieee.parquet`):

| Step | Detail |
|------|--------|
| Join | `train_transaction` ⟵ `train_identity` (left join on `TransactionID`); 590,540 rows, ~24% have identity data |
| Kept raw columns | `TransactionAmt`, `ProductCD`, `card1–6`, `addr1/2`, `dist1/2`, `P/R_emaildomain`, `C1–C14`, `D1–D15`, `M1–M9`, identity `id_01–id_11`, `id_12/15/16/28/29/30/31/33/35/36/37/38`, `DeviceType`, `DeviceInfo` |
| `V*` columns | drop those >25% missing, then keep the **60** highest-variance remaining |
| Low-cardinality categoricals | one-hot, top 12 levels + `__other` (`ProductCD`, `card4/6`, `M1–M9`, `DeviceType`, several `id_*`) |
| High-cardinality categoricals | frequency encoding (`card1/2/3/5`, `addr1/2`, email domains, `DeviceInfo`, `id_30/31/33`) |
| Numerics | median imputation + an explicit `<col>__isna` indicator when >5% missing |
| Result | **224 features**, fraud prevalence **3.5%** |

**What IEEE-CIS does NOT have:** customer transaction history, a "normal hour",
device/beneficiary provenance, or any behavioural-consistency field. The `C*`
(counting) and `D*` (timedelta) columns are used downstream **only as coarse
velocity / recency proxies**, and every value they produce is flagged
`is_proxy=True`. SafeTriage therefore runs on IEEE-CIS in **reduced mode**
(no true behavioural-consistency layer).

**Engineered features** (`safetriage/features.py`, fit on the TRAIN split only):
`amt_log`, `amt_cents` + `amt_is_round` (a known IEEE-CIS signal), cyclical
`TransactionDT` hour / day-of-week, C-column aggregates (`C_sum/mean/nonzero/max`),
D-column `D_min/D_mean`, and per-key aggregates for `card1`, `card2`, `addr1`,
`P_emaildomain`, `R_emaildomain`: log-count encoding, `amount / key-mean`, and the
`(amount − key-mean) / key-std` z-score. Raw keys are retained in the parquet as
`k_card1 … k_remail` (not model features) purely so these aggregates can be fit
train-only. ~26 engineered columns on top of the ~224 base features.

**Role:** realistic fraud-model performance and an honest calibration
demonstration (large labelled set), plus a supplementary reduced Baseline vs
SafeTriage comparison.

---

## Dataset B — controlled synthetic behavioural dataset (primary experimental vehicle)

**Generator:** `safetriage/data/generate_synthetic_dataset.py` → `data/synthetic.csv`
(seed 42, **3,000 records**, fraud prevalence **~3.9%**).

**Fields** — transaction (`TransactionID`, `TransactionAmt`, `ProductCD`,
`TransactionDT`), account history (`account_age_days`,
`historical_avg_amount`, `historical_median_amount`, `normal_transaction_hour`),
device (`DeviceType`, `DeviceInfo`, `is_new_device`, `device_age_days`),
beneficiary (`beneficiary_id`, `is_new_beneficiary`, `beneficiary_age_days`,
`beneficiary_transaction_count`), behaviour (`amount_deviation_ratio`,
`time_consistency`, `location_consistency`, `device_consistency`,
`beneficiary_consistency`, `velocity_1h_count`, `velocity_24h_count`,
`velocity_24h_amt_sum`), security (`recent_password_reset`,
`beneficiary_added_minutes_ago`, `typing_cadence_score`), graph proxy
(`shared_device_with_flagged_account`, `hop_distance_to_known_mule`), labels
(`isFraud`, `fraud_typology` ∈ {NONE, ACCOUNT_TAKEOVER, SYNTHETIC_IDENTITY,
PUSH_PAYMENT, MONEY_MULE}).

**Deliberate ambiguity** (the whole research problem depends on this):

* Each typology expresses its signature signal on only ~86% of its
  non-camouflaged rows (`SIGNAL_EXPRESSION_P`).
* ~22% of fraud rows are camouflaged to *look normal*
  (`FRAUD_LOOKS_NORMAL_FRAC`); ~22% of legitimate rows are injected with 1–2
  suspicious values (`LEGIT_LOOKS_SUSPICIOUS_FRAC`).
* A built-in check asserts **no single feature separates the classes**
  (max single-feature ROC-AUC ≈ 0.75); after feature engineering + tuning the
  full multivariate XGBoost model lands at **CV ROC-AUC ≈ 0.88 / PR-AUC ≈ 0.56**
  — clearly imperfect, which is the point. The tuner runs in a shallow, heavily
  regularised "small" regime for this dataset so the train-vs-CV ROC-AUC gap
  stays ≈ 0.04 (well inside the 0.10 overfit guard).

**Feature governance:** the four upstream `*_consistency` columns and
`location_consistency` are **withheld from the fraud classifier** (see
`MODEL_FEATURES` / `RESERVED_FOR_POLICY` in the generator). This keeps the
probability-only baseline honest — it cannot "cheat" using the signals SafeTriage
reserves for its interpretation layer. `safetriage/behavior.py` re-derives its
own six consistency components from raw observable fields.

**Engineered features** (`safetriage/features.py`, TRAIN-fit): amount ratios &
logs (`amt_log`, `amt_vs_hist_avg/median`, `amt_abs_dev`), circular time
(`hour_gap_to_normal`, `hour_sin/cos`, `is_night_txn`), account/device/beneficiary
age logs, velocity ratios (`vel_1h_share_24h`, `amt_per_txn_24h`, `amt_share_24h`,
`benef_txn_per_age`), beneficiary recency (`benef_added_hours_ago`,
`benef_added_within_24h`), interactions (`new_device_x_new_benef`,
`new_device_x_pwd_reset`, `low_typing_x_new_device`, `mule_proximity`), and a
train-fit `card_id` frequency. ~22 engineered columns.

**Role:** the only dataset with the customer-history features SafeTriage's
interpretation and decision layers need, so it carries the headline Baseline vs
SafeTriage comparison and drives the dashboard.

---

## Split (both datasets)

60 / 20 / 20, stratified by `isFraud`, seed 42, written to `data/splits/`.
See `EVAL_PROTOCOL.md` for what each split may touch. **This is controlled
experimental data — it is not evidence of production performance.**

# Inside SafeTriage

**A complete engineering walkthrough of the SafeTriage prototype** — architecture, data flow,
internal working, model performance, API contract, and dashboard rendering.

> BTP capstone prototype · 56 source files · 5,541 lines · Python 3.13 + React 18 ·
> 2 datasets (590,540 + 3,000 rows) · 0 external services
>
> All figures in this document are computed on the held-out test split with seed 42.

---

## Contents

| § | Section |
|---|---------|
| [01](#01--the-thesis) | The thesis |
| [02](#02--system-map) | System map |
| [03](#03--repository-layout) | Repository layout |
| [04](#04--the-two-datasets) | The two datasets |
| [05](#05--the-build-pipeline) | The build pipeline |
| [06](#06--prediction-layer) | Prediction layer |
| [07](#07--interpretation-layer) | Interpretation layer |
| [08](#08--decision-layer) | Decision layer |
| [09](#09--explanation-layer) | Explanation layer |
| [10](#10--feedback--self-audit) | Feedback & self-audit |
| [11](#11--the-api) | The API |
| [12](#12--the-dashboard) | The dashboard |
| [13](#13--data-flow-traces) | Data-flow traces |
| [14](#14--measured-results) | Measured results |
| [15](#15--running-it) | Running it |
| [16](#16--real--simulated--future) | Real / simulated / future |

---

## 01 — The thesis

SafeTriage is not an attempt to build a better fraud classifier. The classifier is a component.
The research question is what you do with its output.

A conventional fraud system lets a model score cross a threshold and then blocks the transaction.
On an imbalanced problem with an imperfect model this produces two costs nobody measures in the
usual ROC-AUC leaderboard: **legitimate customers blocked**, and **analyst hours burned** on
everything that lands in manual review.

The question this prototype exists to answer:

> **How can we reduce unnecessary fraud alerts and analyst workload without significantly
> sacrificing fraud detection?**

The answer implemented here is a pipeline that keeps five stages separate, so that no single stage
gets to decide alone. The calibrated model probability is *one input* to a documented policy
engine — never the block/allow decision itself. Alongside it the engine weighs whether the
transaction is *behaviourally consistent with that customer's own history*, how much two
independently-trained models *disagree*, and the current *review capacity* and *cost* trade-off.

The output is one of four actions, not a binary:

| Decision | Meaning |
|---|---|
| **AUTO_CLEAR** | Let it through. No analyst touch, no customer friction. |
| **STEP_UP_REVIEW** | Queue for an analyst with an evidence checklist. Not blocked — this is the reclassification that does the false-positive work. |
| **SOFT_BLOCK** | Hold pending review. Reversible. Used when model risk is high but behaviour still looks like the customer. |
| **ESCALATE** | Confident, strongly anomalous, models agree. Priority analyst. |

The claim being tested is that moving cases from "blocked" to "reviewed" buys a very large
false-positive reduction for a small, measured, tunable recall cost — and that the operating point
is a dial the institution sets, not a property of the model.

---

## 02 — System map

Five stages. Each one is a separate module with a separate output, and the boundary between them
is deliberate.

```
  ┌──────────────┐   ┌──────────────────┐   ┌──────────────┐   ┌────────────────┐   ┌────────────┐
  │ 1 PREDICTION │ → │ 2 INTERPRETATION │ → │ 3 DECISION   │ → │ 4 HUMAN        │ → │ 5 SELF-    │
  │              │   │                  │   │              │   │   FEEDBACK     │   │   AUDIT    │
  │ XGBoost +    │   │ behavioural      │   │ 4-action     │   │ analyst verdict│   │ both blind │
  │ calibration  │   │ consistency +    │   │ rule table + │   │ + reason tag,  │   │ spots,     │
  │ (+ RF for    │   │ disagreement +   │   │ capacity +   │   │ stored against │   │ test split │
  │  disagree)   │   │ SHAP + reason    │   │ cost model   │   │ original output│   │ only       │
  └──────────────┘   └──────────────────┘   └──────────────┘   └────────────────┘   └────────────┘

  calibrated probability  →  ONE input to the policy engine, never the block/allow decision
```

| Stage | Does | Files |
|---|---|---|
| **1 Prediction** | XGBoost produces a raw score; a Random Forest scores in parallel. The raw score is then calibrated into a probability that means what it says. | `models/train.py`, `features.py` |
| **2 Interpretation** | Behavioural consistency (computed with no reference to the fraud score) plus a model-disagreement proxy, SHAP attributions and a reason code. | `behavior.py`, `uncertainty.py`, `explain.py` |
| **3 Decision** | A priority-ordered rule table combines all signals with review capacity and a cost model to choose one of four actions. | `policy.py`, `config/policy_config.yaml` |
| **4 Human feedback** | The analyst verdict and reason tag are stored against the original model output and policy decision — the dataset a future recalibration would learn from. | `db.py`, `api/main.py` |
| **5 Self-audit** | Samples both blind spots — what was auto-cleared, and what was blocked — from the held-out test split only. | `audit.py` |

### Component responsibilities

| Layer | Module | Produces |
|---|---|---|
| Data | `data/preprocess_ieee.py`, `data/generate_synthetic_dataset.py`, `data/splits.py` | processed parquet, synthetic CSV, seeded split index files |
| Features | `features.py` | train-fit encoders (`fe_<ds>.joblib`), frozen column layout |
| Tuning | `tuning.py` | best hyper-parameters + full trial record |
| Prediction | `models/train.py` | XGBoost, Random Forest, calibrator, calibration curve, overfit check |
| Behaviour | `behavior.py` | six-component consistency score (+ IEEE proxy) |
| Uncertainty | `uncertainty.py` | normalised model-disagreement proxy |
| Policy | `policy.py` | decision, firing rule, plain-language explanation, capacity + cost |
| Explanation | `explain.py`, `graph_proxy.py`, `next_best_evidence.py` | SHAP top-k, reason codes, evidence split, analyst checklist, network view |
| Orchestration | `scoring.py` | the single `/score` object; batch scorer |
| Persistence | `db.py` | SQLite: transactions, feedback, audit_runs, audit_samples |
| Audit | `audit.py` | stratified + control samples of both blind-spot populations |
| Evaluation | `evaluate.py` | baseline vs SafeTriage on the sealed test split |
| Serving | `api/main.py` | FastAPI, 10 routes |
| Interface | `frontend/` | React dashboard, 10 pages |

---

## 03 — Repository layout

Source is tracked; everything derived is gitignored and rebuilt by one command.

```
BTP Claude Pro/
├── ieee-fraud-detection/          raw input, never modified, not tracked
├── config/
│   ├── seed.py                    SEED = 42, single source of truth
│   └── policy_config.yaml         every threshold, weight, cost and training knob
├── safetriage/                    the library — importable, no I/O on import
│   ├── data/
│   │   ├── preprocess_ieee.py     590,540 rows → 224 base features
│   │   ├── generate_synthetic_dataset.py
│   │   └── splits.py              60/20/20 stratified, test-split guard
│   ├── models/
│   │   ├── train.py               FE → tune → fit → calibrate → overfit check
│   │   └── artifacts/             30 files: weights, encoders, metrics, curves
│   ├── features.py                feature engineering, train-fit only
│   ├── tuning.py                  Optuna TPE / HalvingRandomSearchCV
│   ├── behavior.py   uncertainty.py   policy.py
│   ├── explain.py    graph_proxy.py   next_best_evidence.py
│   ├── scoring.py    db.py   audit.py   evaluate.py   metrics.py
├── api/
│   └── main.py                    FastAPI — 10 routes
├── scripts/
│   ├── reset_state.py             wipe every derived artifact
│   ├── build_splits.py            step 1 — the splits
│   ├── run_pipeline.py            everything, one command
│   ├── policy_simulation.py       capacity/cost sweep
│   └── smoke_api.py               asserts every endpoint returns real data
├── frontend/
│   ├── vite.config.js             /api proxy → 127.0.0.1:8000 (dev + preview)
│   └── src/
│       ├── App.jsx                router, dataset context, API status
│       ├── lib/    api.js  constants.js  format.js
│       ├── components/  ui.jsx    shared primitives
│       └── pages/                 10 pages, one file each
└── data/                          generated — gitignored
    ├── processed_ieee.parquet · synthetic.csv
    ├── splits/ {ieee,synthetic}_{train,val,test}.idx
    ├── evaluation_*.json · policy_simulation_*.json
    └── safetriage.db              SQLite, 6,000 scored transactions
```

> **Why nothing derived is tracked.** The IEEE-CIS dataset is competition data that shouldn't be
> redistributed, model weights are large binaries, and every derived file is reproducible from
> seed 42. A fresh clone runs `python scripts/run_pipeline.py --reset --force --epochs 20` and
> lands on the same numbers.

---

## 04 — The two datasets

They answer different questions and are never mixed, relabelled, or averaged together.

### Dataset A — IEEE-CIS Fraud Detection

The real-world benchmark. 590,540 transactions from the Kaggle competition, left-joined with the
identity table (~24% of rows carry identity data), 3.499% fraud prevalence.

| Preprocessing step | Detail |
|---|---|
| Columns kept | `TransactionAmt`, `ProductCD`, `card1–6`, `addr1/2`, `dist1/2`, email domains, `C1–C14`, `D1–D15`, `M1–M9`, identity `id_01–id_11` and 12 categorical `id_*`, `DeviceType`, `DeviceInfo` |
| `V*` columns | drop those >25% missing, keep the 60 highest-variance survivors |
| Low-cardinality categoricals | one-hot, top 12 levels + `__other` → 74 columns |
| High-cardinality categoricals | frequency encoding → 12 columns |
| Numerics | median impute + an explicit `<col>__isna` flag where >5% missing → 32 flags |
| Raw keys retained | `k_card1 … k_remail` — not model features; kept un-encoded so per-key aggregates can be fit on train only |
| **Result** | **224 base features**, 252 after engineering |

> **The honest caveat.** IEEE-CIS has **no customer-history or behavioural-consistency fields** —
> no "normal hour", no device or beneficiary provenance. The `C*` (counting) and `D*` (timedelta)
> columns are used downstream only as coarse velocity/recency proxies, and every value they
> produce is flagged `is_proxy: true`. SafeTriage therefore runs on IEEE-CIS in **reduced mode** —
> a weaker version of the system, not the intended one.

### Dataset B — controlled synthetic behavioural data

The only dataset carrying the customer-history features SafeTriage's interpretation layer needs.
3,000 records, seed 42, 3.93% prevalence (118 fraud rows) across four typologies plus `NONE`:

| Typology | Rows | Signature signals |
|---|---:|---|
| NONE (legitimate) | 2,882 | — |
| SYNTHETIC_IDENTITY | 36 | young account, new device, new beneficiary, mid typing cadence |
| PUSH_PAYMENT | 29 | brand-new beneficiary added minutes ago, large amount — but the real customer on a known device |
| ACCOUNT_TAKEOVER | 28 | new device, recent password reset, velocity burst, low typing-cadence match |
| MONEY_MULE | 25 | 24h fan-in volume, shared device with a flagged account, 1–3 hops to a known mule |

#### Deliberate ambiguity

The research problem only exists when the score is uncertain, so the generator engineers overlap
rather than avoiding it:

- A non-camouflaged fraud row expresses its typology signature only with probability
  `SIGNAL_EXPRESSION_P = 0.92`.
- **18%** of fraud rows are camouflaged to look normal (`FRAUD_LOOKS_NORMAL_FRAC`).
- **22%** of legitimate rows are injected with one or two suspicious values
  (`LEGIT_LOOKS_SUSPICIOUS_FRAC`).
- A built-in check aborts generation if any single feature separates the classes. Measured maximum
  single-feature ROC-AUC: **0.750**; the full multivariate model reaches **0.876**.

#### Feature governance

The four upstream `*_consistency` columns and `location_consistency` are **withheld from the
classifier** (`RESERVED_FOR_POLICY` in the generator). This keeps the probability-only baseline
honest — it cannot cheat using the signals SafeTriage reserves for its interpretation layer.
`behavior.py` re-derives its own six components from raw observable fields instead.

### The split protocol

60/20/20, stratified by `isFraud`, seed 42, written once to `data/splits/` and reused. The Kaggle
`test_*.csv` files carry no label column, so the held-out test split is carved out of
`train_transaction.csv`.

| Split | IEEE rows | Synthetic rows | What may touch it |
|---|---:|---:|---|
| train | 354,324 | 1,800 | feature-encoder fitting, the hyper-parameter search (CV *inside* train), model fitting (65% sub-slice), calibrator fitting (35% sub-slice) |
| validation | 118,108 | 600 | early stopping, calibration-method choice, the baseline threshold, derived policy cut-points, the disagreement normaliser, the capacity sweep |
| test | 118,108 | 600 | **only** the baseline-vs-SafeTriage evaluation and self-audit sampling |

The guard is in code, not just in prose: `splits.load_trainable_indices()` raises a
`PermissionError` if asked for `"test"`, and only `evaluate.py` and `audit.py` call
`load_test_indices()`.

---

## 05 — The build pipeline

What `run_pipeline.py --reset --force --epochs 20` actually does, in order. Roughly 30–40 minutes
end to end; the IEEE hyper-parameter search dominates.

| # | Where | What |
|---:|---|---|
| 01 | `reset_state()` | Delete every derived artifact — model weights, calibrators, feature encoders, split index files, the SQLite database, generated datasets, all evaluation/simulation JSON and PNGs. Raw input under `ieee-fraud-detection/` is untouched. |
| 02 | `preprocess_ieee.main()` | Join transaction + identity tables, select and encode columns, write `data/processed_ieee.parquet` (~50 MB) and a feature manifest. ~30 s. |
| 03 | `generate_synthetic_dataset.main()` | Generate 3,000 behavioural records, run the separability check, write `data/synthetic.csv`. ~1 s. |
| 04 | `splits.make_splits()` — IEEE first | Stratified 60/20/20 per dataset, index files written to `data/splits/`. Deterministic given the seed. |
| 05 | `features.fit_and_save_features()` | Fit every encoder on `df.iloc[train_idx]` only, transform the whole frame, persist the statistics to `fe_<ds>.joblib` and the exact column layout to `feature_spec_<ds>.json`. |
| 06 | `tuning.tune()` | 20 Optuna TPE trials × 3-fold stratified CV, objective = mean PR-AUC, run on the train split only. IEEE is tuned on a 120,000-row stratified subsample with all fraud rows kept. Median pruner cuts unpromising trials early. **The slow step** — ~20 min for IEEE, ~1 min for synthetic. |
| 07 | `train_dataset()` — fit | Refit XGBoost on the train sub-slice with the tuned parameters, fit the Random Forest, fit the calibrator on the held-back 35% of train, choose sigmoid vs isotonic, compute the reliability curve, Brier and ECE on validation. |
| 08 | overfit guard + full refit | Compute the train-minus-CV ROC-AUC gap and flag it if it exceeds 0.10. Then refit both models on **100% of rows** with the same tuned hyper-parameters as `xgb_<ds>_full` / `rf_<ds>_full` — serving-only artifacts, never evaluated. |
| 09 | `evaluate.evaluate()` | Held-out test split. Pick the baseline threshold on validation to hit the target recall, freeze it, apply to test. Score the test split through the full SafeTriage pipeline. Write `data/evaluation_<ds>.json` including three real demo transactions re-scored in full. |
| 10 | `policy_simulation.simulate()` | Sweep analyst capacity across 11 points on the validation split; record recall, false positives, review volume, concessions and expected cost at each. Writes JSON + PNG. |
| 11 | `db.init_db(reset=True)` + `_store_dataset()` | Rebuild SQLite. Score and store all 3,000 synthetic rows *with SHAP*, plus the first 3,000 IEEE test rows without SHAP (`IEEE_STORE_SAMPLE`). 6,000 rows total. |
| 12 | `audit.run_audit()` ×2 | Sample both blind-spot populations from the test split for each dataset; write the run summary and the sampled rows. |
| 13 | seed feedback | Insert three sample analyst decisions so the feedback endpoints have real data immediately. |

### Individual steps

```bash
# the same thing, one stage at a time
python scripts/reset_state.py                    # wipe all prior state
python scripts/build_splits.py                   # step 1 — splits, IEEE first
python -m safetriage.models.train --epochs 20    # step 2 — FE + tuning + fit + calibrate
python -m safetriage.evaluate                    # step 3 — held-out evaluation
python -m safetriage.audit synthetic             # step 4 — a self-audit pass
python scripts/policy_simulation.py              # capacity/cost sweep
```

> **On "epochs".** XGBoost has no epochs — each boosting round adds one tree, and the round count
> is set by early stopping (reported as `boosting_rounds_used`). The `--epochs` flag is the number
> of *hyper-parameter configurations* the tuner evaluates by cross-validation. The naming is
> preserved because that's what was asked for; the metrics file states the distinction explicitly.

---

## 06 — Prediction layer

Two models, one calibrator, and a guard that makes overfitting a reported number rather than a hope.

### Feature engineering — `features.py`

Two rules keep this honest. **Everything is fit on the train split only** — frequency encodings and
per-key aggregates are learned from `df.iloc[train_idx]` and then *applied* to validation, test and
live inference, with unseen keys falling back to a global default. And **the exact training-time
column layout is frozen** to `feature_spec_<ds>.json`, so `transform_for_model()` reindexes onto it
and inference can never silently gain or lose a column.

| Dataset | Engineered columns |
|---|---|
| **Synthetic** (+22) | amount ratios and logs (`amt_log`, `amt_vs_hist_avg`, `amt_vs_hist_median`, `amt_abs_dev`); circular time (`hour_gap_to_normal`, `hour_sin/cos`, `is_night_txn`); age logs; velocity ratios (`vel_1h_share_24h`, `amt_per_txn_24h`, `amt_share_24h`, `benef_txn_per_age`); beneficiary recency; interactions (`new_device_x_new_benef`, `new_device_x_pwd_reset`, `low_typing_x_new_device`, `mule_proximity`); train-fit `card_id` frequency |
| **IEEE-CIS** (+28) | `amt_log`, `amt_cents` and `amt_is_round` (a known IEEE-CIS signal); cyclical `TransactionDT` hour/day-of-week; C-column aggregates (`C_sum/mean/nonzero/max`); `D_min/D_mean`; and for each of `card1`, `card2`, `addr1`, `P_emaildomain`, `R_emaildomain` — log count encoding, amount ÷ key-mean, and the key z-score |

### Hyper-parameter search — `tuning.py`

Optuna TPE sampler with a median pruner when Optuna is importable, otherwise scikit-learn's
`HalvingRandomSearchCV`. Objective is **mean PR-AUC over stratified k-fold CV** — the right metric
for a rare positive class, where ROC-AUC flatters everything. The search runs on the train split only.

The search space is **size-aware**, which is what keeps the small dataset from memorising its 71
training fraud rows:

| Parameter | Large regime (IEEE) | Small regime (<20k rows or <400 fraud) |
|---|---|---|
| `max_depth` | 3 – 8 | 3 – 5 |
| `n_estimators` | 150 – 1200 | 100 – 600 |
| `learning_rate` | 0.01 – 0.12 (log) | 0.015 – 0.07 (log) |
| `min_child_weight` | 1 – 20 | 4 – 25 *(floor raised)* |
| `reg_lambda` | 0.01 – 20 | 0.5 – 20 *(floor raised)* |
| `gamma` | 0.001 – 5 | 0.05 – 4 *(floor raised)* |

### Calibration

The calibrator is fit on a 35% holdout carved out of **train** — never on validation, never on
test — so the reliability curve reported on validation sits on genuinely unseen data. Sigmoid
(Platt) is preferred because it produces a smooth monotone map; isotonic is only permitted on the
large dataset, and only when it cuts holdout Brier by ≥10% without hurting ranking. On a small
positive class isotonic overfits into flat plateaus that destroy the probability spread the policy
thresholds depend on.

Everything downstream consumes the **calibrated** probability. The raw score is retained in the
response purely so the dashboard can show the before/after.

### The overfit guard

After fitting, the model is scored two ways: train ROC-AUC on the fit slice, and 5-fold
cross-validated ROC-AUC over train ∪ validation. The gap between them is written to
`model_metrics_<ds>.json → overfit_check` and flagged when it exceeds `training.overfit_gap_warn`
(0.10).

| Dataset | Train ROC-AUC | CV ROC-AUC | Gap | Warning |
|---|---:|---:|---:|---|
| IEEE-CIS | 1.000 | 0.957 | +0.043 | PASS |
| Synthetic | 0.904 | 0.876 | +0.035 | PASS |

### Measured model performance

| Metric | IEEE-CIS | Synthetic |
|---|---:|---:|
| Features (base + engineered) | 252 | 57 |
| Tuning trials / pruned | 20 / 4 | 20 / 6 |
| Tuned on rows | 120,000 | 1,800 |
| Best CV PR-AUC (tuner) | 0.844 | 0.515 |
| Boosting rounds used | 878 | 520 |
| **CV ROC-AUC (XGBoost)** | **0.957** | **0.876** |
| **CV PR-AUC (XGBoost)** | **0.926** | **0.563** |
| CV ROC-AUC (Random Forest) | 0.904 | 0.853 |
| Calibration method | sigmoid | sigmoid |
| Brier — raw → calibrated | 0.0133 → 0.0139 | 0.0370 → 0.0338 |
| ECE — raw → calibrated | 0.0019 → 0.0025 | 0.0026 → 0.0121 |

The headline discrimination number is cross-validated over train ∪ validation rather than read off
the single validation split, because that split holds only ~24 synthetic fraud rows — far too few
for a stable AUC. On IEEE the CV runs on a seeded 60,000-row stratified subsample with all fraud
rows retained, purely for tractability.

---

## 07 — Interpretation layer

Two signals computed with no reference to the fraud probability, so they can genuinely contradict it.

### Behavioural consistency — `behavior.py`

A score in `[0,1]` where **1.0 means the transaction looks like this customer's own history** and
0.0 means strongly out of pattern. Six components, each a documented bounded transform of
observable fields — no learned weights — combined with the weights in `policy_config.yaml`:

| Component | Weight | Derived from |
|---|---:|---|
| `amount_consistency` | 0.22 | amount ÷ the customer's historical median, decaying past 1.0 |
| `beneficiary_consistency` | 0.22 | new-beneficiary flag, beneficiary age, transaction count, minutes since added |
| `device_consistency` | 0.20 | new-device flag and device age |
| `time_consistency` | 0.13 | circular hour distance from the customer's normal transaction hour |
| `velocity_consistency` | 0.13 | 1h and 24h counts against a personal baseline |
| `location_consistency` | 0.10 | consumed directly — the only coarse location signal available |

`behavioural_anomaly_score = 1 − consistency`. The two worked cases that motivate keeping this
separate from the fraud score:

- **High probability + high consistency → de-escalate.** The model is confident, but nothing about
  the transaction is unusual for this customer. That combination is as likely to be a model error
  or a legitimate large purchase as it is fraud, so it becomes a soft block or a review rather than
  a hard block.
- **High probability + high anomaly → escalate.** Two independent signals agree. This is where
  confident automated action is defensible.

For IEEE rows, `consistency_ieee_proxy()` returns a coarse score from min-max normalised `C*`
(velocity proxy) and `D*` (recency proxy) columns, always tagged `is_proxy: true` so nothing
downstream mistakes it for the real thing.

### Model-disagreement uncertainty — `uncertainty.py`

```
raw_gap     = |p_xgb_calibrated − p_random_forest|
uncertainty = clip( (raw_gap − p05) / (p95 − p05), 0, 1 )
```

where `p05` and `p95` come from the gap distribution measured on the validation split and stored in
`uncertainty_norm_<ds>.json`. `confidence_score = 1 − uncertainty`.

> **Naming discipline.** This is labelled a **model-disagreement proxy** everywhere it appears — in
> the module docstring, the API response, and the dashboard. It is *not* epistemic or Bayesian
> uncertainty. Two models agreeing does not mean the prediction is correct; it means they are not a
> useful place to ask for a second opinion.

---

## 08 — Decision layer

A readable, priority-ordered rule table. No magic numbers live in the code — every threshold comes
from `config/policy_config.yaml`.

### The rule table — first match wins

| # | Condition | Decision | Firing rule |
|---:|---|---|---|
| 1 | uncertainty ≥ 0.50 *and* p ≥ p_low | STEP_UP_REVIEW | `MODEL_DISAGREEMENT` |
| 2 | p ≥ p_high *and* anomaly ≥ 0.60 | ESCALATE | `HIGH_RISK_HIGH_ANOMALY` |
| 3 | p ≥ p_high *and* consistency ≥ 0.70 | SOFT_BLOCK | `HIGH_RISK_BEHAVIOURALLY_CONSISTENT` |
| 4 | p ≥ p_high (residual) | SOFT_BLOCK | `HIGH_RISK_DEFAULT` |
| 5 | p_low ≤ p < p_high, consistency high, anomaly and uncertainty low | AUTO_CLEAR | `MEDIUM_RISK_BEHAVIOURALLY_CONSISTENT` |
| 6 | p_low ≤ p < p_high, otherwise | STEP_UP_REVIEW | `MEDIUM_RISK_REVIEW` |
| 7 | p < p_low *but* anomaly ≥ 0.60 and uncertainty low | STEP_UP_REVIEW | `LOW_RISK_BUT_ANOMALOUS` |
| 8 | p < p_low, no strong anomaly | AUTO_CLEAR | `LOW_RISK_CONSISTENT` |

Rule 3 is the one that carries the thesis: *high model risk plus behaviour that matches the customer
produces a reversible hold, not a blind hard block.* Rule 7 is its mirror — low model risk plus
strongly anomalous behaviour still earns a human look, to catch model false negatives.

### Where p_low and p_high come from

Not hand-set. They are **derived per dataset** from the calibrated-probability distribution on the
validation split — the 78th and 96th percentiles — and written to `policy_thresholds_<ds>.json`.
"High risk" therefore means "in the top few percent of calibrated risk for this data", which stays
meaningful across datasets with different prevalence and different model strength. A static
fallback exists in config if the derived file is missing.

### Capacity awareness

`apply_capacity()` models a real constraint: a fraud team has a finite number of reviews per day.
With `analyst_capacity_per_1k = 120` and a 10% headroom reserve, the projected review budget for a
batch is computed, and if `STEP_UP_REVIEW` volume exceeds it the **lowest-marginal-value** items —
ranked by calibrated probability, then behavioural anomaly — are downgraded to `AUTO_CLEAR` and
tagged `capacity_concession`. `SOFT_BLOCK` and `ESCALATE` are never downgraded.

> **This matters for reading the numbers.** Capacity concessions are counted as missed detections in
> the evaluation, because in that operating scenario they are. On the synthetic test split the model
> routes 18 of 23 fraud cases to review, but capacity throttling means some of those are conceded —
> which is exactly why the Policy Lab exposes capacity as a slider rather than burying it as a
> constant.

### Cost model

| Parameter | Default | Represents |
|---|---:|---|
| `fraud_loss_cost` | 100.0 | missed fraud that reaches completion |
| `false_positive_cost` | 8.0 | a legitimate transaction hard-blocked — support contact, goodwill, churn risk |
| `customer_friction_cost` | 2.0 | a step-up challenge shown to a legitimate customer |
| `review_cost` | 1.0 | one analyst manual review — the unit |

Units are relative cost points, not currency. The Policy Lab lets you move all four and recomputes
expected cost across the capacity sweep from the stored confusion counts, without re-scoring.

---

## 09 — Explanation layer

Two explanations, kept deliberately distinct: why the *model* scored it that way, and why the
*policy* acted that way.

### Model-level — SHAP

`shap.TreeExplainer` with `tree_path_dependent` perturbation, so no background dataset is needed and
results are deterministic and fast. Returns the top 5 signed contributions in XGBoost margin space;
positive pushes toward fraud, negative away. `shap_top_features_batch()` computes the whole matrix
in one call for the batch scorer. `evidence_split()` then partitions those features into
**supporting** and **counter** evidence in plain language.

### Reason codes — deterministic rules

Not a model output. A small ordered rule set over raw fields and scores:

| Code | Fires when |
|---|---|
| `MODEL_DISAGREEMENT_REVIEW` | uncertainty ≥ the configured threshold |
| `ATO_NEW_DEVICE_HIGH_VELOCITY` | new device and ≥4 transactions in the last hour |
| `UNUSUAL_AMOUNT_NEW_BENEFICIARY` | new or minutes-old beneficiary and amount ≥2× the customer's usual |
| `MULE_NETWORK_PROXIMITY` | shared device with a flagged account, or ≤2 hops to a known mule |
| `HIGH_RISK_BUT_BEHAVIORALLY_CONSISTENT` | p ≥ p_high but consistency ≥ 0.70 |
| `KNOWN_DEVICE_NORMAL_BEHAVIOR` | p < p_low and consistency ≥ 0.70 |
| `REVIEW_MIXED_SIGNALS` | fallback |

### Next best evidence — `next_best_evidence.py`

For any case routed to a human, a deterministic ordered checklist of what to pull next, keyed off
the transaction's own fields: beneficiary 90-day history, device fingerprint reuse, the auth
timeline after a password reset, the last 1h/24h transactions, the trailing amount distribution,
shared-device neighbours, side-by-side XGBoost vs Random-Forest attributions, and prior analyst
decisions on this customer. Each item carries a rationale and a named source system.

> **Labelled honestly.** Fixed templates over the transaction's own fields. No LLM, no agent, no
> ranking model trained on historical analyst resolutions. The module docstring and the dashboard
> both say so.

### Graph proxy — `graph_proxy.py`

Builds a small node/edge structure per transaction from two synthetic columns —
`shared_device_with_flagged_account` and `hop_distance_to_known_mule` — so the investigation view
can draw a mule-network-style picture: the account, its device, its beneficiary, any flagged account
sharing that device, and the hop chain to a known mule. There is no graph database, no GraphSAGE,
no message passing. Every response carries `illustrative: true` and a note saying exactly that.

---

## 10 — Feedback & self-audit

### SQLite schema — `db.py`

| Table | Holds |
|---|---|
| `transactions` | one row per scored transaction — all six scores, decision, firing rule, typology, reason code, capacity-concession flag, ground-truth label, split, model and policy version, plus the full raw feature dict and score object as JSON |
| `feedback` | analyst verdict (`CONFIRM_FRAUD` / `MARK_LEGITIMATE` / `NEED_MORE_EVIDENCE`), reason tag, free note, and a snapshot of the original model output and policy decision at the time |
| `audit_runs` | one row per self-audit pass, with the summary JSON |
| `audit_samples` | the sampled transactions — population, sample type, calibrated probability, decision, proxy outcome label |

Storing the original model output alongside the verdict is the point: that pairing is the dataset a
future recalibration would train on. The MVP captures it; it does not yet learn from it.

### Self-audit — `audit.py`

Fraud systems have two blind spots, and conventional evaluation looks at neither. SafeTriage samples
both, **drawing only from the sealed test split**:

- **AUTO_CLEAR population** — hidden **false negatives**, fraud that was waved through. Sampled 90
  risk-stratified (quantile bins of calibrated probability, so both low- and high-risk clears are
  represented) plus 45 random control.
- **BLOCK / ESCALATE population** — hidden **false positives**, legitimate customers who were
  actioned. Sampled 60 risk-stratified plus 30 random control, or the whole population if it is
  smaller.

Because there are no verified human re-review labels, the outcome uses the dataset's own `isFraud`
as an **audit proxy**, tagged `is_proxy: 1` on every row. Every rate is computed from the sample —
nothing is hardcoded. This surfaces uncomfortable numbers rather than hiding them: on IEEE, the
risk-stratified block sample has consistently shown a double-digit proxy false-positive rate, which
is the finding, not a bug.

---

## 11 — The API

FastAPI, 10 routes, everything read from model artifacts and SQLite. No external services, no API
keys.

| Route | Parameters | Returns |
|---|---|---|
| `POST /score` | `transaction_id` or a full `transaction` payload, `dataset`, `persist` | the complete scoring object (below) |
| `GET /transactions` | `dataset`, `decision`, `split`, `limit` (default 100, max 2000), `offset`, `order_by`, `desc` | paginated scored rows, JSON fields hydrated |
| `GET /transactions/{id}` | — | one hydrated row |
| `GET /kpis` | `dataset` | volumes, decision distribution, the four rates, capacity concessions, score means, audit proxies, model and policy version |
| `GET /audit/sample` | `dataset` | latest audit run — summary plus every sampled row |
| `GET /feedback` | `limit` | stored analyst decisions |
| `POST /feedback` | `transaction_id`, `analyst_decision`, `reason_tag`, `note` | the stored row, echoed with its id |
| `GET /evaluation` | `dataset` or `all` | baseline vs SafeTriage, deltas, cost breakdown, notes, three demo cases |
| `GET /calibration` | `dataset` or `all` | reliability curves (raw + calibrated), Brier, ECE, plus the full model metrics |
| `GET /policy/simulate` | `dataset`, and any of the four cost parameters | the capacity sweep; supplying a cost recomputes expected cost from stored confusion counts |

### The `/score` contract

Twenty-two fields spanning all three layers. This is what one transaction looks like coming out of
the whole pipeline:

```jsonc
{
  "transaction_id", "dataset",

  // prediction
  "fraud_probability",          // raw XGBoost
  "calibrated_probability",     // what every decision actually uses

  // interpretation
  "behavioral_consistency_score", "behavioral_anomaly_score",
  "behavioral_components",        // the six sub-scores
  "is_behavioral_proxy",          // true on IEEE (reduced mode)
  "uncertainty", "confidence_score",

  // decision
  "decision",                   // one of the four actions
  "policy_rule",                // which rule fired
  "decision_explanation",       // plain-language, with the numbers in it

  // explanation
  "predicted_typology", "fraud_reason_code",
  "top_features",               // SHAP: feature, value, signed contribution
  "supporting_evidence", "counter_evidence",
  "next_best_evidence",         // analyst checklist
  "graph_proxy",                // nodes + edges (synthetic only)

  // provenance
  "model_version",              // content hash of xgb + calibrator + rf + encoders
  "policy_version",             // config name + content hash
  "latency_ms"
}
```

`model_version` and `policy_version` are content hashes, so any retrain or config edit changes them
visibly in the dashboard header — you can always tell which model produced a given score.

### Which model serves

Two sets of weights exist. The **evaluated** models (fit on train) serve by default, so
per-transaction scores stay consistent with `evaluation_<ds>.json`. The **full-refit** models (fit
on 100% of rows with the same tuned hyper-parameters) are written for completeness and served only
if `serving.use_full_refit` is set true — at which point the dashboard's numbers would no longer
match the held-out evaluation, which is why it defaults to false.

---

## 12 — The dashboard

React 18 + Vite 6 + Tailwind 4 + Recharts. Ten pages, one file each. Every number on screen comes
from a live API call — there is no fixture data anywhere in the frontend.

### How rendering works

The mechanism is deliberately plain. `main.jsx` mounts `App.jsx` inside a `BrowserRouter`.
`App.jsx` holds one piece of global state — the selected dataset — in a React context, renders the
sidebar, the header (with a live API-status indicator that pings `/kpis` every 15 seconds), and a
`<Routes>` block mapping each path to a page component.

Each page calls `useApi(path, deps)` from `lib/api.js`. That hook holds `{data, error, loading}`,
fires `fetch("/api" + path)` on mount and whenever its dependencies change, and guards against
out-of-order responses with a sequence counter. Pages render one of three states explicitly —
`<Loading />`, `<ErrorState />` (which tells you the API may not be running and how to start it), or
the real content. There is no placeholder branch.

The `/api` prefix is a proxy, declared in `vite.config.js` for **both** the dev server and
`vite preview`, rewriting to `http://127.0.0.1:8000`. Both are needed — a built app served by
`vite preview` without the proxy would render the shell with every data call failing, which is
precisely the "looks mocked" failure mode.

### What each page renders

| Page | Route | Calls | Renders |
|---|---|---|---|
| **Overview** | `/` | `/kpis`, `/audit/sample` | four KPI tiles (volume, auto-clear rate, review rate, hard-action rate), a decision-distribution bar chart coloured by the risk ramp, the two audit-proxy rates, and score means including capacity concessions |
| **Live Transaction Feed** | `/feed` | `/transactions?limit=400` | a sortable table — id, split, calibrated probability, consistency, disagreement, decision chip, typology, reason code, ground truth. Sorting re-fetches with a new `order_by`. A "Play next 40" button reveals rows in batches client-side; a decision filter re-queries. Clicking a row routes to Investigation |
| **Investigation** | `/investigation/:id` | `/transactions/{id}`, `POST /score` | the deepest page: four score tiles, the policy decision with its firing rule and explanation, a horizontal diverging SHAP bar chart, the six behavioural components as bars, supporting vs counter evidence, the next-best-evidence checklist, an inline SVG of the graph proxy, the full raw record, and three feedback buttons that POST and confirm inline |
| **False Positive Lab** | `/fp-lab` | `/evaluation` | the headline comparison — four delta tiles, a grouped bar chart of recall / precision / FP rate / block rate, the full metric table, the evaluation's own caveats, and the three demo cases with their real scores and top SHAP features |
| **Review Queue** | `/review-queue` | `/transactions?decision=STEP_UP_REVIEW`, `POST /score` on expand | the analyst work queue, highest calibrated risk first. Expanding a row scores it live and shows the decision explanation plus the evidence checklist, with the three feedback actions and a link through to full investigation |
| **Self-Audit** | `/self-audit` | `/audit/sample` | a proxy-labelling banner, the two blind-spot populations side by side with stratified vs control rates and population sizes, and the full sampled-transaction table |
| **Calibration + Uncertainty** | `/calibration` | `/calibration` | Brier and ECE before/after as tiles, the reliability curve plotting raw and calibrated against the diagonal, cross-validated discrimination for both models, and the disagreement-gap distribution with its "this is a proxy" warning |
| **Policy Lab** | `/policy-lab` | `/policy/simulate` | four cost sliders that re-query with overrides, a capacity slider that selects a point on the sweep, six tiles for the selected operating point, and a four-line chart of recall / FP rate / review rate / normalised cost against capacity with a marker on the selection |
| **Fraud Detection Cycle** | `/cycle` | `/kpis`, `/audit/sample`, `/feedback` | the ten-step closed loop as a numbered timeline, with live counts injected into the steps that have them — transactions scored, mean calibrated probability, decision mix, concessions, feedback rows, audit population sizes |
| **Architecture** | `/architecture` | `/kpis` | the five-stage pipeline strip and three columns — LIVE, SIMULATED, FUTURE — plus the active model and policy version hashes |

### Shared primitives — `components/ui.jsx`

`SectionCard`, `StatTile`, `DecisionBadge`, `DataTable`, `KeyValue`, `Banner`, `Loading`,
`ErrorState`. Decision colours live in one place — `lib/constants.js → DECISION_STYLE` — mapping
each of the four actions to a chip class and a chart fill, so a decision looks identical in a table
cell, a chart bar and a page header.

---

## 13 — Data-flow traces

### Trace A — the Live Feed loads 400 rows

| # | Where | What |
|---:|---|---|
| 01 | `pages/LiveFeed.jsx : 17` | Builds the query string. The row cap is one literal here — `&limit=400` — alongside `dataset`, `order_by`, `desc` and an optional `decision` filter. |
| 02 | `lib/api.js → useApi → apiGet` | `fetch("/api" + path)`, with a sequence counter discarding responses that arrive out of order after a re-sort. |
| 03 | `frontend/vite.config.js` | The dev/preview server strips `/api` and forwards to `http://127.0.0.1:8000/transactions?…`. |
| 04 | `api/main.py : 120` | `get_transactions()` validates — `limit` defaults to 100 and is capped at 2000 by `Query(100, le=2000)`; the page's 400 passes. |
| 05 | `safetriage/db.py : 128` | `fetch_transactions()` builds `SELECT * FROM transactions … ORDER BY <col> DESC LIMIT ? OFFSET ?`, with `order_by` checked against an allow-list to prevent injection. |
| 06 | `data/safetriage.db` | SQLite returns 400 of the 3,000 rows held for that dataset. |
| 07 | `api/main.py → _hydrate()` | Parses the stored `raw_json` and `output_json` back into nested objects before serialising the response. |
| 08 | `pages/LiveFeed.jsx : 24` | `items.slice(0, shown)` — the 400 are then paginated client-side in batches of 40, which is what the replay button steps through. |

> **Why 400 and not all 3,000.** Every row carries its full raw feature dict and score object as
> JSON, so 3,000 of them is a multi-megabyte response. The feed is for sorting and skimming, not
> bulk export — and because re-sorting re-queries the server, you always get the top 400 *for that
> ordering*. Aggregates are not capped: `/kpis` reads with `limit=100000`, so the Overview counts
> cover every stored row.

### Trace B — one transaction through `POST /score`

| # | Where | What |
|---:|---|---|
| 01 | `api/main.py : 78` | If given a `transaction_id`, loads the stored raw record from SQLite; if given a full payload, uses it directly. |
| 02 | `features.transform_for_model()` | Loads `fe_<ds>.joblib`, applies the train-fit encoders, engineers the derived columns, and reindexes onto the frozen column layout — missing columns filled with zero, extra columns dropped. |
| 03 | `scoring._models()` | Cached load of XGBoost, the calibrator and the Random Forest. Raw probability, calibrated probability and the RF probability are computed. |
| 04 | `uncertainty.disagreement_uncertainty()` | Normalises the gap between the calibrated and RF probabilities against the stored validation distribution. |
| 05 | `behavior.behavioural_frame()` | Six bounded components computed from raw observables, weighted into a consistency score — with no access to the fraud probability. |
| 06 | `policy.decide()` | Loads thresholds (derived per dataset), walks the priority-ordered rule table, returns decision, firing rule and a plain-language explanation with the actual numbers interpolated into it. |
| 07 | `explain` · `next_best_evidence` · `graph_proxy` | SHAP top-5, evidence split, reason code, typology guess, analyst checklist, and the illustrative network. |
| 08 | `scoring.score_transaction()` returns | The 22-field object, with `latency_ms` measured across the whole call — typically a few hundred milliseconds, dominated by SHAP. |

### User flow — the intended session

1. **Overview** — read the shape of the traffic: how much was auto-cleared, how much needs a human,
   what the audit says about both blind spots.
2. **Review Queue** — work the actual queue, highest risk first. Expand a case, read why the policy
   routed it, see what evidence to pull, record a verdict.
3. **Investigation** — for anything ambiguous, open the full view: SHAP, behavioural breakdown,
   network proximity, raw record, then decide.
4. **False Positive Lab** — step back and check whether the whole arrangement is actually beating a
   plain threshold, on data the models never saw.
5. **Policy Lab** — if the trade-off is wrong, move capacity and the four cost parameters and watch
   recall, false positives and expected cost move together.
6. **Self-Audit** — periodically check what the system is getting wrong in the two places normal
   monitoring cannot see.

---

## 14 — Measured results

Held-out test split, seed 42. The baseline threshold is chosen on validation to hit the target
recall, then frozen and applied to test. Every number here is computed by `evaluate.py`, not
asserted.

### IEEE-CIS — 118,108 test transactions, 4,132 fraud

| | Baseline → SafeTriage | Change |
|---|---|---|
| False positives | 3,565 → 446 | **87.5% fewer** |
| Recall | 0.796 → 0.859 | **+6.3 pts** |
| Precision | 0.480 → 0.888 | **+41 pts** |
| Expected cost | 112,820 → 89,580 | **−21%** |

On the real benchmark SafeTriage is strictly better on both axes at once: it catches *more* fraud
than the threshold baseline while blocking 87.5% fewer legitimate customers. The cost is 86.3
analyst reviews per 1,000 transactions where the baseline had none.

### Synthetic — 600 test transactions, 23 fraud

| | Baseline → SafeTriage | Change |
|---|---|---|
| False positives | 128 → 1 | **99.2% fewer** |
| Recall | 0.739 → 0.609 | −13.0 pts |
| Precision | 0.117 → 0.933 | **+82 pts** |
| Expected cost | 1,624 → 1,072 | **−34%** |

On the deliberately non-separable dataset the trade-off is real and visible: 13 recall points given
up for a 99.2% cut in false positives and a third off expected cost. With only 23 fraud rows in the
test split, a single case is worth 4.3 recall points — these figures carry wide error bars and the
document says so rather than rounding them into confidence.

### Full comparison

| Metric | IEEE baseline | IEEE SafeTriage | Synth baseline | Synth SafeTriage |
|---|---:|---:|---:|---:|
| Precision | 0.480 | 0.888 | 0.117 | 0.933 |
| Recall | 0.796 | 0.859 | 0.739 | 0.609 |
| False positive rate | 3.13% | 0.39% | 22.18% | 0.17% |
| False discovery rate | 52.0% | 11.2% | 88.3% | 6.7% |
| True positives | 3,289 | 3,549 | 17 | 14 |
| False positives | 3,565 | 446 | 128 | 1 |
| Missed fraud | 843 | 583 | 6 | 9 |
| Hard-block rate | 5.8% | 2.2% | 24.2% | 0.2% |
| Review rate | 0.0% | 8.6% | 0.0% | 10.7% |
| Auto-clear rate | 94.2% | 89.2% | 75.8% | 89.2% |
| Analyst workload / 1k | 0.0 | 86.3 | 0.0 | 106.7 |
| Fraud caught / 1k reviewed | n/a | 140.2 | n/a | 218.8 |

### How the comparison is defined

- **Baseline** — probability threshold only. At or above the threshold is a hard block; below it is
  cleared. No reviews exist.
- **SafeTriage** — `ESCALATE` and `SOFT_BLOCK` count as actioned as fraud. `STEP_UP_REVIEW` is
  resolved using the ground-truth label as a proxy for the analyst working the queue. `AUTO_CLEAR`
  is cleared.
- A legitimate transaction routed to `STEP_UP_REVIEW` is **not** counted as a false positive — it
  was not blocked. It is counted as analyst workload and customer friction, both reported
  separately. That reclassification *is* the thesis, so it is stated explicitly in the evaluation
  output rather than buried.

> **Two things that flatter this comparison.**
> **The analyst is modelled as perfect** — review outcomes resolve by ground truth; a real queue has
> error, latency and inconsistency, so "fraud caught in review" is an upper bound.
> **The baseline is deliberately naive** — a threshold picked for a fixed recall target is the
> textbook comparison, and on a weak model it blocks an unrealistic share of traffic, which is the
> motivation for the project, but also means a stronger baseline would narrow the gap.

---

## 15 — Running it

### One-time build

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --reset --force --epochs 20
```

Wipes all prior state, rebuilds both datasets, splits, feature encoders, tuned models, calibrators,
evaluation, capacity sweep, the database and both audit runs. Expect 30–40 minutes, almost all of it
in the IEEE hyper-parameter search. It ends with a `PIPELINE COMPLETE` block reporting stored
transaction counts and audit population sizes.

### Terminal 1 — the API

```bash
python -m uvicorn api.main:app --reload
# interactive docs at http://127.0.0.1:8000/docs
```

### Terminal 2 — the dashboard

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

### Verify

```bash
python scripts/smoke_api.py   # expects: ALL ENDPOINT CHECKS PASSED
```

This asserts every route returns real data — that the decision mix sums to one, that raw and
calibrated probabilities differ, that SHAP is populated, that both audit populations exist, that a
feedback POST round-trips, and that a cost override actually moves the simulation.

### If something goes wrong

| Symptom | Cause and fix |
|---|---|
| Dashboard renders but every panel shows an error | The API isn't running, or you used `npm run preview` on a build predating the preview proxy. Start uvicorn; the header dot turns green within 15 seconds. |
| `404 … run scripts/run_pipeline.py first` | Artifacts or the database are missing. Run the pipeline. |
| `ModuleNotFoundError: config` | A script was run from inside `scripts/`. Run from the project root; the scripts insert the root on `sys.path` themselves. |
| `overfit_warning: true` in model metrics | The train-minus-CV gap exceeded 0.10. Tighten the relevant regime in `tuning.py` — lower `max_depth`, raise the `min_child_weight` floor — and retrain that dataset. |
| Optuna not installed | Nothing breaks. `tuning.py` falls back to `HalvingRandomSearchCV` automatically and records which method it used. |
| Port 8000 already in use | An earlier uvicorn is still running. Stop it, or pass `--port 8001` and set `SAFETRIAGE_API` before starting Vite. |
| `UnicodeEncodeError` on Windows | The console codepage can't render a character. Prefix with `PYTHONIOENCODING=utf-8`. |

---

## 16 — Real / simulated / future

The distinction that keeps the prototype honest about what it has actually demonstrated.

### LIVE — really runs, really computed

dataset processing · XGBoost · Random Forest · feature engineering (train-fit) · Optuna
hyper-parameter search · probability calibration · Brier + ECE · overfit guard · behavioural
consistency · disagreement proxy · policy engine · capacity + cost model · SHAP · reason codes ·
FastAPI · SQLite · feedback capture · self-audit sampling · held-out evaluation · capacity
simulation

### SIMULATED — works, but stands in for a real system

transaction replay (static file, not a stream) · graph view (two columns, no graph DB) ·
next-best-evidence (fixed templates) · analyst explanation text (templates, no LLM) · analyst
resolution in evaluation (ground-truth proxy) · audit outcomes (proxy labels, no verified re-review)

### FUTURE — not built

Kafka / Redpanda · Redis feature store · graph database + GraphSAGE · LLM analyst assistant ·
feedback-driven recalibration · SAP HANA / AI Core / XSUAA · drift monitoring · verified audit
labels · privacy & security hardening

### What this prototype does not prove

- **IEEE-CIS lacks the features SafeTriage is built around.** On the only real dataset here, the
  interpretation layer runs in reduced mode on flagged proxies — a weaker version of the system, not
  the intended one.
- **The synthetic dataset is an instrument, not evidence.** Its typologies, prevalence and the exact
  amount of class overlap are authored. It makes the trade-off visible and measurable; it says
  nothing about real traffic.
- **Small positive class.** 23 fraud cases in the synthetic test split means every rate there is
  noisy.
- **Model disagreement is not uncertainty.** It marks where two models diverge, not where the
  prediction is genuinely unreliable.
- **Thresholds are derived, not optimised.** Probability cut-points come from validation
  percentiles; the consistency, anomaly, uncertainty and cost parameters are reasoned defaults, not
  fitted against an objective.
- **Feedback is stored, not learned from.** The recalibration loop is architected and captured; it
  is not run.

### The contribution being claimed

Not SHAP, not calibration, not XGBoost, not stratified sampling — all of those are standard, and the
README says so explicitly. The claim is about **integration**: that wiring these components around a
false-positive-aware decision problem, with the model's probability demoted to one input among four,
produces a measurable improvement on the axes that actually cost an institution money — and that the
improvement can be demonstrated end-to-end on a held-out split, with the trade-off exposed as a dial
rather than hidden in a threshold.

---

*SafeTriage · BTP capstone prototype · seed 42 · all figures computed on the held-out test split ·
runs fully locally, no external services or API keys*

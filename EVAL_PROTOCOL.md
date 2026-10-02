# SafeTriage — Evaluation Protocol (Phase 0)

This file is the contract every other phase follows. If a later script violates
something here, that script is wrong.

## Random seed

* Single source of truth: `config/seed.py` → `SEED = 42`.
* Every stochastic step imports it: synthetic data generation, the
  train/val/test split, XGBoost / Random-Forest training, the calibration-fit
  sub-split, cross-validation folds, and audit sampling.
* `set_all_seeds()` seeds Python `random`, NumPy, and `PYTHONHASHSEED`.

## Train / validation / test split

* Ratio **60 / 20 / 20**, **stratified by `isFraud`**, seeded.
* Implemented in `safetriage/data/splits.py`; index files are written once to
  `data/splits/<dataset>_{train,val,test}.idx` and reused.
* The IEEE-CIS Kaggle `test_*.csv` files have **no label column**, so they cannot
  be used for evaluation. The held-out test split is carved out of
  `train_transaction.csv`.

### What each split may be used for

| Split | Allowed uses |
|-------|--------------|
| **train** | **feature-engineering fit** (frequency encodings, per-key aggregates), the **hyper-parameter search** (k-fold CV *inside* the train split), fitting XGBoost + Random Forest (on a 65% sub-slice), fitting the probability calibrator (on the remaining 35% sub-slice) |
| **validation** | XGBoost early stopping, calibration-method selection, the **baseline probability threshold**, the derived policy probability cut-points, the disagreement-proxy normaliser, the capacity simulation |
| **test** | **only** Baseline vs SafeTriage evaluation (Phase 9) and self-audit sampling (Phase 6) |

**Feature engineering** (`safetriage/features.py`): all learned encoders are fit on
`df.iloc[train_idx]` only and applied to val/test/inference; unseen keys fall back
to a global default. Fitted stats are persisted to `fe_<dataset>.joblib`.

**Hyper-parameter search** (`safetriage/tuning.py`): Optuna TPE + median pruner
(or `HalvingRandomSearchCV` if Optuna is absent), objective = CV mean PR-AUC,
run on the **train split only** (large datasets on a seeded stratified subsample,
all fraud rows kept). Best params are written to `tuning_<dataset>.json`.

**"Whole dataset used for training"**: after the held-out evaluation is done, a
second artifact `xgb_<dataset>_full.joblib` is refit on 100% of the rows with the
same tuned hyper-parameters. It is **serving-only and never evaluated** — the API
still serves the train-fit model by default (`serving.use_full_refit: false`) so
the dashboard stays consistent with `evaluation_<dataset>.json`.

**Overfit guard**: `model_metrics_<dataset>.json → overfit_check` reports
`train_roc_auc`, `cv_roc_auc` and the gap; `overfit_warning: true` when the gap
exceeds `training.overfit_gap_warn` (0.10).

Enforcement:
* `splits.load_trainable_indices()` raises if asked for `"test"`.
* `splits.load_test_indices()` is called only by `safetriage/evaluate.py` and
  `safetriage/audit.py`.
* `splits.assert_no_test_leakage()` is available as a defence-in-depth check.
* The calibrator is fit on a sub-slice of **train**, never on validation or test,
  so the reported calibration curve (on validation) is on unseen data.

## Dependencies

Pinned exactly in `requirements.txt` (verified to import on Python 3.13.7,
Windows). Key versions: numpy 2.2.5, pandas 2.2.3, scikit-learn 1.6.1,
xgboost 3.4.1, shap 0.52.0, fastapi 0.139.2, matplotlib 3.11.1, pyarrow 18.1.0.

## IEEE-CIS access

The IEEE-CIS Fraud Detection dataset **is present** under
`ieee-fraud-detection/` (`train_transaction.csv` 590,540 rows,
`train_identity.csv` 144,233 rows). The Phase-0 fallback (proceed synthetic-only
if the data cannot be obtained) is therefore **not triggered**. Both datasets are
used; see `DATASET_README.md`.

## Definition of Done (MVP acceptance)

- [x] All 9 API endpoints return real computed data (no stubs) — `scripts/smoke_api.py` passes
- [x] All 10 frontend pages render with real backend data, no placeholder content — verified live
- [x] Baseline vs SafeTriage metrics computed from the test set, not asserted (`safetriage/evaluate.py`)
- [x] Calibration curve and Brier score are real, computed values (`calibration_<ds>.json` + PNG)
- [x] At least one self-audit pass has run on both AUTO_CLEAR and BLOCK/ESCALATE populations (both datasets)
- [x] Feedback loop stores analyst decisions end-to-end (seeded + UI-submitted rows in SQLite)
- [x] README clearly separates LIVE / SIMULATED / FUTURE
- [x] Fixed seed and train/val/test split documented and enforced (`config/seed.py`, `splits.py`)

## Reproduce

```bash
pip install -r requirements.txt
pip install optuna==4.1.0                       # optional, faster tuning

python scripts/reset_state.py                   # wipe all previous weights + splits + data
python scripts/build_splits.py                  # step 1: fresh 60/20/20 splits (IEEE first)
python -m safetriage.models.train --epochs 20   # step 2: FE + 20-trial tuning + train + calibrate
python -m safetriage.evaluate                   # step 3: held-out test evaluation

# …or all of the above in one go:
python scripts/run_pipeline.py --reset --force --epochs 20
python -m uvicorn api.main:app --reload
```

Re-running with the same seed reproduces identical split index files. Tuning is
seeded too (TPESampler / HalvingRandomSearchCV `random_state=SEED`), so best
params and metrics are reproducible on the same machine.

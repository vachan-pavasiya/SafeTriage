# SafeTriage — Build Log

Per-phase status. "Simplified" = deliberately scoped down for the MVP;
"Assumption" = a choice made where the spec left room.

## Phase 0 — Environment & Evaluation Protocol — DONE
- `config/seed.py` (SEED=42), `requirements.txt` (exact pins, verified on Python 3.13.7).
- `safetriage/data/splits.py`: 60/20/20 stratified, seeded, persisted to `data/splits/`.
  Test split guarded (`load_trainable_indices` refuses `"test"`; only `evaluate.py`
  / `audit.py` call `load_test_indices`).
- `EVAL_PROTOCOL.md` written incl. Definition of Done.
- **Assumption:** calibrator is fit on a 35% sub-slice of *train* (not validation),
  so the validation calibration curve is on unseen data.

## Phase 1 — Data — DONE
- Dataset A: `preprocess_ieee.py` → `data/processed_ieee.parquet` (590,540 rows,
  224 features, 3.5% fraud). IEEE-CIS present locally → no fallback.
- Dataset B: `generate_synthetic_dataset.py` → `data/synthetic.csv` (3,000 rows,
  3.9% fraud, 5 typologies). Built-in separability check: max single-feature
  ROC-AUC ≈ 0.70.
- **Assumption:** the four `*_consistency` columns + `location_consistency` are
  withheld from the classifier (`RESERVED_FOR_POLICY`) so the baseline can't cheat.
- **Simplified:** IEEE has no behavioural fields; `C*`/`D*` used as flagged proxies.

## Phase 2b — Feature engineering + hyper-parameter tuning — DONE (upgrade)
- `safetriage/features.py` rewritten as the FE hub. Encoders fit on **train only**,
  persisted to `fe_<ds>.joblib`; final column layout in `feature_spec_<ds>.json`.
  - synthetic: +~22 engineered cols (amount ratios/logs, circular hour, velocity
    ratios, interactions, train-fit `card_id` frequency).
  - IEEE: +~26 engineered cols (`amt_cents`/round-amount, `TransactionDT` hour/dow
    cyclical, C/D aggregates, and train-fit per-`card1`/`card2`/`addr1`/email
    aggregates → count-enc, amount-vs-key-mean, key z-score). `preprocess_ieee.py`
    now retains raw keys `k_card1…k_remail` (not model features) for this.
- `safetriage/tuning.py` (new): Optuna TPE + MedianPruner, else
  `HalvingRandomSearchCV`. Objective = CV mean PR-AUC. Runs on the **train split
  only**; large datasets tuned on a 120k stratified subsample (all fraud kept).
  Output → `tuning_<ds>.json`.
- `train.py` reworked: IEEE first; split reused from `build_splits.py`; tuned
  XGBoost refit with early stopping; **overfit guard** (`train_roc_auc - cv_roc_auc`
  gap, warns > 0.10) in `model_metrics_<ds>.json → overfit_check`; **full-refit**
  `xgb_<ds>_full` / `rf_<ds>_full` on 100% of rows (serving-only, never evaluated).
- New scripts: `scripts/reset_state.py` (wipe all derived state),
  `scripts/build_splits.py` (step-1 split). `run_pipeline.py` gains
  `--reset --epochs --cv-folds`.
- Config: `training:` (hpo_trials 20, cv_folds 3, tune_subsample_cap, overfit
  guard) and `serving.use_full_refit` (default false) in `policy_config.yaml`.
- `requirements.txt`: `optuna==4.1.0` (optional; graceful fallback).

### Phase 2b follow-up — overfit fix + verification (this session)
- The first 2b run flagged `overfit_warning: true` on synthetic
  (train ROC-AUC 0.997 vs CV 0.863, gap 0.133 > 0.10). Fixes:
  - `tuning.py` now has a **size-aware search space**: datasets with < 20k rows
    or < 400 fraud tune in a *small* regime (`max_depth` 3–5, `min_child_weight`
    floor 4–25, `reg_lambda` 0.5–20, `gamma` ≥ 0.05). IEEE keeps the large space.
  - `train.py` step 4: the small regime no longer early-stops the final fit on the
    tiny (~24-fraud) validation split — it trusts the CV-selected `n_estimators`
    (early-stop kept only as a runaway guard). Fixes a 16-round under-fit.
  - Result: synthetic train-vs-CV gap **0.133 → 0.035**, CV ROC-AUC **0.876**,
    PR-AUC 0.563; IEEE unchanged (gap 0.043, CV ROC-AUC 0.957, PR-AUC 0.926).
  - `analyst_capacity_per_1k` default **80 → 120** (sized to the review demand the
    tuned models actually produce; the Policy Lab still sweeps 10–200).
- Frontend: `vite.config.js` gains `preview.proxy` (the built app served by
  `vite preview` had no `/api` proxy — likely why it "felt disconnected"). Header
  gains a live **API connected / offline** status dot (`App.jsx`).
- Verified end-to-end: `scripts/smoke_api.py` → ALL ENDPOINT CHECKS PASSED;
  browser network log shows real `GET /api/*` → 200 on every page.

### Held-out test evaluation after the fix
| | IEEE-CIS (reduced) | Synthetic |
|---|---|---|
| CV ROC-AUC / PR-AUC | 0.957 / 0.926 | 0.876 / 0.563 |
| overfit gap (warn > 0.10) | +0.043 | +0.035 |
| baseline recall / FP | 0.796 / 3565 | 0.739 / 128 |
| SafeTriage recall / FP | 0.859 / 446 | 0.609 / 1 |
| false-positive reduction | 87.5% | 99.2% |
| expected cost | 112,820 → 89,580 | 1,624 → 1,072 |

## Phase 2 — Models + Calibration — DONE (baseline, now superseded by 2b)
- XGBoost primary, RandomForest secondary (disagreement proxy only).
- Calibration: sigmoid vs isotonic; sigmoid preferred, isotonic only allowed on
  the large dataset when it cuts holdout Brier ≥10% without hurting AUC.
- Reports Brier + ECE + reliability curve (validation) + PNG, per dataset.
- **Assumption:** synthetic XGBoost uses `scale_pos_weight=1` (no reweighting) —
  reweighting forced steppy isotonic fits on ~20 holdout positives; IEEE uses
  sqrt-damped weight (plenty of positives, improves ECE).
- **Assumption:** headline discrimination is reported as 5-fold CV over train+val
  (single 20% val split has too few positives to be stable); IEEE CV is a seeded
  60k stratified subsample (all fraud kept).

## Phase 3 — Behavioural consistency + uncertainty — DONE
- `behavior.py`: six documented bounded components → `behavioural_consistency_score`
  (weights from `policy_config.yaml`). `consistency_ieee_proxy()` for reduced mode.
- `uncertainty.py`: normalised |p_xgb_cal − p_rf|, always labelled a
  "model-disagreement proxy", normaliser from the validation gap distribution.

## Phase 4 — Policy engine + cost/capacity — DONE
- `policy.py`: 4 actions, priority-ordered rule table, all thresholds from config.
  Probability cut-points **derived per dataset** from validation calibrated-prob
  percentiles (written to `policy_thresholds_<ds>.json`).
- `apply_capacity()`: downgrades lowest-value reviews to AUTO_CLEAR past budget.
- `expected_cost()`; `scripts/policy_simulation.py` sweeps capacity → JSON + PNG.

## Phase 5 — Explainability + graph + next-best-evidence — DONE
- `explain.py`: SHAP TreeExplainer (batched), top-5 features, deterministic reason
  codes, supporting/counter evidence.
- `graph_proxy.py`: illustrative node/edge structure (clearly labelled — no graph DB).
- `next_best_evidence.py`: deterministic template checklist (labelled — no LLM).

## Phase 6 — Self-audit + feedback — DONE
- `db.py`: SQLite schema (transactions, feedback, audit_runs, audit_samples).
- `audit.py`: samples AUTO_CLEAR + BLOCK_ESCALATE populations from the **test
  split only**, risk-stratified + random control, outcome = `isFraud` used as an
  **audit proxy** (`is_proxy=1`), rates computed not hardcoded.
- **Simplified:** feedback is stored end-to-end; live recalibration is FUTURE.
- **Note:** synthetic BLOCK_ESCALATE population is small (~a dozen) by design —
  SafeTriage hard-blocks very little. The audit still runs on it.

## Phase 7 — Backend API — DONE
- `api/main.py` (FastAPI): `/score`, `/transactions`(+`/{id}`), `/audit/sample`,
  `/kpis`, `/feedback` (GET+POST), `/evaluation`, `/calibration`,
  `/policy/simulate`. All return real computed data.
- `scripts/run_pipeline.py` builds everything with one command;
  `scripts/smoke_api.py` asserts every endpoint. **All endpoint checks pass.**

## Phase 9 — Baseline vs SafeTriage — DONE
- `evaluate.py`: test split only; baseline threshold frozen from validation.
  Metrics: precision, recall, FPR, FDR, review/auto-clear/escalation rates,
  analyst workload per 1k, fraud caught per 1k reviewed, expected cost.
- Demo cases A/B/C selected from real test rows and re-scored in full.
- Runs on synthetic (headline) and IEEE (reduced mode).

## Phase 10 — Documentation — DONE (backend pass)
- `EVAL_PROTOCOL.md`, `DATASET_README.md`, `ARCHITECTURE.md`, `LIMITATIONS.md`,
  `README.md`, this file.

## Phase 8 — Frontend — DONE
- Vite 6 + React 18 + React Router 6 + Tailwind v4 (`@tailwindcss/vite` 4.1.11) +
  Recharts 2.13. Hand-scaffolded; dev proxy `/api` → `:8000`. `npm run build` clean.
- 10 pages, all verified live against the running API:
  Overview, Live Transaction Feed, Investigation (SHAP + behavioural components +
  graph proxy + next-best-evidence + feedback POST), False Positive Lab
  (baseline-vs-SafeTriage table + chart + demo cases A/B/C), Review Queue,
  Self-Audit (both blind-spot populations + sample table), Calibration + Uncertainty
  (reliability curve, Brier/ECE, CV AUC, disagreement stats), Policy Lab (capacity +
  cost sliders → live sweep chart), Fraud Detection Cycle (10 steps + live counts),
  Architecture (LIVE / SIMULATED / FUTURE).
- **Assumption:** Tailwind bumped from 4.0.0 → 4.1.11 (4.0.0 had a CSS-generate bug).
- **Assumption:** demo cases render inline (not as links) — some IEEE demo rows are
  outside the 3k-row stored sample, so a navigate could 404.
- **Simplified:** "replay" reveals already-loaded rows in batches (no live stream).
- Verified: feedback submitted from the Investigation UI stored end-to-end
  ("Saved feedback #5: CONFIRM_FRAUD").

# SafeTriage

**A False-Positive-Aware Human-AI Fraud Triage and Self-Auditing System**
*BTP capstone / research prototype — runs fully locally, no external services or API keys.*

---

## The problem

Conventional fraud systems let a model score cross a threshold and then block the
transaction. On an imbalanced problem with an imperfect model this produces a
large number of **false positives** — legitimate customers blocked — and a heavy
**analyst workload** from everything that lands in manual review.

SafeTriage's research question is **not** "build a better classifier". It is:

> **How can we reduce unnecessary fraud alerts and analyst workload without
> significantly sacrificing fraud detection?**

The answer here is a pipeline that keeps the stages separate:

```
PREDICTION → INTERPRETATION → DECISION → HUMAN FEEDBACK → AUDIT
```

The calibrated model probability is **one input** to a documented policy engine —
never the block/allow decision itself. The policy engine also weighs whether the
transaction is *behaviourally consistent with that customer's own history*, how
much two models *disagree*, and the current *review capacity* and *cost* trade-off.
It chooses one of four actions: **AUTO_CLEAR**, **STEP_UP_REVIEW**, **SOFT_BLOCK**,
**ESCALATE**.

## What the measurement shows

Baseline (probability threshold picked for ~80% recall on validation) vs full
SafeTriage, both on the **held-out test split**. Numbers are computed by
`safetriage/evaluate.py`, not asserted — regenerate with `python -m safetriage.evaluate`.

| | Synthetic (headline) | IEEE-CIS (reduced mode) |
|---|---|---|
| Baseline recall / false positives | see `data/evaluation_synthetic.json` | see `data/evaluation_ieee.json` |
| SafeTriage recall / false positives | " | " |
| False-positive reduction | " (≈99%) | " (≈95%) |
| Analyst workload (reviews / 1k) | " | " |
| Expected cost | " | " |

The trade-off is real and reported honestly: SafeTriage gives up some recall
(single-digit to ~13 points on these datasets) to cut false positives by
90–99% and lower expected cost. The operating point is tunable — see the Policy
Lab / `scripts/policy_simulation.py`.

## Claimed contributions

1. A fraud-triage architecture that **separates prediction from decision**, with
   the calibrated probability as only one policy input.
2. A **behavioural-consistency layer** computed independently of the fraud score
   and used to *de-escalate* confident-but-consistent cases and *escalate*
   confident-and-anomalous ones.
3. A **capacity- and cost-aware policy engine** with documented thresholds and a
   runnable capacity/cost simulation.
4. A **model-disagreement uncertainty proxy** used to route ambiguous cases to
   humans rather than to an automated block or clear.
5. A **self-audit loop** that samples both blind-spot populations (auto-cleared
   *and* blocked/escalated) from the held-out test set.
6. A **human-feedback schema** that captures analyst verdicts as the basis for
   future recalibration.

The novelty is the **integration around the false-positive-aware decision
problem** — not the individual pieces. SHAP, probability calibration, XGBoost,
stratified sampling and cost curves are all standard; SafeTriage is a claim about
how to wire them together so the model does not decide alone.

## Research gaps addressed

1. Fraud research optimises AUC/recall, not analyst workload or customer friction.
2. Model scores are used directly as decisions instead of as one signal.
3. Calibration is reported but rarely fed into the *decision* logic.
4. "Anomalous" and "fraudulent" are conflated — a transaction can be unusual and
   legitimate, or typical and fraudulent.
5. Uncertainty is ignored at decision time; ambiguous cases are auto-actioned.
6. Review capacity is treated as unlimited in offline evaluation.
7. Auto-cleared traffic is never audited for hidden false negatives.
8. Blocked traffic is never audited for hidden false positives.
9. Analyst feedback is not captured in a form a model can later learn from.

## Existing work vs SafeTriage

| | Typical fraud ML | SafeTriage |
|---|---|---|
| Model output | decision | one policy input |
| Objective | recall / AUC | false-positive & workload reduction at bounded recall loss |
| Calibration | reported | drives the policy thresholds |
| Behaviour vs fraud | merged as features | separate consistency layer at the decision step |
| Uncertainty | unused | routes ambiguous cases to humans |
| Capacity / cost | assumed infinite / ignored | explicit, simulated |
| Blind-spot audit | none | samples auto-clear *and* block populations |
| Feedback | ad hoc | structured store for recalibration |

## Tech stack

Python 3.13 · XGBoost · scikit-learn (RandomForest, calibration) · SHAP ·
pandas / NumPy · FastAPI + Uvicorn · SQLite · matplotlib.
Frontend (Phase 8, after checkpoint): React + Vite + Tailwind + Recharts.
Exact versions in `requirements.txt`.

## Run it locally

```bash
pip install -r requirements.txt
pip install optuna==4.1.0          # optional — faster hyper-parameter search

# clean slate → fresh splits → feature engineering + 20-trial tuning + train
# + calibration → evaluation → capacity sim → SQLite → self-audit
python scripts/run_pipeline.py --reset --force --epochs 20

# serve the API  (terminal 1)
python -m uvicorn api.main:app --reload
#   docs at http://127.0.0.1:8000/docs

# verify every endpoint returns real data
python scripts/smoke_api.py
```

Or the individual training steps (IEEE-CIS is always trained first):

```bash
python scripts/reset_state.py                    # delete all prior weights + splits + data
python scripts/build_splits.py                   # step 1: 60/20/20 stratified splits
python -m safetriage.models.train --epochs 20    # step 2: FE (train-fit) + tuning + train + calibrate
python -m safetriage.evaluate                    # step 3: held-out test evaluation
```

`--epochs N` = number of hyper-parameter configurations the tuner evaluates by
cross-validation (XGBoost has no "epochs"; its boosting rounds are set by early
stopping and reported in `model_metrics_<ds>.json`). Feature encoders and the
tuner see the **train split only**; `model_metrics_<ds>.json → overfit_check`
reports the train-vs-CV AUC gap.

Dashboard (terminal 2):

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173  (proxies /api → :8000)
# or: npm run build && npm run preview
```

The 10 pages — Overview, Live Transaction Feed, Investigation, False Positive Lab,
Review Queue, Self-Audit, Calibration + Uncertainty, Policy Lab, Fraud Detection
Cycle, Architecture — every number comes from a live API call; there is no
placeholder content. A header switch toggles the synthetic / IEEE-CIS dataset.

Individual stages:

```bash
python -m safetriage.data.generate_synthetic_dataset
python -m safetriage.data.preprocess_ieee
python -m safetriage.models.train                 # both datasets
python -m safetriage.evaluate                      # baseline vs SafeTriage
python -m safetriage.audit synthetic              # self-audit pass
python scripts/policy_simulation.py                # capacity/cost sweep
```

## LIVE / SIMULATED / FUTURE

**LIVE** — dataset processing, XGBoost, Random Forest, calibration (Brier + ECE),
behavioural consistency, model-disagreement uncertainty, policy engine, SHAP,
FastAPI, SQLite, feedback storage, self-audit, evaluation, capacity simulation.

**SIMULATED** — transaction replay (scored from a static file), graph
visualisation (illustrative, no graph DB / GraphSAGE), next-best-evidence
(fixed templates), analyst-assistant explanation text (templates, no LLM),
analyst resolution in evaluation (ground-truth proxy).

**FUTURE** — Kafka/Redpanda, Redis feature store, graph database + GraphSAGE,
LLM analyst assistant, live feedback-driven recalibration, SAP HANA/AI Core/XSUAA,
drift monitoring, verified audit labels, security & privacy hardening.

## Limitations

See **`LIMITATIONS.md`** — in short: IEEE-CIS lacks the behavioural features
SafeTriage needs; the synthetic data is a controlled instrument, not evidence of
production performance; the evaluation's analyst is modelled as perfect; audit
outcomes are proxies; model disagreement is not true uncertainty; graph and
next-best-evidence are illustrative; feedback is stored but not yet learned from.

## Repository map

| Path | What |
|---|---|
| `config/` | seed + `policy_config.yaml` (all thresholds & costs, documented) |
| `safetriage/data/` | synthetic generator, IEEE preprocessing, seeded splits |
| `safetriage/models/` | training + calibration + saved artifacts |
| `safetriage/*.py` | behaviour, uncertainty, policy, explain, graph_proxy, next_best_evidence, scoring, db, audit, evaluate, metrics |
| `api/main.py` | FastAPI backend (9 endpoints) |
| `scripts/` | `run_pipeline.py`, `policy_simulation.py`, `smoke_api.py` |
| `EVAL_PROTOCOL.md` `DATASET_README.md` `ARCHITECTURE.md` `LIMITATIONS.md` `BUILD_LOG.md` | documentation |
| `frontend/` | React + Vite + Tailwind + Recharts dashboard (10 pages) |

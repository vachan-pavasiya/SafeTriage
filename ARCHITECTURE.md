# SafeTriage — Architecture

## The pipeline (never a single model deciding)

```
              ┌─────────────┐   ┌──────────────────┐   ┌──────────────┐   ┌───────────────┐   ┌─────────┐
  txn  ─────► │ PREDICTION  │─► │ INTERPRETATION   │─► │  DECISION    │─► │ HUMAN FEEDBACK│─► │  AUDIT  │
              │ XGBoost +   │   │ behavioural      │   │ policy engine│   │ analyst verdict│  │ blind-  │
              │ calibration │   │ consistency +    │   │ 4 actions +  │   │ + reason tag  │   │ spot    │
              │ (+ RF for   │   │ disagreement     │   │ capacity +   │   │ (recalib set) │   │ sampling│
              │  disagree)  │   │ + SHAP + reason  │   │ cost model   │   │               │   │ (proxy) │
              └─────────────┘   └──────────────────┘   └──────────────┘   └───────────────┘   └─────────┘
   calibrated probability  →  ONE input to the policy engine, never the block/allow decision
```

Decisions: `AUTO_CLEAR` · `STEP_UP_REVIEW` · `SOFT_BLOCK` · `ESCALATE`.

## Components

| Layer | Module | Output |
|-------|--------|--------|
| Data | `safetriage/data/` | `synthetic.csv`, `processed_ieee.parquet`, seeded splits |
| Prediction | `safetriage/models/train.py` | `xgb_*`, `rf_*`, `calibrator_*`, calibration + metrics JSON/PNG |
| Behaviour | `safetriage/behavior.py` | 6-component consistency score (+ IEEE proxy) |
| Uncertainty | `safetriage/uncertainty.py` | normalised model-disagreement proxy |
| Policy | `safetriage/policy.py` | decision + firing rule + explanation; capacity + cost |
| Explain | `safetriage/explain.py` | SHAP top-5, reason code, supporting/counter evidence |
| Graph | `safetriage/graph_proxy.py` | illustrative node/edge structure |
| Evidence | `safetriage/next_best_evidence.py` | deterministic analyst checklist |
| Scoring | `safetriage/scoring.py` | the full `/score` object; batch scorer |
| Persistence | `safetriage/db.py` | SQLite: transactions, feedback, audit_runs, audit_samples |
| Audit | `safetriage/audit.py` | risk-stratified + control samples of two blind-spot populations |
| Evaluation | `safetriage/evaluate.py` | Baseline vs SafeTriage on the held-out test split |
| API | `api/main.py` | FastAPI, 9 endpoints |

## LIVE / SIMULATED / FUTURE

### LIVE (really runs, really computed)
Dataset processing · XGBoost · Random Forest · probability calibration
(sigmoid/isotonic, Brier + ECE) · behavioural-consistency scoring · model-disagreement
uncertainty · policy engine (thresholds, capacity, cost) · SHAP explanations ·
reason codes · FastAPI backend · SQLite persistence · analyst-feedback storage ·
self-audit sampling · Baseline vs SafeTriage evaluation · capacity simulation.

### SIMULATED (works, but stands in for a real system)
Live transaction replay (records are scored from a static file) · graph
visualisation (illustrative structure from two columns, no graph DB / GraphSAGE) ·
next-best-evidence (fixed templates, no ranking model) · "analyst assistant"
explanation text (template sentences, no LLM) · analyst resolution of review
cases in evaluation (modelled with the ground-truth label as a proxy).

### FUTURE (not built)
Kafka / Redpanda streaming · Redis feature store · graph database + GraphSAGE ·
LLM analyst assistant · live feedback-driven recalibration · SAP HANA / AI Core /
XSUAA · drift monitoring · verified human audit labels · privacy & security
hardening.

"""Phase 7 — SafeTriage backend API (FastAPI).

Runs fully locally: no external services, no API keys. All data is read from the
model artifacts and the SQLite database produced by ``scripts/run_pipeline.py``.

    python scripts/run_pipeline.py            # build artifacts + DB (once)
    python -m uvicorn api.main:app --reload   # serve

Endpoints
    POST /score            score a transaction (by id, or full feature payload)
    GET  /transactions     paginated scored transactions (filter by decision/split)
    GET  /transactions/{id}
    GET  /audit/sample     latest self-audit run (both blind-spot populations)
    GET  /kpis             volumes, decision mix, audit-proxy FP/FN
    GET  /feedback         stored analyst decisions
    POST /feedback         record an analyst decision
    GET  /evaluation       baseline vs SafeTriage (held-out test set)
    GET  /calibration      reliability curve + Brier + ECE (raw vs calibrated)
    GET  /policy/simulate   capacity / cost sweep
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from safetriage import ARTIFACTS_DIR, DATA_DIR, policy_version
from safetriage import db
from safetriage.policy import PolicyThresholds, apply_capacity, expected_cost
from safetriage.scoring import model_version, score_transaction

app = FastAPI(title="SafeTriage API", version="1.0.0",
              description="False-positive-aware human-AI fraud triage — local research prototype.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATASETS = ("synthetic", "ieee")
_JSON_FIELDS = ("raw_json", "output_json")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _hydrate(row: dict) -> dict:
    out = dict(row)
    for f in _JSON_FIELDS:
        if out.get(f):
            try:
                out[f.replace("_json", "")] = json.loads(out.pop(f))
            except (json.JSONDecodeError, TypeError):
                out.pop(f, None)
    return out


def _read_json_artifact(path: Path):
    if not path.exists():
        raise HTTPException(404, f"{path.name} not found — run scripts/run_pipeline.py first")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# /score
# --------------------------------------------------------------------------- #
class ScoreRequest(BaseModel):
    transaction_id: str | None = Field(None, description="score a transaction already in the DB")
    dataset: str = "synthetic"
    transaction: dict | None = Field(None, description="raw feature payload for a fresh score")
    persist: bool = False


@app.post("/score")
def post_score(req: ScoreRequest):
    if req.dataset not in DATASETS:
        raise HTTPException(400, f"dataset must be one of {DATASETS}")

    if req.transaction is not None:
        raw = req.transaction
    elif req.transaction_id:
        stored = db.get_transaction(req.transaction_id)
        if not stored:
            raise HTTPException(404, f"unknown transaction_id {req.transaction_id!r}")
        raw = json.loads(stored["raw_json"])
        req.dataset = stored["dataset"]
    else:
        raise HTTPException(400, "provide either transaction_id or transaction")

    result = score_transaction(raw, req.dataset, with_shap=True, with_graph=(req.dataset == "synthetic"))

    if req.persist:
        tid = f"{req.dataset}-{raw.get('TransactionID', result['transaction_id'])}"
        db.upsert_transactions([{
            "transaction_id": tid, "dataset": req.dataset, "split": None,
            "created_at": db.now_iso(),
            "fraud_probability": result["fraud_probability"],
            "calibrated_probability": result["calibrated_probability"],
            "behavioural_consistency_score": result["behavioral_consistency_score"],
            "behavioural_anomaly_score": result["behavioral_anomaly_score"],
            "uncertainty": result["uncertainty"], "confidence_score": result["confidence_score"],
            "decision": result["decision"], "policy_rule": result["policy_rule"],
            "predicted_typology": result["predicted_typology"],
            "fraud_reason_code": result["fraud_reason_code"], "capacity_concession": 0,
            "is_fraud_label": int(raw["isFraud"]) if "isFraud" in raw else None,
            "model_version": result["model_version"], "policy_version": result["policy_version"],
            "latency_ms": result["latency_ms"],
            "raw_json": json.dumps(raw, default=str), "output_json": json.dumps(result, default=str),
        }])
    return result


# --------------------------------------------------------------------------- #
# /transactions
# --------------------------------------------------------------------------- #
@app.get("/transactions")
def get_transactions(
    decision: str | None = None,
    split: str | None = None,
    dataset: str = "synthetic",
    limit: int = Query(100, le=2000),
    offset: int = 0,
    order_by: str = "calibrated_probability",
    desc: bool = True,
):
    rows = db.fetch_transactions(decision=decision, split=split, dataset=dataset,
                                 limit=limit, offset=offset, order_by=order_by, desc=desc)
    return {"count": len(rows), "dataset": dataset, "items": [_hydrate(r) for r in rows]}


@app.get("/transactions/{transaction_id}")
def get_transaction(transaction_id: str):
    row = db.get_transaction(transaction_id)
    if not row:
        raise HTTPException(404, f"unknown transaction_id {transaction_id!r}")
    return _hydrate(row)


# --------------------------------------------------------------------------- #
# /audit/sample
# --------------------------------------------------------------------------- #
@app.get("/audit/sample")
def get_audit_sample(dataset: str = "synthetic"):
    run = db.latest_audit_run(dataset)
    if not run:
        raise HTTPException(404, "no audit run — run scripts/run_pipeline.py first")
    return run


# --------------------------------------------------------------------------- #
# /kpis
# --------------------------------------------------------------------------- #
@app.get("/kpis")
def get_kpis(dataset: str = "synthetic"):
    rows = db.fetch_transactions(dataset=dataset, limit=100000)
    if not rows:
        raise HTTPException(404, "no transactions — run scripts/run_pipeline.py first")
    df = pd.DataFrame(rows)
    n = len(df)
    dist = df["decision"].value_counts().to_dict()
    frauds = int(df["is_fraud_label"].fillna(0).sum())

    def rate(d):
        return round(dist.get(d, 0) / n, 5)

    audit = db.latest_audit_run(dataset)
    audit_proxy = {}
    if audit:
        for pop, pdata in audit["summary"].get("populations", {}).items():
            audit_proxy[pop] = pdata.get("samples", {})

    return {
        "dataset": dataset,
        "n_transactions": n,
        "n_fraud_labelled": frauds,
        "fraud_prevalence": round(frauds / n, 5),
        "decision_distribution": {k: int(v) for k, v in dist.items()},
        "auto_clear_rate": rate("AUTO_CLEAR"),
        "review_rate": rate("STEP_UP_REVIEW"),
        "soft_block_rate": rate("SOFT_BLOCK"),
        "escalation_rate": rate("ESCALATE"),
        "capacity_concessions": int(df["capacity_concession"].fillna(0).sum()),
        "analyst_workload_per_1k": round(dist.get("STEP_UP_REVIEW", 0) / n * 1000.0, 2),
        "mean_calibrated_probability": round(float(df["calibrated_probability"].mean()), 5),
        "mean_behavioural_consistency": round(float(df["behavioural_consistency_score"].mean()), 5),
        "mean_uncertainty": round(float(df["uncertainty"].mean()), 5),
        "audit_proxy": audit_proxy,
        "model_version": model_version(dataset),
        "policy_version": policy_version(),
    }


# --------------------------------------------------------------------------- #
# /feedback
# --------------------------------------------------------------------------- #
class FeedbackRequest(BaseModel):
    transaction_id: str
    analyst_decision: str = Field(..., description="MARK_LEGITIMATE | CONFIRM_FRAUD | NEED_MORE_EVIDENCE")
    reason_tag: str | None = None
    note: str | None = None


@app.get("/feedback")
def get_feedback(limit: int = Query(200, le=2000)):
    rows = db.fetch_feedback(limit=limit)
    return {"count": len(rows), "items": rows}


@app.post("/feedback")
def post_feedback(req: FeedbackRequest):
    try:
        row = db.insert_feedback(req.transaction_id, req.analyst_decision,
                                 reason_tag=req.reason_tag, note=req.note)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    return {"stored": row}


# --------------------------------------------------------------------------- #
# /evaluation  /calibration
# --------------------------------------------------------------------------- #
@app.get("/evaluation")
def get_evaluation(dataset: str = "synthetic"):
    if dataset == "all":
        return {ds: _read_json_artifact(DATA_DIR / f"evaluation_{ds}.json") for ds in DATASETS}
    return _read_json_artifact(DATA_DIR / f"evaluation_{dataset}.json")


@app.get("/calibration")
def get_calibration(dataset: str = "synthetic"):
    if dataset == "all":
        return {ds: _read_json_artifact(ARTIFACTS_DIR / f"calibration_{ds}.json") for ds in DATASETS}
    data = _read_json_artifact(ARTIFACTS_DIR / f"calibration_{dataset}.json")
    mm = ARTIFACTS_DIR / f"model_metrics_{dataset}.json"
    if mm.exists():
        data["model_metrics"] = json.loads(mm.read_text())
    return data


# --------------------------------------------------------------------------- #
# /policy/simulate
# --------------------------------------------------------------------------- #
@app.get("/policy/simulate")
def get_policy_simulate(
    dataset: str = "synthetic",
    fraud_loss_cost: float | None = None,
    false_positive_cost: float | None = None,
    review_cost: float | None = None,
    customer_friction_cost: float | None = None,
):
    """Precomputed capacity sweep (validation split). If any cost parameter is
    supplied, the expected-cost column is recomputed from the stored confusion
    counts — no re-scoring needed."""
    data = _read_json_artifact(DATA_DIR / f"policy_simulation_{dataset}.json")
    overrides = {k: v for k, v in {
        "fraud_loss_cost": fraud_loss_cost, "false_positive_cost": false_positive_cost,
        "review_cost": review_cost, "customer_friction_cost": customer_friction_cost,
    }.items() if v is not None}
    if overrides:
        cfg = {"costs": {**data["cost_params"], **overrides}}
        for p in data["points"]:
            p["expected_cost"] = expected_cost(
                missed_fraud=p["missed_fraud"], false_blocks=p["false_positives"],
                reviews=p["review_volume"], step_up_challenges=p["legit_sent_to_review"],
                cfg=cfg,
            )["total_expected_cost"]
        data["cost_params"] = cfg["costs"]
        data["cost_overrides_applied"] = overrides
    return data


@app.get("/")
def root():
    return {"service": "SafeTriage API", "docs": "/docs",
            "endpoints": ["/score", "/transactions", "/audit/sample", "/kpis",
                          "/feedback", "/evaluation", "/calibration", "/policy/simulate"]}

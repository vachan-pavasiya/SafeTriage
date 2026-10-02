"""End-to-end scoring pipeline: PREDICTION -> INTERPRETATION -> DECISION.

``score_transaction(raw, dataset)`` returns the full object served by
``POST /score``. ``score_dataframe(df, dataset)`` is the vectorised path used to
populate the database in scripts/run_pipeline.py.

Order of operations (all separated, none of them "decides" on its own):
  1. XGBoost raw probability            (safetriage.models)
  2. calibrated probability             (calibrator artifact)
  3. Random-Forest probability          -> model-disagreement uncertainty proxy
  4. behavioural consistency / anomaly  (safetriage.behavior)
  5. policy decision                    (safetriage.policy)
  6. SHAP attributions + reason code    (safetriage.explain)
  7. next-best-evidence + graph proxy   (safetriage.next_best_evidence / graph_proxy)
"""

from __future__ import annotations

import time
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from safetriage import ARTIFACTS_DIR, artifact_version, load_policy_config, policy_version
from safetriage.behavior import behavioural_frame, consistency_ieee_proxy
from safetriage.explain import evidence_split, reason_code, shap_top_features, shap_top_features_batch
from safetriage.features import transform_for_model
from safetriage.graph_proxy import transaction_graph
from safetriage.next_best_evidence import recommend
from safetriage.policy import PolicyThresholds, decide, decide_frame
from safetriage.uncertainty import disagreement_uncertainty


def _artifact_suffix() -> str:
    # "" -> evaluated model (fit on TRAIN, consistent with evaluation_<ds>.json)
    # "_full" -> refit on the whole dataset (config serving.use_full_refit)
    try:
        return "_full" if load_policy_config().get("serving", {}).get("use_full_refit") else ""
    except Exception:
        return ""


@lru_cache(maxsize=4)
def _models(dataset: str):
    sfx = _artifact_suffix()
    xgb_p = ARTIFACTS_DIR / f"xgb_{dataset}{sfx}.joblib"
    rf_p = ARTIFACTS_DIR / f"rf_{dataset}{sfx}.joblib"
    if not xgb_p.exists():  # full-refit artifacts absent -> fall back to evaluated
        xgb_p = ARTIFACTS_DIR / f"xgb_{dataset}.joblib"
        rf_p = ARTIFACTS_DIR / f"rf_{dataset}.joblib"
    return (
        joblib.load(xgb_p),
        joblib.load(ARTIFACTS_DIR / f"calibrator_{dataset}.joblib"),
        joblib.load(rf_p),
    )


@lru_cache(maxsize=4)
def model_version(dataset: str) -> str:
    sfx = _artifact_suffix()
    return "m_" + artifact_version(
        ARTIFACTS_DIR / f"xgb_{dataset}{sfx}.joblib",
        ARTIFACTS_DIR / f"calibrator_{dataset}.joblib",
        ARTIFACTS_DIR / f"rf_{dataset}{sfx}.joblib",
        ARTIFACTS_DIR / f"fe_{dataset}.joblib",
    )


TYPOLOGY_BY_REASON = {
    "ATO_NEW_DEVICE_HIGH_VELOCITY": "ACCOUNT_TAKEOVER",
    "MULE_NETWORK_PROXIMITY": "MONEY_MULE",
    "MODEL_DISAGREEMENT_REVIEW": "UNCERTAIN",
    "KNOWN_DEVICE_NORMAL_BEHAVIOR": "NONE",
}


def predict_typology(raw: dict, reason: str, calibrated_probability: float, p_low: float) -> str:
    if reason in TYPOLOGY_BY_REASON:
        return TYPOLOGY_BY_REASON[reason]
    if reason == "UNUSUAL_AMOUNT_NEW_BENEFICIARY":
        if float(raw.get("account_age_days", 9999)) < 75:
            return "SYNTHETIC_IDENTITY"
        return "PUSH_PAYMENT"
    if calibrated_probability < p_low:
        return "NONE"
    return "REVIEW_UNCLASSIFIED"


# --------------------------------------------------------------------------- #
# Single transaction
# --------------------------------------------------------------------------- #
def score_transaction(raw: dict, dataset: str = "synthetic", *, with_shap: bool = True,
                      with_graph: bool = True) -> dict:
    t0 = time.perf_counter()
    xgb, cal, rf = _models(dataset)
    th = PolicyThresholds.from_config(dataset=dataset)

    X = transform_for_model(dataset, raw)
    p_raw = float(xgb.predict_proba(X)[:, 1][0])
    p_cal = float(cal.predict_proba(X)[:, 1][0])
    p_rf = float(rf.predict_proba(X)[:, 1][0])
    u = float(disagreement_uncertainty([p_cal], [p_rf], dataset)[0])

    if dataset == "synthetic":
        b = behavioural_frame(pd.DataFrame([raw])).iloc[0].to_dict()
    else:
        b = consistency_ieee_proxy(pd.DataFrame([raw])).iloc[0].to_dict()
    consistency = float(b["behavioural_consistency_score"])
    anomaly = float(b["behavioural_anomaly_score"])

    dec = decide(p_cal, consistency, anomaly, u, th)

    top_features = shap_top_features(dataset, X) if with_shap else []
    support, counter = evidence_split(top_features)
    reason = reason_code(raw, calibrated_probability=p_cal, behavioural_consistency=consistency,
                         uncertainty=u, top_features=top_features, th=th, dataset=dataset)
    typ = predict_typology(raw, reason, p_cal, th.p_low)
    nbe = recommend(raw, reason_code=reason, decision=dec["decision"], uncertainty=u)

    out = {
        "transaction_id": str(raw.get("TransactionID", raw.get("transaction_id", "unknown"))),
        "dataset": dataset,
        "fraud_probability": round(p_raw, 5),
        "calibrated_probability": round(p_cal, 5),
        "behavioral_consistency_score": round(consistency, 5),
        "behavioral_anomaly_score": round(anomaly, 5),
        "uncertainty": round(u, 5),
        "confidence_score": round(1.0 - u, 5),
        "decision": dec["decision"],
        "policy_rule": dec["policy_rule"],
        "predicted_typology": typ,
        "fraud_reason_code": reason,
        "top_features": top_features,
        "supporting_evidence": support,
        "counter_evidence": counter,
        "next_best_evidence": nbe,
        "decision_explanation": dec["decision_explanation"],
        "behavioral_components": {
            k: b[k] for k in b
            if (k.endswith("_consistency") or k.endswith("_proxy")) and k != "is_proxy"
        },
        "is_behavioral_proxy": bool(b.get("is_proxy", False)),
        "model_version": model_version(dataset),
        "policy_version": policy_version(),
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }
    if with_graph and dataset == "synthetic":
        out["graph_proxy"] = transaction_graph(raw)
    return out


# --------------------------------------------------------------------------- #
# Batch (DB seeding / evaluation)
# --------------------------------------------------------------------------- #
def score_dataframe(df: pd.DataFrame, dataset: str = "synthetic", *, with_shap: bool = False,
                    with_graph: bool = False) -> pd.DataFrame:
    """Vectorised scoring. Returns a frame aligned to ``df.index`` with the
    numeric scores + policy decision; per-row SHAP / reason / NBE / graph are
    added only when requested (they loop and are slower).
    """
    xgb, cal, rf = _models(dataset)
    X = transform_for_model(dataset, df)
    p_raw = xgb.predict_proba(X)[:, 1]
    p_cal = cal.predict_proba(X)[:, 1]
    p_rf = rf.predict_proba(X)[:, 1]
    u = disagreement_uncertainty(p_cal, p_rf, dataset)

    if dataset == "synthetic":
        bf = behavioural_frame(df)
    else:
        bf = consistency_ieee_proxy(df)

    scores = pd.DataFrame({
        "fraud_probability": np.round(p_raw, 5),
        "calibrated_probability": np.round(p_cal, 5),
        "behavioral_consistency_score": bf["behavioural_consistency_score"].to_numpy(),
        "behavioral_anomaly_score": bf["behavioural_anomaly_score"].to_numpy(),
        "uncertainty": np.round(u, 5),
        "confidence_score": np.round(1.0 - u, 5),
    }, index=df.index)

    dec = decide_frame(
        scores.rename(columns={
            "behavioral_consistency_score": "behavioural_consistency_score",
            "behavioral_anomaly_score": "behavioural_anomaly_score",
        }),
        dataset=dataset,
    )
    out = pd.concat([scores, dec], axis=1)

    th = PolicyThresholds.from_config(dataset=dataset)
    if with_shap or with_graph:
        shap_lists = shap_top_features_batch(dataset, X) if with_shap else [[] for _ in range(len(df))]
        reasons, typs, supports, counters, nbes, graphs, tops = [], [], [], [], [], [], []
        for pos, (idx, raw) in enumerate(zip(df.index, df.to_dict("records"))):
            tf = shap_lists[pos]
            s, c = evidence_split(tf)
            rc = reason_code(raw, calibrated_probability=float(out.at[idx, "calibrated_probability"]),
                             behavioural_consistency=float(out.at[idx, "behavioral_consistency_score"]),
                             uncertainty=float(out.at[idx, "uncertainty"]), top_features=tf, th=th, dataset=dataset)
            reasons.append(rc)
            typs.append(predict_typology(raw, rc, float(out.at[idx, "calibrated_probability"]), th.p_low))
            supports.append(s); counters.append(c); tops.append(tf)
            nbes.append(recommend(raw, reason_code=rc, decision=out.at[idx, "decision"],
                                  uncertainty=float(out.at[idx, "uncertainty"])))
            graphs.append(transaction_graph(raw) if (with_graph and dataset == "synthetic") else None)
        out["fraud_reason_code"] = reasons
        out["predicted_typology"] = typs
        out["supporting_evidence"] = supports
        out["counter_evidence"] = counters
        out["top_features"] = tops
        out["next_best_evidence"] = nbes
        out["graph_proxy"] = graphs
    return out

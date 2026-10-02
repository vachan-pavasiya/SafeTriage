"""One command to build everything the API needs.

    python scripts/run_pipeline.py                       # build missing pieces
    python scripts/run_pipeline.py --force               # rebuild data + models
    python scripts/run_pipeline.py --reset --force       # wipe ALL derived state first
    python scripts/run_pipeline.py --reset --force --epochs 30 --cv-folds 4

Steps: [reset] -> IEEE preprocessing -> synthetic data -> fresh 60/20/20 splits
-> feature engineering (train-fit) + hyper-parameter tuning + train (IEEE first,
then synthetic) -> evaluate (both) -> capacity simulation -> (re)build SQLite
-> score + store transactions -> self-audit (both) -> seed sample feedback.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config.seed import SEED, set_all_seeds
from safetriage import ARTIFACTS_DIR, DATA_DIR, SPLITS_DIR, policy_version
from safetriage.audit import run_audit
from safetriage.data import generate_synthetic_dataset, preprocess_ieee
from safetriage.data.splits import load_indices
from safetriage.db import count_transactions, fetch_transactions, init_db, insert_feedback, upsert_transactions
from safetriage.evaluate import evaluate
from safetriage.models import train as train_mod
from safetriage.scoring import model_version, score_dataframe

IEEE_STORE_SAMPLE = 3000  # rows of IEEE-CIS to score + store (feed + spot checks)


def _split_lookup(dataset: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for name in ("train", "val", "test"):
        for i in load_indices(dataset, name):
            out[int(i)] = name
    return out


def reset_state() -> None:
    """Delete every derived artifact: model weights, splits, DB, generated data,
    evaluation/simulation JSON. Raw inputs (ieee-fraud-detection/) are untouched."""
    import shutil

    removed = []
    for d in (ARTIFACTS_DIR, SPLITS_DIR):
        if d.exists():
            shutil.rmtree(d)
            removed.append(str(d))
        d.mkdir(parents=True, exist_ok=True)
    if DATA_DIR.exists():
        for p in DATA_DIR.iterdir():
            if p.is_file() and p.suffix in {".csv", ".parquet", ".db", ".json", ".png"}:
                p.unlink()
                removed.append(p.name)
    print(f"[reset] removed {len(removed)} derived items (weights, splits, DB, generated data).")


def _json_safe(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _store_dataset(dataset: str, df: pd.DataFrame, positions: np.ndarray, with_shap: bool) -> int:
    """Score rows at ``positions`` (positional into df) and upsert them."""
    sub = df.iloc[positions].reset_index(drop=True)
    scored = score_dataframe(sub, dataset, with_shap=with_shap, with_graph=(dataset == "synthetic"))
    from safetriage.policy import apply_capacity
    scored = apply_capacity(scored)

    split_by_pos = {}
    for name in ("train", "val", "test"):
        s = set(load_indices(dataset, name).tolist())
        for local_i, global_i in enumerate(positions):
            if int(global_i) in s:
                split_by_pos[local_i] = name

    from safetriage.db import now_iso
    mv, pv = model_version(dataset), policy_version()
    rows = []
    for local_i in range(len(sub)):
        s = scored.iloc[local_i]
        src = sub.iloc[local_i]
        rows.append({
            "transaction_id": f"{dataset}-{int(src['TransactionID'])}",
            "dataset": dataset,
            "split": split_by_pos.get(local_i),
            "created_at": now_iso(),
            "fraud_probability": float(s["fraud_probability"]),
            "calibrated_probability": float(s["calibrated_probability"]),
            "behavioural_consistency_score": float(s["behavioral_consistency_score"]),
            "behavioural_anomaly_score": float(s["behavioral_anomaly_score"]),
            "uncertainty": float(s["uncertainty"]),
            "confidence_score": float(s["confidence_score"]),
            "decision": str(s["decision"]),
            "policy_rule": str(s["policy_rule"]),
            "predicted_typology": str(s.get("predicted_typology", "") or ""),
            "fraud_reason_code": str(s.get("fraud_reason_code", "") or ""),
            "capacity_concession": int(s.get("capacity_concession", 0)),
            "is_fraud_label": int(src["isFraud"]),
            "model_version": mv,
            "policy_version": pv,
            "latency_ms": None,
            "raw_json": json.dumps({k: _json_safe(v) for k, v in src.to_dict().items()}),
            "output_json": json.dumps({
                k: _json_safe(s[k]) for k in (
                    "fraud_probability", "calibrated_probability", "behavioral_consistency_score",
                    "behavioral_anomaly_score", "uncertainty", "confidence_score", "decision",
                    "policy_rule", "predicted_typology", "fraud_reason_code", "top_features",
                    "supporting_evidence", "counter_evidence", "next_best_evidence", "graph_proxy")
                if k in s
            }),
        })
    return upsert_transactions(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="wipe ALL derived state (weights, splits, DB, data) first")
    ap.add_argument("--force", action="store_true", help="rebuild data + models even if present")
    ap.add_argument("--epochs", type=int, default=None, help="hyper-parameter tuning trials (default: config hpo_trials)")
    ap.add_argument("--cv-folds", type=int, default=None, help="CV folds per tuning trial (default: config cv_folds)")
    args = ap.parse_args()
    set_all_seeds(SEED)

    if args.reset:
        reset_state()
        args.force = True

    # 1. data (IEEE first) --------------------------------------------------- #
    if args.force or not (DATA_DIR / "processed_ieee.parquet").exists():
        preprocess_ieee.main()
    if args.force or not (DATA_DIR / "synthetic.csv").exists():
        generate_synthetic_dataset.main()

    # 2. splits + feature engineering + tuning + train (IEEE first) --------- #
    need_models = args.force or not (
        (ARTIFACTS_DIR / "xgb_synthetic.joblib").exists()
        and (ARTIFACTS_DIR / "xgb_ieee.joblib").exists()
    )
    if need_models:
        train_mod.main(["ieee", "synthetic"], n_trials=args.epochs, cv_folds=args.cv_folds)

    # 3. evaluation + simulation ------------------------------------------- #
    for ds in ("ieee", "synthetic"):
        evaluate(ds)
    from scripts.policy_simulation import simulate
    for ds in ("ieee", "synthetic"):
        simulate(ds)

    # 4. database ------------------------------------------------------------ #
    init_db(reset=True)

    syn = pd.read_csv(DATA_DIR / "synthetic.csv")
    n_syn = _store_dataset("synthetic", syn, np.arange(len(syn)), with_shap=True)

    ieee = pd.read_parquet(DATA_DIR / "processed_ieee.parquet")
    rng = np.random.default_rng(SEED)
    test_idx = set(load_indices("ieee", "test").tolist())
    pos = np.array(sorted(test_idx))[:IEEE_STORE_SAMPLE] if len(test_idx) >= IEEE_STORE_SAMPLE else np.sort(
        rng.choice(len(ieee), size=IEEE_STORE_SAMPLE, replace=False))
    n_ieee = _store_dataset("ieee", ieee, pos, with_shap=False)

    # 5. self-audit -------------------------------------------------------------- #
    audit_syn = run_audit("synthetic")
    audit_ieee = run_audit("ieee")

    # 6. seed sample analyst feedback -------------------------------------------- #
    fb = 0
    review_rows = fetch_transactions(decision="STEP_UP_REVIEW", dataset="synthetic", limit=6)
    demo = [("CONFIRM_FRAUD", "confirmed_beneficiary_scam"),
            ("MARK_LEGITIMATE", "known_customer_large_purchase"),
            ("NEED_MORE_EVIDENCE", "await_device_check")]
    for i, r in enumerate(review_rows[:3]):
        d, tag = demo[i % len(demo)]
        insert_feedback(r["transaction_id"], d, reason_tag=tag, note="seed sample (run_pipeline)")
        fb += 1

    print("\n================  PIPELINE COMPLETE  ================")
    print(f" transactions stored : synthetic={n_syn}  ieee={n_ieee}  (db total={count_transactions()})")
    print(f" audit runs          : synthetic + ieee  (auto-clear & block/escalate populations)")
    print(f"   synthetic AUTO_CLEAR pop size = "
          f"{audit_syn['populations'].get('AUTO_CLEAR', {}).get('population_size')}, "
          f"BLOCK_ESCALATE pop size = {audit_syn['populations'].get('BLOCK_ESCALATE', {}).get('population_size')}")
    print(f" sample feedback rows : {fb}")
    print(f" evaluation + simulation JSON written to {DATA_DIR}")
    print(" start the API with:  python -m uvicorn api.main:app --reload")


if __name__ == "__main__":
    main()

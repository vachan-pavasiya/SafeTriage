"""Phase 6 — self-audit sampling.

Samples two blind-spot populations, **only from the held-out test split**:

* ``AUTO_CLEAR``      — hidden false-negative risk (fraud we waved through)
* ``BLOCK_ESCALATE``  — hidden false-positive risk (legit we blocked / escalated)

Each population is sampled two ways: a **risk-stratified** sample (quantile bins
of calibrated probability, so low-and-high-risk auto-clears are both represented)
and a **random control** sample.

There are no verified human audit labels here, so the "audit outcome" uses the
dataset's own ``isFraud`` as an **audit proxy** (``is_proxy=1``). All rates are
computed from the sample — nothing is hardcoded.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config.seed import SEED
from safetriage import DATA_DIR
from safetriage.db import init_db, insert_audit_run
from safetriage.data.splits import load_test_indices
from safetriage.policy import apply_capacity
from safetriage.scoring import score_dataframe

DEFAULTS = {
    "auto_clear_stratified": 90,
    "auto_clear_control": 45,
    "block_escalate_stratified": 60,
    "block_escalate_control": 30,
    "n_bins": 5,
}


def _load_test(dataset: str) -> pd.DataFrame:
    if dataset == "synthetic":
        df = pd.read_csv(DATA_DIR / "synthetic.csv")
    else:
        df = pd.read_parquet(DATA_DIR / "processed_ieee.parquet")
    test_idx = load_test_indices(dataset)
    return df.iloc[test_idx].reset_index(drop=True)


def _stratified(pool: pd.DataFrame, n: int, n_bins: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(pool) <= n:
        return pool
    try:
        bins = pd.qcut(pool["calibrated_probability"], q=min(n_bins, pool["calibrated_probability"].nunique()),
                       duplicates="drop")
    except ValueError:
        return pool.sample(n=n, random_state=SEED)
    per = max(1, n // max(1, bins.nunique()))
    picks = []
    for _, grp in pool.groupby(bins, observed=True):
        take = min(len(grp), per)
        picks.append(grp.sample(n=take, random_state=int(rng.integers(1 << 31))))
    out = pd.concat(picks)
    if len(out) < n:  # top up randomly
        rest = pool.drop(out.index)
        out = pd.concat([out, rest.sample(n=min(len(rest), n - len(out)), random_state=SEED)])
    return out.head(n)


def run_audit(dataset: str = "synthetic", *, config: dict | None = None) -> dict:
    cfg = {**DEFAULTS, **(config or {})}
    rng = np.random.default_rng(SEED)
    init_db()

    df = _load_test(dataset)
    scored = score_dataframe(df, dataset, with_shap=False, with_graph=False)
    scored = apply_capacity(scored)
    scored["isFraud"] = df["isFraud"].to_numpy()
    scored["transaction_id"] = df["TransactionID"].astype(str).to_numpy()

    populations = {
        "AUTO_CLEAR": scored[scored["decision"] == "AUTO_CLEAR"],
        "BLOCK_ESCALATE": scored[scored["decision"].isin(["SOFT_BLOCK", "ESCALATE"])],
    }
    size_key = {"AUTO_CLEAR": ("auto_clear_stratified", "auto_clear_control"),
                "BLOCK_ESCALATE": ("block_escalate_stratified", "block_escalate_control")}

    samples: list[dict] = []
    summary: dict = {"dataset": dataset, "test_n": int(len(df)), "populations": {}}

    for pop_name, pool in populations.items():
        strat_k, ctrl_k = size_key[pop_name]
        pop_summary = {"population_size": int(len(pool)), "samples": {}}
        if len(pool) == 0:
            summary["populations"][pop_name] = pop_summary
            continue

        strat = _stratified(pool, cfg[strat_k], cfg["n_bins"], rng)
        remaining = pool.drop(strat.index)
        ctrl = remaining.sample(n=min(len(remaining), cfg[ctrl_k]), random_state=SEED) if len(remaining) else pool.sample(
            n=min(len(pool), cfg[ctrl_k]), random_state=SEED)

        for stype, sdf in (("risk_stratified", strat), ("random_control", ctrl)):
            proxy_fraud_rate = float(sdf["isFraud"].mean()) if len(sdf) else 0.0
            if pop_name == "AUTO_CLEAR":
                metric = {"proxy_missed_fraud_rate": round(proxy_fraud_rate, 5),
                          "proxy_missed_fraud_count": int(sdf["isFraud"].sum())}
            else:
                metric = {"proxy_false_positive_rate": round(1.0 - proxy_fraud_rate, 5),
                          "proxy_false_positive_count": int((sdf["isFraud"] == 0).sum())}
            metric["n_sampled"] = int(len(sdf))
            pop_summary["samples"][stype] = metric
            for _, r in sdf.iterrows():
                samples.append({
                    "population": pop_name,
                    "sample_type": stype,
                    "transaction_id": str(r["transaction_id"]),
                    "calibrated_probability": float(r["calibrated_probability"]),
                    "decision": str(r["decision"]),
                    "audit_proxy_label": int(r["isFraud"]),
                    "is_proxy": 1,
                })
        summary["populations"][pop_name] = pop_summary

    run_id = f"audit_{dataset}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    summary["run_id"] = run_id
    summary["note"] = ("Rates are computed against ground-truth isFraud used as an AUDIT PROXY "
                       "(no verified human re-review). Test split only.")
    insert_audit_run(run_id, dataset, summary, samples)
    return summary


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(run_audit(sys.argv[1] if len(sys.argv) > 1 else "synthetic"), indent=2))

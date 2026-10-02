"""Phase 9 — Baseline vs SafeTriage, on the held-out TEST split only.

Baseline   : probability-threshold-only. The threshold is picked on the
             VALIDATION split to reach ``evaluation.baseline_target_recall`` and
             then frozen; >= threshold => hard block, else clear. No reviews.

SafeTriage : calibrated probability + behavioural consistency + uncertainty +
             policy engine + capacity. Resolution model:
               ESCALATE / SOFT_BLOCK       -> actioned as fraud
               STEP_UP_REVIEW              -> resolved by the proxy label
                                             (models an analyst working the queue)
               AUTO_CLEAR                  -> cleared
             A legitimate transaction sent to STEP_UP_REVIEW is NOT counted as a
             false positive (it was not blocked) — it is counted as analyst
             workload and customer friction. That reclassification is the whole
             false-positive-reduction thesis, so it is reported explicitly.

Every number is computed here; nothing is asserted. Demo cases A/B/C are real
test transactions selected to match a pattern, then re-scored in full.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from safetriage import DATA_DIR, load_policy_config
from safetriage.data.splits import load_indices, load_test_indices
from safetriage.policy import apply_capacity, expected_cost
from safetriage.scoring import score_dataframe, score_transaction

REVIEW = "STEP_UP_REVIEW"
HARD = ("SOFT_BLOCK", "ESCALATE")


def _load(dataset: str) -> pd.DataFrame:
    if dataset == "synthetic":
        return pd.read_csv(DATA_DIR / "synthetic.csv")
    return pd.read_parquet(DATA_DIR / "processed_ieee.parquet")


def _baseline_threshold(dataset: str, df: pd.DataFrame, target_recall: float) -> float:
    """Lowest threshold on the VAL split that still reaches target recall."""
    val = load_indices(dataset, "val")
    scored = score_dataframe(df.iloc[val], dataset)
    p = scored["calibrated_probability"].to_numpy()
    y = df.iloc[val]["isFraud"].to_numpy()
    grid = np.unique(np.round(np.quantile(p, np.linspace(0, 1, 400)), 6))
    best = grid[0]
    for t in grid:
        recall = ((p >= t) & (y == 1)).sum() / max(1, (y == 1).sum())
        if recall >= target_recall:
            best = t
    return float(best)


def _confusion(y: np.ndarray, actioned: np.ndarray) -> dict:
    tp = int(((actioned == 1) & (y == 1)).sum())
    fp = int(((actioned == 1) & (y == 0)).sum())
    fn = int(((actioned == 0) & (y == 1)).sum())
    tn = int(((actioned == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fdr = fp / (fp + tp) if (fp + tp) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 5), "recall": round(rec, 5),
            "false_positive_rate": round(fpr, 5), "false_discovery_rate": round(fdr, 5)}


def evaluate(dataset: str = "synthetic") -> dict:
    cfg = load_policy_config()
    target_recall = cfg["evaluation"]["baseline_target_recall"]
    df = _load(dataset)

    thr = _baseline_threshold(dataset, df, target_recall)

    test = load_test_indices(dataset)
    tdf = df.iloc[test].reset_index(drop=True)
    y = tdf["isFraud"].to_numpy()
    n = len(tdf)

    scored = apply_capacity(score_dataframe(tdf, dataset))
    p = scored["calibrated_probability"].to_numpy()
    decision = scored["decision"].to_numpy()

    # ---- baseline -------------------------------------------------------- #
    base_actioned = (p >= thr).astype(int)
    base = _confusion(y, base_actioned)
    base.update({
        "threshold": round(thr, 6),
        "block_rate": round(float(base_actioned.mean()), 5),
        "review_rate": 0.0,
        "auto_clear_rate": round(float((base_actioned == 0).mean()), 5),
        "analyst_workload_per_1k": 0.0,
        "fraud_caught_per_1k_reviewed": None,
    })

    # ---- SafeTriage ---------------------------------------------------------- #
    is_review = decision == REVIEW
    is_hard = np.isin(decision, HARD)
    is_clear = decision == "AUTO_CLEAR"
    review_catches_fraud = is_review & (y == 1)
    st_actioned = (is_hard | review_catches_fraud).astype(int)
    st = _confusion(y, st_actioned)
    reviews = int(is_review.sum())
    legit_in_review = int((is_review & (y == 0)).sum())
    st.update({
        "block_rate": round(float(is_hard.mean()), 5),
        "escalate_rate": round(float((decision == "ESCALATE").mean()), 5),
        "soft_block_rate": round(float((decision == "SOFT_BLOCK").mean()), 5),
        "review_rate": round(float(is_review.mean()), 5),
        "auto_clear_rate": round(float(is_clear.mean()), 5),
        "capacity_concessions": int(scored["capacity_concession"].sum()),
        "analyst_workload_per_1k": round(reviews / n * 1000.0, 2),
        "legit_sent_to_review": legit_in_review,
        "legit_review_friction_rate": round(legit_in_review / max(1, (y == 0).sum()), 5),
        "fraud_caught_in_review": int(review_catches_fraud.sum()),
        "fraud_caught_per_1k_reviewed": round(int(review_catches_fraud.sum()) / reviews * 1000.0, 2) if reviews else None,
    })

    # ---- cost ------------------------------------------------------------ #
    cost_base = expected_cost(missed_fraud=base["fn"], false_blocks=base["fp"], reviews=0, step_up_challenges=0, cfg=cfg)
    cost_st = expected_cost(missed_fraud=st["fn"], false_blocks=st["fp"], reviews=reviews,
                            step_up_challenges=legit_in_review, cfg=cfg)

    deltas = {
        "recall": round(st["recall"] - base["recall"], 5),
        "false_positives_abs": st["fp"] - base["fp"],
        "false_positive_rate": round(st["false_positive_rate"] - base["false_positive_rate"], 5),
        "false_positive_reduction_pct": round(
            (base["fp"] - st["fp"]) / base["fp"] * 100.0, 2) if base["fp"] else None,
        "hard_block_rate": round(st["block_rate"] - base["block_rate"], 5),
        "expected_cost": round(cost_st["total_expected_cost"] - cost_base["total_expected_cost"], 2),
    }

    result = {
        "dataset": dataset,
        "reduced_safetriage": dataset != "synthetic",
        "test_n": int(n),
        "test_fraud": int(y.sum()),
        "baseline": base,
        "safetriage": st,
        "deltas": deltas,
        "cost": {"baseline": cost_base, "safetriage": cost_st,
                 "cost_params": cfg["costs"]},
        "notes": [
            "Test split only; baseline threshold frozen from validation.",
            "STEP_UP_REVIEW outcomes use ground-truth isFraud as a proxy for analyst resolution.",
            "A legit transaction routed to review is friction/workload, not a false positive.",
        ] + (["IEEE-CIS has no behavioural-consistency fields; SafeTriage runs in REDUCED mode "
              "with C*/D* proxies flagged is_proxy."] if dataset != "synthetic" else []),
        "demo_cases": _demo_cases(dataset, tdf, scored, y, p, decision, thr),
    }
    out_path = DATA_DIR / f"evaluation_{dataset}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    return result


def _demo_cases(dataset, tdf, scored, y, p, decision, thr) -> dict:
    """Pick real test rows matching each pattern, then re-score them in full."""
    cases: dict = {}

    # A: baseline would block, SafeTriage routes to review, actually legitimate
    mask_a = (p >= thr) & (decision == REVIEW) & (y == 0)
    if mask_a.any():
        i = int(np.argmax(np.where(mask_a, p, -1)))
        cases["A_false_positive_avoided"] = _case(dataset, tdf.iloc[i], "Baseline blocks; SafeTriage reviews and clears a legitimate customer.")

    # B: genuine fraud actioned by SafeTriage (escalate > hard block > caught in review)
    for mask_b, headline in (
        ((decision == "ESCALATE") & (y == 1), "Genuine high-risk fraud escalated to a priority analyst."),
        (np.isin(decision, HARD) & (y == 1), "Genuine high-risk fraud soft-blocked pending review."),
        ((decision == REVIEW) & (y == 1), "Genuine fraud routed to step-up review and caught by the analyst."),
    ):
        if mask_b.any():
            i = int(np.argmax(np.where(mask_b, p, -1)))
            cases["B_genuine_fraud_actioned"] = _case(dataset, tdf.iloc[i], headline)
            break

    # C: normal customer auto-cleared (low risk, high consistency)
    consistency = scored["behavioral_consistency_score"].to_numpy()
    mask_c = (decision == "AUTO_CLEAR") & (y == 0) & (p < thr)
    if mask_c.any():
        score_c = np.where(mask_c, consistency - p, -9)
        i = int(np.argmax(score_c))
        cases["C_normal_customer_auto_cleared"] = _case(dataset, tdf.iloc[i], "Low risk, behaviourally consistent — auto-cleared with no analyst touch.")
    return cases


def _case(dataset: str, row: pd.Series, headline: str) -> dict:
    scored = score_transaction(row.to_dict(), dataset, with_shap=True, with_graph=(dataset == "synthetic"))
    scored["_headline"] = headline
    scored["_ground_truth_isFraud"] = int(row["isFraud"])
    if "fraud_typology" in row:
        scored["_ground_truth_typology"] = str(row["fraud_typology"])
    return scored


if __name__ == "__main__":
    import sys

    for ds in (sys.argv[1:] or ["synthetic", "ieee"]):
        r = evaluate(ds)
        b, s = r["baseline"], r["safetriage"]
        print(f"\n=== {ds} (test n={r['test_n']}, fraud={r['test_fraud']}) ===")
        print(f"  baseline   recall={b['recall']:.3f}  FP={b['fp']}  FPR={b['false_positive_rate']:.4f}  "
              f"block_rate={b['block_rate']:.3f}")
        print(f"  safetriage recall={s['recall']:.3f}  FP={s['fp']}  FPR={s['false_positive_rate']:.4f}  "
              f"block_rate={s['block_rate']:.3f}  review_rate={s['review_rate']:.3f}  "
              f"workload/1k={s['analyst_workload_per_1k']}")
        print(f"  delta      recall={r['deltas']['recall']:+.3f}  FP {b['fp']}->{s['fp']} "
              f"({r['deltas']['false_positive_reduction_pct']}% fewer)  "
              f"cost {r['cost']['baseline']['total_expected_cost']:.0f}->{r['cost']['safetriage']['total_expected_cost']:.0f}")

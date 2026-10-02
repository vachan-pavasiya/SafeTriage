"""Phase 4 — capacity sweep.

Shows how moving the analyst review capacity changes recall, false positives,
review volume, escalation rate and expected cost. Runs on the VALIDATION split
(never test). Writes data/policy_simulation_<dataset>.json and a PNG.

Run:  python scripts/policy_simulation.py [synthetic|ieee]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from safetriage import ARTIFACTS_DIR, DATA_DIR, load_policy_config
from safetriage.data.splits import load_indices
from safetriage.policy import apply_capacity, expected_cost
from safetriage.scoring import score_dataframe

REVIEW = "STEP_UP_REVIEW"
HARD = ("SOFT_BLOCK", "ESCALATE")


def _load(dataset: str) -> pd.DataFrame:
    if dataset == "synthetic":
        return pd.read_csv(DATA_DIR / "synthetic.csv")
    return pd.read_parquet(DATA_DIR / "processed_ieee.parquet")


def simulate(dataset: str = "synthetic", capacities=None) -> dict:
    cfg = load_policy_config()
    df = _load(dataset)
    val = load_indices(dataset, "val")
    vdf = df.iloc[val].reset_index(drop=True)
    y = vdf["isFraud"].to_numpy()
    n = len(vdf)

    scored = score_dataframe(vdf, dataset)  # capacity applied per-point below
    caps = capacities or [10, 20, 30, 40, 50, 60, 80, 100, 125, 150, 200]

    points = []
    for cap in caps:
        s = apply_capacity(scored, analyst_capacity_per_1k=cap)
        dec = s["decision"].to_numpy()
        is_review = dec == REVIEW
        is_hard = np.isin(dec, HARD)
        review_catch = is_review & (y == 1)
        actioned = (is_hard | review_catch).astype(int)
        tp = int(((actioned == 1) & (y == 1)).sum())
        fp = int((is_hard & (y == 0)).sum())
        fn = int(((actioned == 0) & (y == 1)).sum())
        reviews = int(is_review.sum())
        legit_review = int((is_review & (y == 0)).sum())
        cost = expected_cost(missed_fraud=fn, false_blocks=fp, reviews=reviews,
                             step_up_challenges=legit_review, cfg=cfg)
        points.append({
            "analyst_capacity_per_1k": cap,
            "recall": round(tp / max(1, (y == 1).sum()), 5),
            "true_positives": tp,
            "missed_fraud": fn,
            "legit_sent_to_review": legit_review,
            "false_positives": fp,
            "false_positive_rate": round(fp / max(1, (y == 0).sum()), 5),
            "review_volume": reviews,
            "review_rate": round(reviews / n, 5),
            "analyst_workload_per_1k": round(reviews / n * 1000.0, 2),
            "escalation_rate": round(float((dec == "ESCALATE").mean()), 5),
            "auto_clear_rate": round(float((dec == "AUTO_CLEAR").mean()), 5),
            "capacity_concessions": int(s["capacity_concession"].sum()),
            "expected_cost": cost["total_expected_cost"],
        })

    out = {"dataset": dataset, "split": "validation", "n": n, "fraud": int(y.sum()),
           "cost_params": cfg["costs"], "points": points}
    (DATA_DIR / f"policy_simulation_{dataset}.json").write_text(json.dumps(out, indent=2))
    _plot(out, ARTIFACTS_DIR / f"policy_simulation_{dataset}.png")
    return out


def _plot(out: dict, path) -> None:
    pts = out["points"]
    cap = [p["analyst_capacity_per_1k"] for p in pts]
    fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
    ax1.plot(cap, [p["recall"] for p in pts], "o-", color="#1f6f4f", label="recall")
    ax1.plot(cap, [p["false_positive_rate"] for p in pts], "s-", color="#c1440e", label="false-positive rate")
    ax1.plot(cap, [p["review_rate"] for p in pts], "^-", color="#2b6cb0", label="review rate")
    ax1.set_xlabel("analyst capacity (reviews per 1,000 transactions)")
    ax1.set_ylabel("rate")
    ax1.legend(loc="center right", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(cap, [p["expected_cost"] for p in pts], "d--", color="#6b46c1", label="expected cost")
    ax2.set_ylabel("expected cost (points)", color="#6b46c1")
    ax1.set_title(f"Capacity sweep — {out['dataset']} (validation)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["synthetic", "ieee"]):
        r = simulate(ds)
        print(f"[{ds}] capacity sweep:")
        for p in r["points"]:
            print(f"  cap={p['analyst_capacity_per_1k']:>4}  recall={p['recall']:.3f}  "
                  f"FP={p['false_positives']:>4}  reviews={p['review_volume']:>4}  "
                  f"concessions={p['capacity_concessions']:>4}  cost={p['expected_cost']:.0f}")

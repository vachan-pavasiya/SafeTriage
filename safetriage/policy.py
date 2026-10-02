"""Phase 4 — the policy engine.

The calibrated probability is ONE input here, never the decision. The engine
combines it with behavioural consistency, behavioural anomaly and the
model-disagreement proxy to choose one of four actions:

    AUTO_CLEAR       let it through, no analyst touch
    STEP_UP_REVIEW   send to an analyst queue (soft friction / evidence gather)
    SOFT_BLOCK       hold the transaction pending review (reversible)
    ESCALATE         high-confidence, strongly anomalous -> priority analyst

All thresholds come from ``config/policy_config.yaml`` (no magic numbers here).
The engine is also **capacity-aware**: :func:`apply_capacity` downgrades the
lowest-value STEP_UP_REVIEW items to AUTO_CLEAR when projected review volume
exceeds the analyst budget, and records the concession.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from safetriage import ARTIFACTS_DIR, load_policy_config, policy_version

DECISIONS = ("AUTO_CLEAR", "STEP_UP_REVIEW", "SOFT_BLOCK", "ESCALATE")
REVIEW_DECISIONS = ("STEP_UP_REVIEW", "SOFT_BLOCK", "ESCALATE")


@dataclass(frozen=True)
class PolicyThresholds:
    p_low: float
    p_high: float
    consistency_high: float
    anomaly_high: float
    uncertainty_high: float

    @classmethod
    def from_config(cls, cfg: dict | None = None, dataset: str | None = None) -> "PolicyThresholds":
        cfg = cfg or load_policy_config()
        p_low, p_high = cfg["probability"]["p_low"], cfg["probability"]["p_high"]
        if dataset:
            derived = ARTIFACTS_DIR / f"policy_thresholds_{dataset}.json"
            if derived.exists():
                d = json.loads(derived.read_text())
                p_low, p_high = d["p_low"], d["p_high"]
        return cls(
            p_low=p_low,
            p_high=p_high,
            consistency_high=cfg["behaviour"]["consistency_high"],
            anomaly_high=cfg["behaviour"]["anomaly_high"],
            uncertainty_high=cfg["uncertainty"]["uncertainty_high"],
        )


def decide(p: float, consistency: float, anomaly: float, uncertainty: float,
           th: PolicyThresholds | None = None) -> dict:
    """Single capacity-agnostic decision. Returns decision + firing rule + text."""
    th = th or PolicyThresholds.from_config()

    # rule priority order — first match wins
    if uncertainty >= th.uncertainty_high and p >= th.p_low:
        d, rule = "STEP_UP_REVIEW", "MODEL_DISAGREEMENT"
        why = (f"Models disagree materially (uncertainty {uncertainty:.2f} ≥ "
               f"{th.uncertainty_high}) at non-trivial risk (p {p:.2f}); route to a human.")
    elif p >= th.p_high and anomaly >= th.anomaly_high:
        d, rule = "ESCALATE", "HIGH_RISK_HIGH_ANOMALY"
        why = (f"High calibrated risk (p {p:.2f} ≥ {th.p_high}) AND behaviour strongly "
               f"anomalous (anomaly {anomaly:.2f} ≥ {th.anomaly_high}); priority analyst.")
    elif p >= th.p_high and consistency >= th.consistency_high:
        d, rule = "SOFT_BLOCK", "HIGH_RISK_BEHAVIOURALLY_CONSISTENT"
        why = (f"High model risk (p {p:.2f}) but behaviour matches the customer "
               f"(consistency {consistency:.2f} ≥ {th.consistency_high}); hold for review, "
               f"not a blind hard block.")
    elif p >= th.p_high:
        d, rule = "SOFT_BLOCK", "HIGH_RISK_DEFAULT"
        why = f"High calibrated risk (p {p:.2f} ≥ {th.p_high}); hold pending review."
    elif p >= th.p_low:
        if consistency >= th.consistency_high and anomaly < th.anomaly_high and uncertainty < th.uncertainty_high:
            d, rule = "AUTO_CLEAR", "MEDIUM_RISK_BEHAVIOURALLY_CONSISTENT"
            why = (f"Medium risk (p {p:.2f}) but consistent behaviour, models agree; "
                   f"clear without an analyst.")
        else:
            d, rule = "STEP_UP_REVIEW", "MEDIUM_RISK_REVIEW"
            why = f"Medium risk (p {p:.2f}) with elevated anomaly/uncertainty; step-up review."
    else:  # p < p_low
        if anomaly >= th.anomaly_high and uncertainty < th.uncertainty_high:
            d, rule = "STEP_UP_REVIEW", "LOW_RISK_BUT_ANOMALOUS"
            why = (f"Low model risk (p {p:.2f}) but strongly anomalous behaviour "
                   f"(anomaly {anomaly:.2f}); review to catch model false negatives.")
        else:
            d, rule = "AUTO_CLEAR", "LOW_RISK_CONSISTENT"
            why = f"Low calibrated risk (p {p:.2f} < {th.p_low}) and no strong anomaly; auto-clear."

    return {"decision": d, "policy_rule": rule, "decision_explanation": why,
            "policy_version": policy_version()}


def decide_frame(df: pd.DataFrame, cfg: dict | None = None, dataset: str | None = None) -> pd.DataFrame:
    """Vectorised :func:`decide` over columns
    ``calibrated_probability, behavioural_consistency_score,
    behavioural_anomaly_score, uncertainty``.
    """
    th = PolicyThresholds.from_config(cfg, dataset=dataset)
    recs = [
        decide(row.calibrated_probability, row.behavioural_consistency_score,
               row.behavioural_anomaly_score, row.uncertainty, th)
        for row in df.itertuples(index=False)
    ]
    return pd.DataFrame(recs, index=df.index)


# --------------------------------------------------------------------------- #
# Capacity model
# --------------------------------------------------------------------------- #
def apply_capacity(df: pd.DataFrame, *, analyst_capacity_per_1k: float | None = None,
                   headroom_fraction: float | None = None, cfg: dict | None = None) -> pd.DataFrame:
    """Downgrade the lowest-value STEP_UP_REVIEW rows to AUTO_CLEAR when the
    projected review volume exceeds the analyst budget.

    ``df`` needs columns ``decision`` and ``calibrated_probability`` (and,
    optionally, ``behavioural_anomaly_score`` for tie-breaking). SOFT_BLOCK and
    ESCALATE are never downgraded. Returns a copy with ``decision`` updated and a
    boolean ``capacity_concession`` column.
    """
    cfg = cfg or load_policy_config()
    cap = cfg["capacity"]["analyst_capacity_per_1k"] if analyst_capacity_per_1k is None else analyst_capacity_per_1k
    headroom = cfg["capacity"]["headroom_fraction"] if headroom_fraction is None else headroom_fraction

    out = df.copy()
    out["capacity_concession"] = False
    n = len(out)
    if n == 0:
        return out

    budget = int(round(cap * n / 1000.0 * (1.0 - headroom)))
    forced = out["decision"].isin(["SOFT_BLOCK", "ESCALATE"]).sum()
    review_budget = max(0, budget - int(forced))

    step_idx = out.index[out["decision"] == "STEP_UP_REVIEW"]
    if len(step_idx) <= review_budget:
        return out

    # keep the highest-value reviews: rank by prob then anomaly, drop the rest
    rank_cols = ["calibrated_probability"]
    for anom in ("behavioural_anomaly_score", "behavioral_anomaly_score"):
        if anom in out.columns:
            rank_cols.append(anom)
            break
    order = out.loc[step_idx].sort_values(rank_cols, ascending=False).index
    drop = order[review_budget:]
    out.loc[drop, "decision"] = "AUTO_CLEAR"
    out.loc[drop, "capacity_concession"] = True
    out.loc[drop, "policy_rule"] = "CAPACITY_CONCESSION"
    return out


# --------------------------------------------------------------------------- #
# Cost model
# --------------------------------------------------------------------------- #
def expected_cost(*, missed_fraud: int, false_blocks: int, reviews: int,
                  step_up_challenges: int, cfg: dict | None = None) -> dict:
    """Relative expected cost from a confusion summary + volume counts."""
    c = (cfg or load_policy_config())["costs"]
    total = (missed_fraud * c["fraud_loss_cost"]
             + false_blocks * c["false_positive_cost"]
             + reviews * c["review_cost"]
             + step_up_challenges * c["customer_friction_cost"])
    return {
        "missed_fraud_cost": round(missed_fraud * c["fraud_loss_cost"], 2),
        "false_block_cost": round(false_blocks * c["false_positive_cost"], 2),
        "review_cost": round(reviews * c["review_cost"], 2),
        "friction_cost": round(step_up_challenges * c["customer_friction_cost"], 2),
        "total_expected_cost": round(total, 2),
    }

"""Phase 5 — "Next Best Evidence" (deterministic, template-based).

For every case the policy engine sends to a human (STEP_UP_REVIEW / SOFT_BLOCK /
ESCALATE) this returns an ordered checklist of what an analyst should pull next,
derived by fixed rules from the transaction's own fields and reason code.

**Prototype evidence recommendation — no LLM, no agent, no learning.** A
production version would rank these against historical analyst resolutions.
"""

from __future__ import annotations

TEMPLATES = {
    "new_beneficiary": {
        "evidence": "Beneficiary 90-day history (other payers, account age, name match)",
        "rationale": "Beneficiary is new or was added minutes before this payment.",
        "source": "payments_ledger",
    },
    "new_device": {
        "evidence": "Device fingerprint reuse across accounts + geo of first-seen",
        "rationale": "Authentication came from a device not seen on this account before.",
        "source": "device_intelligence",
    },
    "password_reset": {
        "evidence": "Auth timeline: password reset, MFA change, and login IPs in last 72h",
        "rationale": "A recent credential reset precedes this transaction (ATO pattern).",
        "source": "identity_events",
    },
    "velocity": {
        "evidence": "All transactions by this account in the last 1h / 24h with amounts",
        "rationale": "Transaction velocity is above this customer's normal range.",
        "source": "transaction_stream",
    },
    "amount": {
        "evidence": "Customer's trailing 6-month amount distribution vs this amount",
        "rationale": "Amount is well outside the customer's historical pattern.",
        "source": "customer_profile",
    },
    "graph": {
        "evidence": "Shared-device / shared-beneficiary neighbours and any prior fraud flags",
        "rationale": "Transaction is close (in the proxy graph) to a flagged or mule account.",
        "source": "graph_proxy",
    },
    "disagreement": {
        "evidence": "Side-by-side XGBoost vs Random-Forest feature attributions",
        "rationale": "The two models disagree; a human tie-breaker is needed.",
        "source": "model_explainer",
    },
    "prior_decisions": {
        "evidence": "Past analyst decisions on this customer / beneficiary and their outcomes",
        "rationale": "Prior human context reduces repeat review effort.",
        "source": "case_history",
    },
}


def recommend(raw: dict, *, reason_code: str, decision: str, uncertainty: float = 0.0) -> list[dict]:
    if decision == "AUTO_CLEAR":
        return []

    keys: list[str] = []
    if float(raw.get("is_new_beneficiary", 0)) >= 0.5 or float(raw.get("beneficiary_added_minutes_ago", 1e9)) < 240:
        keys.append("new_beneficiary")
    if float(raw.get("is_new_device", 0)) >= 0.5:
        keys.append("new_device")
    if float(raw.get("recent_password_reset", 0)) >= 0.5:
        keys.append("password_reset")
    if float(raw.get("velocity_1h_count", 0)) >= 3 or float(raw.get("velocity_24h_count", 0)) >= 8:
        keys.append("velocity")
    if float(raw.get("amount_deviation_ratio", 1.0)) >= 2.0:
        keys.append("amount")
    if float(raw.get("shared_device_with_flagged_account", 0)) >= 0.5 or float(raw.get("hop_distance_to_known_mule", 99)) <= 3:
        keys.append("graph")
    if reason_code == "MODEL_DISAGREEMENT_REVIEW" or uncertainty >= 0.5:
        keys.append("disagreement")
    keys.append("prior_decisions")  # always useful, ranked last

    seen, out = set(), []
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        item = dict(TEMPLATES[k])
        item["key"] = k
        out.append(item)
    return out

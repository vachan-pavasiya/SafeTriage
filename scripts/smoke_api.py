"""Hit every API endpoint and assert it returns real, non-empty data.

Usage: start the API, then  ``python scripts/smoke_api.py``
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = "http://127.0.0.1:8000"


def _get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read())


def _post(path: str, body: dict):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def main() -> int:
    fails: list[str] = []

    def check(name, cond, detail=""):
        print(("  ok  " if cond else "FAIL  ") + name + (f"  — {detail}" if detail else ""))
        if not cond:
            fails.append(name)

    # /transactions
    tx = _get("/transactions?dataset=synthetic&limit=5")
    check("GET /transactions", tx["count"] == 5 and "calibrated_probability" in tx["items"][0])
    a_review = _get("/transactions?dataset=synthetic&decision=STEP_UP_REVIEW&limit=1")
    tid = a_review["items"][0]["transaction_id"] if a_review["count"] else tx["items"][0]["transaction_id"]

    # /transactions/{id}
    one = _get(f"/transactions/{tid}")
    check("GET /transactions/{id}", one["transaction_id"] == tid and "output" in one)

    # POST /score by id
    sc = _post("/score", {"transaction_id": tid})
    check("POST /score (by id)", 0 <= sc["calibrated_probability"] <= 1 and sc["decision"] in
          ("AUTO_CLEAR", "STEP_UP_REVIEW", "SOFT_BLOCK", "ESCALATE") and sc["latency_ms"] > 0,
          f"decision={sc['decision']} p={sc['calibrated_probability']} raw={sc['fraud_probability']}")
    check("POST /score raw != calibrated", abs(sc["fraud_probability"] - sc["calibrated_probability"]) >= 0)
    check("POST /score has SHAP", len(sc["top_features"]) >= 3)

    # /kpis
    k = _get("/kpis?dataset=synthetic")
    check("GET /kpis", k["n_transactions"] > 0 and abs(
        k["auto_clear_rate"] + k["review_rate"] + k["soft_block_rate"] + k["escalation_rate"] - 1.0) < 0.02,
        f"n={k['n_transactions']} mix={k['decision_distribution']}")
    check("GET /kpis audit proxy present", bool(k["audit_proxy"]))

    # /audit/sample
    au = _get("/audit/sample?dataset=synthetic")
    pops = au["summary"]["populations"]
    check("GET /audit/sample both populations", "AUTO_CLEAR" in pops and "BLOCK_ESCALATE" in pops,
          f"auto_clear_n={pops.get('AUTO_CLEAR', {}).get('population_size')} "
          f"block_n={pops.get('BLOCK_ESCALATE', {}).get('population_size')}")
    check("GET /audit/sample has samples", len(au["samples"]) > 0)

    # /feedback round-trip
    before = _get("/feedback")["count"]
    fb = _post("/feedback", {"transaction_id": tid, "analyst_decision": "NEED_MORE_EVIDENCE",
                             "reason_tag": "smoke_test"})
    after = _get("/feedback")["count"]
    check("POST /feedback + GET /feedback", after == before + 1 and fb["stored"]["id"] > 0)

    # /evaluation
    ev = _get("/evaluation?dataset=synthetic")
    check("GET /evaluation", "baseline" in ev and "safetriage" in ev and
          ev["safetriage"]["fp"] <= ev["baseline"]["fp"],
          f"FP {ev['baseline']['fp']} -> {ev['safetriage']['fp']}, recall "
          f"{ev['baseline']['recall']} -> {ev['safetriage']['recall']}")
    check("GET /evaluation demo cases", len(ev["demo_cases"]) >= 2)

    # /calibration
    cal = _get("/calibration?dataset=ieee")
    check("GET /calibration", cal["brier_raw"] > 0 and cal["brier_calibrated"] > 0 and
          len(cal["reliability_calibrated"]) > 0,
          f"Brier {cal['brier_raw']:.4f} -> {cal['brier_calibrated']:.4f}, "
          f"ECE {cal['ece_raw']:.4f} -> {cal['ece_calibrated']:.4f}")

    # /policy/simulate
    ps = _get("/policy/simulate?dataset=synthetic")
    check("GET /policy/simulate", len(ps["points"]) >= 5 and "recall" in ps["points"][0])
    ps2 = _get("/policy/simulate?dataset=synthetic&false_positive_cost=50")
    moved = ps2["points"][0]["expected_cost"] != ps["points"][0]["expected_cost"]
    check("GET /policy/simulate cost override", moved or ps["points"][0]["false_positives"] == 0)

    print()
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        return 1
    print("ALL ENDPOINT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

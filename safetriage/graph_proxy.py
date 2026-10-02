"""Phase 5 — illustrative graph view (NOT a real graph model).

This builds a small node/edge structure per transaction from two synthetic
columns — ``shared_device_with_flagged_account`` and
``hop_distance_to_known_mule`` — so the dashboard can draw a mule-network-style
picture. It is **illustrative only**: there is no graph database, no GraphSAGE,
no message passing. Real graph analysis is listed under FUTURE work.
"""

from __future__ import annotations

import hashlib


def _nid(*parts: object) -> str:
    return "n_" + hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8]


def transaction_graph(raw: dict) -> dict:
    """Return {nodes: [...], edges: [...], illustrative: True, note: str}."""
    acct = _nid("account", raw.get("card_id", "acct"))
    benef = _nid("beneficiary", raw.get("beneficiary_id", "benef"))
    device = _nid("device", raw.get("DeviceInfo", "dev"), raw.get("DeviceType", ""))

    nodes = [
        {"id": acct, "label": str(raw.get("card_id", "account")), "type": "account", "risk": "subject"},
        {"id": benef, "label": str(raw.get("beneficiary_id", "beneficiary")), "type": "beneficiary",
         "risk": "new" if float(raw.get("is_new_beneficiary", 0)) >= 0.5 else "known"},
        {"id": device, "label": f"{raw.get('DeviceType', 'device')} / {raw.get('DeviceInfo', '')}".strip(" /"),
         "type": "device", "risk": "new" if float(raw.get("is_new_device", 0)) >= 0.5 else "known"},
    ]
    edges = [
        {"source": acct, "target": device, "label": "authenticated_from"},
        {"source": acct, "target": benef, "label": "paid"},
    ]

    if float(raw.get("shared_device_with_flagged_account", 0)) >= 0.5:
        flagged = _nid("flagged", raw.get("card_id", "acct"))
        nodes.append({"id": flagged, "label": "flagged account", "type": "account", "risk": "flagged"})
        edges.append({"source": flagged, "target": device, "label": "also_used_device"})

    hop = float(raw.get("hop_distance_to_known_mule", 99))
    if hop <= 5:
        prev = benef
        for h in range(1, int(hop) + 1):
            step = _nid("hop", raw.get("beneficiary_id", "b"), h)
            is_mule = h == int(hop)
            nodes.append({"id": step, "label": "known mule" if is_mule else f"intermediary {h}",
                          "type": "account", "risk": "mule" if is_mule else "intermediary"})
            edges.append({"source": prev, "target": step, "label": "transfers_to"})
            prev = step

    return {
        "nodes": nodes,
        "edges": edges,
        "hop_distance_to_known_mule": None if hop >= 99 else int(hop),
        "shared_device_with_flagged_account": bool(float(raw.get("shared_device_with_flagged_account", 0)) >= 0.5),
        "illustrative": True,
        "note": "Illustrative proxy from two synthetic columns — not a graph DB or GraphSAGE.",
    }

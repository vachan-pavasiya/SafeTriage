"""Phase 6/7 — SQLite persistence (no external services).

Tables
------
transactions   one row per scored transaction (scores + decision + ground-truth
               label + split + full JSON payloads)
feedback       analyst decisions on review cases (the future-recalibration set)
audit_runs     one self-audit pass
audit_samples  the sampled transactions in a run (AUTO_CLEAR + BLOCK/ESCALATE
               populations, risk-stratified + random-control)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable

from safetriage import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id            TEXT PRIMARY KEY,
    dataset                   TEXT NOT NULL,
    split                     TEXT,
    created_at                TEXT NOT NULL,
    fraud_probability         REAL,
    calibrated_probability    REAL,
    behavioural_consistency_score REAL,
    behavioural_anomaly_score REAL,
    uncertainty               REAL,
    confidence_score          REAL,
    decision                  TEXT,
    policy_rule               TEXT,
    predicted_typology        TEXT,
    fraud_reason_code         TEXT,
    capacity_concession       INTEGER DEFAULT 0,
    is_fraud_label            INTEGER,
    model_version             TEXT,
    policy_version            TEXT,
    latency_ms                REAL,
    raw_json                  TEXT,
    output_json               TEXT
);
CREATE INDEX IF NOT EXISTS ix_txn_decision ON transactions(decision);
CREATE INDEX IF NOT EXISTS ix_txn_split ON transactions(split);
CREATE INDEX IF NOT EXISTS ix_txn_dataset ON transactions(dataset);

CREATE TABLE IF NOT EXISTS feedback (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id            TEXT NOT NULL,
    analyst_decision          TEXT NOT NULL,
    reason_tag                TEXT,
    note                      TEXT,
    original_model_output     TEXT,
    original_policy_decision  TEXT,
    created_at                TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_fb_txn ON feedback(transaction_id);

CREATE TABLE IF NOT EXISTS audit_runs (
    run_id                    TEXT PRIMARY KEY,
    dataset                   TEXT NOT NULL,
    created_at                TEXT NOT NULL,
    summary_json              TEXT
);

CREATE TABLE IF NOT EXISTS audit_samples (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                    TEXT NOT NULL,
    population                TEXT NOT NULL,     -- AUTO_CLEAR | BLOCK_ESCALATE
    sample_type               TEXT NOT NULL,     -- risk_stratified | random_control
    transaction_id            TEXT NOT NULL,
    calibrated_probability    REAL,
    decision                  TEXT,
    audit_proxy_label         INTEGER,           -- ground-truth isFraud used as a proxy
    is_proxy                  INTEGER DEFAULT 1,
    created_at                TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_audit_run ON audit_samples(run_id);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(reset: bool = False) -> None:
    if reset and DB_PATH.exists():
        DB_PATH.unlink()
    with connect() as conn:
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------------- #
# transactions
# --------------------------------------------------------------------------- #
_TXN_COLS = [
    "transaction_id", "dataset", "split", "created_at", "fraud_probability",
    "calibrated_probability", "behavioural_consistency_score", "behavioural_anomaly_score",
    "uncertainty", "confidence_score", "decision", "policy_rule", "predicted_typology",
    "fraud_reason_code", "capacity_concession", "is_fraud_label", "model_version",
    "policy_version", "latency_ms", "raw_json", "output_json",
]


def upsert_transactions(rows: Iterable[dict]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    placeholders = ",".join("?" for _ in _TXN_COLS)
    sql = (f"INSERT OR REPLACE INTO transactions ({','.join(_TXN_COLS)}) "
           f"VALUES ({placeholders})")
    payload = [tuple(r.get(c) for c in _TXN_COLS) for r in rows]
    with connect() as conn:
        conn.executemany(sql, payload)
    return len(payload)


def fetch_transactions(*, decision: str | None = None, split: str | None = None,
                       dataset: str | None = None, limit: int = 100, offset: int = 0,
                       order_by: str = "calibrated_probability", desc: bool = True) -> list[dict]:
    where, params = [], []
    for col, val in (("decision", decision), ("split", split), ("dataset", dataset)):
        if val:
            where.append(f"{col} = ?")
            params.append(val)
    allowed = {"calibrated_probability", "uncertainty", "behavioural_anomaly_score", "created_at", "latency_ms"}
    order_by = order_by if order_by in allowed else "calibrated_probability"
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    sql = (f"SELECT * FROM transactions {clause} "
           f"ORDER BY {order_by} {'DESC' if desc else 'ASC'} LIMIT ? OFFSET ?")
    with connect() as conn:
        cur = conn.execute(sql, (*params, limit, offset))
        return [dict(r) for r in cur.fetchall()]


def get_transaction(transaction_id: str) -> dict | None:
    with connect() as conn:
        r = conn.execute("SELECT * FROM transactions WHERE transaction_id = ?", (transaction_id,)).fetchone()
    return dict(r) if r else None


def count_transactions(dataset: str | None = None) -> int:
    with connect() as conn:
        if dataset:
            return conn.execute("SELECT COUNT(*) FROM transactions WHERE dataset = ?", (dataset,)).fetchone()[0]
        return conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]


# --------------------------------------------------------------------------- #
# feedback
# --------------------------------------------------------------------------- #
VALID_FEEDBACK = {"MARK_LEGITIMATE", "CONFIRM_FRAUD", "NEED_MORE_EVIDENCE"}


def insert_feedback(transaction_id: str, analyst_decision: str, *, reason_tag: str | None = None,
                    note: str | None = None) -> dict:
    if analyst_decision not in VALID_FEEDBACK:
        raise ValueError(f"analyst_decision must be one of {sorted(VALID_FEEDBACK)}")
    txn = get_transaction(transaction_id)
    if txn is None:
        raise KeyError(f"unknown transaction_id {transaction_id!r}")
    row = {
        "transaction_id": transaction_id,
        "analyst_decision": analyst_decision,
        "reason_tag": reason_tag,
        "note": note,
        "original_model_output": txn.get("output_json"),
        "original_policy_decision": txn.get("decision"),
        "created_at": now_iso(),
    }
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO feedback (transaction_id, analyst_decision, reason_tag, note, "
            "original_model_output, original_policy_decision, created_at) VALUES "
            "(:transaction_id,:analyst_decision,:reason_tag,:note,:original_model_output,"
            ":original_policy_decision,:created_at)", row,
        )
        row["id"] = cur.lastrowid
    return row


def fetch_feedback(limit: int = 200) -> list[dict]:
    with connect() as conn:
        cur = conn.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def insert_audit_run(run_id: str, dataset: str, summary: dict, samples: list[dict]) -> None:
    ts = now_iso()
    with connect() as conn:
        conn.execute("INSERT OR REPLACE INTO audit_runs (run_id, dataset, created_at, summary_json) VALUES (?,?,?,?)",
                     (run_id, dataset, ts, json.dumps(summary)))
        conn.executemany(
            "INSERT INTO audit_samples (run_id, population, sample_type, transaction_id, "
            "calibrated_probability, decision, audit_proxy_label, is_proxy, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            [(run_id, s["population"], s["sample_type"], s["transaction_id"],
              s.get("calibrated_probability"), s.get("decision"), s.get("audit_proxy_label"),
              int(s.get("is_proxy", 1)), ts) for s in samples],
        )


def latest_audit_run(dataset: str | None = None) -> dict | None:
    with connect() as conn:
        q = "SELECT * FROM audit_runs"
        p: tuple[Any, ...] = ()
        if dataset:
            q += " WHERE dataset = ?"
            p = (dataset,)
        q += " ORDER BY created_at DESC LIMIT 1"
        run = conn.execute(q, p).fetchone()
        if run is None:
            return None
        samples = conn.execute("SELECT * FROM audit_samples WHERE run_id = ?", (run["run_id"],)).fetchall()
    d = dict(run)
    d["summary"] = json.loads(d.pop("summary_json") or "{}")
    d["samples"] = [dict(s) for s in samples]
    return d

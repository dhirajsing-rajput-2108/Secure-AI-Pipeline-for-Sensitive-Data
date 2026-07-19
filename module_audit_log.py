"""
module_audit_log.py
===================
Tamper-evident audit log for the Secure AI Pipeline.

Design
------
Every event is stored as one row in a SQLite table. Rows form a hash chain:

    this_hash = SHA256( canonical_json([prev_hash, ts, actor, role, action, payload_json]) )

The first (genesis) row uses prev_hash = "0"*64. Any modification, deletion,
insertion, or reordering after the fact breaks the chain and the verifier
pinpoints the exact break.

Threat model
------------
Attacker has read + write access to the SQLite file. Attacker does NOT know
any secret. Because the chain is fully deterministic from the (visible) row
contents, an attacker who edits a middle row must also recompute every
subsequent this_hash — the verifier catches this by re-walking the chain
from row 1. If any hash does not match its recomputation, the verifier
returns intact=False and reports the first broken row.

What this does NOT defend against
---------------------------------
* An attacker who can also modify this Python code can rewrite the hash
  algorithm. The audit log is not a substitute for filesystem/binary
  integrity controls.
* Total deletion of the SQLite file: the verifier reports zero events
  but has no proof there ever were more. Off-site backups + periodic
  external attestation (e.g. anchoring the head hash into an external
  system) would address this — noted as Future Work in the report.

Public API
----------
* init_audit_log(db_path, reset=False)
* append_event(db_path, actor, role, action, payload) -> int
* list_events(db_path, limit=None) -> list[AuditEvent]
* tail(db_path, n=50) -> list[AuditEvent]
* verify_chain(db_path) -> ChainVerdict
* AuditEvent, ChainVerdict dataclasses

CLI
---
    python module_audit_log.py init
    python module_audit_log.py append --actor alice --role analyst \\
        --action predict --payload '{"model": "config_1_baseline"}'
    python module_audit_log.py tail --n 10
    python module_audit_log.py verify
    python module_audit_log.py tamper-demo

Author: Rajput Dhirajsing Rajeshsing  (2024DA04378)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GENESIS_HASH: str = "0" * 64           # prev_hash for the first row
HASH_LENGTH: int = 64                  # SHA-256 hex digest length
DEFAULT_DB_PATH: str = "artifacts/audit.db"

# Fixed set of roles the auth system may emit; not enforced here, but callers
# should respect the taxonomy so downstream reports are consistent.
CANONICAL_ROLES: tuple[str, ...] = ("analyst", "admin", "system")

# Module-level write lock. SQLite already serialises writers within a single
# process, but this lock also serialises the (read prev_hash, compute hash,
# insert) *sequence* to prevent two threads producing rows with the same
# prev_hash — which the verifier would flag as a broken chain even though
# both writes succeeded.
_APPEND_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AuditEvent:
    id: int
    ts: str                 # ISO 8601 UTC with microseconds
    actor: str
    role: str
    action: str
    payload_json: str       # canonical JSON string
    prev_hash: str
    this_hash: str

    @property
    def payload(self) -> Any:
        return json.loads(self.payload_json)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChainVerdict:
    intact: bool
    n_events: int
    broken_at_event_id: int | None = None
    broken_at_row_index: int | None = None      # 1-based row index, for report readability
    reason: str | None = None
    expected_hash: str | None = None            # what we computed
    stored_hash: str | None = None              # what the DB claims
    head_hash: str | None = None                # last row's this_hash (if chain intact)
    examples: list[dict] = field(default_factory=list)

    def pretty(self) -> str:
        lines = [
            "Audit chain verifier",
            f"  events on record         : {self.n_events:,}",
            f"  chain intact             : {'YES ✓' if self.intact else 'NO ✗'}",
        ]
        if self.intact:
            lines.append(f"  head hash                : {self.head_hash}")
        else:
            lines.append(f"  first broken event id    : {self.broken_at_event_id}")
            lines.append(f"  first broken row index   : {self.broken_at_row_index}")
            lines.append(f"  reason                   : {self.reason}")
            if self.expected_hash and self.stored_hash:
                lines.append(f"  expected this_hash       : {self.expected_hash}")
                lines.append(f"  stored   this_hash       : {self.stored_hash}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Canonical serialisation + hashing
# ---------------------------------------------------------------------------

def _canonical_json(obj: Any) -> str:
    """Deterministic JSON encoding used for hashing.

    * Keys sorted alphabetically
    * No whitespace (compact separators)
    * Non-JSON-serialisable objects fall through str() (rare in this codebase)
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _compute_row_hash(
    prev_hash: str,
    ts: str,
    actor: str,
    role: str,
    action: str,
    payload_json: str,
) -> str:
    """Return this_hash for a row with the given field values.

    The input to SHA-256 is a canonical JSON serialisation of a fixed-order
    tuple. Using JSON of a list (rather than string concatenation with a
    delimiter) is unambiguous — no field can 'inject' delimiter characters
    to collide with another row.
    """
    canonical = _canonical_json([prev_hash, ts, actor, role, action, payload_json])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    """ISO 8601 UTC timestamp with microseconds. Deterministic string form."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS audit_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT NOT NULL,
    actor          TEXT NOT NULL,
    role           TEXT NOT NULL,
    action         TEXT NOT NULL,
    payload_json   TEXT NOT NULL,
    prev_hash      TEXT NOT NULL,
    this_hash      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_events_ts ON audit_events(ts);
"""


def _connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with sensible defaults for the audit log."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")     # better concurrency
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_audit_log(db_path: str | Path = DEFAULT_DB_PATH, *, reset: bool = False) -> None:
    """Create the audit_events table if it does not exist.

    Parameters
    ----------
    db_path :
        SQLite file path. Parent directory is created if missing.
    reset :
        If True, delete the file first — useful for the demo script and
        the tamper-demo CLI, never for production.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if reset and db_path.exists():
        db_path.unlink()
    conn = _connect(db_path)
    try:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def append_event(
    db_path: str | Path,
    *,
    actor: str,
    role: str,
    action: str,
    payload: Any = None,
    ts: str | None = None,
) -> int:
    """Append one event to the log; return its assigned id.

    Thread-safe: an internal lock serialises the (read tail, compute, insert)
    sequence so concurrent writers cannot both attach to the same prev_hash.

    Parameters
    ----------
    payload :
        Any JSON-serialisable object. Non-JSON values fall through str().
        `None` becomes {}.
    ts :
        Override the timestamp. Only useful in tests; defaults to now.
    """
    if not actor:
        raise ValueError("actor must be a non-empty string")
    if not role:
        raise ValueError("role must be a non-empty string")
    if not action:
        raise ValueError("action must be a non-empty string")
    if payload is None:
        payload = {}
    payload_json = _canonical_json(payload)
    ts = ts or _now_iso()

    with _APPEND_LOCK:
        conn = _connect(db_path)
        try:
            cur = conn.cursor()
            # Read the previous row's this_hash (or genesis if empty).
            cur.execute(
                "SELECT this_hash FROM audit_events ORDER BY id DESC LIMIT 1"
            )
            row = cur.fetchone()
            prev_hash = row[0] if row else GENESIS_HASH

            this_hash = _compute_row_hash(
                prev_hash=prev_hash,
                ts=ts,
                actor=actor,
                role=role,
                action=action,
                payload_json=payload_json,
            )
            cur.execute(
                """INSERT INTO audit_events
                   (ts, actor, role, action, payload_json, prev_hash, this_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (ts, actor, role, action, payload_json, prev_hash, this_hash),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()


def list_events(
    db_path: str | Path,
    *,
    limit: int | None = None,
    offset: int = 0,
    order: str = "ASC",
) -> list[AuditEvent]:
    """Return audit events ordered by id.

    order='ASC'  (default) — oldest first, matches chain traversal
    order='DESC'          — newest first, useful for UIs
    """
    if order.upper() not in ("ASC", "DESC"):
        raise ValueError(f"order must be ASC or DESC, got {order!r}")
    sql = (
        f"SELECT id, ts, actor, role, action, payload_json, prev_hash, this_hash "
        f"FROM audit_events ORDER BY id {order.upper()}"
    )
    if limit is not None:
        sql += f" LIMIT {int(limit)} OFFSET {int(offset)}"

    conn = _connect(db_path)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchall()
    finally:
        conn.close()
    return [
        AuditEvent(
            id=r[0], ts=r[1], actor=r[2], role=r[3], action=r[4],
            payload_json=r[5], prev_hash=r[6], this_hash=r[7],
        )
        for r in rows
    ]


def tail(db_path: str | Path, n: int = 50) -> list[AuditEvent]:
    """Return the most recent `n` events (newest last, for readability)."""
    recent = list_events(db_path, limit=n, order="DESC")
    return list(reversed(recent))


def verify_chain(db_path: str | Path) -> ChainVerdict:
    """Walk the entire chain and confirm every this_hash matches its
    recomputation. Detects modification, deletion, reordering, or injection.
    """
    events = list_events(db_path, order="ASC")
    n = len(events)

    if n == 0:
        return ChainVerdict(intact=True, n_events=0, head_hash=None)

    expected_prev = GENESIS_HASH
    for row_index, ev in enumerate(events, start=1):
        # (a) The stored prev_hash must match the chain's running head.
        if ev.prev_hash != expected_prev:
            return ChainVerdict(
                intact=False,
                n_events=n,
                broken_at_event_id=ev.id,
                broken_at_row_index=row_index,
                reason=(
                    f"prev_hash mismatch: stored {ev.prev_hash[:12]}..., "
                    f"expected {expected_prev[:12]}..."
                ),
                expected_hash=expected_prev,
                stored_hash=ev.prev_hash,
            )

        # (b) The stored this_hash must equal the recomputation.
        recomputed = _compute_row_hash(
            prev_hash=ev.prev_hash,
            ts=ev.ts,
            actor=ev.actor,
            role=ev.role,
            action=ev.action,
            payload_json=ev.payload_json,
        )
        if recomputed != ev.this_hash:
            return ChainVerdict(
                intact=False,
                n_events=n,
                broken_at_event_id=ev.id,
                broken_at_row_index=row_index,
                reason=(
                    "this_hash does not match recomputation — row content "
                    "has been altered since insertion"
                ),
                expected_hash=recomputed,
                stored_hash=ev.this_hash,
            )
        expected_prev = ev.this_hash

    return ChainVerdict(intact=True, n_events=n, head_hash=events[-1].this_hash)


# ---------------------------------------------------------------------------
# Helper: pretty-print a row for the CLI / demo
# ---------------------------------------------------------------------------

def _fmt_event(ev: AuditEvent, payload_max: int = 80) -> str:
    payload_str = ev.payload_json
    if len(payload_str) > payload_max:
        payload_str = payload_str[: payload_max - 3] + "..."
    return (
        f"  #{ev.id:<4} {ev.ts}  [{ev.role:<7}] {ev.actor:<12} "
        f"{ev.action:<20} {payload_str}  hash={ev.this_hash[:12]}..."
    )


# ---------------------------------------------------------------------------
# Tamper demo (used both by the CLI subcommand and by the demo script)
# ---------------------------------------------------------------------------

def run_tamper_demo(db_path: str | Path, *, reset: bool = True) -> ChainVerdict:
    """End-to-end scripted demo:
      1. reset the DB, append 3 events
      2. verify -> intact
      3. mutate row 2's action string via raw SQL (simulating an attacker)
      4. verify -> broken at row 2

    Returns the *post-tamper* verdict so the caller can print it.
    """
    if reset:
        init_audit_log(db_path, reset=True)
    else:
        init_audit_log(db_path)

    print("[tamper-demo] appending 3 legitimate events...")
    append_event(db_path, actor="alice",  role="analyst", action="login",   payload={"ip": "127.0.0.1"})
    append_event(db_path, actor="alice",  role="analyst", action="predict", payload={"config": "config_1_baseline", "prob": 0.234})
    append_event(db_path, actor="bob",    role="admin",   action="audit_verify", payload={})

    v = verify_chain(db_path)
    print("[tamper-demo] pre-tamper verdict:")
    print(v.pretty())
    if not v.intact:
        raise RuntimeError("Fresh chain unexpectedly broken — cannot proceed.")

    # ---- attack: mutate row 2's action outside of the append API --------
    print("\n[tamper-demo] MUTATING row 2's action field via raw SQL "
          "(simulating a filesystem attacker)...")
    conn = _connect(db_path)
    try:
        conn.execute(
            "UPDATE audit_events SET action = ? WHERE id = 2",
            ("predict_TAMPERED",),
        )
        conn.commit()
    finally:
        conn.close()

    v2 = verify_chain(db_path)
    print("\n[tamper-demo] post-tamper verdict:")
    print(v2.pretty())
    return v2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SQLite chained-hash audit log for the Secure AI Pipeline."
    )
    p.add_argument("--db", default=DEFAULT_DB_PATH,
                   help=f"Path to the SQLite audit log (default: {DEFAULT_DB_PATH}).")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("init", help="Create (or reset) the audit log.")
    i.add_argument("--reset", action="store_true", help="Delete the file first.")

    a = sub.add_parser("append", help="Append one event.")
    a.add_argument("--actor",  required=True)
    a.add_argument("--role",   required=True, choices=list(CANONICAL_ROLES))
    a.add_argument("--action", required=True)
    a.add_argument("--payload", default="{}",
                   help="JSON-encoded payload (default: {}).")

    t = sub.add_parser("tail", help="Show the most recent events.")
    t.add_argument("--n", type=int, default=10)

    v = sub.add_parser("verify", help="Verify the chain end-to-end.")
    _ = v  # linter

    d = sub.add_parser("tamper-demo",
                       help="Reset the log, append 3 events, mutate one, and re-verify.")
    _ = d
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    db = args.db

    if args.cmd == "init":
        init_audit_log(db, reset=args.reset)
        print(f"[audit] initialised {db}"
              + (" (reset)" if args.reset else ""))
        return 0

    if args.cmd == "append":
        try:
            payload = json.loads(args.payload)
        except json.JSONDecodeError as e:
            print(f"[audit] --payload is not valid JSON: {e}", file=sys.stderr)
            return 2
        event_id = append_event(
            db, actor=args.actor, role=args.role, action=args.action, payload=payload,
        )
        print(f"[audit] appended event id={event_id}")
        return 0

    if args.cmd == "tail":
        events = tail(db, n=args.n)
        if not events:
            print("[audit] (no events)")
            return 0
        print(f"[audit] last {len(events)} events:")
        for ev in events:
            print(_fmt_event(ev))
        return 0

    if args.cmd == "verify":
        v = verify_chain(db)
        print(v.pretty())
        return 0 if v.intact else 1

    if args.cmd == "tamper-demo":
        v = run_tamper_demo(db, reset=True)
        # Exit code 1 to signal "tamper was detected" — deliberate.
        return 0 if not v.intact else 1

    return 1


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "GENESIS_HASH",
    "HASH_LENGTH",
    "DEFAULT_DB_PATH",
    "CANONICAL_ROLES",
    # Data classes
    "AuditEvent",
    "ChainVerdict",
    # Public API
    "init_audit_log",
    "append_event",
    "list_events",
    "tail",
    "verify_chain",
    "run_tamper_demo",
]


if __name__ == "__main__":
    sys.exit(main())

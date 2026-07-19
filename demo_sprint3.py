"""
demo_sprint3.py
===============
End-to-end live demo of Sprint 3 (audit log + FastAPI service) for the VIVA.

Flow
----
    1. Reset the audit log.
    2. Boot the FastAPI service in-process (background thread).
    3. Hit /health from the outside.
    4. Log in as analyst → GET /me, GET /configs.
    5. Send one prediction row → POST /predict.
    6. Log in as admin → GET /audit/tail, GET /audit/verify (intact).
    7. Deliberately tamper via POST /audit/tamper-demo.
    8. GET /audit/verify again → chain is broken; verifier reports where.
    9. Show that non-admin cannot reach /audit/*.

The whole thing runs against http://127.0.0.1:{port} using httpx. Every
step prints a clearly labelled section header so the panel can follow along.

Usage
-----
    python demo_sprint3.py                                  # uses config_1_baseline
    python demo_sprint3.py --config config_2_k_anon_l_div   # any Sprint-1/2 slug
    python demo_sprint3.py --port 8001                      # non-default port
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import httpx
import pandas as pd
import uvicorn

from midsem_demo import RAW_CSV, ensure_dataset
from module_audit_log import DEFAULT_DB_PATH as DEFAULT_AUDIT_DB
from module_audit_log import verify_chain
from module_inference_api import create_app

BANNER = "=" * 78
SUB    = "-" * 78


def banner(title: str) -> None:
    print()
    print(BANNER)
    print(f"  {title}")
    print(BANNER)


def section(title: str) -> None:
    print()
    print(SUB)
    print(f"  {title}")
    print(SUB)


# ---------------------------------------------------------------------------
# Background uvicorn runner
# ---------------------------------------------------------------------------

class ThreadedServer:
    """Run uvicorn.Server in a daemon thread and shut it down cleanly."""

    def __init__(self, app, host: str = "127.0.0.1", port: int = 8000):
        self.config = uvicorn.Config(
            app, host=host, port=port, log_level="warning", access_log=False,
        )
        self.server = uvicorn.Server(self.config)
        self.thread: threading.Thread | None = None
        self.base_url = f"http://{host}:{port}"

    def start(self) -> None:
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        # Poll /health until the server responds.
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                r = httpx.get(f"{self.base_url}/health", timeout=1.0)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.15)
        raise RuntimeError("FastAPI server did not become ready within 20 seconds.")

    def stop(self) -> None:
        self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Convenience: authenticated HTTP client
# ---------------------------------------------------------------------------

def login_client(base_url: str, username: str, password: str) -> httpx.Client:
    """Return an httpx.Client with a Bearer token attached."""
    r = httpx.post(
        f"{base_url}/login",
        data={"username": username, "password": password},
        timeout=5.0,
    )
    r.raise_for_status()
    token = r.json()["access_token"]
    return httpx.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=15.0,
    )


# ---------------------------------------------------------------------------
# The demo
# ---------------------------------------------------------------------------

def run_demo(
    config_slug: str,
    checkpoint_root: Path,
    audit_db_path: Path,
    port: int,
) -> int:
    banner("Sprint 3 Demo — FastAPI Inference Service + Chained-Hash Audit Log")
    print("  Student  : Rajput Dhirajsing Rajeshsing  (2024DA04378)")
    print(f"  Config   : {config_slug}")
    print(f"  Audit DB : {audit_db_path}")
    print(f"  Endpoint : http://127.0.0.1:{port}")

    # ----- 1) reset audit log ----------------------------------------------
    section("1) Reset the audit log for a clean demo")
    if audit_db_path.exists():
        audit_db_path.unlink()
        print(f"[demo] removed old {audit_db_path}")
    else:
        print(f"[demo] no existing audit log at {audit_db_path}")

    # ----- 2) boot the API -------------------------------------------------
    section("2) Boot the FastAPI service (background thread)")
    app = create_app(
        checkpoint_root=checkpoint_root,
        audit_db_path=audit_db_path,
        init_audit=True,
    )
    server = ThreadedServer(app, host="127.0.0.1", port=port)
    server.start()
    print(f"[demo] service is live at {server.base_url}")

    try:
        # ----- 3) public /health ------------------------------------------
        section("3) Public GET /health")
        r = httpx.get(f"{server.base_url}/health")
        r.raise_for_status()
        health = r.json()
        print(json.dumps(health, indent=2))

        available = health.get("available_configs", [])
        if config_slug not in available:
            print(f"\n[demo] ERROR: '{config_slug}' is not in available_configs. "
                  f"Have you run train_all_configs.py yet?")
            return 2

        # ----- 4) analyst login + inference -------------------------------
        section("4) Log in as 'analyst' → GET /me → GET /configs → POST /predict")
        analyst = login_client(server.base_url, "analyst", "analyst_password")
        print("[demo] analyst token acquired")

        me = analyst.get("/me").json()
        print(f"[demo] /me → {me}")

        configs = analyst.get("/configs").json()
        print(f"[demo] /configs → {configs['count']} configs available")
        for c in configs["configs"][:5]:
            print(f"       {c['slug']}  ROC-AUC={c.get('test_roc_auc'):.4f}"
                  if c.get("test_roc_auc") is not None
                  else f"       {c['slug']}  (metrics unreadable)")
        if len(configs["configs"]) > 5:
            print(f"       ... (+{len(configs['configs']) - 5} more)")

        sample = _load_sample_row()
        print("\n[demo] sending one row to POST /predict...")
        r = analyst.post(
            "/predict",
            params={"config": config_slug},
            json={"features": sample},
        )
        if r.status_code != 200:
            print(f"[demo] /predict returned {r.status_code}: {r.text}")
            return 3
        pred = r.json()
        print(f"[demo] prediction   : {pred['prediction']}")
        print(f"       probability  : {pred['probability']:.4f}")
        print(f"       input SHA-256: {pred['input_sha256'][:24]}...")
        print(f"       audit id     : {pred['audit_event_id']}")
        print(f"       model info   : {pred['model_info']}")

        # ----- 5) forbidden: analyst tries to reach admin route ----------
        section("5) RBAC check — analyst is REFUSED access to /audit/verify")
        r = analyst.get("/audit/verify")
        print(f"[demo] /audit/verify (as analyst) → HTTP {r.status_code}")
        print(f"       body: {r.json()}")
        assert r.status_code == 403, "RBAC broken — analyst should not reach admin routes!"

        # ----- 6) admin login, audit tail + verify -----------------------
        section("6) Log in as 'admin' → GET /audit/tail → GET /audit/verify (intact)")
        admin = login_client(server.base_url, "admin", "admin_password")
        print("[demo] admin token acquired")

        tail = admin.get("/audit/tail", params={"n": 10}).json()
        print(f"[demo] /audit/tail (last {tail['count']} events):")
        for ev in tail["events"]:
            payload_str = ev["payload_json"]
            if len(payload_str) > 60:
                payload_str = payload_str[:57] + "..."
            print(f"       #{ev['id']:<3} {ev['ts']}  [{ev['role']:<7}] "
                  f"{ev['actor']:<8}  {ev['action']:<18}  {payload_str}")

        v = admin.get("/audit/verify").json()
        print("\n[demo] /audit/verify verdict:")
        print(f"       intact           : {v['intact']}")
        print(f"       events on record : {v['n_events']}")
        print(f"       head hash        : {v['head_hash']}")
        assert v["intact"], "chain unexpectedly broken before tampering!"

        # ----- 7) TAMPER --------------------------------------------------
        section("7) TAMPERING — POST /audit/tamper-demo (event_id=2)")
        r = admin.post(
            "/audit/tamper-demo",
            json={"event_id": 2, "new_action": "predict_TAMPERED"},
        )
        r.raise_for_status()
        print(f"[demo] {r.json()['message']}")

        # ----- 8) verify again — chain now broken -------------------------
        section("8) Re-run /audit/verify — chain break detected")
        v2 = admin.get("/audit/verify").json()
        print(json.dumps(v2, indent=2))
        assert not v2["intact"], "tamper not detected!"
        assert v2["broken_at_event_id"] == 2
        print(f"\n[demo] SUCCESS: verifier pinpointed the mutated row "
              f"(event_id={v2['broken_at_event_id']}, row {v2['broken_at_row_index']})")

        # ----- 9) confirm from outside too -------------------------------
        section("9) Sanity check — verify_chain() directly against the SQLite file")
        direct = verify_chain(audit_db_path)
        print(direct.pretty())

        analyst.close()
        admin.close()
        banner("Sprint 3 demo complete — audit log, RBAC, and JWT all verified live.")
        return 0

    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Sample row for the /predict call
# ---------------------------------------------------------------------------

def _load_sample_row() -> dict:
    """Grab a single row from the cached UCI CSV, minus the target column."""
    ensure_dataset()
    df = pd.read_csv(RAW_CSV, nrows=1)
    row = df.iloc[0].to_dict()
    row.pop("readmitted", None)
    # Keep encounter_id / patient_nbr — the checkpoint's feature pipeline
    # doesn't need them (they're not in feature_columns_pre_encoding), so
    # the missing-column check ignores extras.
    return row


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config_1_baseline",
                   help="Checkpoint slug to use for the /predict step.")
    p.add_argument("--checkpoint-root", default="artifacts/checkpoints", type=Path,
                   help="Checkpoint root directory (default: artifacts/checkpoints).")
    p.add_argument("--audit-db", default=DEFAULT_AUDIT_DB, type=Path,
                   help=f"Audit SQLite path (default: {DEFAULT_AUDIT_DB}).")
    p.add_argument("--port", default=8000, type=int, help="HTTP port.")
    args = p.parse_args()

    try:
        return run_demo(
            config_slug=args.config,
            checkpoint_root=args.checkpoint_root,
            audit_db_path=args.audit_db,
            port=args.port,
        )
    except KeyboardInterrupt:
        print("\n[demo] interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())

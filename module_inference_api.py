"""
module_inference_api.py
=======================
Authenticated inference service for the Secure AI Pipeline.

Endpoints
---------
    GET  /health                       (public)
    POST /login                        (public — issues JWT, HS256, 30 min)
    GET  /me                           (authenticated)
    GET  /configs                      (authenticated)
    POST /predict?config=<slug>        (analyst + admin)
    GET  /audit/tail?n=50              (admin only)
    GET  /audit/verify                 (admin only)
    POST /audit/tamper-demo            (admin only — for VIVA)

Security model
--------------
* Authentication: OAuth2 password flow -> JWT bearer tokens (HS256).
  Secret is read from env `SECURE_PIPELINE_JWT_SECRET`. A dev fallback is
  used if unset (with a startup warning printed to stderr).
* Passwords are bcrypt-hashed via passlib. The demo user table is small
  and hardcoded; in production it would be a users table in SQLite/Postgres
  behind a secrets manager (noted in the STRIDE chapter).
* RBAC: two roles — `analyst` (predict + read own data) and `admin`
  (predict + audit access + tamper demo). Admin is a strict superset.
* Every `/predict` call writes to the chained-hash audit log. The stored
  payload contains only:  config slug, prediction, probability,
  input SHA-256 (NOT the raw input — PHI is never persisted in audit).

Integration
-----------
Reloads any of the Sprint-1/2 checkpoints via
`module_midsem_training.load_configuration_checkpoint` and calls
`LoadedConfiguration.predict_proba`. Checkpoints are cached in-process
on first use.

Running
-------
    # Direct
    uvicorn module_inference_api:app --host 127.0.0.1 --port 8000

    # Or via the factory (used by demo_sprint3.py)
    from module_inference_api import create_app
    app = create_app(audit_db_path="artifacts/audit.db",
                     checkpoint_root="artifacts/checkpoints")

Author: Rajput Dhirajsing Rajeshsing  (2024DA04378)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

# ----- pipeline modules -----------------------------------------------------
import module_audit_log as audit
from module_midsem_training import (
    LoadedConfiguration,
    load_configuration_checkpoint,
)

logger = logging.getLogger("secure_pipeline.api")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT_ROOT = Path("artifacts/checkpoints")
DEFAULT_AUDIT_DB        = Path("artifacts/audit.db")
DEFAULT_JWT_ALGORITHM   = "HS256"
DEFAULT_TOKEN_TTL_MIN   = 30

# Anthology of demo users. In production, replace with a users table.
# Passwords are bcrypt-hashed on module import (adds ~200 ms to startup).
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

DEMO_USERS_PLAINTEXT: dict[str, dict[str, str]] = {
    "analyst": {"password": "analyst_password", "role": "analyst"},
    "admin":   {"password": "admin_password",   "role": "admin"},
}


def _build_user_table(plaintext: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """Turn plaintext demo users into a bcrypt-hashed lookup table."""
    return {
        username: {
            "hashed_password": _pwd_context.hash(info["password"]),
            "role": info["role"],
        }
        for username, info in plaintext.items()
    }


# ---------------------------------------------------------------------------
# JWT plumbing
# ---------------------------------------------------------------------------

def _resolve_secret() -> str:
    """Return the JWT signing secret from the environment, or a warned-about default."""
    secret = os.environ.get("SECURE_PIPELINE_JWT_SECRET")
    if not secret:
        # Print to stderr so it appears prominently in the demo output.
        print(
            "[module_inference_api] WARNING: SECURE_PIPELINE_JWT_SECRET not set — "
            "using an insecure dev secret. Set this env var before any real deployment.",
            file=sys.stderr,
        )
        secret = "dev-secret-do-not-use-in-production-2024DA04378"
    return secret


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    role: str
    username: str


class PredictRequest(BaseModel):
    features: dict[str, Any] = Field(
        ..., description="One row of raw feature values (column name → value)."
    )


class PredictResponse(BaseModel):
    config: str
    prediction: int
    probability: float
    threshold: float = 0.5
    input_sha256: str
    audit_event_id: int
    model_info: dict[str, Any]


class TamperDemoRequest(BaseModel):
    event_id: int = Field(..., description="ID of the event to mutate.")
    new_action: str = Field(default="TAMPERED_ACTION")


class ChainVerdictResponse(BaseModel):
    intact: bool
    n_events: int
    broken_at_event_id: int | None = None
    broken_at_row_index: int | None = None
    reason: str | None = None
    expected_hash: str | None = None
    stored_hash: str | None = None
    head_hash: str | None = None


# ---------------------------------------------------------------------------
# App state — populated by create_app()
# ---------------------------------------------------------------------------

@dataclass
class AppState:
    checkpoint_root: Path
    audit_db_path: Path
    jwt_secret: str
    jwt_algorithm: str
    token_ttl_minutes: int
    users: dict[str, dict[str, str]]
    loaded_configs: dict[str, LoadedConfiguration]

    def available_config_slugs(self) -> list[str]:
        if not self.checkpoint_root.is_dir():
            return []
        return sorted(
            p.name for p in self.checkpoint_root.iterdir()
            if p.is_dir() and (p / "model.pt").exists()
        )


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _authenticate_user(
    users: dict[str, dict[str, str]], username: str, password: str,
) -> dict[str, str] | None:
    user = users.get(username)
    if user is None:
        return None
    if not _pwd_context.verify(password, user["hashed_password"]):
        return None
    return user


def _create_token(secret: str, algorithm: str, ttl_min: int,
                  *, username: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ttl_min)
    payload = {
        "sub": username,
        "role": role,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, secret, algorithm=algorithm)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    *,
    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
    audit_db_path: str | Path = DEFAULT_AUDIT_DB,
    jwt_secret: str | None = None,
    jwt_algorithm: str = DEFAULT_JWT_ALGORITHM,
    token_ttl_minutes: int = DEFAULT_TOKEN_TTL_MIN,
    users_plaintext: dict[str, dict[str, str]] | None = None,
    init_audit: bool = True,
) -> FastAPI:
    """Build a FastAPI app with the given configuration."""
    state = AppState(
        checkpoint_root=Path(checkpoint_root),
        audit_db_path=Path(audit_db_path),
        jwt_secret=jwt_secret or _resolve_secret(),
        jwt_algorithm=jwt_algorithm,
        token_ttl_minutes=token_ttl_minutes,
        users=_build_user_table(users_plaintext or DEMO_USERS_PLAINTEXT),
        loaded_configs={},
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if init_audit:
            audit.init_audit_log(state.audit_db_path)
            audit.append_event(
                state.audit_db_path,
                actor="system", role="system", action="api_startup",
                payload={
                    "checkpoint_root": str(state.checkpoint_root),
                    "available_configs": state.available_config_slugs(),
                },
            )
        yield
        # No shutdown teardown required — SQLite closes per-request; no
        # persistent connections to drain.

    app = FastAPI(
        title="Secure AI Pipeline — Inference API",
        version="0.3.0-sprint3",
        description=(
            "Authenticated inference service for the four-configuration ablation. "
            "Every prediction is written to a tamper-evident SQLite audit log."
        ),
        lifespan=lifespan,
    )
    # Permissive CORS for the localhost demo. Tighten in production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

    # ----- dependency: current user from JWT -----------------------------
    def get_current_user(token: str = Depends(oauth2_scheme)) -> dict[str, str]:
        credentials_exc = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            payload = jwt.decode(token, state.jwt_secret, algorithms=[state.jwt_algorithm])
        except JWTError:
            raise credentials_exc
        username = payload.get("sub")
        role = payload.get("role")
        if not username or not role:
            raise credentials_exc
        # Confirm the user still exists (a real deployment could revoke here).
        if username not in state.users:
            raise credentials_exc
        return {"username": username, "role": role}

    def require_role(*allowed: str):
        def dep(user: dict[str, str] = Depends(get_current_user)) -> dict[str, str]:
            if user["role"] not in allowed:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=(
                        f"Role '{user['role']}' cannot access this endpoint. "
                        f"Allowed roles: {sorted(allowed)}."
                    ),
                )
            return user
        return dep

    # ----- helper: load a checkpoint on demand ---------------------------
    def _load(config_slug: str) -> LoadedConfiguration:
        if config_slug in state.loaded_configs:
            return state.loaded_configs[config_slug]
        ckpt_dir = state.checkpoint_root / config_slug
        if not ckpt_dir.is_dir():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Config '{config_slug}' not found. "
                    f"Available: {state.available_config_slugs() or 'none'}."
                ),
            )
        try:
            loaded = load_configuration_checkpoint(ckpt_dir)
        except Exception as exc:
            logger.exception("failed to load checkpoint %s", config_slug)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load checkpoint '{config_slug}': {exc}",
            )
        state.loaded_configs[config_slug] = loaded
        return loaded

    # ---------------- Endpoints ------------------------------------------

    @app.get("/health")
    def health():
        try:
            v = audit.verify_chain(state.audit_db_path)
            audit_intact = v.intact
            audit_events = v.n_events
        except Exception:
            audit_intact = None
            audit_events = None
        return {
            "status": "ok",
            "service": "secure-ai-pipeline-inference",
            "version": app.version,
            "checkpoint_root": str(state.checkpoint_root),
            "audit_db_path": str(state.audit_db_path),
            "audit_chain_intact": audit_intact,
            "audit_events": audit_events,
            "available_configs": state.available_config_slugs(),
            "loaded_configs": sorted(state.loaded_configs.keys()),
        }

    @app.post("/login", response_model=LoginResponse)
    def login(form: OAuth2PasswordRequestForm = Depends(), request: Request = None):
        user = _authenticate_user(state.users, form.username, form.password)
        if user is None:
            # Log the failed attempt as an audit event.
            audit.append_event(
                state.audit_db_path,
                actor=form.username or "unknown",
                role="system",
                action="login_failed",
                payload={"remote": request.client.host if request and request.client else "unknown"},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = _create_token(
            state.jwt_secret, state.jwt_algorithm, state.token_ttl_minutes,
            username=form.username, role=user["role"],
        )
        audit.append_event(
            state.audit_db_path,
            actor=form.username,
            role=user["role"],
            action="login",
            payload={"remote": request.client.host if request and request.client else "unknown"},
        )
        return LoginResponse(
            access_token=token,
            expires_in_minutes=state.token_ttl_minutes,
            role=user["role"],
            username=form.username,
        )

    @app.get("/me")
    def me(current: dict[str, str] = Depends(get_current_user)):
        return {"username": current["username"], "role": current["role"]}

    @app.get("/configs")
    def configs(current: dict[str, str] = Depends(get_current_user)):
        available = state.available_config_slugs()
        # Include a summary of each config's metrics for admin dashboards.
        summaries: list[dict[str, Any]] = []
        for slug in available:
            meta_path = state.checkpoint_root / slug / "metadata.json"
            try:
                with open(meta_path) as fh:
                    meta = json.load(fh)
                summary = {
                    "slug": slug,
                    "config_name": meta.get("config_name"),
                    "test_roc_auc": meta.get("metrics", {}).get("test_roc_auc"),
                    "test_accuracy": meta.get("metrics", {}).get("test_accuracy"),
                    "loaded": slug in state.loaded_configs,
                }
                if "dp" in meta:
                    summary["dp"] = {
                        "epsilon_target": meta["dp"].get("target_epsilon"),
                        "epsilon_spent":  meta["dp"].get("epsilon_spent"),
                    }
                summaries.append(summary)
            except Exception:
                summaries.append({"slug": slug, "config_name": "(unreadable)", "loaded": False})
        return {"configs": summaries, "count": len(summaries)}

    @app.post("/predict", response_model=PredictResponse)
    def predict(
        req: PredictRequest,
        config: str = Query(..., description="Checkpoint slug (e.g. 'config_1_baseline')."),
        current: dict[str, str] = Depends(require_role("analyst", "admin")),
    ):
        loaded = _load(config)

        # Validate that all required feature columns are present.
        required = set(loaded.metadata.get("feature_columns_pre_encoding", []))
        # Drop target + system columns from the required set — they are
        # optional in inference input.
        required = required - {loaded.metadata.get("target_column")}
        sent = set(req.features.keys())
        missing = sorted(required - sent)
        if missing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "Missing required feature columns.",
                    "missing": missing[:20],
                    "n_missing": len(missing),
                },
            )

        # Run inference on a one-row DataFrame.
        try:
            df = pd.DataFrame([req.features])
            probs = loaded.predict_proba(df)
        except Exception as exc:
            logger.exception("inference failed for config %s", config)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Inference failed: {exc}",
            )
        prob = float(probs[0])
        pred = int(prob >= 0.5)

        # Hash the input row for audit purposes — never store the raw PHI.
        canonical = json.dumps(req.features, sort_keys=True, separators=(",", ":"), default=str)
        input_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        # Log the prediction to the audit chain.
        event_id = audit.append_event(
            state.audit_db_path,
            actor=current["username"],
            role=current["role"],
            action="predict",
            payload={
                "config": config,
                "input_sha256": input_sha256,
                "prediction": pred,
                "probability": prob,
                "threshold": 0.5,
            },
        )

        model_info = {
            "config_name": loaded.metadata.get("config_name"),
            "test_roc_auc": loaded.metadata.get("metrics", {}).get("test_roc_auc"),
            "seed": loaded.metadata.get("seed"),
        }
        if "dp" in loaded.metadata:
            model_info["dp"] = {
                "epsilon_target": loaded.metadata["dp"].get("target_epsilon"),
                "epsilon_spent":  loaded.metadata["dp"].get("epsilon_spent"),
            }

        return PredictResponse(
            config=config,
            prediction=pred,
            probability=prob,
            input_sha256=input_sha256,
            audit_event_id=event_id,
            model_info=model_info,
        )

    @app.get("/audit/tail")
    def audit_tail(
        n: int = Query(50, ge=1, le=1000),
        current: dict[str, str] = Depends(require_role("admin")),
    ):
        events = audit.tail(state.audit_db_path, n=n)
        # Log this admin action too — audit access is itself auditable.
        audit.append_event(
            state.audit_db_path,
            actor=current["username"], role=current["role"],
            action="audit_tail",
            payload={"n": n},
        )
        return {
            "events": [e.to_dict() for e in events],
            "count": len(events),
        }

    @app.get("/audit/verify", response_model=ChainVerdictResponse)
    def audit_verify(current: dict[str, str] = Depends(require_role("admin"))):
        v = audit.verify_chain(state.audit_db_path)
        # NB: we log this AFTER verifying — logging before would add an
        # event that the verifier hasn't yet seen. Order matters.
        audit.append_event(
            state.audit_db_path,
            actor=current["username"], role=current["role"],
            action="audit_verify",
            payload={"intact": v.intact, "n_events": v.n_events},
        )
        return ChainVerdictResponse(
            intact=v.intact,
            n_events=v.n_events,
            broken_at_event_id=v.broken_at_event_id,
            broken_at_row_index=v.broken_at_row_index,
            reason=v.reason,
            expected_hash=v.expected_hash,
            stored_hash=v.stored_hash,
            head_hash=v.head_hash,
        )

    @app.post("/audit/tamper-demo")
    def audit_tamper_demo(
        req: TamperDemoRequest,
        current: dict[str, str] = Depends(require_role("admin")),
    ):
        """FOR THE VIVA DEMO ONLY — deliberately mutate a row so the panel
        can watch the verifier detect it on the next /audit/verify call."""
        import sqlite3
        conn = sqlite3.connect(str(state.audit_db_path))
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE audit_events SET action = ? WHERE id = ?",
                (req.new_action, req.event_id),
            )
            if cur.rowcount == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No audit event with id {req.event_id}.",
                )
            conn.commit()
        finally:
            conn.close()
        # We *deliberately* do NOT audit-log this event — the whole point
        # is that the mutation is invisible to the log itself. The next
        # /audit/verify call will expose it.
        return {
            "message": (
                f"Row {req.event_id} action mutated to '{req.new_action}'. "
                "Call GET /audit/verify to see the chain break."
            ),
            "mutated_event_id": req.event_id,
        }

    return app


# ---------------------------------------------------------------------------
# Module-level app object for `uvicorn module_inference_api:app`
# ---------------------------------------------------------------------------

app = create_app()


__all__ = [
    "app",
    "create_app",
    "AppState",
    "LoginResponse",
    "PredictRequest",
    "PredictResponse",
    "ChainVerdictResponse",
    "TamperDemoRequest",
    "DEFAULT_CHECKPOINT_ROOT",
    "DEFAULT_AUDIT_DB",
]

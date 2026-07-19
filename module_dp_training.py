"""
module_dp_training.py
=====================
Differentially private training for Configurations 3 and 4.

    Configuration 3 — DP-SGD only (no data-level defence)
    Configuration 4 — DP-SGD + k-anonymity (k=5) + l-diversity (l=3)

Both use the *same* MLP architecture, feature engineering, and split logic
as Configurations 1 and 2 (imported from module_midsem_training) so the
four-configuration ablation is genuinely controlled — only the training
mechanism changes.

Privacy accounting
------------------
Opacus's PrivacyEngine with Rényi DP accountant (Abadi et al. 2016,
Mironov 2017). Noise multiplier is auto-calibrated by
`make_private_with_epsilon` to spend exactly `target_epsilon` at
`target_delta` after `epochs` epochs with the given batch sampling rate.

The actually-spent ε at training end is recorded in the checkpoint —
if early stopping fires at epoch T < epochs, the spent ε is proportionally
lower than the target, which is *better* privacy than what we paid for.

Checkpoint contract
-------------------
DP checkpoints use the *same* on-disk layout as Configs 1/2:

    artifacts/checkpoints/<slug>/
    ├── model.pt              # underlying MLPClassifier state_dict (unwrapped)
    ├── preprocessor.joblib   # fitted sklearn Pipeline
    ├── training_dataframe.parquet
    ├── split_indices.npz
    └── metadata.json         # + top-level "dp" sub-dict with ε accounting

This means `module_midsem_training.load_configuration_checkpoint()` reloads
DP checkpoints without any code changes — the FastAPI service (Sprint 3)
and the attack modules (Sprint 4) treat all configs uniformly.

Author: Rajput Dhirajsing Rajeshsing  (2024DA04378)
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# --- shared plumbing from Sprint 1 -----------------------------------------
from module_midsem_training import (
    DEFAULT_SEED,
    DROP_FROM_FEATURES,
    POSITIVE_LABEL,
    TARGET_COLUMN,
    EarlyStopping,
    MLPClassifier,
    TrainingArtifacts,          # imported for downstream type parity
    TrainingMetrics,
    _make_loader,               # internal but stable — same file, same version
    _persist_artifacts,         # internal but stable
    _slugify,
    build_feature_pipeline,
    default_checkpoint_dir,
    prepare_xy,
    seed_everything,
)

# ---------------------------------------------------------------------------
# Opacus — imported lazily with a clear error message if missing
# ---------------------------------------------------------------------------

try:
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    from opacus.validators import ModuleValidator
    _OPACUS_AVAILABLE = True
    _OPACUS_IMPORT_ERROR: Exception | None = None
except ImportError as _e:
    _OPACUS_AVAILABLE = False
    _OPACUS_IMPORT_ERROR = _e


def _require_opacus() -> None:
    """Raise a helpful error if Opacus is not installed."""
    if not _OPACUS_AVAILABLE:
        raise RuntimeError(
            "Opacus is required for DP-SGD training but is not installed.\n"
            "Install it with:\n"
            "    pip install 'opacus>=1.5,<1.6'\n"
            f"Original import error: {_OPACUS_IMPORT_ERROR}"
        )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TARGET_DELTA: float = 1e-5
DEFAULT_MAX_GRAD_NORM: float = 1.0
DEFAULT_EPSILONS: tuple[float, ...] = (1.0, 3.0, 8.0)

# Physical batch size ceiling under BatchMemoryManager. 64 keeps per-sample
# gradient memory ~73 MB per Linear layer on our feature width, which is
# comfortable on 8 GB laptops (the target hardware in the mid-sem report).
DEFAULT_MAX_PHYSICAL_BATCH_SIZE: int = 64

# Longer default for DP because training curves are noisier and often
# still improving at epoch 12; early stopping (patience 5) trims where
# appropriate.
DEFAULT_DP_EPOCHS: int = 15
DEFAULT_DP_PATIENCE: int = 5

# Slug prefixes for the six DP checkpoints produced by a full sweep.
CONFIG_3_SLUG_PREFIX = "config_3_dp_only"
CONFIG_4_SLUG_PREFIX = "config_4_dp_kanon"


def dp_checkpoint_slug(prefix: str, epsilon: float) -> str:
    """Return e.g. 'config_3_dp_only_eps1' for prefix 'config_3_dp_only', ε=1."""
    # Format ε as int when possible ("1" not "1.0"); otherwise keep one decimal.
    if float(epsilon).is_integer():
        eps_tag = f"eps{int(epsilon)}"
    else:
        eps_tag = f"eps{epsilon:.1f}".replace(".", "p")
    return f"{prefix}_{eps_tag}"


# ---------------------------------------------------------------------------
# DPTrainingMetrics — composes TrainingMetrics + DP accounting
# ---------------------------------------------------------------------------

@dataclass
class DPTrainingMetrics:
    """Result of a DP-SGD training run.

    Composes a base `TrainingMetrics` (utility numbers, unchanged shape)
    with a DP-specific accounting block. Downstream code that only cares
    about utility can access `.base` and treat it like a Sprint-1 result;
    code that cares about privacy (Pareto plot, report tables) reads the
    additional fields directly.
    """
    base: TrainingMetrics
    # DP accounting fields
    dp_target_epsilon: float
    dp_target_delta: float
    dp_epsilon_spent: float
    dp_noise_multiplier: float
    dp_max_grad_norm: float
    dp_sample_rate: float
    dp_epochs_charged: int
    dp_accountant: str = "rdp"

    # ---------- convenient passthroughs to base metrics ----------
    @property
    def config_name(self) -> str: return self.base.config_name
    @property
    def test_accuracy(self) -> float: return self.base.test_accuracy
    @property
    def test_f1(self) -> float: return self.base.test_f1
    @property
    def test_roc_auc(self) -> float: return self.base.test_roc_auc
    @property
    def checkpoint_dir(self) -> str | None: return self.base.checkpoint_dir

    def pretty(self) -> str:
        """Human-readable summary — used by the sweep orchestrator."""
        base_txt = self.base.pretty()
        dp_block = (
            "\n  --- Differential Privacy Accounting ---\n"
            f"  target (ε, δ)           : ({self.dp_target_epsilon:.2f}, {self.dp_target_delta:.0e})\n"
            f"  ε spent (RDP)           : {self.dp_epsilon_spent:.4f}\n"
            f"  noise multiplier σ      : {self.dp_noise_multiplier:.4f}\n"
            f"  clipping norm C         : {self.dp_max_grad_norm:.2f}\n"
            f"  sample rate q           : {self.dp_sample_rate:.5f}\n"
            f"  epochs charged          : {self.dp_epochs_charged}"
        )
        return base_txt + dp_block

    def to_dict(self) -> dict:
        d = asdict(self.base)
        d["dp"] = {
            "target_epsilon": self.dp_target_epsilon,
            "target_delta": self.dp_target_delta,
            "epsilon_spent": self.dp_epsilon_spent,
            "noise_multiplier": self.dp_noise_multiplier,
            "max_grad_norm": self.dp_max_grad_norm,
            "sample_rate": self.dp_sample_rate,
            "epochs_charged": self.dp_epochs_charged,
            "accountant": self.dp_accountant,
        }
        return d


@dataclass
class DPTrainingArtifacts:
    """Full-artifact bundle returned when `return_artifacts=True`."""
    metrics: DPTrainingMetrics
    model: MLPClassifier                  # plain MLPClassifier (unwrapped)
    pipeline: Pipeline
    training_dataframe: pd.DataFrame
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    metadata: dict
    checkpoint_dir: Path | None = None


# ---------------------------------------------------------------------------
# Metadata builder — extends Sprint 1's format with a "dp" sub-dict
# ---------------------------------------------------------------------------

def _build_dp_metadata(
    config_name: str,
    feature_columns: list[str],
    n_features_after_encoding: int,
    seed: int,
    hyperparameters: dict,
    metrics: TrainingMetrics,
    row_counts: dict,
    train_losses: list[float],
    val_losses: list[float],
    # DP-specific
    target_epsilon: float,
    target_delta: float,
    epsilon_spent: float,
    noise_multiplier: float,
    max_grad_norm: float,
    sample_rate: float,
    epochs_charged: int,
    accountant: str,
) -> dict:
    """Assemble the metadata.json payload — identical schema to Sprint 1
    plus a top-level "dp" sub-dict for the privacy accounting."""
    return {
        "config_name": config_name,
        "config_slug": _slugify(config_name),
        "target_column": TARGET_COLUMN,
        "positive_label": POSITIVE_LABEL,
        "drop_from_features": list(DROP_FROM_FEATURES),
        "feature_columns_pre_encoding": feature_columns,
        "n_features_after_encoding": int(n_features_after_encoding),
        "seed": int(seed),
        "hyperparameters": hyperparameters,
        "metrics": {
            "test_accuracy": metrics.test_accuracy,
            "test_f1": metrics.test_f1,
            "test_roc_auc": metrics.test_roc_auc,
            "best_epoch": metrics.best_epoch,
            "best_val_loss": metrics.best_val_loss,
            "train_loss_by_epoch": train_losses,
            "val_loss_by_epoch": val_losses,
        },
        "row_counts": row_counts,
        "dp": {
            "target_epsilon": float(target_epsilon),
            "target_delta": float(target_delta),
            "epsilon_spent": float(epsilon_spent),
            "noise_multiplier": float(noise_multiplier),
            "max_grad_norm": float(max_grad_norm),
            "sample_rate": float(sample_rate),
            "epochs_charged": int(epochs_charged),
            "accountant": accountant,
        },
    }


# ---------------------------------------------------------------------------
# Core DP training routine
# ---------------------------------------------------------------------------

def train_dp_configuration(
    df: pd.DataFrame,
    config_name: str,
    *,
    target_epsilon: float,
    target_delta: float = DEFAULT_TARGET_DELTA,
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
    epochs: int = DEFAULT_DP_EPOCHS,
    batch_size: int = 512,
    max_physical_batch_size: int = DEFAULT_MAX_PHYSICAL_BATCH_SIZE,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    hidden: int = 128,
    dropout: float = 0.2,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
    early_stopping: bool = True,
    patience: int = DEFAULT_DP_PATIENCE,
    min_delta: float = 1e-4,
    persist_to: str | Path | None = None,
    return_artifacts: bool = False,
    accountant: str = "rdp",
) -> DPTrainingMetrics | DPTrainingArtifacts:
    """
    Train the MLP with DP-SGD (Abadi et al. 2016, Opacus 1.5).

    Parameters
    ----------
    df :
        Training dataframe. Must contain the target column `readmitted`.
        Direct identifiers are expected to already be dropped by the caller.
    config_name :
        Human-readable label recorded in metrics + metadata.
    target_epsilon :
        Privacy budget to spend during training. Opacus auto-calibrates
        the noise multiplier to hit this after `epochs` epochs.
    target_delta :
        Failure probability δ. Default 1e-5 matches the mid-sem report.
    max_grad_norm :
        Per-example gradient clipping norm C. Default 1.0 (standard).
    max_physical_batch_size :
        Ceiling for physical batch size inside BatchMemoryManager. The
        logical (privacy-accounted) batch is `batch_size`; the physical
        micro-batches keep per-sample gradient memory bounded.
    weight_decay :
        Default 0.0 for DP-SGD: L2 regularisation interferes with the
        noise calibration. Set explicitly to override.
    early_stopping :
        If True (default), monitor val_loss with `patience` epochs of
        no improvement, then restore the best-observed weights. Note the
        privacy accountant is *not* rewound on early stopping — the ε
        actually spent is proportional to epochs run, which is fine
        (early ε spend is a floor, not a ceiling; less spend = better).
    persist_to :
        Optional directory. If given, writes the checkpoint bundle
        (compatible with `load_configuration_checkpoint`).
    return_artifacts :
        If True, return a `DPTrainingArtifacts` object; otherwise return
        `DPTrainingMetrics`.
    accountant :
        Opacus accountant. "rdp" (default) matches Abadi et al. Others
        available in Opacus: "gdp", "prv".
    """
    _require_opacus()

    if target_epsilon <= 0:
        raise ValueError(f"target_epsilon must be > 0, got {target_epsilon}")
    if target_delta <= 0 or target_delta >= 1:
        raise ValueError(f"target_delta must be in (0, 1), got {target_delta}")

    seed_everything(seed)

    # -----------------------------------------------------------------------
    # 1) Feature engineering + 70/15/15 stratified split
    # -----------------------------------------------------------------------
    X_df, y = prepare_xy(df)
    positional_idx = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        positional_idx, test_size=0.15, random_state=seed, stratify=y,
    )
    y_train_val = y[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.15 / 0.85,
        random_state=seed,
        stratify=y_train_val,
    )

    X_train = X_df.iloc[train_idx].reset_index(drop=True)
    X_val   = X_df.iloc[val_idx].reset_index(drop=True)
    X_test  = X_df.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_val   = y[val_idx]
    y_test  = y[test_idx]

    pipeline = build_feature_pipeline(X_train)
    X_train_enc = pipeline.fit_transform(X_train).astype(np.float32)
    X_val_enc   = pipeline.transform(X_val).astype(np.float32)
    X_test_enc  = pipeline.transform(X_test).astype(np.float32)
    n_features  = X_train_enc.shape[1]

    # -----------------------------------------------------------------------
    # 2) Model / loss / optimiser
    # -----------------------------------------------------------------------
    model = MLPClassifier(in_features=n_features, hidden=hidden, dropout=dropout)

    # Validate + auto-fix any Opacus-incompatible modules (BN, etc.). Our MLP
    # is Linear/ReLU/Dropout only, so this is a no-op — the check is defensive.
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=True)

    # Class imbalance handling — same as non-DP configs.
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight_val = neg / max(pos, 1.0)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    train_loader = _make_loader(X_train_enc, y_train, batch_size, shuffle=True)
    val_loader   = _make_loader(X_val_enc,   y_val,   batch_size, shuffle=False)

    # -----------------------------------------------------------------------
    # 3) Attach the PrivacyEngine
    # -----------------------------------------------------------------------
    privacy_engine = PrivacyEngine(accountant=accountant)
    try:
        dp_model, dp_optimizer, dp_train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            epochs=epochs,
            max_grad_norm=max_grad_norm,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Opacus could not calibrate noise for target ε={target_epsilon} at δ={target_delta} "
            f"over {epochs} epochs with batch size {batch_size} on {len(y_train):,} rows. "
            f"Try increasing `epochs`, relaxing `target_epsilon`, or reducing `batch_size`. "
            f"Underlying error: {exc}"
        ) from exc

    noise_multiplier = float(dp_optimizer.noise_multiplier)
    sample_rate = float(batch_size) / float(len(y_train))

    if verbose:
        print(
            f"  [{config_name}] PrivacyEngine attached — "
            f"σ={noise_multiplier:.4f}, C={max_grad_norm}, "
            f"q={sample_rate:.5f}, accountant={accountant}"
        )

    # -----------------------------------------------------------------------
    # 4) Training loop with BatchMemoryManager + optional early stopping
    # -----------------------------------------------------------------------
    es: EarlyStopping | None = (
        EarlyStopping(patience=patience, min_delta=min_delta) if early_stopping else None
    )
    train_losses: list[float] = []
    val_losses:   list[float] = []
    epochs_run = 0
    stopped_early_flag = False

    for epoch in range(1, epochs + 1):
        # ---- train ----
        dp_model.train()
        running, n_seen = 0.0, 0
        with BatchMemoryManager(
            data_loader=dp_train_loader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=dp_optimizer,
        ) as memory_safe_loader:
            for xb, yb in memory_safe_loader:
                dp_optimizer.zero_grad()
                logits = dp_model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                dp_optimizer.step()
                running += float(loss.item()) * xb.size(0)
                n_seen  += xb.size(0)
        train_loss = running / max(n_seen, 1)
        train_losses.append(train_loss)

        # ---- val ----
        dp_model.eval()
        running, n_seen = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = dp_model(xb)
                loss = criterion(logits, yb)
                running += float(loss.item()) * xb.size(0)
                n_seen  += xb.size(0)
        val_loss = running / max(n_seen, 1)
        val_losses.append(val_loss)

        epochs_run = epoch

        if verbose:
            try:
                eps_so_far = privacy_engine.get_epsilon(delta=target_delta)
            except Exception:
                eps_so_far = float("nan")
            marker = ""
            if es is not None and val_loss <= es.best_loss + 1e-12:
                marker = "  <-- best"
            print(
                f"  [{config_name}] epoch {epoch:>2}/{epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"ε_spent={eps_so_far:.3f}{marker}"
            )

        if es is not None:
            # Early stopping tracks the UNDERLYING module's state so that
            # `restore_best` yields a plain MLPClassifier state_dict on
            # reload — no GradSampleModule wrapper prefixes.
            underlying = dp_model._module
            should_stop = es.step(val_loss, epoch, underlying)
            if should_stop:
                stopped_early_flag = True
                if verbose:
                    print(
                        f"  [{config_name}] early stopping at epoch {epoch}; "
                        f"restoring weights from epoch {es.best_epoch} "
                        f"(val_loss={es.best_loss:.4f})"
                    )
                break

    # -----------------------------------------------------------------------
    # 5) Restore best weights (on the underlying module)
    # -----------------------------------------------------------------------
    if es is not None:
        underlying = dp_model._module
        es.restore_best(underlying)
        best_epoch = es.best_epoch or epochs_run
        best_val_loss = es.best_loss if es.best_state is not None else val_losses[-1]
    else:
        best_epoch = int(np.argmin(val_losses)) + 1 if val_losses else 0
        best_val_loss = float(min(val_losses)) if val_losses else float("inf")

    # -----------------------------------------------------------------------
    # 6) Query the accountant for ε actually spent
    # -----------------------------------------------------------------------
    try:
        dp_epsilon_spent = float(privacy_engine.get_epsilon(delta=target_delta))
    except Exception as exc:
        warnings.warn(
            f"Could not query final ε from Opacus accountant: {exc}. "
            f"Falling back to target_epsilon."
        )
        dp_epsilon_spent = float(target_epsilon)

    # -----------------------------------------------------------------------
    # 7) Test-set evaluation
    # -----------------------------------------------------------------------
    dp_model.eval()
    with torch.no_grad():
        logits = dp_model(torch.tensor(X_test_enc, dtype=torch.float32))
        probs  = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)

    base_metrics = TrainingMetrics(
        config_name=config_name,
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        n_test_rows=len(y_test),
        n_features_after_encoding=n_features,
        epochs=epochs,
        train_loss_by_epoch=train_losses,
        val_loss_by_epoch=val_losses,
        test_accuracy=float(accuracy_score(y_test, preds)),
        test_f1=float(f1_score(y_test, preds, zero_division=0)),
        test_roc_auc=float(roc_auc_score(y_test, probs)),
        epochs_run=epochs_run,
        best_epoch=best_epoch,
        best_val_loss=float(best_val_loss),
        stopped_early=stopped_early_flag,
        early_stopping_patience=(patience if early_stopping else 0),
        seed=seed,
        pos_weight=float(pos_weight_val),
    )

    dp_metrics = DPTrainingMetrics(
        base=base_metrics,
        dp_target_epsilon=float(target_epsilon),
        dp_target_delta=float(target_delta),
        dp_epsilon_spent=float(dp_epsilon_spent),
        dp_noise_multiplier=float(noise_multiplier),
        dp_max_grad_norm=float(max_grad_norm),
        dp_sample_rate=float(sample_rate),
        dp_epochs_charged=int(epochs_run),
        dp_accountant=accountant,
    )

    # -----------------------------------------------------------------------
    # 8) Metadata + persistence (same on-disk contract as Sprint 1)
    # -----------------------------------------------------------------------
    hyperparameters = {
        "hidden": hidden,
        "dropout": dropout,
        "epochs_planned": epochs,
        "epochs_run": epochs_run,
        "batch_size": batch_size,
        "max_physical_batch_size": max_physical_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "early_stopping": bool(early_stopping),
        "early_stopping_patience": (patience if early_stopping else 0),
        "min_delta": (min_delta if early_stopping else 0.0),
        "pos_weight": float(pos_weight_val),
        "dp_sgd": True,
    }
    row_counts = {
        "train": int(len(y_train)),
        "val":   int(len(y_val)),
        "test":  int(len(y_test)),
    }
    metadata = _build_dp_metadata(
        config_name=config_name,
        feature_columns=list(X_df.columns),
        n_features_after_encoding=n_features,
        seed=seed,
        hyperparameters=hyperparameters,
        metrics=base_metrics,
        row_counts=row_counts,
        train_losses=train_losses,
        val_losses=val_losses,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        epsilon_spent=dp_epsilon_spent,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        sample_rate=sample_rate,
        epochs_charged=epochs_run,
        accountant=accountant,
    )

    checkpoint_dir_path: Path | None = None
    if persist_to is not None:
        checkpoint_dir_path = Path(persist_to)
        # Save the UNDERLYING module so `load_configuration_checkpoint` gets a
        # plain MLPClassifier — no GradSampleModule wrapper prefixes.
        underlying_model = dp_model._module
        _persist_artifacts(
            checkpoint_dir=checkpoint_dir_path,
            model=underlying_model,
            pipeline=pipeline,
            training_dataframe=df.reset_index(drop=True),
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            metadata=metadata,
        )
        base_metrics.checkpoint_dir = str(checkpoint_dir_path)
        metadata["checkpoint_dir"] = str(checkpoint_dir_path)

    if return_artifacts:
        return DPTrainingArtifacts(
            metrics=dp_metrics,
            model=dp_model._module,
            pipeline=pipeline,
            training_dataframe=df.reset_index(drop=True),
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            metadata=metadata,
            checkpoint_dir=checkpoint_dir_path,
        )
    return dp_metrics


# ---------------------------------------------------------------------------
# Public per-configuration wrappers
# ---------------------------------------------------------------------------

def train_config_3_dp_only(
    df_original: pd.DataFrame,
    target_epsilon: float,
    *,
    target_delta: float = DEFAULT_TARGET_DELTA,
    persist_to: str | Path | None = None,
    return_artifacts: bool = False,
    **kwargs,
) -> DPTrainingMetrics | DPTrainingArtifacts:
    """Config 3 — DP-SGD only, no data-level defence.

    Direct identifiers (encounter_id, patient_nbr) are dropped, matching
    Config 1's data-preparation exactly. The only difference from Config 1
    is that training uses DP-SGD with the given (ε, δ) budget.
    """
    df = df_original.drop(columns=["encounter_id", "patient_nbr"], errors="ignore")
    return train_dp_configuration(
        df,
        config_name=(
            f"Config 3 — DP-SGD only (ε={target_epsilon:g}, δ={target_delta:.0e})"
        ),
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        persist_to=persist_to,
        return_artifacts=return_artifacts,
        **kwargs,
    )


def train_config_4_dp_plus_kanon(
    df_anon: pd.DataFrame,
    target_epsilon: float,
    *,
    target_delta: float = DEFAULT_TARGET_DELTA,
    persist_to: str | Path | None = None,
    return_artifacts: bool = False,
    **kwargs,
) -> DPTrainingMetrics | DPTrainingArtifacts:
    """Config 4 — DP-SGD on the k-anonymous + l-diverse dataframe.

    The caller is expected to have already run `de_identify(k=5, l=3)`
    on the raw data and to pass the resulting `.dataframe` here — this
    keeps the DP module free of a dependency on module_deidentification.
    """
    return train_dp_configuration(
        df_anon,
        config_name=(
            f"Config 4 — DP-SGD + k-anon + l-div (ε={target_epsilon:g}, δ={target_delta:.0e})"
        ),
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        persist_to=persist_to,
        return_artifacts=return_artifacts,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Sweep helper — one call runs the full ε ∈ {1, 3, 8} grid
# ---------------------------------------------------------------------------

def sweep_epsilons(
    df: pd.DataFrame,
    config_prefix: str,
    trainer_fn,
    *,
    epsilons: Iterable[float] = DEFAULT_EPSILONS,
    root: str | Path = "artifacts/checkpoints",
    target_delta: float = DEFAULT_TARGET_DELTA,
    **kwargs,
) -> dict[float, DPTrainingMetrics]:
    """Run `trainer_fn` (train_config_3_dp_only or train_config_4_dp_plus_kanon)
    for every ε in `epsilons`. Persists one checkpoint per ε.

    Returns a dict mapping ε → DPTrainingMetrics.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    results: dict[float, DPTrainingMetrics] = {}
    for eps in epsilons:
        slug = dp_checkpoint_slug(config_prefix, eps)
        checkpoint_dir = root / slug
        print()
        print("+" + "-" * 76 + "+")
        print(f"|  ε = {eps}  →  checkpoint {slug}".ljust(77) + "|")
        print("+" + "-" * 76 + "+")
        metrics = trainer_fn(
            df,
            target_epsilon=float(eps),
            target_delta=target_delta,
            persist_to=checkpoint_dir,
            **kwargs,
        )
        results[float(eps)] = metrics
        print()
        print(metrics.pretty())
    return results


# ---------------------------------------------------------------------------
# Convenience: load DP checkpoint back into a DPTrainingMetrics view
# ---------------------------------------------------------------------------

def load_dp_metadata(checkpoint_dir: str | Path) -> dict:
    """Read the metadata.json from a DP checkpoint.

    Raises ValueError if the checkpoint is not a DP config (no "dp" key).
    """
    checkpoint_dir = Path(checkpoint_dir)
    meta_path = checkpoint_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {checkpoint_dir}")
    with open(meta_path) as fh:
        meta = json.load(fh)
    if "dp" not in meta:
        raise ValueError(
            f"Checkpoint at {checkpoint_dir} is not a DP config "
            f"(no 'dp' section in metadata.json). "
            f"Use module_midsem_training.load_configuration_checkpoint instead."
        )
    return meta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Dataclasses
    "DPTrainingMetrics",
    "DPTrainingArtifacts",
    # Training entry points
    "train_dp_configuration",
    "train_config_3_dp_only",
    "train_config_4_dp_plus_kanon",
    "sweep_epsilons",
    # Constants
    "DEFAULT_TARGET_DELTA",
    "DEFAULT_MAX_GRAD_NORM",
    "DEFAULT_EPSILONS",
    "DEFAULT_MAX_PHYSICAL_BATCH_SIZE",
    "DEFAULT_DP_EPOCHS",
    "DEFAULT_DP_PATIENCE",
    "CONFIG_3_SLUG_PREFIX",
    "CONFIG_4_SLUG_PREFIX",
    # Helpers
    "dp_checkpoint_slug",
    "load_dp_metadata",
]

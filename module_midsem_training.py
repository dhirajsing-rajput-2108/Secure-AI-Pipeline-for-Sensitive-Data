"""
module_midsem_training.py
=========================
Training scaffolding for the four-configuration ablation.

Sprint-1 upgrades over the mid-semester version
------------------------------------------------
1.  Early stopping on validation loss (default ON; patience = 3).
    Directly addresses Section 5.4 of the mid-sem report, which observed
    that val_loss bottoms out around epoch 2 and climbs thereafter.
2.  Best-weights restoration — the returned model has the parameters that
    achieved the lowest val_loss, not the last-epoch parameters.
3.  Full checkpoint persistence — every training run can optionally write
    a self-describing bundle to disk (model + preprocessor + splits +
    metadata) that downstream sprints (FastAPI service, MIA attack,
    attribute-inference attack, Pareto plot) will load from.
4.  A `LoadedConfiguration` handle for post-hoc inference and attack
    evaluation, with a `.reconstruct_splits()` method that recovers the
    exact train/val/test partition used during training.

Backward-compatibility guarantees
---------------------------------
* Every public name that existed in the mid-sem version still exists.
* Every field of `TrainingMetrics` that existed still exists (new fields
  have default values, so old code that constructs `TrainingMetrics`
  positionally is unaffected).
* `train_one_configuration`, `train_config_1_baseline`,
  `train_config_2_k_anon_l_div`, and `train_config_2_k_anonymised`
  still accept the same arguments and still return `TrainingMetrics`.
  Persistence is opt-in via a new `persist_to=` keyword.

Author: Rajput Dhirajsing Rajeshsing  (2024DA04378)
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Reproducibility
# =============================================================================

DEFAULT_SEED: int = 42


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy and PyTorch so two runs produce the same numbers.

    On CPU-only machines this is deterministic. If you later port to CUDA,
    the cudnn flags below tighten determinism at some speed cost.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Target + feature engineering
# =============================================================================

# UCI Diabetes target column has three values:
#   "<30" -> patient readmitted within 30 days  (positive class)
#   ">30" -> patient readmitted, but later       (negative)
#   "NO"  -> never readmitted                    (negative)
TARGET_COLUMN: str = "readmitted"
POSITIVE_LABEL: str = "<30"


def binarise_target(series: pd.Series) -> np.ndarray:
    """Map the three-class 'readmitted' column to a 0/1 NumPy array."""
    return (series.astype(str).str.strip() == POSITIVE_LABEL).astype(np.int64).values


# Columns that are not features. The training pipeline never sees them.
DROP_FROM_FEATURES: tuple[str, ...] = (
    TARGET_COLUMN,
    "weight",             # ~97% missing in the source data; not useful
    "payer_code",         # billing artefact, not clinical
    "medical_specialty",  # very high cardinality; sparse signal
)


def _split_columns_by_dtype(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) inferred from dtypes."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_feature_pipeline(df_features: pd.DataFrame) -> Pipeline:
    """Build a ColumnTransformer that one-hot encodes categoricals and
    standard-scales numerics, wrapped in an sklearn Pipeline."""
    numeric_cols, categorical_cols = _split_columns_by_dtype(df_features)
    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("features", transformer)])


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Split off the target and drop non-feature columns; return (X_df, y).

    Also normalises the UCI missing-value sentinel "?" to explicit string
    category "Missing" so OneHotEncoder treats it as its own value.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"target column {TARGET_COLUMN!r} not present")

    y = binarise_target(df[TARGET_COLUMN])
    feature_cols = [c for c in df.columns if c not in DROP_FROM_FEATURES]
    X = df[feature_cols].copy()

    for c in X.columns:
        if X[c].dtype == object or pd.api.types.is_string_dtype(X[c]):
            X[c] = X[c].replace("?", "Missing").astype(str)

    return X, y


# =============================================================================
# MLP
# =============================================================================

class MLPClassifier(nn.Module):
    """Small two-hidden-layer MLP for binary classification on tabular data.

    Architecture is deliberately simple — the dissertation is about the
    privacy stack, not squeezing the last point of accuracy out of the
    model. `hidden` and `dropout` are stored on the module so we can
    round-trip them through the checkpoint file without external metadata.
    """

    def __init__(self, in_features: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.hidden = hidden
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # shape (batch,)


# =============================================================================
# Early stopping
# =============================================================================

class EarlyStopping:
    """Standard patience-based early stopping on a monitored scalar (val_loss).

    Stores the best-so-far model state on CPU and restores it on request.
    Deliberately simple: no LR-schedule coupling, no distributed logic.
    """

    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = float("inf")
        self.best_epoch: int = 0
        self.best_state: dict[str, torch.Tensor] | None = None
        self.wait: int = 0
        self.should_stop: bool = False

    def step(self, val_loss: float, epoch: int, model: nn.Module) -> bool:
        """Return True if training should stop after this epoch."""
        improved = val_loss < (self.best_loss - self.min_delta)
        if improved:
            self.best_loss = val_loss
            self.best_epoch = epoch
            # Detach + clone so we do not keep autograd graphs around.
            self.best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
        return self.should_stop

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# =============================================================================
# TrainingMetrics — backward compatible + new fields
# =============================================================================

@dataclass
class TrainingMetrics:
    """Result object returned by every training call.

    Preserves every field from the mid-sem version. New fields all have
    defaults so callers written against the old dataclass keep working.
    """
    config_name: str
    n_train_rows: int
    n_val_rows: int
    n_test_rows: int
    n_features_after_encoding: int
    epochs: int                                       # epochs planned (max)
    train_loss_by_epoch: list[float] = field(default_factory=list)
    val_loss_by_epoch: list[float] = field(default_factory=list)
    test_accuracy: float = 0.0
    test_f1: float = 0.0
    test_roc_auc: float = 0.0
    # --- Sprint-1 additions (all with defaults for backward compat) -----
    epochs_run: int = 0
    best_epoch: int = 0                               # 1-indexed
    best_val_loss: float = float("inf")
    stopped_early: bool = False
    early_stopping_patience: int = 0
    checkpoint_dir: str | None = None
    seed: int = DEFAULT_SEED
    pos_weight: float = 1.0

    def pretty(self) -> str:
        lines = [
            f"[{self.config_name}]",
            f"  train / val / test rows : "
            f"{self.n_train_rows:,} / {self.n_val_rows:,} / {self.n_test_rows:,}",
            f"  encoded feature count   : {self.n_features_after_encoding:,}",
        ]
        if self.stopped_early:
            lines.append(
                f"  epochs run              : {self.epochs_run}/{self.epochs}"
                f" (early-stopped, patience={self.early_stopping_patience})"
            )
        else:
            lines.append(f"  epochs run              : {self.epochs_run}/{self.epochs}")

        if self.best_epoch:
            lines.append(f"  best epoch (val loss)   : {self.best_epoch}")
            lines.append(f"  best val loss           : {self.best_val_loss:.4f}")

        if self.train_loss_by_epoch:
            lines.append(f"  last train loss         : {self.train_loss_by_epoch[-1]:.4f}")
        if self.val_loss_by_epoch:
            lines.append(f"  last  val  loss         : {self.val_loss_by_epoch[-1]:.4f}")

        lines.extend([
            f"  test accuracy           : {self.test_accuracy:.4f}",
            f"  test F1 (positive class): {self.test_f1:.4f}",
            f"  test ROC-AUC            : {self.test_roc_auc:.4f}",
        ])
        if self.checkpoint_dir:
            lines.append(f"  checkpoint              : {self.checkpoint_dir}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Persistence — checkpoint + LoadedConfiguration
# =============================================================================

_CHECKPOINT_FILES = {
    "model":        "model.pt",
    "preprocessor": "preprocessor.joblib",
    "dataframe":    "training_dataframe.parquet",
    "splits":       "split_indices.npz",
    "metadata":     "metadata.json",
}


def default_checkpoint_dir(config_slug: str, root: str | Path = "artifacts/checkpoints") -> Path:
    """Canonical location for a config's persisted artifacts."""
    return Path(root) / config_slug


def _slugify(config_name: str) -> str:
    """Convert 'Config 1 — Baseline (no defence)' to a Windows-safe slug.

    Keeps only [a-z0-9_]; every other character becomes an underscore.
    Guarantees the result is a valid directory name on POSIX and Windows.
    """
    import re
    s = config_name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)   # anything not [a-z0-9] becomes _
    s = re.sub(r"_+", "_", s).strip("_")  # collapse runs of _ and trim
    return s or "config"


@dataclass
class TrainingArtifacts:
    """Everything a training call produces, bundled for downstream sprints."""
    metrics: TrainingMetrics
    model: MLPClassifier
    pipeline: Pipeline
    training_dataframe: pd.DataFrame
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    metadata: dict
    checkpoint_dir: Path | None = None


@dataclass
class LoadedConfiguration:
    """A trained configuration reloaded from a checkpoint directory."""
    model: MLPClassifier
    pipeline: Pipeline
    metadata: dict
    checkpoint_dir: Path

    # -----------------------------------------------------------------
    # Inference helpers (used by FastAPI service + attack modules)
    # -----------------------------------------------------------------
    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        """Apply prepare_xy-style cleaning + the fitted encoder to a
        raw dataframe. Ignores the target column if present."""
        X = df.copy()
        if TARGET_COLUMN in X.columns:
            X = X.drop(columns=[TARGET_COLUMN])
        drop_present = [c for c in DROP_FROM_FEATURES if c in X.columns and c != TARGET_COLUMN]
        if drop_present:
            X = X.drop(columns=drop_present)
        for c in X.columns:
            if X[c].dtype == object or pd.api.types.is_string_dtype(X[c]):
                X[c] = X[c].replace("?", "Missing").astype(str)
        return self.pipeline.transform(X).astype(np.float32)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return P(readmitted<30) for every row in df."""
        X_enc = self._prepare(df)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X_enc, dtype=torch.float32))
            return torch.sigmoid(logits).numpy()

    def predict_logits(self, df: pd.DataFrame) -> np.ndarray:
        """Return raw logits — useful for calibration / attack studies."""
        X_enc = self._prepare(df)
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(X_enc, dtype=torch.float32)).numpy()

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(df) >= threshold).astype(int)

    # -----------------------------------------------------------------
    # Split reconstruction (required by MIA / attribute inference)
    # -----------------------------------------------------------------
    def reconstruct_splits(self) -> dict[str, Any]:
        """Reload the exact train/val/test splits used during training.

        Returns a dict with:
            df_train, df_val, df_test           : raw pandas dataframes
            X_train_enc, X_val_enc, X_test_enc  : float32 encoded arrays
            y_train, y_val, y_test              : int64 label arrays
        This is what the MIA attack (Sprint 4) uses to line up
        "IN" (training) and "OUT" (test) confidence vectors.
        """
        df = pd.read_parquet(self.checkpoint_dir / _CHECKPOINT_FILES["dataframe"])
        splits = np.load(self.checkpoint_dir / _CHECKPOINT_FILES["splits"])
        train_idx = splits["train_idx"]
        val_idx   = splits["val_idx"]
        test_idx  = splits["test_idx"]

        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val   = df.iloc[val_idx].reset_index(drop=True)
        df_test  = df.iloc[test_idx].reset_index(drop=True)

        X_train_df, y_train = prepare_xy(df_train)
        X_val_df,   y_val   = prepare_xy(df_val)
        X_test_df,  y_test  = prepare_xy(df_test)

        X_train_enc = self.pipeline.transform(X_train_df).astype(np.float32)
        X_val_enc   = self.pipeline.transform(X_val_df).astype(np.float32)
        X_test_enc  = self.pipeline.transform(X_test_df).astype(np.float32)

        return {
            "df_train": df_train, "df_val": df_val, "df_test": df_test,
            "X_train_enc": X_train_enc, "X_val_enc": X_val_enc, "X_test_enc": X_test_enc,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
        }


def load_configuration_checkpoint(checkpoint_dir: str | Path) -> LoadedConfiguration:
    """Reload a training bundle saved by `train_and_persist_configuration`."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")

    # -------- metadata ---------------------------------------------------
    with open(checkpoint_dir / _CHECKPOINT_FILES["metadata"], "r") as fh:
        metadata: dict = json.load(fh)

    # -------- model ------------------------------------------------------
    # weights_only=False is required because we bundle small dict metadata
    # with the state_dict; the file is trusted because we wrote it.
    bundle = torch.load(
        checkpoint_dir / _CHECKPOINT_FILES["model"],
        map_location="cpu",
        weights_only=False,
    )
    arch = bundle["arch"]
    model = MLPClassifier(
        in_features=arch["in_features"],
        hidden=arch["hidden"],
        dropout=arch["dropout"],
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    # -------- preprocessor ----------------------------------------------
    pipeline: Pipeline = joblib.load(checkpoint_dir / _CHECKPOINT_FILES["preprocessor"])

    return LoadedConfiguration(
        model=model,
        pipeline=pipeline,
        metadata=metadata,
        checkpoint_dir=checkpoint_dir,
    )


def _persist_artifacts(
    checkpoint_dir: Path,
    model: MLPClassifier,
    pipeline: Pipeline,
    training_dataframe: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    metadata: dict,
) -> None:
    """Write all five checkpoint files atomically-ish (best-effort)."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # model
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": {
                "in_features": model.in_features,
                "hidden": model.hidden,
                "dropout": model.dropout,
            },
            "class_name": "MLPClassifier",
        },
        checkpoint_dir / _CHECKPOINT_FILES["model"],
    )

    # preprocessor
    joblib.dump(pipeline, checkpoint_dir / _CHECKPOINT_FILES["preprocessor"])

    # training dataframe (parquet keeps dtypes cleanly)
    training_dataframe.to_parquet(
        checkpoint_dir / _CHECKPOINT_FILES["dataframe"], index=False
    )

    # split indices
    np.savez_compressed(
        checkpoint_dir / _CHECKPOINT_FILES["splits"],
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    # metadata JSON
    with open(checkpoint_dir / _CHECKPOINT_FILES["metadata"], "w") as fh:
        json.dump(metadata, fh, indent=2, default=str)


# =============================================================================
# DataLoaders
# =============================================================================

def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# =============================================================================
# Core training routine
# =============================================================================

def train_one_configuration(
    df: pd.DataFrame,
    config_name: str,
    *,
    epochs: int = 12,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden: int = 128,
    dropout: float = 0.2,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
    early_stopping: bool = True,
    patience: int = 3,
    min_delta: float = 1e-4,
    persist_to: str | Path | None = None,
    return_artifacts: bool = False,
) -> TrainingMetrics | TrainingArtifacts:
    """Train the MLP on the supplied dataframe.

    Parameters
    ----------
    df :
        Training dataframe. Must contain the target column `readmitted`.
        Direct identifiers (encounter_id, patient_nbr) should be dropped
        by the caller (helper wrappers do this).
    config_name :
        Human-readable label recorded in the metrics + metadata.
    epochs :
        Maximum epochs. Early stopping may cut this short.
    early_stopping :
        If True (default), monitor val_loss with `patience` epochs of
        no improvement, then restore the best-observed weights before
        evaluating on the test set. Directly fixes Section 5.4 of the
        mid-sem report.
    persist_to :
        Optional directory. If given, writes the full checkpoint bundle
        (model, preprocessor, dataframe, indices, metadata) there.
    return_artifacts :
        If True, return a `TrainingArtifacts` object instead of a bare
        `TrainingMetrics`. `TrainingArtifacts.metrics` still holds the
        metrics, so callers can extract them either way.

    Returns
    -------
    TrainingMetrics (default) or TrainingArtifacts.
    """
    seed_everything(seed)

    # ----- target + features -------------------------------------------------
    X_df, y = prepare_xy(df)

    # ----- 70 / 15 / 15 stratified split -----
    # We split on integer positional indices so we can serialise them.
    positional_idx = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        positional_idx, test_size=0.15, random_state=seed, stratify=y
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

    # ----- fit encoder on TRAIN only ----------------------------------------
    pipeline = build_feature_pipeline(X_train)
    X_train_enc = pipeline.fit_transform(X_train).astype(np.float32)
    X_val_enc   = pipeline.transform(X_val).astype(np.float32)
    X_test_enc  = pipeline.transform(X_test).astype(np.float32)
    n_features  = X_train_enc.shape[1]

    # ----- class imbalance handling -----------------------------------------
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight_val = neg / max(pos, 1.0)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)

    # ----- model / loss / optimiser -----------------------------------------
    model = MLPClassifier(in_features=n_features, hidden=hidden, dropout=dropout)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    train_loader = _make_loader(X_train_enc, y_train, batch_size, shuffle=True)
    val_loader   = _make_loader(X_val_enc,   y_val,   batch_size, shuffle=False)

    es = EarlyStopping(patience=patience, min_delta=min_delta) if early_stopping else None
    train_losses: list[float] = []
    val_losses:   list[float] = []
    epochs_run = 0
    stopped_early_flag = False

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        running, n_seen = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * xb.size(0)
            n_seen  += xb.size(0)
        train_loss = running / max(n_seen, 1)
        train_losses.append(train_loss)

        # ---- val ----
        model.eval()
        running, n_seen = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                running += float(loss.item()) * xb.size(0)
                n_seen  += xb.size(0)
        val_loss = running / max(n_seen, 1)
        val_losses.append(val_loss)

        epochs_run = epoch

        if verbose:
            marker = ""
            if es is not None and val_loss <= es.best_loss + 1e-12:
                marker = "  <-- best"
            print(
                f"  [{config_name}] epoch {epoch:>2}/{epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}{marker}"
            )

        if es is not None:
            should_stop = es.step(val_loss, epoch, model)
            if should_stop:
                stopped_early_flag = True
                if verbose:
                    print(
                        f"  [{config_name}] early stopping at epoch {epoch}; "
                        f"restoring weights from epoch {es.best_epoch} "
                        f"(val_loss={es.best_loss:.4f})"
                    )
                break

    # ----- restore best weights (if ES was on) -----------------------------
    if es is not None:
        es.restore_best(model)
        best_epoch = es.best_epoch or epochs_run
        best_val_loss = es.best_loss if es.best_state is not None else val_losses[-1]
    else:
        best_epoch = int(np.argmin(val_losses)) + 1 if val_losses else 0
        best_val_loss = float(min(val_losses)) if val_losses else float("inf")

    # ----- test-set evaluation ----------------------------------------------
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test_enc, dtype=torch.float32))
        probs  = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)

    metrics = TrainingMetrics(
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

    metadata = {
        "config_name": config_name,
        "config_slug": _slugify(config_name),
        "target_column": TARGET_COLUMN,
        "positive_label": POSITIVE_LABEL,
        "drop_from_features": list(DROP_FROM_FEATURES),
        "feature_columns_pre_encoding": list(X_df.columns),
        "n_features_after_encoding": int(n_features),
        "seed": int(seed),
        "hyperparameters": {
            "hidden": hidden,
            "dropout": dropout,
            "epochs_planned": epochs,
            "epochs_run": epochs_run,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "early_stopping": bool(early_stopping),
            "early_stopping_patience": (patience if early_stopping else 0),
            "min_delta": (min_delta if early_stopping else 0.0),
            "pos_weight": float(pos_weight_val),
        },
        "metrics": {
            "test_accuracy": metrics.test_accuracy,
            "test_f1": metrics.test_f1,
            "test_roc_auc": metrics.test_roc_auc,
            "best_epoch": metrics.best_epoch,
            "best_val_loss": metrics.best_val_loss,
            "train_loss_by_epoch": train_losses,
            "val_loss_by_epoch": val_losses,
        },
        "row_counts": {
            "train": int(len(y_train)),
            "val":   int(len(y_val)),
            "test":  int(len(y_test)),
        },
    }

    # ----- optional persistence --------------------------------------------
    checkpoint_dir_path: Path | None = None
    if persist_to is not None:
        checkpoint_dir_path = Path(persist_to)
        _persist_artifacts(
            checkpoint_dir=checkpoint_dir_path,
            model=model,
            pipeline=pipeline,
            training_dataframe=df.reset_index(drop=True),
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            metadata=metadata,
        )
        metrics.checkpoint_dir = str(checkpoint_dir_path)
        metadata["checkpoint_dir"] = str(checkpoint_dir_path)

    if return_artifacts:
        return TrainingArtifacts(
            metrics=metrics,
            model=model,
            pipeline=pipeline,
            training_dataframe=df.reset_index(drop=True),
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            metadata=metadata,
            checkpoint_dir=checkpoint_dir_path,
        )
    return metrics


# =============================================================================
# Public per-configuration helpers
# =============================================================================

def train_config_1_baseline(
    df_original: pd.DataFrame,
    *,
    persist_to: str | Path | None = None,
    return_artifacts: bool = False,
    **kwargs,
) -> TrainingMetrics | TrainingArtifacts:
    """Config 1 — no privacy defence. Original data, direct identifiers dropped."""
    df = df_original.drop(columns=["encounter_id", "patient_nbr"], errors="ignore")
    return train_one_configuration(
        df,
        config_name="Config 1 — Baseline (no defence)",
        persist_to=persist_to,
        return_artifacts=return_artifacts,
        **kwargs,
    )


def train_config_2_k_anon_l_div(
    df_anon: pd.DataFrame,
    *,
    persist_to: str | Path | None = None,
    return_artifacts: bool = False,
    **kwargs,
) -> TrainingMetrics | TrainingArtifacts:
    """Config 2 — train on the k-anonymous + l-diverse dataframe."""
    return train_one_configuration(
        df_anon,
        config_name="Config 2 — k-anonymity (k=5) + l-diversity (l=3)",
        persist_to=persist_to,
        return_artifacts=return_artifacts,
        **kwargs,
    )


# Backwards-compatible alias for older code that still imports this name.
train_config_2_k_anonymised = train_config_2_k_anon_l_div


# =============================================================================
# Public helper: full-artifact training (convenience for Sprint 2+ code)
# =============================================================================

def train_and_persist_configuration(
    df: pd.DataFrame,
    config_name: str,
    checkpoint_dir: str | Path,
    **kwargs,
) -> TrainingArtifacts:
    """Shortcut that always persists and always returns TrainingArtifacts."""
    return train_one_configuration(  # type: ignore[return-value]
        df,
        config_name=config_name,
        persist_to=checkpoint_dir,
        return_artifacts=True,
        **kwargs,
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Core
    "MLPClassifier",
    "TrainingMetrics",
    "TrainingArtifacts",
    "LoadedConfiguration",
    "EarlyStopping",
    # Feature engineering
    "TARGET_COLUMN",
    "POSITIVE_LABEL",
    "DROP_FROM_FEATURES",
    "binarise_target",
    "prepare_xy",
    "build_feature_pipeline",
    # Training entry points
    "train_one_configuration",
    "train_config_1_baseline",
    "train_config_2_k_anon_l_div",
    "train_config_2_k_anonymised",   # legacy alias
    "train_and_persist_configuration",
    # Persistence
    "default_checkpoint_dir",
    "load_configuration_checkpoint",
    # Reproducibility
    "seed_everything",
    "DEFAULT_SEED",
]

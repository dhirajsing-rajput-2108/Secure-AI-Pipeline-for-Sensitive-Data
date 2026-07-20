"""
module_midsem_training.py
=========================
Demo Pieces 3 & 4 — train identical MLPs on (i) the original dataset and
(ii) the k-anonymised dataset, then compare predictive utility.

Both configurations share:
  * the same target-binarisation rule  (`<30` -> positive class)
  * the same feature-engineering pipeline (one-hot encoding for categoricals,
    standard-scaling for numerics)
  * the same train/validation/test split with a fixed random seed
  * the same MLP architecture, optimiser, loss, batch size, and epoch count

The *only* thing that differs is the input dataframe — this guarantees
that any observed utility gap is attributable to k-anonymity, not to
preprocessing noise.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict, field
from typing import Iterable

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


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy, and PyTorch so two runs produce the same numbers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)  # keep speed; we don't need bit-exact


# ---------------------------------------------------------------------------
# Target + feature engineering
# ---------------------------------------------------------------------------

# UCI Diabetes target column has three values:
#   "<30" -> patient readmitted within 30 days  (positive class)
#   ">30" -> patient readmitted, but later      (negative)
#   "NO"  -> never readmitted                   (negative)
# Standard practice in the published literature on this dataset binarises
# the target this way because the clinically interesting outcome is the
# *early* readmission (it is what reimbursement penalties target).
TARGET_COLUMN = "readmitted"


def binarise_target(series: pd.Series) -> np.ndarray:
    """Map the three-class 'readmitted' column to a 0/1 NumPy array."""
    return (series.astype(str).str.strip() == "<30").astype(np.int64).values


# Columns that are not features. The training pipeline never sees them.
DROP_FROM_FEATURES: tuple[str, ...] = (
    TARGET_COLUMN,
    "weight",          # ~97% missing in the source data; not useful
    "payer_code",      # billing artefact, not clinical
    "medical_specialty",  # very high cardinality; sparse signal
)


def _split_columns_by_dtype(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) inferred from dtypes."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_feature_pipeline(
    df_features: pd.DataFrame,
) -> tuple[Pipeline, list[str], list[str]]:
    """
    Build a ColumnTransformer that one-hot encodes categoricals and
    standard-scales numerics. The transformer is wrapped in a Pipeline so
    it has the standard sklearn fit/transform interface.
    """
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
    return Pipeline([("features", transformer)]), numeric_cols, categorical_cols


def prepare_xy(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Split off the target and drop non-feature columns. Returns (X_df, y).
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"target column {TARGET_COLUMN!r} not present in dataframe")

    y = binarise_target(df[TARGET_COLUMN])
    feature_cols = [c for c in df.columns if c not in DROP_FROM_FEATURES]
    X = df[feature_cols].copy()

    # Replace the UCI missing-value sentinel "?" with explicit NaN-equivalent
    # categorical strings so OneHotEncoder treats them as their own category
    # rather than refusing to encode them.
    for c in X.columns:
        if X[c].dtype == object or pd.api.types.is_string_dtype(X[c]):
            X[c] = X[c].replace("?", "Missing").astype(str)

    return X, y


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLPClassifier(nn.Module):
    """
    A small two-hidden-layer MLP for binary classification on tabular data.
    Architecture deliberately kept simple — the dissertation is about the
    privacy stack, not about wringing the last point of accuracy out of
    the model.
    """

    def __init__(self, in_features: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
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
        return self.net(x).squeeze(-1)  # -> shape (batch,)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainingMetrics:
    config_name: str
    n_train_rows: int
    n_val_rows: int
    n_test_rows: int
    n_features_after_encoding: int
    epochs: int
    train_loss_by_epoch: list[float] = field(default_factory=list)
    val_loss_by_epoch: list[float] = field(default_factory=list)
    test_accuracy: float = 0.0
    test_f1: float = 0.0
    test_roc_auc: float = 0.0

    def pretty(self) -> str:
        return (
            f"[{self.config_name}]\n"
            f"  train / val / test rows : {self.n_train_rows:,} / "
            f"{self.n_val_rows:,} / {self.n_test_rows:,}\n"
            f"  encoded feature count   : {self.n_features_after_encoding:,}\n"
            f"  epochs run              : {self.epochs}\n"
            f"  final train loss        : {self.train_loss_by_epoch[-1]:.4f}\n"
            f"  final val   loss        : {self.val_loss_by_epoch[-1]:.4f}\n"
            f"  test accuracy           : {self.test_accuracy:.4f}\n"
            f"  test F1 (positive class): {self.test_f1:.4f}\n"
            f"  test ROC-AUC            : {self.test_roc_auc:.4f}"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_one_configuration(
    df: pd.DataFrame,
    config_name: str,
    *,
    epochs: int = 12,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> TrainingMetrics:
    """
    Train the MLP on the supplied dataframe and return a TrainingMetrics
    object with test-set Accuracy, F1, and ROC-AUC.

    All hyperparameters are passed in, so Config 1 and Config 2 can be
    trained with literally the same call signature — guaranteeing the
    comparison is fair.
    """
    seed_everything(seed)

    # ----- target + features -----
    X_df, y = prepare_xy(df)

    # ----- splits: 70 / 15 / 15, stratified on the target -----
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_df, y, test_size=0.15, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.15 / 0.85,            # 15% of the original total
        random_state=seed, stratify=y_train_val,
    )

    # ----- fit feature pipeline on TRAIN ONLY, then apply to val/test -----
    pipeline, _, _ = build_feature_pipeline(X_train)
    X_train_enc = pipeline.fit_transform(X_train).astype(np.float32)
    X_val_enc   = pipeline.transform(X_val).astype(np.float32)
    X_test_enc  = pipeline.transform(X_test).astype(np.float32)
    n_features  = X_train_enc.shape[1]

    # ----- class imbalance handling: pos_weight in BCEWithLogitsLoss -----
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)

    # ----- model, loss, optimiser -----
    model = MLPClassifier(in_features=n_features)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # ----- loaders -----
    train_loader = _make_loader(X_train_enc, y_train, batch_size, shuffle=True)
    val_loader   = _make_loader(X_val_enc,   y_val,   batch_size, shuffle=False)

    train_losses: list[float] = []
    val_losses:   list[float] = []

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        running = 0.0
        n_seen  = 0
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
        running = 0.0
        n_seen  = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                running += float(loss.item()) * xb.size(0)
                n_seen  += xb.size(0)
        val_loss = running / max(n_seen, 1)
        val_losses.append(val_loss)

        if verbose:
            print(
                f"  [{config_name}] epoch {epoch:>2}/{epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            )

    # ----- test-set evaluation -----
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
    )
    return metrics


# ---------------------------------------------------------------------------
# Public helpers — Config 1 (baseline) and Config 2 (k-anonymised)
# ---------------------------------------------------------------------------

def train_config_1_baseline(df_original: pd.DataFrame, **kwargs) -> TrainingMetrics:
    """Config 1 — no defence applied. Original data, identifiers dropped."""
    df = df_original.drop(
        columns=["encounter_id", "patient_nbr"], errors="ignore"
    )
    return train_one_configuration(df, config_name="Config 1 — Baseline", **kwargs)


def train_config_2_k_anonymised(df_anon: pd.DataFrame, **kwargs) -> TrainingMetrics:
    """Config 2 — train on the k-anonymised dataframe produced by
    module_deidentification.de_identify()."""
    return train_one_configuration(df_anon, config_name="Config 2 — k-anonymity (k=5)", **kwargs)


__all__ = [
    "MLPClassifier",
    "TrainingMetrics",
    "binarise_target",
    "prepare_xy",
    "build_feature_pipeline",
    "train_one_configuration",
    "train_config_1_baseline",
    "train_config_2_k_anonymised",
    "seed_everything",
]

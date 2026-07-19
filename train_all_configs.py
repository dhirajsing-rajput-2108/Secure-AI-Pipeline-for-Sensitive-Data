"""
train_all_configs.py
====================
Canonical training entrypoint for the second half of the semester.

Runs every configuration end-to-end with early stopping enabled and
persists complete checkpoint bundles under artifacts/checkpoints/.
Downstream sprints (FastAPI inference service, MIA / attribute-inference
attacks, Pareto plot) all read from these checkpoints — this script is
their single source of truth.

Currently trains:
    Config 1  —  Baseline (no defence)                → config_1_baseline/
    Config 2  —  k-anonymity (k=5) + l-diversity (l=3) → config_2_k_anon_l_div/

Sprint 2 will extend this file to add:
    Config 3  —  DP-SGD only (ε ∈ {1, 3, 8})           → config_3_dp_only_eps*/
    Config 4  —  DP-SGD + k-anon + l-div                → config_4_dp_kanon_eps*/

Usage
-----
    # First run (downloads the UCI dataset ~20 MB via midsem_demo.ensure_dataset)
    python train_all_configs.py

    # Faster smoke test
    python train_all_configs.py --quick

    # Custom output root
    python train_all_configs.py --root artifacts/checkpoints_v2

    # Skip Config 2 (train baseline only)
    python train_all_configs.py --only-config 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

# Local modules
from midsem_demo import ensure_dataset, RAW_CSV                  # dataset acquisition
from module_deidentification import de_identify, DEFAULT_LDIV_ATTRIBUTE
from module_midsem_training import (
    default_checkpoint_dir,
    load_configuration_checkpoint,
    train_and_persist_configuration,
)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

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
# Configurations trained by this script
# ---------------------------------------------------------------------------

DEFAULT_EPOCHS_FULL  = 30    # generous ceiling; ES will cut it short
DEFAULT_EPOCHS_QUICK = 6
DEFAULT_PATIENCE     = 3

CONFIG_1_SLUG = "config_1_baseline"
CONFIG_2_SLUG = "config_2_k_anon_l_div"


# ---------------------------------------------------------------------------
# Individual training runs
# ---------------------------------------------------------------------------

def train_config_1(
    df_original: pd.DataFrame,
    root: Path,
    epochs: int,
    patience: int,
) -> Path:
    banner("Config 1 — Baseline MLP on the original dataset (no defence)")
    ckpt_dir = default_checkpoint_dir(CONFIG_1_SLUG, root=root)
    print(f"[config-1] input rows: {len(df_original):,}")
    print(f"[config-1] checkpoint dir: {ckpt_dir}")
    t0 = time.time()
    artifacts = train_and_persist_configuration(
        df=df_original.drop(columns=["encounter_id", "patient_nbr"], errors="ignore"),
        config_name="Config 1 — Baseline (no defence)",
        checkpoint_dir=ckpt_dir,
        epochs=epochs,
        patience=patience,
        early_stopping=True,
        verbose=True,
    )
    print(f"[config-1] finished in {time.time() - t0:.1f}s")
    print()
    print(artifacts.metrics.pretty())
    return ckpt_dir


def train_config_2(
    df_original: pd.DataFrame,
    root: Path,
    epochs: int,
    patience: int,
) -> Path:
    banner("Config 2 — MLP on k-anon + l-diverse data (k=5, l=3)")

    section("2a) Run de-identification pipeline")
    t0 = time.time()
    result = de_identify(
        df_original,
        k=5,
        l=3,
        sensitive_attribute=DEFAULT_LDIV_ATTRIBUTE,
    )
    print(f"[config-2] de-identification finished in {time.time() - t0:.2f}s")
    print(f"[config-2] {result.suppression_stats.n_input_rows:,} rows in  →  "
          f"{result.suppression_stats.n_output_rows:,} rows out")
    print(f"[config-2] verifier verdict: "
          f"k={'PASS' if result.verifier_report.k_passes else 'FAIL'} / "
          f"l={'PASS' if result.verifier_report.l_passes else 'FAIL'}")
    if not result.verifier_report.passes:
        raise SystemExit(
            "de_identify() produced a dataset that fails the verifier — refusing to proceed."
        )

    section("2b) Train MLP on the anonymised dataframe")
    ckpt_dir = default_checkpoint_dir(CONFIG_2_SLUG, root=root)
    print(f"[config-2] checkpoint dir: {ckpt_dir}")
    t0 = time.time()
    artifacts = train_and_persist_configuration(
        df=result.dataframe,
        config_name="Config 2 — k-anonymity (k=5) + l-diversity (l=3)",
        checkpoint_dir=ckpt_dir,
        epochs=epochs,
        patience=patience,
        early_stopping=True,
        verbose=True,
    )
    print(f"[config-2] finished in {time.time() - t0:.1f}s")
    print()
    print(artifacts.metrics.pretty())
    return ckpt_dir


# ---------------------------------------------------------------------------
# Comparison + summary
# ---------------------------------------------------------------------------

def print_comparison(ckpt_dirs: dict[str, Path]) -> dict:
    banner("Utility summary — all configurations trained")

    rows = [("Config", "Test Acc", "Test F1", "Test ROC-AUC", "Epochs run", "Best epoch")]
    summary: dict[str, dict] = {}
    for slug, ckpt_dir in ckpt_dirs.items():
        loaded = load_configuration_checkpoint(ckpt_dir)
        m = loaded.metadata["metrics"]
        hp = loaded.metadata["hyperparameters"]
        rows.append((
            loaded.metadata["config_name"],
            f"{m['test_accuracy']:.4f}",
            f"{m['test_f1']:.4f}",
            f"{m['test_roc_auc']:.4f}",
            f"{hp['epochs_run']}/{hp['epochs_planned']}",
            f"{m['best_epoch']}",
        ))
        summary[slug] = {
            "config_name": loaded.metadata["config_name"],
            "test_accuracy": m["test_accuracy"],
            "test_f1": m["test_f1"],
            "test_roc_auc": m["test_roc_auc"],
            "epochs_run": hp["epochs_run"],
            "epochs_planned": hp["epochs_planned"],
            "best_epoch": m["best_epoch"],
            "best_val_loss": m["best_val_loss"],
            "checkpoint_dir": str(ckpt_dir),
        }

    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    for i, r in enumerate(rows):
        line = "  " + "  ".join(str(r[j]).ljust(widths[j]) for j in range(len(r)))
        print(line)
        if i == 0:
            print("  " + "  ".join("-" * w for w in widths))
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train all pipeline configurations and persist checkpoints."
    )
    parser.add_argument(
        "--root", type=Path, default=Path("artifacts/checkpoints"),
        help="Output root directory for checkpoints.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override max epochs (default: 30 full, 6 quick).",
    )
    parser.add_argument(
        "--patience", type=int, default=DEFAULT_PATIENCE,
        help="Early-stopping patience (default: 3).",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Sample 20k rows + 6 epochs for a fast smoke run.",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Fail if the UCI dataset is not already cached.",
    )
    parser.add_argument(
        "--only-config", type=int, choices=[1, 2], default=None,
        help="Train only Config N (skip the others).",
    )
    args = parser.parse_args()

    banner("Secure AI Pipeline — train_all_configs.py")
    print("  Student   : Rajput Dhirajsing Rajeshsing  (2024DA04378)")
    print("  Sprint 1  : Config 1 (baseline)  +  Config 2 (k=5, l=3)")
    print("              — both with early stopping + full checkpoint persistence")

    epochs = args.epochs or (DEFAULT_EPOCHS_QUICK if args.quick else DEFAULT_EPOCHS_FULL)
    print(f"  Config    : max epochs = {epochs}   patience = {args.patience}   "
          f"root = {args.root}")

    # ----- dataset ---------------------------------------------------------
    ensure_dataset(skip_download=args.skip_download)
    df_original = pd.read_csv(RAW_CSV)
    print(f"\n[setup] loaded {len(df_original):,} rows × {len(df_original.columns)} columns")
    if args.quick:
        df_original = df_original.sample(
            n=min(20000, len(df_original)), random_state=42
        ).reset_index(drop=True)
        print(f"[setup] --quick mode: sub-sampled to {len(df_original):,} rows")

    args.root.mkdir(parents=True, exist_ok=True)

    ckpt_dirs: dict[str, Path] = {}
    if args.only_config in (None, 1):
        ckpt_dirs[CONFIG_1_SLUG] = train_config_1(
            df_original, root=args.root, epochs=epochs, patience=args.patience,
        )
    if args.only_config in (None, 2):
        ckpt_dirs[CONFIG_2_SLUG] = train_config_2(
            df_original, root=args.root, epochs=epochs, patience=args.patience,
        )

    if not ckpt_dirs:
        print("[warn] no configs trained (check --only-config).")
        return 1

    summary = print_comparison(ckpt_dirs)

    summary_path = args.root / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] summary written to {summary_path}")

    banner("All requested configurations trained and persisted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

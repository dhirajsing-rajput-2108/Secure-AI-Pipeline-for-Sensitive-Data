"""
train_dp_configs.py
===================
Sprint-2 orchestrator: trains Configurations 3 and 4 across ε ∈ {1, 3, 8}
and persists six checkpoints under artifacts/checkpoints/, using the same
on-disk contract as Configs 1 and 2 so downstream sprints load them
with `module_midsem_training.load_configuration_checkpoint`.

Produced checkpoints (default sweep):

    artifacts/checkpoints/
    ├── config_1_baseline/            (from Sprint 1)
    ├── config_2_k_anon_l_div/         (from Sprint 1)
    ├── config_3_dp_only_eps1/         ← NEW
    ├── config_3_dp_only_eps3/         ← NEW
    ├── config_3_dp_only_eps8/         ← NEW
    ├── config_4_dp_kanon_eps1/        ← NEW
    ├── config_4_dp_kanon_eps3/        ← NEW
    └── config_4_dp_kanon_eps8/        ← NEW

Usage
-----
    # Full sweep — Configs 3 & 4 at ε ∈ {1, 3, 8}. Slow: ~1-2 hours on CPU.
    python train_dp_configs.py

    # Faster smoke test: 20k-row sample, 4 epochs.
    python train_dp_configs.py --quick

    # Custom ε grid.
    python train_dp_configs.py --epsilons 1 3 8

    # Only Config 3 (skip 4), only ε=8.
    python train_dp_configs.py --only-config 3 --epsilons 8

Runtime notes
-------------
DP-SGD is slower than non-DP training because per-sample gradients are
computed and clipped. On an Intel i5 8th-gen CPU with 8 GB RAM (the mid-
sem report's target hardware), one full config-3 run at ε=1 with the
default 15 epochs takes ~15-25 minutes. A full sweep of 6 checkpoints
therefore takes 1.5–2.5 hours. Use `--quick` for iteration.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

# Sprint-1 modules
from midsem_demo import ensure_dataset, RAW_CSV
from module_deidentification import de_identify, DEFAULT_LDIV_ATTRIBUTE
from module_midsem_training import load_configuration_checkpoint

# Sprint-2 module
from module_dp_training import (
    CONFIG_3_SLUG_PREFIX,
    CONFIG_4_SLUG_PREFIX,
    DEFAULT_DP_EPOCHS,
    DEFAULT_DP_PATIENCE,
    DEFAULT_EPSILONS,
    DEFAULT_TARGET_DELTA,
    dp_checkpoint_slug,
    sweep_epsilons,
    train_config_3_dp_only,
    train_config_4_dp_plus_kanon,
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
# Sweep runners
# ---------------------------------------------------------------------------

def run_config_3_sweep(
    df_original: pd.DataFrame,
    root: Path,
    epsilons: list[float],
    target_delta: float,
    epochs: int,
    patience: int,
) -> dict[float, dict]:
    banner(f"Config 3 — DP-SGD only  (sweep over ε ∈ {epsilons})")
    print(f"[config-3] input rows: {len(df_original):,}")
    print(f"[config-3] identifiers dropped inside trainer; features same as Config 1")

    t0 = time.time()
    results = sweep_epsilons(
        df=df_original,
        config_prefix=CONFIG_3_SLUG_PREFIX,
        trainer_fn=train_config_3_dp_only,
        epsilons=epsilons,
        root=root,
        target_delta=target_delta,
        epochs=epochs,
        patience=patience,
        early_stopping=True,
        verbose=True,
    )
    print(f"\n[config-3] finished full sweep in {time.time() - t0:.1f}s")
    return {eps: r.to_dict() for eps, r in results.items()}


def run_config_4_sweep(
    df_original: pd.DataFrame,
    root: Path,
    epsilons: list[float],
    target_delta: float,
    epochs: int,
    patience: int,
) -> dict[float, dict]:
    banner(f"Config 4 — DP-SGD + k-anon (k=5) + l-div (l=3)  (sweep over ε ∈ {epsilons})")

    section("4a) Run de-identification pipeline (shared across all ε values)")
    t0 = time.time()
    result = de_identify(
        df_original,
        k=5,
        l=3,
        sensitive_attribute=DEFAULT_LDIV_ATTRIBUTE,
    )
    print(f"[config-4] de-identification finished in {time.time() - t0:.2f}s")
    print(f"[config-4] {result.suppression_stats.n_input_rows:,} rows in  →  "
          f"{result.suppression_stats.n_output_rows:,} rows out")
    print(f"[config-4] verifier: "
          f"k={'PASS' if result.verifier_report.k_passes else 'FAIL'} / "
          f"l={'PASS' if result.verifier_report.l_passes else 'FAIL'}")
    if not result.verifier_report.passes:
        raise SystemExit(
            "de_identify() produced a dataset that fails the verifier — refusing to proceed."
        )

    section(f"4b) DP-SGD sweep on the anonymised dataframe (ε ∈ {epsilons})")
    t0 = time.time()
    results = sweep_epsilons(
        df=result.dataframe,
        config_prefix=CONFIG_4_SLUG_PREFIX,
        trainer_fn=train_config_4_dp_plus_kanon,
        epsilons=epsilons,
        root=root,
        target_delta=target_delta,
        epochs=epochs,
        patience=patience,
        early_stopping=True,
        verbose=True,
    )
    print(f"\n[config-4] finished full sweep in {time.time() - t0:.1f}s")
    return {eps: r.to_dict() for eps, r in results.items()}


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------

def load_all_dp_summaries(root: Path, sweep_results: dict) -> dict:
    """Re-read every DP checkpoint metadata.json from disk to build a
    canonical summary table."""
    summary: dict[str, dict] = {}
    for prefix, per_config in sweep_results.items():
        for eps in per_config:
            slug = dp_checkpoint_slug(prefix, eps)
            ckpt = root / slug
            loaded = load_configuration_checkpoint(ckpt)
            m = loaded.metadata["metrics"]
            hp = loaded.metadata["hyperparameters"]
            dp = loaded.metadata["dp"]
            summary[slug] = {
                "config_name": loaded.metadata["config_name"],
                "epsilon_target": dp["target_epsilon"],
                "epsilon_spent": dp["epsilon_spent"],
                "delta": dp["target_delta"],
                "noise_multiplier": dp["noise_multiplier"],
                "max_grad_norm": dp["max_grad_norm"],
                "sample_rate": dp["sample_rate"],
                "test_accuracy": m["test_accuracy"],
                "test_f1": m["test_f1"],
                "test_roc_auc": m["test_roc_auc"],
                "epochs_run": hp["epochs_run"],
                "epochs_planned": hp["epochs_planned"],
                "best_epoch": m["best_epoch"],
                "best_val_loss": m["best_val_loss"],
                "checkpoint_dir": str(ckpt),
            }
    return summary


def print_dp_comparison(summary: dict) -> None:
    banner("DP-SGD sweep summary — all configurations")
    rows = [(
        "Config", "ε target", "ε spent", "σ (noise)",
        "Test Acc", "Test F1", "Test ROC-AUC", "Epochs",
    )]
    for slug in sorted(summary.keys()):
        s = summary[slug]
        rows.append((
            s["config_name"],
            f"{s['epsilon_target']:.2f}",
            f"{s['epsilon_spent']:.3f}",
            f"{s['noise_multiplier']:.3f}",
            f"{s['test_accuracy']:.4f}",
            f"{s['test_f1']:.4f}",
            f"{s['test_roc_auc']:.4f}",
            f"{s['epochs_run']}/{s['epochs_planned']}",
        ))

    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    for i, r in enumerate(rows):
        line = "  " + "  ".join(str(r[j]).ljust(widths[j]) for j in range(len(r)))
        print(line)
        if i == 0:
            print("  " + "  ".join("-" * w for w in widths))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train DP-SGD Configs 3 and 4 across an ε sweep."
    )
    parser.add_argument(
        "--root", type=Path, default=Path("artifacts/checkpoints"),
        help="Output root directory for checkpoints.",
    )
    parser.add_argument(
        "--epsilons", type=float, nargs="+", default=list(DEFAULT_EPSILONS),
        help="ε values to sweep (default: 1 3 8).",
    )
    parser.add_argument(
        "--delta", type=float, default=DEFAULT_TARGET_DELTA,
        help=f"δ for privacy accounting (default: {DEFAULT_TARGET_DELTA:.0e}).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=f"Max epochs (default: {DEFAULT_DP_EPOCHS} full, 4 quick).",
    )
    parser.add_argument(
        "--patience", type=int, default=DEFAULT_DP_PATIENCE,
        help=f"Early-stopping patience (default: {DEFAULT_DP_PATIENCE}).",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="20k-row sample + 4 epochs for a fast smoke run.",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Fail if the UCI dataset is not already cached.",
    )
    parser.add_argument(
        "--only-config", type=int, choices=[3, 4], default=None,
        help="Train only Config N (skip the others).",
    )
    args = parser.parse_args()

    banner("Secure AI Pipeline — train_dp_configs.py")
    print("  Student   : Rajput Dhirajsing Rajeshsing  (2024DA04378)")
    print("  Sprint 2  : Config 3 (DP-SGD only)  +  Config 4 (DP + k-anon + l-div)")
    print(f"              ε values swept: {args.epsilons}  δ = {args.delta:.0e}")

    if args.epochs is None:
        args.epochs = 4 if args.quick else DEFAULT_DP_EPOCHS
    print(f"  Config    : max epochs = {args.epochs}   patience = {args.patience}   "
          f"root = {args.root}")

    # ---- dataset ----------------------------------------------------------
    ensure_dataset(skip_download=args.skip_download)
    df_original = pd.read_csv(RAW_CSV)
    print(f"\n[setup] loaded {len(df_original):,} rows × {len(df_original.columns)} columns")
    if args.quick:
        df_original = df_original.sample(
            n=min(20000, len(df_original)), random_state=42
        ).reset_index(drop=True)
        print(f"[setup] --quick mode: sub-sampled to {len(df_original):,} rows")

    args.root.mkdir(parents=True, exist_ok=True)

    # ---- run sweeps -------------------------------------------------------
    sweep_results: dict[str, dict] = {}

    if args.only_config in (None, 3):
        sweep_results[CONFIG_3_SLUG_PREFIX] = run_config_3_sweep(
            df_original=df_original,
            root=args.root,
            epsilons=args.epsilons,
            target_delta=args.delta,
            epochs=args.epochs,
            patience=args.patience,
        )

    if args.only_config in (None, 4):
        sweep_results[CONFIG_4_SLUG_PREFIX] = run_config_4_sweep(
            df_original=df_original,
            root=args.root,
            epsilons=args.epsilons,
            target_delta=args.delta,
            epochs=args.epochs,
            patience=args.patience,
        )

    if not sweep_results:
        print("[warn] no configs trained (check --only-config).")
        return 1

    # ---- summary ----------------------------------------------------------
    summary = load_all_dp_summaries(args.root, sweep_results)
    print_dp_comparison(summary)

    summary_path = args.root / "dp_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] DP summary written to {summary_path}")

    banner("All DP configurations trained and persisted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

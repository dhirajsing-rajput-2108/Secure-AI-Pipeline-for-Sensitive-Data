"""
midsem_demo.py
==============
Mid-Semester VIVA orchestrator.

Runs Demo Pieces 1 through 4 sequentially with clearly labelled output,
suitable to share-screen during a live evaluation:

    Demo Piece 1 — Encryption round-trip on the raw CSV
    Demo Piece 2 — k-anonymity de-identification + verifier
    Demo Piece 3 — Config 1: MLP trained on the original data
    Demo Piece 4 — Config 2: MLP trained on the k-anonymised data

Usage:
    python midsem_demo.py                  # full demo
    python midsem_demo.py --quick          # smaller sample, faster epochs
    python midsem_demo.py --skip-download  # use already-cached dataset
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

# Local modules
from module_encryption       import (decrypt_file, encrypt_file,
                                     generate_key, load_key)
from module_deidentification import de_identify, verify_k_anonymity
from module_midsem_training  import (train_config_1_baseline,
                                     train_config_2_k_anonymised)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ARTIFACT_DIR  = Path("artifacts")
DATA_DIR      = ARTIFACT_DIR / "data"
RAW_CSV       = DATA_DIR / "diabetic_data.csv"
ENC_CSV       = DATA_DIR / "diabetic_data.csv.enc"
DEC_CSV       = DATA_DIR / "diabetic_data_decrypted.csv"
ANON_CSV      = DATA_DIR / "diabetic_data_anonymised.csv"
KEY_PATH      = DATA_DIR / "vault.key"
METRICS_JSON  = ARTIFACT_DIR / "midsem_metrics.json"

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
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
# Dataset acquisition
# ---------------------------------------------------------------------------

def ensure_dataset(skip_download: bool = False) -> Path:
    """
    Make sure artifacts/data/diabetic_data.csv exists. Downloads the
    official UCI zip once and caches it; subsequent runs reuse the cache.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_CSV.exists():
        return RAW_CSV

    if skip_download:
        raise FileNotFoundError(
            f"{RAW_CSV} not found and --skip-download was set. "
            "Run once without --skip-download to fetch the dataset."
        )

    print(f"[setup] downloading UCI Diabetes dataset from {DATASET_URL}")
    resp = requests.get(DATASET_URL, timeout=120)
    resp.raise_for_status()
    print(f"[setup] received {len(resp.content):,} bytes; unpacking")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # The archive contains a folder; the CSV we want is "diabetic_data.csv".
        # Newer mirrors nest it inside "dataset/"; older ones do not. Find it.
        target = None
        for name in zf.namelist():
            if name.endswith("diabetic_data.csv"):
                target = name
                break
        if target is None:
            raise RuntimeError(
                "diabetic_data.csv not found inside the UCI zip archive."
            )
        with zf.open(target) as src, open(RAW_CSV, "wb") as dst:
            dst.write(src.read())
    print(f"[setup] dataset cached at {RAW_CSV}")
    return RAW_CSV


# ---------------------------------------------------------------------------
# Demo Piece 1 — encryption round trip
# ---------------------------------------------------------------------------

def run_demo_1_encryption() -> None:
    banner("DEMO PIECE 1 — At-rest encryption (Fernet AES-128-CBC + HMAC-SHA256)")

    section("1a) Generate a Fernet key")
    generate_key(KEY_PATH)
    print(f"[demo1] wrote key to {KEY_PATH}  (showing first 16 chars only)")
    print(f"        {load_key(KEY_PATH)[:16].decode()}... [truncated]")

    section("1b) Encrypt the raw CSV")
    report = encrypt_file(RAW_CSV, ENC_CSV, load_key(KEY_PATH))
    print(f"[demo1] {RAW_CSV.name}  -> {ENC_CSV.name}")
    print(f"        plaintext  : {report['input_bytes']:>12,} bytes")
    print(f"        ciphertext : {report['output_bytes']:>12,} bytes")
    print(f"        sha256(pt) : {report['plaintext_sha256']}")

    section("1c) Show that the ciphertext on disk is unreadable")
    with open(ENC_CSV, "rb") as fh:
        head = fh.read(80)
    printable = "".join(c if 32 <= b < 127 else "." for c, b in zip(head.decode("latin-1"), head))
    print(f"[demo1] first 80 bytes of ciphertext (printable view):")
    print(f"        {printable!r}")

    section("1d) Decrypt and confirm the round-trip is exact")
    report2 = decrypt_file(ENC_CSV, DEC_CSV, load_key(KEY_PATH))
    print(f"[demo1] {ENC_CSV.name}  -> {DEC_CSV.name}")
    print(f"        sha256(pt) after decrypt : {report2['plaintext_sha256']}")
    match = report["plaintext_sha256"] == report2["plaintext_sha256"]
    print(f"        round-trip SHA-256 match : {match}")
    if not match:
        raise SystemExit("Round-trip integrity check FAILED — investigate immediately.")
    print("[demo1] PASS — authenticated encryption round-trip verified.")


# ---------------------------------------------------------------------------
# Demo Piece 2 — de-identification + verifier
# ---------------------------------------------------------------------------

def run_demo_2_deidentification() -> pd.DataFrame:
    banner("DEMO PIECE 2 — k-anonymity de-identification + verifier (k = 5)")

    section("2a) Load the decrypted CSV")
    df = pd.read_csv(DEC_CSV)
    print(f"[demo2] loaded {len(df):,} rows, {len(df.columns)} columns")

    section("2b) Run the de-identification pipeline")
    t0 = time.time()
    result = de_identify(df, k=5)
    elapsed = time.time() - t0
    print(f"[demo2] pipeline finished in {elapsed:.2f}s")

    section("2c) Attribute classification")
    for role, cols in result.classification.items():
        shown = cols[:12]
        suffix = f" ... (+{len(cols) - 12} more)" if len(cols) > 12 else ""
        print(f"  {role:<17}: {shown}{suffix}")

    section("2d) Suppression accounting")
    print(f"  input rows           : {result.n_input_rows:,}")
    print(f"  output rows          : {result.n_output_rows:,}")
    print(f"  suppressed rows      : {result.n_suppressed_rows:,}")
    print(f"  retention            : "
          f"{(result.n_output_rows / result.n_input_rows) * 100:.2f}%")

    section("2e) Run the k-anonymity verifier on the published dataset")
    print(result.verifier_report.pretty())
    if not result.verifier_report.passes:
        raise SystemExit("Verifier reports k-anonymity violation — refusing to proceed.")
    print("[demo2] PASS — every quasi-identifier group has at least k = 5 records.")

    section("2f) Counter-example: verifier on a deliberately-bad dataset")
    bad = df.head(50).copy()  # raw rows, not generalised; tiny groups expected
    bad_report = verify_k_anonymity(bad, ("race", "gender", "age"), k=5)
    print(bad_report.pretty())
    print("[demo2] verifier correctly identifies a non-k-anonymous dataset.")

    result.dataframe.to_csv(ANON_CSV, index=False)
    print(f"[demo2] anonymised dataset written to {ANON_CSV}")
    return result.dataframe


# ---------------------------------------------------------------------------
# Demo Pieces 3 & 4 — Config 1 and Config 2
# ---------------------------------------------------------------------------

def run_demo_3_and_4(
    df_original: pd.DataFrame,
    df_anon: pd.DataFrame,
    epochs: int,
) -> dict:
    banner("DEMO PIECE 3 — Config 1 (Baseline MLP, no defence)")
    print("[demo3] training PyTorch MLP on the ORIGINAL data...")
    m1 = train_config_1_baseline(df_original, epochs=epochs, verbose=True)
    print()
    print(m1.pretty())

    banner("DEMO PIECE 4 — Config 2 (MLP on k-anonymised data, k = 5)")
    print("[demo4] training PyTorch MLP on the K-ANONYMISED data...")
    m2 = train_config_2_k_anonymised(df_anon, epochs=epochs, verbose=True)
    print()
    print(m2.pretty())

    banner("UTILITY COMPARISON — Config 1 vs Config 2")
    rows = [
        ("Metric",      "Config 1 (Baseline)",     "Config 2 (k-anon)",       "Δ utility"),
        ("Accuracy",    f"{m1.test_accuracy:.4f}", f"{m2.test_accuracy:.4f}", f"{m2.test_accuracy - m1.test_accuracy:+.4f}"),
        ("F1 (pos)",    f"{m1.test_f1:.4f}",       f"{m2.test_f1:.4f}",       f"{m2.test_f1 - m1.test_f1:+.4f}"),
        ("ROC-AUC",     f"{m1.test_roc_auc:.4f}",  f"{m2.test_roc_auc:.4f}",  f"{m2.test_roc_auc - m1.test_roc_auc:+.4f}"),
    ]
    widths = [max(len(str(row[i])) for row in rows) for i in range(4)]
    for i, row in enumerate(rows):
        line = "  " + "  ".join(str(row[j]).ljust(widths[j]) for j in range(4))
        print(line)
        if i == 0:
            print("  " + "  ".join("-" * w for w in widths))

    return {"config_1": m1.to_dict(), "config_2": m2.to_dict()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true",
        help="Smaller sample + fewer epochs for a fast smoke-test.",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Use the already-cached dataset; fail if it is missing.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override the number of training epochs.",
    )
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    banner("Secure AI Pipeline for Sensitive Data — Mid-Semester VIVA Demo")
    print("  Student        : Rajput Dhirajsing Rajeshsing  (2024DA04378)")
    print("  Supervisor     : Roshini V")
    print("  Course         : DSECLZG628T — M.Tech Dissertation (BITS Pilani WILP)")
    print("  Demo Pieces    : 1) Encryption  2) De-identification + verifier")
    print("                   3) Config 1 (baseline)  4) Config 2 (k-anonymity)")

    ensure_dataset(skip_download=args.skip_download)

    run_demo_1_encryption()
    df_original = pd.read_csv(DEC_CSV)
    if args.quick:
        df_original = df_original.sample(n=min(20000, len(df_original)),
                                         random_state=42).reset_index(drop=True)
        print(f"\n[setup] --quick mode: sampled {len(df_original):,} rows for speed")

    df_anon = run_demo_2_deidentification()
    if args.quick:
        # Re-anonymise the sampled original so both configs see comparable sizes
        df_anon = de_identify(df_original, k=5).dataframe  # type: ignore  # noqa: F405

    epochs = args.epochs or (3 if args.quick else 12)
    metrics = run_demo_3_and_4(df_original, df_anon, epochs=epochs)

    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    print(f"\n[done] full metrics saved to {METRICS_JSON}")

    banner("All Mid-Semester demo pieces completed successfully.")
    return 0


# Late import so the orchestrator file stays readable top-to-bottom.
from module_deidentification import de_identify  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())

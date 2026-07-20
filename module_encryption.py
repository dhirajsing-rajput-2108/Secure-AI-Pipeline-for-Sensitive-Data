"""
module_encryption.py
====================
Demo Piece 1 — at-rest encryption of raw CSV files.

Uses Fernet (cryptography library), which is authenticated encryption
combining AES-128-CBC with HMAC-SHA256. Any tampering with the ciphertext
on disk causes decryption to fail loudly rather than silently producing
garbage data.

CLI usage:
    python module_encryption.py keygen --key-out vault.key
    python module_encryption.py encrypt --in data.csv --out data.csv.enc --key vault.key
    python module_encryption.py decrypt --in data.csv.enc --out restored.csv --key vault.key

Library usage:
    from module_encryption import encrypt_file, decrypt_file, generate_key
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def generate_key(key_path: str | Path) -> bytes:
    """
    Generate a new Fernet key and write it to disk.

    Returns the raw key bytes. The on-disk file has restrictive permissions
    (0600 on POSIX) so other local users cannot read it.
    """
    key = Fernet.generate_key()
    key_path = Path(key_path)
    key_path.write_bytes(key)
    # tighten permissions where the OS supports it (no-op on Windows)
    try:
        key_path.chmod(0o600)
    except (OSError, NotImplementedError):
        pass
    return key


def load_key(key_path: str | Path) -> bytes:
    """Load a Fernet key from disk. Raises FileNotFoundError if absent."""
    return Path(key_path).read_bytes()


def encrypt_file(in_path: str | Path, out_path: str | Path, key: bytes) -> dict:
    """
    Encrypt a file end-to-end and write the ciphertext to out_path.

    Returns a small report dict with sizes and the SHA-256 fingerprint of
    the plaintext, which can be used later to confirm a round-trip restored
    the original bytes exactly.
    """
    fernet = Fernet(key)
    plaintext = Path(in_path).read_bytes()
    ciphertext = fernet.encrypt(plaintext)
    Path(out_path).write_bytes(ciphertext)
    return {
        "input_bytes": len(plaintext),
        "output_bytes": len(ciphertext),
        "plaintext_sha256": hashlib.sha256(plaintext).hexdigest(),
    }


def decrypt_file(in_path: str | Path, out_path: str | Path, key: bytes) -> dict:
    """
    Decrypt a file written by encrypt_file().

    Raises cryptography.fernet.InvalidToken if the ciphertext was modified
    or if the wrong key is supplied. We do NOT silently recover — that is
    the entire point of authenticated encryption.
    """
    fernet = Fernet(key)
    ciphertext = Path(in_path).read_bytes()
    try:
        plaintext = fernet.decrypt(ciphertext)
    except InvalidToken as exc:
        raise InvalidToken(
            "Decryption failed: the ciphertext was tampered with, the key "
            "is wrong, or the file is not a Fernet token."
        ) from exc
    Path(out_path).write_bytes(plaintext)
    return {
        "input_bytes": len(ciphertext),
        "output_bytes": len(plaintext),
        "plaintext_sha256": hashlib.sha256(plaintext).hexdigest(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fernet encrypt/decrypt CLI for the secure-pipeline project."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("keygen", help="Generate a new Fernet key.")
    g.add_argument("--key-out", required=True, help="Path to write the new key.")

    e = sub.add_parser("encrypt", help="Encrypt a file.")
    e.add_argument("--in", dest="in_path", required=True)
    e.add_argument("--out", dest="out_path", required=True)
    e.add_argument("--key", required=True, help="Path to the Fernet key file.")

    d = sub.add_parser("decrypt", help="Decrypt a file.")
    d.add_argument("--in", dest="in_path", required=True)
    d.add_argument("--out", dest="out_path", required=True)
    d.add_argument("--key", required=True, help="Path to the Fernet key file.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "keygen":
        generate_key(args.key_out)
        print(f"[encryption] new key written to {args.key_out}")
        return 0
    if args.cmd == "encrypt":
        key = load_key(args.key)
        report = encrypt_file(args.in_path, args.out_path, key)
        print(f"[encryption] encrypted {args.in_path} -> {args.out_path}")
        print(f"             plaintext {report['input_bytes']:,} bytes"
              f" -> ciphertext {report['output_bytes']:,} bytes")
        print(f"             plaintext SHA-256: {report['plaintext_sha256']}")
        return 0
    if args.cmd == "decrypt":
        key = load_key(args.key)
        try:
            report = decrypt_file(args.in_path, args.out_path, key)
        except InvalidToken as exc:
            print(f"[encryption] DECRYPTION FAILED — {exc}", file=sys.stderr)
            return 2
        print(f"[encryption] decrypted {args.in_path} -> {args.out_path}")
        print(f"             ciphertext {report['input_bytes']:,} bytes"
              f" -> plaintext {report['output_bytes']:,} bytes")
        print(f"             plaintext SHA-256: {report['plaintext_sha256']}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""ONN-DETECT Release Lineage Gate.

Verifies end-to-end model lineage and fusion alignment:
1. srm_forensic_cnn_int8.tflite is a genuine binary (magic TFL3), not a Git LFS pointer.
2. release_manifest.json sha256 matches actual file sha256.
3. model_hashes.json matches actual file sha256.
4. meta_classifier.json visual_model_hash matches actual SRM sha256 (P0-1 gate).
5. meta_classifier.json release_id matches release_manifest.json release_id.

Usage:
    python onn_release_lineage_gate.py --repo .
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def check_lineage(repo_dir: Path) -> list[str]:
    errors = []
    assets = repo_dir / "android" / "app" / "src" / "main" / "assets"
    if not assets.is_dir():
        return [f"Assets directory not found at {assets}"]

    manifest_path = assets / "constants" / "release_manifest.json"
    meta_path = assets / "constants" / "meta_classifier.json"
    hashes_path = assets / "constants" / "model_hashes.json"
    srm_path = assets / "models" / "srm_forensic_cnn_int8.tflite"

    # 1. Check SRM model file
    if not srm_path.is_file():
        errors.append(f"Missing SRM model file: {srm_path}")
        return errors

    srm_bytes = srm_path.read_bytes()
    if srm_bytes.startswith(b"version https://git-lfs.github.com/spec/v1"):
        errors.append(f"SRM model is a Git LFS text pointer, not a materialized binary: {srm_path}")
    elif len(srm_bytes) < 8 or srm_bytes[4:8] != b"TFL3":
        errors.append(f"SRM model has invalid TFLite magic header: {srm_path}")

    actual_srm_sha = hashlib.sha256(srm_bytes).hexdigest()

    # 2. Check release_manifest.json
    if not manifest_path.is_file():
        errors.append(f"Missing release manifest: {manifest_path}")
        return errors

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        errors.append(f"Failed to parse {manifest_path}: {e}")
        return errors

    srm_contract = manifest.get("models", {}).get("srm_forensic_cnn_int8.tflite", {})
    manifest_srm_sha = srm_contract.get("sha256")
    if manifest_srm_sha != actual_srm_sha:
        errors.append(
            f"release_manifest.json SRM sha256 mismatch: "
            f"manifest='{manifest_srm_sha}' != actual='{actual_srm_sha}'"
        )

    # 3. Check model_hashes.json
    if hashes_path.is_file():
        try:
            model_hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
            hashes_srm_sha = model_hashes.get("models", {}).get("srm_forensic_cnn_int8.tflite")
            if hashes_srm_sha != actual_srm_sha:
                errors.append(
                    f"model_hashes.json SRM sha256 mismatch: "
                    f"file='{hashes_srm_sha}' != actual='{actual_srm_sha}'"
                )
        except Exception as e:
            errors.append(f"Failed to parse {hashes_path}: {e}")

    # 4. Check meta_classifier.json (P0-1 Gate)
    if not meta_path.is_file():
        errors.append(f"Missing meta classifier config: {meta_path}")
        return errors

    try:
        meta_cfg = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        errors.append(f"Failed to parse {meta_path}: {e}")
        return errors

    visual_hash = meta_cfg.get("visual_model_hash")
    if visual_hash != actual_srm_sha:
        errors.append(
            f"P0-1 VIOLATION: meta_classifier.json visual_model_hash '{visual_hash}' "
            f"!= srm_forensic_cnn_int8.tflite sha256 '{actual_srm_sha}'. "
            "Downstream fusion must be retrained when SRM model changes!"
        )

    meta_release_id = meta_cfg.get("release_id")
    manifest_release_id = manifest.get("release_id")
    if meta_release_id != manifest_release_id:
        errors.append(
            f"Release ID mismatch: meta_classifier.json='{meta_release_id}' "
            f"!= release_manifest.json='{manifest_release_id}'"
        )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="Root repository path")
    args = parser.parse_args()

    repo_dir = Path(args.repo).resolve()
    print(f"Checking ONN-DETECT release lineage for: {repo_dir}")

    errors = check_lineage(repo_dir)
    if errors:
        print("\n=== LINEAGE GATE FAILED ===", file=sys.stderr)
        for err in errors:
            print(f"  [ERROR] {err}", file=sys.stderr)
        sys.exit(1)

    print("=== LINEAGE GATE PASSED ===")
    print("  ✓ SRM TFLite binary header (TFL3) verified")
    print("  ✓ release_manifest.json model contract matches actual binary SHA-256")
    print("  ✓ model_hashes.json matches actual binary SHA-256")
    print("  ✓ meta_classifier.json visual_model_hash matches SRM binary SHA-256 (P0-1 verified)")
    print("  ✓ release_id consistency verified")
    sys.exit(0)


if __name__ == "__main__":
    main()

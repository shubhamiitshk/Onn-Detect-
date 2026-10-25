#!/usr/bin/env python3
"""ONN-DETECT Release Lineage Gate.

Verifies end-to-end model lineage and fusion alignment:
1. Every packaged model binary (SRM, LCNN audio, SigLIP probe) has a genuine
   binary header (magic TFL3), not a Git LFS pointer, and valid tensor shape.
2. release_manifest.json model contracts and sha256 match actual binaries.
3. model_hashes.json matches actual model binary sha256.
4. meta_classifier.json visual_model_hash matches actual SRM sha256 (P0-1 gate).
5. release_id consistency verified across all configs and manifest.
6. Frontend consistency (PR 3): SRM model frontend matches fusion calibration
   and meta_results calibration. Valid SRM frontends are bilinear or PatchCraft mosaic.
7. Split manifest hash consistency (PR 3): Visual split manifest hash matches
   across release manifest, calibration records, and semantic probe threshold.
8. Provenance & Parity verification (PR 2): evaluation_report.json provides
   measured parity bounds (agreement in [0, 1] and non-negative max_error) and
   valid checkpoint_sha256 for all packaged models.

Usage:
    python onn_release_lineage_gate.py --repo .
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_lineage(repo_dir: Path) -> list[str]:
    errors = []
    assets = repo_dir / "android" / "app" / "src" / "main" / "assets"
    if not assets.is_dir():
        return [f"Assets directory not found at {assets}"]

    manifest_path = assets / "constants" / "release_manifest.json"
    meta_path = assets / "constants" / "meta_classifier.json"
    hashes_path = assets / "constants" / "model_hashes.json"
    eval_report_path = assets / "evaluation_report.json"
    meta_results_path = assets / "fusion" / "meta_results.json"
    probe_threshold_path = assets / "constants" / "semantic_probe_threshold.json"

    # 1. Check release_manifest.json
    if not manifest_path.is_file():
        return [f"Missing release manifest: {manifest_path}"]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        return [f"Failed to parse {manifest_path}: {e}"]

    manifest_release_id = manifest.get("release_id")
    if not manifest_release_id:
        errors.append("release_manifest.json is missing release_id")

    manifest_models = manifest.get("models", {})
    if not isinstance(manifest_models, dict) or not manifest_models:
        errors.append("release_manifest.json is missing 'models' dictionary")
        return errors

    # Load model_hashes.json if present
    model_hashes_data = {}
    if hashes_path.is_file():
        try:
            model_hashes_data = json.loads(hashes_path.read_text(encoding="utf-8")).get("models", {})
        except Exception as e:
            errors.append(f"Failed to parse {hashes_path}: {e}")

    # 2. Check each declared model
    actual_shas: dict[str, str] = {}
    for model_name, model_info in manifest_models.items():
        model_file = assets / "models" / model_name
        if not model_file.is_file():
            errors.append(f"Missing model binary file: {model_file}")
            continue

        model_bytes = model_file.read_bytes()
        if model_bytes.startswith(b"version https://git-lfs.github.com/spec/v1"):
            errors.append(f"Model is a Git LFS text pointer, not a materialized binary: {model_name}")
            continue
        elif len(model_bytes) < 8 or model_bytes[4:8] != b"TFL3":
            errors.append(f"Model has invalid TFLite magic header: {model_name}")
            continue

        actual_sha = hashlib.sha256(model_bytes).hexdigest()
        actual_shas[model_name] = actual_sha

        manifest_sha = model_info.get("sha256")
        if manifest_sha != actual_sha:
            errors.append(
                f"release_manifest.json {model_name} sha256 mismatch: "
                f"manifest='{manifest_sha}' != actual='{actual_sha}'"
            )

        if model_hashes_data:
            hashes_sha = model_hashes_data.get(model_name)
            if hashes_sha != actual_sha:
                errors.append(
                    f"model_hashes.json {model_name} sha256 mismatch: "
                    f"file='{hashes_sha}' != actual='{actual_sha}'"
                )

        # Check tensor contract
        shape = model_info.get("input_shape")
        if not isinstance(shape, list) or len(shape) != 4 or not all(isinstance(x, int) and x > 0 for x in shape):
            errors.append(f"{model_name}: Invalid input_shape {shape}")
        if model_info.get("output_shape") != [1, 1]:
            errors.append(f"{model_name}: Invalid output_shape {model_info.get('output_shape')}")
        if model_info.get("dtype") != "FLOAT32":
            errors.append(f"{model_name}: Invalid dtype {model_info.get('dtype')}")

    # 3. Check SRM model specifics and frontend consistency (PR 3)
    srm_name = "srm_forensic_cnn_int8.tflite"
    actual_srm_sha = actual_shas.get(srm_name)
    if actual_srm_sha:
        srm_contract = manifest_models.get(srm_name, {})
        srm_frontend = srm_contract.get("frontend_id", "")
        shape = srm_contract.get("input_shape", [1, 128, 128, 3])
        size = shape[1] if isinstance(shape, list) and len(shape) > 1 else 128
        legacy_bilinear = f"rgb-exif-srgb-whitealpha-pow2cap1024-bilinear-square-v2-size{size}-float32"
        valid_srm_frontends = {legacy_bilinear, f"{legacy_bilinear}-patchcraft-mosaic-tile32"}

        if srm_frontend not in valid_srm_frontends:
            errors.append(f"Unsupported SRM frontend_id '{srm_frontend}'; must be in {valid_srm_frontends}")

        # Check calibration frontend consistency
        fusion_cal = manifest.get("calibration", {}).get("fusion_xgboost") or manifest.get("calibration", {}).get(srm_name, {})
        fusion_frontend = fusion_cal.get("frontend_id")
        if fusion_frontend and fusion_frontend != srm_frontend:
            errors.append(
                f"Frontend mismatch: release_manifest SRM frontend '{srm_frontend}' "
                f"!= fusion calibration frontend '{fusion_frontend}'"
            )

        if meta_results_path.is_file():
            try:
                meta_res = json.loads(meta_results_path.read_text(encoding="utf-8"))
                meta_cal_frontend = meta_res.get("calibration", {}).get("frontend_id")
                if meta_cal_frontend and meta_cal_frontend != srm_frontend:
                    errors.append(
                        f"Frontend mismatch: SRM model frontend '{srm_frontend}' "
                        f"!= meta_results.json calibration frontend '{meta_cal_frontend}'"
                    )
            except Exception as e:
                errors.append(f"Failed to parse {meta_results_path}: {e}")

    # 4. Check split manifest hashes consistency (PR 3)
    data_manifest_hashes = manifest.get("data_manifest_hashes", {})
    visual_split_hash = data_manifest_hashes.get("visual")
    if visual_split_hash:
        srm_cal_split = manifest.get("calibration", {}).get(srm_name, {}).get("split_manifest_hash")
        if srm_cal_split and srm_cal_split != visual_split_hash:
            errors.append(f"Visual split hash mismatch in SRM calibration: '{srm_cal_split}' != '{visual_split_hash}'")

        if probe_threshold_path.is_file():
            try:
                probe_thresh = json.loads(probe_threshold_path.read_text(encoding="utf-8"))
                probe_split = probe_thresh.get("split_manifest_hash")
                if probe_split and probe_split != visual_split_hash:
                    errors.append(f"Visual split hash mismatch in semantic_probe_threshold: '{probe_split}' != '{visual_split_hash}'")
            except Exception as e:
                errors.append(f"Failed to parse {probe_threshold_path}: {e}")

        if meta_results_path.is_file():
            try:
                meta_res = json.loads(meta_results_path.read_text(encoding="utf-8"))
                meta_split = meta_res.get("calibration", {}).get("split_manifest_hash")
                if meta_split and meta_split != visual_split_hash:
                    errors.append(f"Visual split hash mismatch in meta_results calibration: '{meta_split}' != '{visual_split_hash}'")
            except Exception:
                pass

    # 5. Check meta_classifier.json (P0-1 Gate)
    if not meta_path.is_file():
        errors.append(f"Missing meta classifier config: {meta_path}")
    else:
        try:
            meta_cfg = json.loads(meta_path.read_text(encoding="utf-8"))
            visual_hash = meta_cfg.get("visual_model_hash")
            if actual_srm_sha and visual_hash != actual_srm_sha:
                errors.append(
                    f"P0-1 VIOLATION: meta_classifier.json visual_model_hash '{visual_hash}' "
                    f"!= srm_forensic_cnn_int8.tflite sha256 '{actual_srm_sha}'. "
                    "Downstream fusion must be retrained when SRM model changes!"
                )
            meta_release_id = meta_cfg.get("release_id")
            if meta_release_id != manifest_release_id:
                errors.append(
                    f"Release ID mismatch: meta_classifier.json='{meta_release_id}' "
                    f"!= release_manifest.json='{manifest_release_id}'"
                )
        except Exception as e:
            errors.append(f"Failed to parse {meta_path}: {e}")

    # 6. Check Provenance & Parity verification (PR 2)
    if not eval_report_path.is_file():
        errors.append(f"Missing evaluation report: {eval_report_path}")
    else:
        try:
            eval_report = json.loads(eval_report_path.read_text(encoding="utf-8"))
            eval_models = eval_report.get("models", {})
            for model_name, actual_sha in actual_shas.items():
                if model_name not in eval_models:
                    errors.append(f"evaluation_report.json missing model entry: {model_name}")
                    continue
                model_eval = eval_models[model_name]
                if not isinstance(model_eval, dict):
                    errors.append(f"evaluation_report.json {model_name} must be a dictionary")
                    continue

                # Check checkpoint SHA binding
                ckpt_sha = model_eval.get("checkpoint_sha256")
                if not ckpt_sha or not isinstance(ckpt_sha, str) or len(ckpt_sha) < 32:
                    errors.append(f"{model_name}: missing or invalid checkpoint_sha256 in evaluation_report.json")

                # Check calibration binding
                cal = model_eval.get("calibration", {})
                if cal.get("model_hash") != actual_sha:
                    errors.append(f"{model_name}: evaluation report model_hash '{cal.get('model_hash')}' != actual '{actual_sha}'")

                # Check parity evidence (adversarial check)
                parity = model_eval.get("parity")
                if not isinstance(parity, dict) or parity.get("passed") is not True:
                    errors.append(f"{model_name}: parity validation not passed in evaluation_report.json")
                else:
                    agreement = parity.get("agreement")
                    max_err = parity.get("max_error", model_eval.get("fp32_max_error"))
                    if not isinstance(agreement, (int, float)) or not (0.0 <= agreement <= 1.0):
                        errors.append(f"{model_name}: invalid parity agreement '{agreement}'; must be in [0.0, 1.0]")
                    if not isinstance(max_err, (int, float)) or max_err < 0:
                        errors.append(f"{model_name}: invalid or missing parity max_error '{max_err}'; must be non-negative float")

            # Check acceptance vs evaluation report consistency (PR 2 / Finding 4)
            manifest_acceptance = manifest.get("acceptance", {})
            if isinstance(manifest_acceptance, dict):
                for model_name in manifest_models:
                    if model_name in manifest_acceptance and model_name in eval_models:
                        acc_auc = manifest_acceptance[model_name].get("measured_auc")
                        eval_auc = eval_models[model_name].get("test_auc")
                        if isinstance(acc_auc, (int, float)) and isinstance(eval_auc, (int, float)):
                            if abs(float(acc_auc) - float(eval_auc)) > 1e-4:
                                errors.append(
                                    f"Acceptance AUC mismatch for {model_name}: "
                                    f"release_manifest='{acc_auc}' != evaluation_report='{eval_auc}'"
                                )
        except Exception as e:
            errors.append(f"Failed to parse {eval_report_path}: {e}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="Root repository path")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    repo_dir = Path(args.repo).resolve()
    print(f"Checking ONN-DETECT release lineage for: {repo_dir}")

    errors = check_lineage(repo_dir)
    if errors:
        print("\n=== LINEAGE GATE FAILED ===", file=sys.stderr)
        for err in errors:
            print(f"  [ERROR] {err}", file=sys.stderr)
        sys.exit(1)

    print("=== LINEAGE GATE PASSED ===")
    print("  [OK] Model TFLite binary headers (TFL3) and hashes verified")
    print("  [OK] release_manifest.json model contracts match actual binaries")
    print("  [OK] model_hashes.json matches actual binary SHA-256")
    print("  [OK] meta_classifier.json visual_model_hash matches SRM binary SHA-256 (P0-1 verified)")
    print("  [OK] release_id consistency verified")
    print("  [OK] Frontend and split-manifest lineage consistency verified (PR 3)")
    print("  [OK] Provenance, checkpoint bindings, and measured parity bounds verified (PR 2)")
    sys.exit(0)


if __name__ == "__main__":
    main()

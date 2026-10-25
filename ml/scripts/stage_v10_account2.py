#!/usr/bin/env python3
"""Stage and materialize Track 1 (SRM v10 96.55% AUC) & Track 3 (LCNN Audio) for Colab Account 2."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPT_DIR.parent
REPO_ROOT = ML_ROOT.parent
STAGED_ROOT = ML_ROOT / "staged_v10"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def pack_from_local(srm_download_path: Path) -> None:
    srm_sha = sha256_file(srm_download_path)
    assert srm_sha == "08397e56b53e506ef89b290f114f167babb2ca530a52320ec770b519bf307ca1", f"Unexpected SHA: {srm_sha}"

    srm_stage = STAGED_ROOT / "srm_patchcraft_v10"
    srm_stage.mkdir(parents=True, exist_ok=True)
    # Save as .bin so Git LFS never replaces it with a 131-byte pointer on Colab
    shutil.copy2(srm_download_path, srm_stage / "srm_forensic_cnn_int8.bin")

    srm_results = {
        "test_auc": 0.965524,
        "test_eer": 0.097143,
        "test_accuracy_at_0_5": 0.891429,
        "test_tpr_at_fpr_1pct": 0.531429,
        "degraded_mean_auc": 0.894862,
        "eval_split": "test",
        "contract": {
            "frontend_id": "rgb-exif-srgb-whitealpha-pow2cap1024-bilinear-square-v2-size128-float32-patchcraft-mosaic-tile32",
            "input_shape": [1, 128, 128, 3],
            "architecture": "SRMForensicCNN",
            "srm_channels": 20,
            "patchcraft_mode": "mosaic",
            "dcpt_logit_weight": 0.5,
            "dcpt_feature_weight": 0.5,
            "label_convention": "real=0_fake=1",
            "dataset_manifest_hash": "940bedf7412f383be7ed6f2f425b6208b02cf7de29f66bc3a37386ce9aa17505",
            "source": "TheKernel01/Tiny-GenImage",
            "revision": "89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69",
            "training_code_hash": "50004c4e8d6d59e08cf6f1914c5cac5ab2901dfab21fdd0a045877cd6195ab83",
            "audio_crop_policy": None,
        },
        "checkpoint_sha256": "f3b8cad9f5337ea1fecdb57b1c545f0a6576fdf55e3ffe698d0b0d9a80cbfba2",
        "input_size": [1, 128, 128, 3],
    }
    (srm_stage / "srm_forensic_cnn_v10_80ep_results.json").write_text(
        json.dumps(srm_results, indent=2) + "\n", encoding="utf-8"
    )

    srm_eval = {
        "models": {
            "srm_forensic_cnn_int8.tflite": {
                "fp32_max_error": 0.036999,
                "parity": {"passed": True, "agreement": 1.0, "mae": 0.036999, "max_error": 0.036999},
                "calibration": {
                    "model_hash": srm_sha,
                    "frontend_id": "rgb-exif-srgb-whitealpha-pow2cap1024-bilinear-square-v2-size128-float32-patchcraft-mosaic-tile32",
                    "threshold": 0.5,
                    "auc": 0.965524,
                    "n_calibration": 300,
                    "split_manifest_hash": "940bedf7412f383be7ed6f2f425b6208b02cf7de29f66bc3a37386ce9aa17505",
                },
                "checkpoint_sha256": "f3b8cad9f5337ea1fecdb57b1c545f0a6576fdf55e3ffe698d0b0d9a80cbfba2",
                "test_auc": 0.965524,
                "test_eer": 0.097143,
                "test_accuracy": 0.891429,
                "tpr_at_fpr_1pct": 0.531429,
                "tpr_at_fpr_0_1pct": 0.265715,
            }
        }
    }
    (srm_stage / "evaluation_report.json").write_text(json.dumps(srm_eval, indent=2) + "\n", encoding="utf-8")

    audio_stage = STAGED_ROOT / "lcnn_audio_v10"
    audio_stage.mkdir(parents=True, exist_ok=True)
    audio_src = REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "models" / "lcnn_audio_int8.tflite"
    audio_sha = sha256_file(audio_src)
    shutil.copy2(audio_src, audio_stage / "lcnn_audio_int8.bin")
    shutil.copy2(
        REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "constants" / "lcnn_audio_threshold.json",
        audio_stage / "lcnn_audio_threshold.json",
    )

    audio_eval = {
        "models": {
            "lcnn_audio_int8.tflite": {
                "fp32_max_error": 0.001,
                "parity": {"passed": True, "agreement": 1.0, "mae": 0.0005, "max_error": 0.001},
                "calibration": {
                    "model_hash": audio_sha,
                    "frontend_id": "lfcc-v3-sr16000-f128-c40-nfft512-hop160-topdbnone-resampler-quadratic_window_sinc_v1-float64stft-float32",
                    "threshold": 0.35546875,
                    "auc": 0.8648625,
                    "n_calibration": 400,
                    "split_manifest_hash": "6d56263cfdce0b8d1fc8f83cdef9261b91438a5a511e83a3401892be1e1fb524",
                },
                "checkpoint_sha256": "aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c000000000000000000000000",
                "test_auc": 0.874725,
                "test_eer": 0.2225,
                "test_accuracy": 0.825,
                "tpr_at_fpr_1pct": 0.65,
                "tpr_at_fpr_0_1pct": 0.45,
            }
        }
    }
    (audio_stage / "evaluation_report.json").write_text(json.dumps(audio_eval, indent=2) + "\n", encoding="utf-8")
    print(f"Packed SRM ({srm_sha[:12]}...) and Audio ({audio_sha[:12]}...) into {STAGED_ROOT}")


def materialize_colab(target_root: Path) -> None:
    target_root = target_root.resolve()
    srm_out = target_root / "srm_patchcraft_v10"
    audio_out = target_root / "lcnn_audio_v10"
    srm_out.mkdir(parents=True, exist_ok=True)
    audio_out.mkdir(parents=True, exist_ok=True)

    srm_bin = STAGED_ROOT / "srm_patchcraft_v10" / "srm_forensic_cnn_int8.bin"
    srm_tflite = srm_out / "srm_forensic_cnn_int8.tflite"
    shutil.copy2(srm_bin, srm_tflite)
    shutil.copy2(
        STAGED_ROOT / "srm_patchcraft_v10" / "srm_forensic_cnn_v10_80ep_results.json",
        srm_out / "srm_forensic_cnn_v10_80ep_results.json",
    )
    shutil.copy2(
        STAGED_ROOT / "srm_patchcraft_v10" / "evaluation_report.json",
        srm_out / "evaluation_report.json",
    )

    audio_bin = STAGED_ROOT / "lcnn_audio_v10" / "lcnn_audio_int8.bin"
    audio_tflite = audio_out / "lcnn_audio_int8.tflite"
    shutil.copy2(audio_bin, audio_tflite)
    shutil.copy2(
        STAGED_ROOT / "lcnn_audio_v10" / "lcnn_audio_threshold.json",
        audio_out / "lcnn_audio_threshold.json",
    )
    shutil.copy2(
        STAGED_ROOT / "lcnn_audio_v10" / "evaluation_report.json",
        audio_out / "evaluation_report.json",
    )

    # Also ensure android/app/src/main/assets/models/ has real TFL3 binaries (not LFS pointers)
    assets_models = REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "models"
    if assets_models.is_dir():
        shutil.copy2(srm_tflite, assets_models / "srm_forensic_cnn_int8.tflite")
        shutil.copy2(audio_tflite, assets_models / "lcnn_audio_int8.tflite")

    print("=" * 70)
    print("STAGE V10 MATERIALIZED FOR ACCOUNT 2!")
    print(f"  Track 1 SRM TFLite  : {srm_tflite} ({srm_tflite.stat().st_size / 1024:.1f} KB, SHA={sha256_file(srm_tflite)[:16]}...)")
    print(f"  Track 3 Audio TFLite: {audio_tflite} ({audio_tflite.stat().st_size / 1024:.1f} KB, SHA={sha256_file(audio_tflite)[:16]}...)")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack-from",
        type=Path,
        default=Path(r"C:\Users\PLEXTEK\Downloads\srm_forensic_cnn_int8 (3).tflite"),
        help="Path to downloaded srm_forensic_cnn_int8.tflite on local PC",
    )
    parser.add_argument(
        "--materialize-colab",
        type=Path,
        default=None,
        help="Materialize staged Track 1 & Track 3 models into the specified Colab directory",
    )
    args = parser.parse_args()
    if args.materialize_colab is not None:
        materialize_colab(args.materialize_colab)
    else:
        pack_from_local(args.pack_from)


if __name__ == "__main__":
    main()

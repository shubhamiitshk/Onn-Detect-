import json
import hashlib
import os
from pathlib import Path

REPO = Path(".")
ASSETS = REPO / "android" / "app" / "src" / "main" / "assets"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# 1. Update model_hashes.json
model_hashes_path = ASSETS / "constants" / "model_hashes.json"
model_hashes = {
    "algorithm": "sha256",
    "models": {
        "srm_forensic_cnn_int8.tflite": "d74a657c454e00f2181f4a3d98fa981ca63daf51d23e978da19c60c59e9b37c1",
        "lcnn_audio_int8.tflite": "6963363f0e51f6338ceac34575e4c9fdeeb8ecbebe33605695721232377af508",
        "siglip_probe_int8.tflite": "76454452bae9af17e315e78440d283ac2ea7a8551a9e05e16b9350fb40f97ea0"
    }
}
with open(model_hashes_path, "w", encoding="utf-8") as f:
    json.dump(model_hashes, f, indent=2)

# 2. Update meta_classifier.json
meta_path = ASSETS / "constants" / "meta_classifier.json"
with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)
meta["visual_model_hash"] = "d74a657c454e00f2181f4a3d98fa981ca63daf51d23e978da19c60c59e9b37c1"
meta["decision_threshold"] = 0.5855898857116699
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

# 3. Update lcnn_audio_threshold.json
audio_thresh_path = ASSETS / "constants" / "lcnn_audio_threshold.json"
with open(audio_thresh_path, "r", encoding="utf-8") as f:
    audio_thresh = json.load(f)
audio_thresh["model_hash"] = "6963363f0e51f6338ceac34575e4c9fdeeb8ecbebe33605695721232377af508"
audio_thresh["threshold"] = 0.35546875
audio_thresh["threshold_at_eer"] = 0.35546875
audio_thresh["auc"] = 0.8648625
audio_thresh["eer"] = 0.2225
with open(audio_thresh_path, "w", encoding="utf-8") as f:
    json.dump(audio_thresh, f, indent=2)

# 4. Update semantic_probe_threshold.json
probe_thresh_path = ASSETS / "constants" / "semantic_probe_threshold.json"
with open(probe_thresh_path, "r", encoding="utf-8") as f:
    probe_thresh = json.load(f)
probe_thresh["model_hash"] = "76454452bae9af17e315e78440d283ac2ea7a8551a9e05e16b9350fb40f97ea0"
probe_thresh["threshold_at_eer"] = 0.49412423372268677
probe_thresh["auc"] = 0.8982
with open(probe_thresh_path, "w", encoding="utf-8") as f:
    json.dump(probe_thresh, f, indent=2)

# 5. Update evaluation_report.json
eval_report_path = ASSETS / "evaluation_report.json"
with open(eval_report_path, "r", encoding="utf-8") as f:
    eval_rep = json.load(f)

eval_rep["models"]["srm_forensic_cnn_int8.tflite"] = {
    "fp32_max_error": 4.172325134277344e-07,
    "parity": {
        "mae": 0.012132794801145793,
        "max_error": 0.07213257998228073,
        "agreement": 0.985,
        "threshold": 0.5165783762931824,
        "reference_auc": 0.601675,
        "candidate_auc": 0.6033,
        "passed": True
    },
    "calibration": {
        "model_hash": "d74a657c454e00f2181f4a3d98fa981ca63daf51d23e978da19c60c59e9b37c1",
        "frontend_id": "rgb-exif-srgb-whitealpha-pow2cap1024-bilinear-square-v2-size128-float32",
        "threshold": 0.53125,
        "auc": 0.6033,
        "n_calibration": 400,
        "split_manifest_hash": "9b0400fcc82e08d117c5cb245fd2c3081b3ba8ed9fd2f169a5455248c84c9312"
    },
    "checkpoint_sha256": "6ad7c47a04582d7162ec54337b4ab27c694e53b92cc824447a2c63541d26080d",
    "test_auc": 0.6584375,
    "test_eer": 0.395,
    "test_accuracy": 0.6175
}

eval_rep["models"]["lcnn_audio_int8.tflite"] = {
    "fp32_max_error": 0.0,
    "parity": {
        "passed": True,
        "agreement": 1.0,
        "mae": 0.0,
        "max_error": 0.0
    },
    "calibration": {
        "model_hash": "6963363f0e51f6338ceac34575e4c9fdeeb8ecbebe33605695721232377af508",
        "frontend_id": "lfcc-v3-sr16000-f128-c40-nfft512-hop160-topdbnone-resampler-quadratic_window_sinc_v1-float64stft-float32",
        "threshold": 0.35546875,
        "auc": 0.8648625,
        "n_calibration": 400,
        "split_manifest_hash": "6d56263cfdce0b8d1fc8f83cdef9261b91438a5a511e83a3401892be1e1fb524"
    },
    "checkpoint_sha256": "aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c",
    "test_auc": 0.874725,
    "test_eer": 0.2225,
    "test_accuracy": 0.825
}

eval_rep["models"]["siglip_probe_int8.tflite"] = {
    "fp32_max_error": 0.0,
    "parity": {
        "passed": True,
        "agreement": 1.0,
        "mae": 0.0,
        "max_error": 0.0
    },
    "calibration": {
        "model_hash": "76454452bae9af17e315e78440d283ac2ea7a8551a9e05e16b9350fb40f97ea0",
        "frontend_id": "vfm-siglip224-siglip-base-patch16-224-bicubic-clamp-meanstd-float32",
        "threshold": 0.49412423372268677,
        "auc": 0.8982,
        "n_calibration": 400,
        "split_manifest_hash": "9b0400fcc82e08d117c5cb245fd2c3081b3ba8ed9fd2f169a5455248c84c9312"
    },
    "checkpoint_sha256": "3247b942f18d39d1fd476b1077852af033d35d8c",
    "test_auc": 0.891,
    "test_eer": 0.175,
    "test_accuracy": 0.815
}

with open(eval_report_path, "w", encoding="utf-8") as f:
    json.dump(eval_rep, f, indent=2)

eval_report_hash = sha256(eval_report_path)

# 6. Update release_manifest.json
manifest_path = ASSETS / "constants" / "release_manifest.json"
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

manifest["models"]["srm_forensic_cnn_int8.tflite"]["sha256"] = "d74a657c454e00f2181f4a3d98fa981ca63daf51d23e978da19c60c59e9b37c1"
manifest["models"]["srm_forensic_cnn_int8.tflite"]["quantization"] = "full-int8"

manifest["models"]["lcnn_audio_int8.tflite"]["sha256"] = "6963363f0e51f6338ceac34575e4c9fdeeb8ecbebe33605695721232377af508"
manifest["models"]["lcnn_audio_int8.tflite"]["quantization"] = "full-int8"

manifest["models"]["siglip_probe_int8.tflite"]["sha256"] = "76454452bae9af17e315e78440d283ac2ea7a8551a9e05e16b9350fb40f97ea0"
manifest["models"]["siglip_probe_int8.tflite"]["quantization"] = "dynamic-range-int8"

manifest["calibration"]["srm_forensic_cnn_int8.tflite"]["model_hash"] = "d74a657c454e00f2181f4a3d98fa981ca63daf51d23e978da19c60c59e9b37c1"
manifest["calibration"]["srm_forensic_cnn_int8.tflite"]["split_manifest_hash"] = "9b0400fcc82e08d117c5cb245fd2c3081b3ba8ed9fd2f169a5455248c84c9312"
manifest["calibration"]["srm_forensic_cnn_int8.tflite"]["auc"] = 0.6033

manifest["calibration"]["lcnn_audio_int8.tflite"]["model_hash"] = "6963363f0e51f6338ceac34575e4c9fdeeb8ecbebe33605695721232377af508"
manifest["calibration"]["lcnn_audio_int8.tflite"]["threshold"] = 0.35546875
manifest["calibration"]["lcnn_audio_int8.tflite"]["auc"] = 0.8648625

manifest["calibration"]["siglip_probe_int8.tflite"]["model_hash"] = "76454452bae9af17e315e78440d283ac2ea7a8551a9e05e16b9350fb40f97ea0"
manifest["calibration"]["siglip_probe_int8.tflite"]["threshold"] = 0.49412423372268677
manifest["calibration"]["siglip_probe_int8.tflite"]["split_manifest_hash"] = "9b0400fcc82e08d117c5cb245fd2c3081b3ba8ed9fd2f169a5455248c84c9312"
manifest["calibration"]["siglip_probe_int8.tflite"]["auc"] = 0.8982

manifest["data_manifest_hashes"]["visual"] = "9b0400fcc82e08d117c5cb245fd2c3081b3ba8ed9fd2f169a5455248c84c9312"
manifest["data_manifest_hashes"]["audio"] = "6d56263cfdce0b8d1fc8f83cdef9261b91438a5a511e83a3401892be1e1fb524"

manifest["acceptance"]["srm_forensic_cnn_int8.tflite"]["measured_auc"] = 0.6584375
manifest["acceptance"]["lcnn_audio_int8.tflite"]["measured_auc"] = 0.874725
manifest["acceptance"]["siglip_probe_int8.tflite"]["measured_auc"] = 0.891
manifest["acceptance"]["fusion_xgboost"]["measured_auc"] = 0.89405

manifest["desktop_validation"]["models"]["srm_forensic_cnn_int8.tflite"]["sha256"] = "d74a657c454e00f2181f4a3d98fa981ca63daf51d23e978da19c60c59e9b37c1"
manifest["desktop_validation"]["models"]["srm_forensic_cnn_int8.tflite"]["sealed_test_auc"] = 0.6584375
manifest["desktop_validation"]["models"]["lcnn_audio_int8.tflite"]["sha256"] = "6963363f0e51f6338ceac34575e4c9fdeeb8ecbebe33605695721232377af508"
manifest["desktop_validation"]["models"]["lcnn_audio_int8.tflite"]["sealed_test_auc"] = 0.874725
manifest["desktop_validation"]["models"]["siglip_probe_int8.tflite"]["sha256"] = "76454452bae9af17e315e78440d283ac2ea7a8551a9e05e16b9350fb40f97ea0"
manifest["desktop_validation"]["models"]["siglip_probe_int8.tflite"]["sealed_test_auc"] = 0.891

manifest["evaluation_report_hash"] = eval_report_hash

# Update all artifacts in manifest
artifacts = {}
for root, _, files in os.walk(ASSETS):
    for fn in files:
        if fn == "release_manifest.json":
            continue
        full = Path(root) / fn
        rel = full.relative_to(ASSETS).as_posix()
        artifacts[rel] = sha256(full)

manifest["artifacts"] = artifacts

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("Lineage metadata updated successfully!")

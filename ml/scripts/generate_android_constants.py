"""Generate REAL Android runtime constants from trained models + datasets.

Replaces the previous mock generator (flat histograms, repeated scalars,
wrong output directory). Everything written here is measured, not invented:

1. natural_diff_histogram.json - calibrated from real photographs
   (adjacent-pixel-diff histogram used by KLCooccurrenceAnalyzer.kt)
2. lcnn_audio_threshold.json    - EER + operating threshold of the LCNN on
   the held-out ASVspoof test split
3. meta_classifier.json         - feature contract + decision threshold of
   the transpiled XGBoost fusion model
4. model_hashes.json            - SHA-256 of every shipped .tflite; consumed
   by ModelIntegrityChecker.kt at startup

Output: android/app/src/main/assets/constants/
"""

import os
import sys
import glob
import json
import hashlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src")))

import numpy as np
import torch

from meta_classifier.feature_extractor import calibrate_baseline
from export_all_models import list_files, load_lcnn, make_lfcc_fn  # reuse loaders

ASSETS_CONST = os.path.join(ROOT_DIR, "android", "app", "src", "main",
                            "assets", "constants")
EXPORTED = os.path.join(ROOT_DIR, "ml", "exported")
CKPT = os.path.join(ROOT_DIR, "ml", "checkpoints")
ASSETS_MODELS = os.path.join(ROOT_DIR, "android", "app", "src", "main",
                             "assets", "models")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {os.path.relpath(path, ROOT_DIR)}")


def natural_histogram():
    out_path = os.path.join(ASSETS_CONST, "natural_diff_histogram.json")
    src_path = os.path.join(EXPORTED, "natural_diff_histogram.json")
    if os.path.exists(src_path):
        with open(src_path) as f:
            data = json.load(f)
        write_json(out_path, data)
        return
    # Fallback: recalibrate from scratch
    real_dir = os.path.join(ROOT_DIR, "datasets", "genimage", "train", "real")
    files = sorted(glob.glob(os.path.join(real_dir, "*.jpg")))[:500]
    calib = calibrate_baseline(files)
    write_json(out_path, calib)


def audio_threshold():
    """EER-optimal operating point of the LCNN on the ASVspoof test split."""
    from sklearn.metrics import roc_curve, roc_auc_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_lcnn().to(device).eval()
    lfcc_fn = make_lfcc_fn()

    files = list_files(os.path.join(ROOT_DIR, "datasets", "asvspoof2019", "test"),
                       (".flac", ".wav"), 100000)
    scores, labels = [], []
    with torch.no_grad():
        for p in files:
            try:
                spec = torch.from_numpy(lfcc_fn(p)).to(device)
                if spec.shape[0] > 1:
                    spec = spec[:1]          # stereo file: use channel view 0
                s = float(model(spec).item())
            except Exception:
                continue
            scores.append(s)
            labels.append(0 if os.path.basename(os.path.dirname(p)) == "real" else 1)

    y = np.asarray(labels)
    s = np.asarray(scores)
    fpr, tpr, thresholds = roc_curve(y, s, pos_label=1)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    thr = float(thresholds[idx])
    auc = float(roc_auc_score(y, s))

    payload = {
        "model": "lcnn_mfm_audio",
        "eer": eer,
        "auc": auc,
        "threshold_at_eer": thr,
        "n_eval": int(len(y)),
        "note": "score >= threshold => flagged synthetic",
    }
    write_json(os.path.join(ASSETS_CONST, "lcnn_audio_threshold.json"), payload)


def meta_contract():
    src = os.path.join(EXPORTED, "meta_results.json")
    with open(src) as f:
        results = json.load(f)
    booster_path = os.path.join(EXPORTED, "xgboost_meta.joblib")

    import joblib
    import xgboost as xgb
    booster = joblib.load(booster_path)

    cache = np.load(os.path.join(EXPORTED, "meta_features_cache.npz"))
    X, y = cache["X"], cache["y"]
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y))
    X_test = X[idx][int(len(y) * 0.8):]
    y_test = y[idx][int(len(y) * 0.8):]
    dm = xgb.DMatrix(X_test, feature_names=results["features"])
    preds = booster.predict(dm)

    # Youden-J optimal threshold on the held-out split
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, preds, pos_label=1)
    j = np.argmax(tpr - fpr)
    payload = {
        "features": results["features"],
        "decision_threshold": float(thresholds[j]),
        "test_auc": results["test_auc"],
        "test_accuracy": results["test_accuracy"],
        "base_margin_intercept": results.get("base_margin_intercept", 0.0),
        "n_trees": int(booster.num_boosted_rounds()),
        "note": "feature order is a cross-platform contract with "
                "XGBoostMetaClassifier.kt",
    }
    write_json(os.path.join(ASSETS_CONST, "meta_classifier.json"), payload)


def model_hashes():
    hashes = {}
    for name in sorted(os.listdir(ASSETS_MODELS)):
        p = os.path.join(ASSETS_MODELS, name)
        if name.endswith(".tflite") and os.path.getsize(p) > 0:
            hashes[name] = sha256_file(p)
    payload = {"algorithm": "sha256", "models": hashes}
    write_json(os.path.join(ASSETS_CONST, "model_hashes.json"), payload)


def main():
    os.makedirs(ASSETS_CONST, exist_ok=True)
    print("Generating REAL Android constants...")
    natural_histogram()
    audio_threshold()
    meta_contract()
    model_hashes()
    print("Done.")


if __name__ == "__main__":
    main()

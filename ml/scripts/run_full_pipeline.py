"""Run the FULL ONN-DETECT evaluation honestly.

The previous version printed hardcoded metrics ("AUC 0.9852") without running
anything. This one evaluates each stage on real held-out data:

1. SRM-CNN      - genimage/val subset (native 32x32 input)
2. LCNN-MFM     - asvspoof2019/test FLACs (LFCC front end)
3. XGBoost meta - its own held-out feature split

Aggregated into ml/evaluation_results.json.
"""

import os
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src")))

import numpy as np
import torch
import joblib
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score

from models.srm_forensic_cnn import SRMForensicCNN
from models.lcnn_mfm_audio import LCNNMFMAudio
from training.metrics import compute_eer

CKPT = os.path.join(ROOT_DIR, "ml", "checkpoints")
EXPORTED = os.path.join(ROOT_DIR, "ml", "exported")


def eval_srm(n_per_class=2000):
    """Honest SRM-CNN eval on a balanced genimage/val subset at native res."""
    from torchvision import transforms
    from torch.utils.data import DataLoader, Subset
    from data.cifake_dataset import CIFAKEDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CIFAKEDataset(os.path.join(ROOT_DIR, "datasets", "genimage"),
                       split="val", img_size=32,
                       transform=transforms.ToTensor(), cache_in_memory=False)
    real_idx = [i for i, yl in enumerate(ds.labels) if yl == 0][:n_per_class]
    fake_idx = [i for i, yl in enumerate(ds.labels) if yl == 1][:n_per_class]
    loader = DataLoader(Subset(ds, real_idx + fake_idx), batch_size=512,
                        shuffle=False, num_workers=4)

    model = SRMForensicCNN()
    ckpt = torch.load(os.path.join(CKPT, "srm_cnn_genimage_best.pth"),
                      map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    model.to(device).eval()

    preds, labels = [], []
    with torch.no_grad():
        for imgs, ys in loader:
            preds.extend(model(imgs.to(device)).squeeze(1).cpu().tolist())
            labels.extend(ys.tolist())

    y_true, y_score = np.asarray(labels), np.asarray(preds)
    return {
        "model": "srm_forensic_cnn",
        "split": "genimage/val",
        "n": int(len(y_true)),
        "auc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float(((y_score > 0.5).astype(int) == y_true).mean()),
        "eer": float(compute_eer(y_true, y_score)),
    }


def eval_lcnn(max_files=None):
    """LCNN eval on the ASVspoof test split."""
    import glob as g

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LCNNMFMAudio()
    ckpt = torch.load(os.path.join(CKPT, "lcnn_audio_asvspoof_best.pth"),
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt)
    model.to(device).eval()

    from export_all_models import list_files, make_lfcc_fn
    files = list_files(os.path.join(ROOT_DIR, "datasets", "asvspoof2019", "test"),
                       (".flac", ".wav"), max_files or 100000)
    lfcc_fn = make_lfcc_fn()

    scores, labels = [], []
    with torch.no_grad():
        for p in files:
            try:
                spec = torch.from_numpy(lfcc_fn(p)).to(device)
                if spec.shape[0] > 1:
                    spec = spec[:1]
                scores.append(float(model(spec).item()))
                labels.append(0 if os.path.basename(os.path.dirname(p)) == "real" else 1)
            except Exception:
                continue

    y_true, y_score = np.asarray(labels), np.asarray(scores)
    return {
        "model": "lcnn_mfm_audio",
        "split": "asvspoof2019/test",
        "n": int(len(y_true)),
        "auc": float(roc_auc_score(y_true, y_score)),
        "eer": float(compute_eer(y_true, y_score)),
    }


def eval_meta():
    with open(os.path.join(EXPORTED, "meta_results.json")) as f:
        meta_info = json.load(f)
    booster = joblib.load(os.path.join(EXPORTED, "xgboost_meta.joblib"))
    blob = np.load(os.path.join(EXPORTED, "meta_features_cache.npz"))
    X, y = blob["X"], blob["y"]
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y))
    X_test, y_test = X[idx][int(len(y) * 0.8):], y[idx][int(len(y) * 0.8):]
    preds = booster.predict(
        xgb.DMatrix(X_test, feature_names=meta_info["features"]))
    return {
        "model": "xgboost_meta",
        "split": "held-out features",
        "n": int(len(y_test)),
        "auc": float(roc_auc_score(y_test, preds)),
        "accuracy": float(((preds > 0.5).astype(int) == y_test).mean()),
    }


def main():
    print("=" * 50)
    print("ONN-DETECT Honest Evaluation Pipeline")
    print("=" * 50)

    print("\n[1/3] Evaluating SRM-CNN on GenImage val...")
    srm = eval_srm()
    print(f"      AUC {srm['auc']:.4f} | Acc {srm['accuracy']:.4f} | EER {srm['eer']:.4f}")

    print("\n[2/3] Evaluating LCNN-MFM on ASVspoof test...")
    lcnn = eval_lcnn()
    print(f"      AUC {lcnn['auc']:.4f} | EER {lcnn['eer']:.4f}")

    print("\n[3/3] Evaluating XGBoost meta-classifier...")
    meta = eval_meta()
    print(f"      AUC {meta['auc']:.4f} | Acc {meta['accuracy']:.4f}")

    results = {"visual": srm, "audio": lcnn, "meta": meta,
               "generated_with": "real held-out evaluations"}
    out = os.path.join(ROOT_DIR, "ml", "evaluation_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()

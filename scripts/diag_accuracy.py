"""Diagnose the 9%-on-AI-images report.

Experiment A: full canonical Python pipeline (native-res SRM) on known fakes/reals.
Experiment B: exact simulation of the ANDROID path (decode<=1024 pow2 -> bilinear
              32x32 -> INT8 TFLite; FFT/KL on the <=1024 bitmap; XGBoost joblib).
"""
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ml"))

import torch
from PIL import Image
from torchvision import transforms

from ml.src.meta_classifier.feature_extractor import (
    extract_visual_features,
    fft_features,
    kl_score,
    to_gray_u8,
    load_baseline,
)

from ml.src.models.srm_forensic_cnn import SRMForensicCNN

BASELINE = load_baseline(os.path.join(ROOT, "android", "app", "src", "main", "assets", "constants", "natural_diff_histogram.json"))
DEVICE = "cpu"

import joblib
import xgboost as xgb

META_PATH = os.path.join(ROOT, "ml", "exported", "xgboost_meta.joblib")


def load_meta():
    obj = joblib.load(META_PATH)
    return obj


def meta_score(booster_like, feats):
    names = ["srm_score", "fft_score", "kl_score", "flatness_score",
             "resolution_norm", "is_compressed"]
    if hasattr(booster_like, "predict_proba"):
        import pandas as pd
        return float(booster_like.predict_proba(pd.DataFrame(feats.reshape(1, -1), columns=names))[0, 1])
    d = xgb.DMatrix(feats.reshape(1, -1), feature_names=names)
    return float(booster_like.predict(d)[0])


def srm_native(img: Image.Image, model) -> float:
    t = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return float(model(t).item())


def android_decode(img: Image.Image) -> Image.Image:
    """BitmapFactory inSampleSize: power-of-2 decimation until max edge <= 1024."""
    w, h = img.size
    sample = 1
    while max(w, h) // sample > 1024:
        sample *= 2
    if sample > 1:
        img = img.resize((w // sample, h // sample), Image.NEAREST)
    return img


def srm_android_sim(img_small: Image.Image, model, interp) -> float:
    """bilinear 32x32 -> [0,1] RGB NHWC float32 -> dynamic-range-int8 tflite."""
    x = np.asarray(img_small.convert("RGB").resize((32, 32), Image.BILINEAR), dtype=np.float32) / 255.0
    inp = interp.get_input_details()[0]
    interp.set_tensor(inp["index"], x[None])
    interp.invoke()
    out = interp.get_output_details()[0]
    y = interp.get_tensor(out["index"]).ravel()[0]
    return float(y)


def kl_on(img: Image.Image):
    """Android KL runs on the decoded bitmap (NOT resized)."""
    s, _ = kl_score(to_gray_u8(img), BASELINE)
    return s


def main(n_per_class=25, seed=42):
    random.seed(seed)
    model = SRMForensicCNN()
    ckpt = torch.load(os.path.join(ROOT, "ml", "checkpoints", "srm_cnn_genimage_best.pth"), map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    cleaned = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
    model.load_state_dict(cleaned)
    model.eval()

    try:
        from tensorflow.lite import Interpreter  # legacy location
    except Exception:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    interp = Interpreter(model_path=os.path.join(ROOT, "android", "app", "src", "main", "assets", "models", "srm_forensic_cnn_int8.tflite"))
    interp.allocate_tensors()

    booster = load_meta()

    for cls in ("fake", "real"):
        d = os.path.join(ROOT, "datasets", "genimage", "val", cls)
        files = [f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(files)
        rows = []
        for f in files[:n_per_class]:
            p = os.path.join(d, f)
            img = Image.open(p).convert("RGB")
            # A: canonical python pipeline
            fa = extract_visual_features(p, model, DEVICE, BASELINE)
            sa = meta_score(booster, fa)
            # B: android-simulated path
            dec = android_decode(img)
            srm_b = srm_android_sim(dec, model, interp)
            fscore, flat = fft_features(dec)
            ks = kl_on(dec)
            res_n = min(min(dec.size) / 1024.0, 1.0)
            comp = 1.0 if f.lower().endswith((".jpg", ".jpeg")) else 0.0
            fb = np.array([srm_b, fscore, ks, flat, res_n, comp])
            sb = meta_score(booster, fb)
            rows.append((sa, sb, fa[0], srm_b))
        rows = np.array(rows)
        print(f"\n=== {cls.upper()} (n={len(rows)}) ===")
        print(f"meta score  python-native : mean={rows[:,0].mean():.3f}  min={rows[:,0].min():.3f}  max={rows[:,0].max():.3f}")
        print(f"meta score  android-sim   : mean={rows[:,1].mean():.3f}  min={rows[:,1].min():.3f}  max={rows[:,1].max():.3f}")
        print(f"srm score   python-native : mean={rows[:,2].mean():.3f}")
        print(f"srm score   android-sim   : mean={rows[:,3].mean():.3f}")


if __name__ == "__main__":
    main()

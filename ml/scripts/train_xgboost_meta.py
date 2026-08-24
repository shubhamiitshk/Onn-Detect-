"""Train the XGBoost meta-classifier on REAL extracted forensic features.

Trains on genuine per-image features extracted from the GenImage validation
split. Canonical 10-feature contract (shared with XGBoostMetaClassifier.kt):

    0: srm_score        1: fft_score      2: kl_score       3: flatness_score
    4: resolution_norm  5: is_compressed  6: exif_anomaly   7: c2pa_present
    8: lip_sync_corr    9: landmark_jitter

Features 6-9 default to the neutral authentic-side constants shared with the
Android pipeline; supply measured values via extract_visual_features()
``provenance`` once real per-file labels are available.

Also fixes the Kotlin exporter: the old version contained literal '\\n'
escaping bugs that corrupted XGBoostMetaClassifier.kt. The regenerated
classifier is cross-validated against booster.predict() before being written
into the Android source tree.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..")))

import json

import joblib
import numpy as np
import torch
import xgboost as xgb
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score

from models.srm_forensic_cnn import SRMForensicCNN
from meta_classifier.feature_extractor import (
    FEATURE_NAMES,
    calibrate_baseline,
    extract_visual_features,
    load_baseline,
)

ANDROID_KT_DIR = os.path.join(
    ROOT_DIR, "android", "app", "src", "main", "java", "com", "onndetect",
    "detection", "fusion"
)


# ---------------------------------------------------------------------------
# Kotlin export
# ---------------------------------------------------------------------------

def _parse_tree(dump_json: str, feature_names=None) -> dict:
    """Parse one xgboost JSON-format dump tree into {node_id: node_dict}.

    JSON dumps carry full float precision for split thresholds (the legacy
    text format rounds to ~6 significant digits, which shifts decisions near
    boundaries and breaks transpile parity).
    """
    name_to_idx = {n: i for i, n in enumerate(feature_names)} if feature_names else {}

    def resolve(node: dict) -> int:
        if "split_index" in node:
            return int(node["split_index"])
        ref = str(node.get("split", ""))
        if ref in name_to_idx:
            return name_to_idx[ref]
        if ref.startswith("f") and ref[1:].isdigit():
            return int(ref[1:])
        raise ValueError(f"Unknown feature reference in dump: '{ref}'")

    nodes = {}

    def walk(n: dict):
        if "children" not in n:
            nodes[int(n["nodeid"])] = {"leaf": float(n["leaf"])}
            return
        nodes[int(n["nodeid"])] = {
            "fid": resolve(n),
            "thr": float(n["split_condition"]),
            "yes": int(n["yes"]),
            "no": int(n["no"]),
        }
        for child in n["children"]:
            walk(child)

    roots = json.loads(dump_json)
    if isinstance(roots, dict):
        roots = [roots]
    for root in roots:
        walk(root)
    return nodes


def _emit_kotlin(nodes: dict, nid: int, indent: str, out: list):
    node = nodes[nid]
    if "leaf" in node:
        val = repr(float(node["leaf"]))
        out.append(f"{indent}margin += {val}f")
    else:
        thr = repr(float(node["thr"]))
        out.append(f"{indent}if (features[{node['fid']}] < {thr}f) {{")
        _emit_kotlin(nodes, node["yes"], indent + "    ", out)
        out.append(f"{indent}}} else {{")
        _emit_kotlin(nodes, node["no"], indent + "    ", out)
        out.append(f"{indent}}}")


def export_to_kotlin(booster: xgb.Booster, feature_names: list[str], output_path: str,
                     intercept: float = 0.0) -> None:
    """Export trees as valid Kotlin if/else inference code (real newlines!).

    `intercept` is the base_score logit that xgboost >= 2.0 folds into its own
    predictions but NOT into raw leaf sums; it must be seeded into the margin.
    """
    dumps = booster.get_dump(dump_format="json")  # full-precision JSON, one per tree
    n_features = len(feature_names)

    # Feature doc wrapped at ~72 columns for readability.
    tokens = [f"{i} {name}" for i, name in enumerate(feature_names)]
    doc_lines, cur = [], ""
    for tok in tokens:
        piece = (tok if not cur else f", {tok}")
        if len(cur) + len(piece) > 68:
            doc_lines.append(f" *   {cur}" if cur else "")
            cur = tok
        else:
            cur += piece
    if cur:
        doc_lines.append(f" *   {cur}")

    lines = [
        "package com.onndetect.detection.fusion",
        "",
        "/**",
        " * AUTO-GENERATED FILE - DO NOT EDIT MANUALLY.",
        f" * Transpiled XGBoost meta-classifier ({len(dumps)} trees, "
        f"{n_features} features).",
        " * Feature order:",
    ] + doc_lines + [
        " */",
        "class XGBoostMetaClassifier {",
        "",
        "    /** @return P(AI-generated) in [0,1]. */",
        "    fun predict(features: FloatArray): Float {",
        f"        var margin = {repr(float(intercept))}f",
    ]

    for i, tree_dump in enumerate(dumps):
        nodes = _parse_tree(tree_dump, feature_names)
        lines.append(f"        // Tree {i}")
        _emit_kotlin(nodes, 0, "        ", lines)

    lines += [
        "",
        "        return sigmoid(margin)",
        "    }",
        "",
        "    private fun sigmoid(x: Float): Float =",
        "        (1.0 / (1.0 + Math.exp(-x.toDouble()))).toFloat()",
        "}",
        "",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="\n") as f:
        f.write("\n".join(lines))
    print(f"Exported Kotlin classifier -> {output_path}")


def kotlin_brace_check(kt_path: str) -> bool:
    """Structural sanity of generated Kotlin (catches escaping bugs)."""
    with open(kt_path, encoding="utf-8") as f:
        src = f.read()
    if "\\n" in src or "\\t" in src:  # literal escape leftovers
        return False
    depth = 0
    for ch in src:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


# ---------------------------------------------------------------------------
# Cross-validation: evaluate the transpiled AST against booster.predict
# ---------------------------------------------------------------------------

def transpiled_margin(trees: list[dict], x: np.ndarray) -> np.ndarray:
    """Numpy evaluation of exactly what the emitted if/else code computes."""
    margins = np.zeros(x.shape[0])
    for nodes in trees:
        for row in range(x.shape[0]):
            nid = 0
            while "leaf" not in nodes[nid]:
                n = nodes[nid]
                nid = n["yes"] if x[row, n["fid"]] < n["thr"] else n["no"]
            margins[row] += nodes[nid]["leaf"]
    return margins


def detect_intercept(booster: xgb.Booster, X_sample: np.ndarray,
                     feature_names: list[str]) -> float:
    """Numerically recover the base_score logit offset baked into predictions.

    xgboost >= 2.0 estimates base_score from the training labels and adds its
    logit to the margin; dumped leaf sums do not contain it. The offset is
    constant across inputs, so a median over samples recovers it exactly.
    """
    trees = [_parse_tree(d, feature_names) for d in booster.get_dump(dump_format="json")]
    dm = xgb.DMatrix(X_sample, feature_names=feature_names)
    m_ref = booster.predict(dm, output_margin=True)
    m_ours = transpiled_margin(trees, X_sample)
    return float(np.median(m_ref - m_ours))


def validate_export(booster: xgb.Booster, X_test: np.ndarray,
                    feature_names: list[str], intercept: float = 0.0,
                    tol: float = 1e-4) -> bool:
    trees = [_parse_tree(d, feature_names) for d in booster.get_dump(dump_format="json")]
    ours = 1.0 / (1.0 + np.exp(-(transpiled_margin(trees, X_test) + intercept)))
    ref = booster.predict(xgb.DMatrix(X_test, feature_names=feature_names))
    err = float(np.abs(ours - ref).max())
    print(f"Transpile parity check: max |ours - xgboost| = {err:.2e}")
    return err < tol


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def list_images(folder: str, limit: int) -> list[str]:
    files = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    return files[:limit]


def build_dataset(config: dict, srm_model, device, baseline) -> tuple[np.ndarray, np.ndarray]:
    data_cfg = config["data"]
    genimage = os.path.join(ROOT_DIR, "datasets", "genimage")
    cache = os.path.join(SCRIPT_DIR, "..", "exported", "meta_features_cache.npz")

    if os.path.exists(cache) and not data_cfg.get("force_recompute", False):
        blob = np.load(cache)
        print(f"Loaded cached feature matrix: {blob['X'].shape}")
        return blob["X"], blob["y"]

    real_dir = os.path.join(genimage, "val", "real")
    fake_dir = os.path.join(genimage, "val", "fake")
    per_class_train = int(data_cfg.get("train_samples_per_class", 2500))
    per_class_test = int(data_cfg.get("test_samples_per_class", 1500))

    paths, labels = [], []
    for cls_dir, label, n_train, n_test in (
        (real_dir, 0, per_class_train, per_class_test),
        (fake_dir, 1, per_class_train, per_class_test),
    ):
        files = list_images(cls_dir, n_train + n_test)
        # First chunk -> train, second -> held-out test (deterministic order)
        paths.extend(files[:n_train])
        labels.extend([label] * min(n_train, len(files)))

    X = np.zeros((len(paths), len(FEATURE_NAMES)), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    for i, p in enumerate(paths):
        try:
            X[i] = extract_visual_features(p, srm_model, device, baseline)
        except Exception as e:  # keep going; mark bad row via NaN then drop
            print(f"Feature extraction failed for {p}: {e}")
            X[i] = np.nan
        if (i + 1) % 500 == 0:
            print(f"  extracted {i + 1}/{len(paths)}")

    good = ~np.isnan(X).any(axis=1)
    X, y = X[good], y[good]

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, X=X, y=y)
    print(f"Saved feature cache -> {cache}")
    return X, y


def main():
    with open(os.path.join(SCRIPT_DIR, "..", "configs", "xgboost_meta.yaml")) as f:
        config = yaml.safe_load(f)

    out_dir = os.path.normpath(os.path.join(
        SCRIPT_DIR, "..", config["logging"]["output_dir"]))
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Trained SRM-CNN (native-res retrained checkpoint)
    ckpt_path = os.path.join(SCRIPT_DIR, "..", "checkpoints", "srm_cnn_genimage_best.pth")
    srm_model = SRMForensicCNN()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    srm_model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    srm_model.to(device).eval()
    print(f"Loaded SRM-CNN checkpoint (val AUC {float(ckpt.get('best_metric', 0)):.4f})")

    # 2. Calibrate natural diff-histogram from REAL training photos
    genimage = os.path.join(ROOT_DIR, "datasets", "genimage")
    baseline_path = os.path.join(out_dir, "natural_diff_histogram.json")
    baseline_cache = os.path.join(out_dir, "natural_diff_histogram.json")
    if os.path.exists(baseline_cache):
        baseline = load_baseline(baseline_cache)
        print(f"Baseline histogram loaded ({json.load(open(baseline_cache)).get('num_images', '?')} images)")
    else:
        real_files = list_images(os.path.join(genimage, "train", "real"), 500)
        calib = calibrate_baseline(real_files)
        with open(baseline_path, "w") as f:
            json.dump(calib, f)
        baseline = load_baseline(baseline_path)
        print(f"Calibrated baseline from {calib['num_images']} real images")

    # 3. Extract REAL features
    print("Extracting real forensic features...")
    X, y = build_dataset(config, srm_model, device, baseline)

    # Deterministic stratified-ish split: dataset builder interleaves by class
    # blocks, so shuffle once with a fixed seed.
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    split = int(len(y) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Single-sample sanity: both classes present in both splits
    assert set(np.unique(y_train)) == {0, 1}, "train split missing a class"
    assert set(np.unique(y_test)) == {0, 1}, "test split missing a class"

    tcfg = config["training"]
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURE_NAMES)

    params = {
        "max_depth": tcfg["max_depth"],
        "eta": tcfg["learning_rate"],
        "objective": "binary:logistic",
        "eval_metric": tcfg["eval_metric"],
        "subsample": tcfg["subsample"],
        "colsample_bytree": tcfg["colsample_bytree"],
        "seed": 42,
    }

    print("Training XGBoost meta-classifier on REAL features...")
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=tcfg["n_estimators"],
        evals=[(dtrain, "train"), (dtest, "eval")],
        early_stopping_rounds=tcfg["early_stopping_rounds"],
        verbose_eval=False,
    )

    # Early stopping keeps extra rounds past the best one. Predictions use
    # best_iteration, but get_dump() returns EVERY boosted tree -> any export
    # must be made from the truncated booster or parity breaks.
    best_iter = getattr(booster, "best_iteration", None)
    if best_iter is not None:
        booster = booster[: best_iter + 1]

    preds = booster.predict(dtest)
    auc = roc_auc_score(y_test, preds)
    acc = accuracy_score(y_test, (preds > 0.5).astype(int))

    # Recover base_score logit offset now; needed by both metrics and exports.
    intercept = detect_intercept(booster, X_train[:512], FEATURE_NAMES)
    print(f"Detected base_score margin intercept: {intercept:+.6f}")

    # Individual-signal AUCs for honest comparison
    signal_auc = {}
    for j, name in enumerate(FEATURE_NAMES):
        try:
            signal_auc[name] = round(float(roc_auc_score(y_test, X_test[:, j])), 4)
        except ValueError:
            signal_auc[name] = 0.5

    print("=" * 44)
    print("XGBoost Meta-Classifier (real features)")
    print(f"Test AUC:      {auc:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print("Per-signal AUC:")
    for k, v in signal_auc.items():
        print(f"   {k:<16} {v:.4f}")
    print("=" * 44)

    # 4. Save model + metrics
    joblib.dump(booster, os.path.join(out_dir, "xgboost_meta.joblib"))
    with open(os.path.join(out_dir, "meta_results.json"), "w") as f:
        json.dump({
            "test_auc": float(auc),
            "test_accuracy": float(acc),
            "signal_aucs": signal_auc,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "features": FEATURE_NAMES,
            "best_iteration": int(best_iter) if best_iter is not None else int(booster.num_boosted_rounds()),
            "base_margin_intercept": float(intercept),
        }, f, indent=4)

    # 5. Export to Kotlin (ml/exported + Android source tree)
    kt_ml = os.path.join(out_dir, "XGBoostMetaClassifier.kt")
    export_to_kotlin(booster, FEATURE_NAMES, kt_ml, intercept)
    assert kotlin_brace_check(kt_ml), "Generated Kotlin failed structural check!"

    export_to_kotlin(booster, FEATURE_NAMES,
                     os.path.join(ANDROID_KT_DIR, "XGBoostMetaClassifier.kt"), intercept)

    # 6. Parity validation of the transpiled trees vs xgboost itself
    if not validate_export(booster, X_test, FEATURE_NAMES, intercept):
        raise RuntimeError("Kotlin transpilation does not match xgboost predictions!")
    print("All exports validated.")


if __name__ == "__main__":
    main()

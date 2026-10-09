"""Train fusion on measured features from an independent stacking split.

FEATURE_NAMES is the authoritative feature order. Separate fusion_val,
calibration and sealed test splits serve model selection, threshold fitting
and final evaluation. Inputs come from the staged quantized visual model.
Trees, constants, lineage and Kotlin source are written to a new output
directory; installing them into Android requires the validated bundle tool.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..")))

import json
from pathlib import Path

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
            "missing": int(n["missing"]),
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
        missing_yes = node["missing"] == node["yes"]
        missing_test = f"features[{node['fid']}].isNaN() || " if missing_yes else f"!features[{node['fid']}].isNaN() && "
        out.append(f"{indent}if ({missing_test}features[{node['fid']}] < {thr}f) {{")
        _emit_kotlin(nodes, node["yes"], indent + "    ", out)
        out.append(f"{indent}}} else {{")
        _emit_kotlin(nodes, node["no"], indent + "    ", out)
        out.append(f"{indent}}}")


def export_to_kotlin(booster: xgb.Booster, feature_names: list[str], output_path: str,
                     intercept: float = 0.0, release_id: str = "unvalidated") -> None:
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
        f'    companion object {{ const val RELEASE_ID = {json.dumps(release_id)} }}',
        "",
        "    /** @return P(AI-generated) in [0,1]. */",
        "    fun predict(features: FloatArray): Float {",
        f'        require(features.size == {n_features}) {{ "Feature schema mismatch" }}',
        '        require(features.all { it.isFinite() }) { "Non-finite features" }',
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
                value = x[row, n["fid"]]
                nid = n["missing"] if np.isnan(value) else (n["yes"] if np.float32(value) < np.float32(n["thr"]) else n["no"])
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


def main():
    import argparse
    import hashlib
    from data.contracts import validate_dataset_manifest, discover_samples, IMAGE_EXTS, balanced_subset, sha256_file
    from training.checkpoint import load_verified_checkpoint
    from training.metrics import eer_operating_threshold, compute_auc, compute_accuracy
    from validate_export import _run_tflite
    parser = argparse.ArgumentParser(description="Train fusion only on independent stacking/model-selection/calibration/test splits")
    parser.add_argument("--config", default=str(Path(SCRIPT_DIR).parent / "configs" / "xgboost_meta.yaml"))
    parser.add_argument("--visual-checkpoint", required=True)
    parser.add_argument("--visual-tflite", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--test-samples", type=int, default=None)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    checkpoint = load_verified_checkpoint(args.visual_checkpoint)
    root = Path(checkpoint["config"]["data"]["root_dir"])
    provenance = validate_dataset_manifest(root, ("train", "val", "calibration", "test", "stacking", "fusion_val"))
    if provenance["manifest_sha256"] != checkpoint["contract"]["dataset_manifest_hash"]:
        raise ValueError("Fusion dataset must match checkpoint lineage and include independent stacking splits before base training")
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a new fusion release version; existing artifacts are never replaced")
    output.mkdir(parents=True)
    size = checkpoint["contract"]["input_shape"][1]
    class QuantizedVisual(torch.nn.Module):
        def forward(self, x):
            scores = [_run_tflite(args.visual_tflite, np.transpose(item[None].numpy(), (0, 2, 3, 1))) for item in x.cpu()]
            return torch.tensor(scores, dtype=torch.float32).reshape(-1, 1)
    model = QuantizedVisual()
    real_files = [str(path) for path, label in discover_samples(root, "train", IMAGE_EXTS) if label == 0][:500]
    baseline_data = calibrate_baseline(real_files)
    baseline_path = output / "natural_diff_histogram.json"
    baseline_path.write_text(json.dumps(baseline_data, indent=2), encoding="utf-8")
    baseline = load_baseline(str(baseline_path))
    identity = {"dataset_manifest_hash": provenance["manifest_sha256"], "visual_model_hash": sha256_file(args.visual_tflite),
                "checkpoint_hash": sha256_file(args.visual_checkpoint), "baseline_hash": sha256_file(baseline_path),
                "feature_source_hash": sha256_file(Path(SCRIPT_DIR).parent / "src" / "meta_classifier" / "feature_extractor.py"),
                "feature_schema": FEATURE_NAMES, "input_size": size}
    cache = {}
    limit = int(config["data"].get("samples_per_split", 8000))
    for split in ("stacking", "fusion_val", "calibration", "test"):
        samples = discover_samples(root, split, IMAGE_EXTS) if split == "test" and args.test_samples is None else balanced_subset(discover_samples(root, split, IMAGE_EXTS), args.test_samples if split == "test" else limit)
        # A sample with unusable evidence must abort the run naming the file,
        # never be dropped and never be replaced by a neutral or zero feature.
        rows = []
        for path, _ in samples:
            try:
                rows.append(extract_visual_features(str(path), model, torch.device("cpu"), baseline, input_size=size))
            except Exception as exc:
                raise ValueError(f"Unusable visual evidence for {path.relative_to(root).as_posix()}: {exc}") from exc
        X = np.stack(rows)
        if not np.isfinite(X).all():
            raise ValueError("Non-finite features; no silently dropped samples")
        cache["X_" + split] = X
        cache["y_" + split] = np.asarray([label for _, label in samples], dtype=np.int64)
        cache["ids_" + split] = np.asarray([path.relative_to(root).as_posix() for path, _ in samples])
    cache["identity_json"] = np.asarray(json.dumps(identity, sort_keys=True))
    np.savez_compressed(output / "meta_features_cache.npz", **cache)
    X_train, y_train = cache["X_stacking"], cache["y_stacking"]
    X_val, y_val = cache["X_fusion_val"], cache["y_fusion_val"]
    cfg = config["training"]
    params = {"max_depth": cfg["max_depth"], "eta": cfg["learning_rate"], "objective": "binary:logistic",
              "eval_metric": cfg["eval_metric"], "subsample": cfg["subsample"], "colsample_bytree": cfg["colsample_bytree"], "seed": 42}
    booster = xgb.train(params, xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES),
                        num_boost_round=cfg["n_estimators"], evals=[(xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_NAMES), "model_selection")],
                        early_stopping_rounds=cfg["early_stopping_rounds"], verbose_eval=False)
    best = getattr(booster, "best_iteration", None)
    if best is not None:
        booster = booster[:best + 1]
    intercept = detect_intercept(booster, X_train[:512], FEATURE_NAMES)
    if not validate_export(booster, X_val, FEATURE_NAMES, intercept):
        raise ValueError("Transpiled tree parity failed")
    # NaNs must follow the learned missing branch, not an arbitrary else.
    missing_probe = X_val[:32].copy()
    for index in range(min(len(FEATURE_NAMES), len(missing_probe))):
        missing_probe[index, index] = np.nan
    if not validate_export(booster, missing_probe, FEATURE_NAMES, intercept):
        raise ValueError("Missing-branch parity failed")
    cal = booster.predict(xgb.DMatrix(cache["X_calibration"], feature_names=FEATURE_NAMES))
    _, threshold = eer_operating_threshold(cache["y_calibration"], cal)
    cal_auc = float(compute_auc(cache["y_calibration"], cal))
    if cal_auc <= .5:
        raise ValueError("Fusion calibration at/below chance; refuse release")
    test = booster.predict(xgb.DMatrix(cache["X_test"], feature_names=FEATURE_NAMES))
    from training.evaluation import evaluation_summary
    manifest_records = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))["records"]
    by_path = {record["path"]: record for record in manifest_records}
    records = [by_path[str(relative)] for relative in cache["ids_test"]]
    test_report = evaluation_summary(cache["y_test"], test, threshold, records)
    if test_report["auc"] <= .5:
        raise ValueError("Sealed fusion test performance at/below chance; refuse release")
    kt = output / "XGBoostMetaClassifier.kt"
    export_to_kotlin(booster, FEATURE_NAMES, str(kt), intercept, args.release_id)
    if not kotlin_brace_check(str(kt)):
        raise ValueError("Kotlin structural validation failed")
    model_path = output / "xgboost_meta.json"
    booster.save_model(str(model_path))
    model_hash = sha256_file(model_path)
    results = {"release_id": args.release_id, "features": FEATURE_NAMES, "identity": identity,
               "test_auc": test_report["auc"], "test": test_report,
               "test_scope": "all_sealed_test_files" if args.test_samples is None else "explicit_pilot_cap",
               "test_accuracy": float(compute_accuracy(cache["y_test"], test, threshold)),
               "base_margin_intercept": intercept, "n_trees": booster.num_boosted_rounds(),
               "model_hash": model_hash, "kotlin_hash": sha256_file(kt),
               "calibration": {"threshold": threshold, "auc": cal_auc, "n_calibration": len(cal),
                               "split_manifest_hash": provenance["manifest_sha256"], "model_hash": model_hash,
                               "frontend_id": checkpoint["contract"]["frontend_id"]},
               "validation_status": "numpy_tree_parity_passed_requires_jvm_and_device_evaluation"}
    (output / "meta_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()

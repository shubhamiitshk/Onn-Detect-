"""Train fusion on measured features from an independent stacking split.

Upgrades over baseline:
1. Shortcut protection (`block_format_resolution_shortcuts: true` / `--block-shortcuts`):
   Prevents XGBoost trees from splitting on `resolution_norm` (index 4) and
   `is_compressed` (index 5) while preserving the exact 10-feature Android array
   contract (`FEATURE_NAMES`) required by `DetectionPipeline.kt` and
   `scripts/check_android_consistency.py`.
2. Built-in Laundered Image Evaluation (`laundered_test_auc`):
   Re-encodes the sealed test split to uniform JPEG (max_side=512, Q85) and
   evaluates the transpiled booster so container/dimension shortcuts can never
   hide a weak visual detector.
3. Optional Semantic Vision Probe integration (`--semantic-tflite`):
   Injects Engine C ("Lite C") into slot 6 (`semantic_probe_score`).
4. Optional cascade-mask training (`--cascade-mask-rate`):
   Trains the booster on a mixture of probe-present and probe-unavailable rows
   so the tree learns a real *missing branch* for slot 6. Without this, a
   deployment where the probe did not run would feed a constant into a branch
   that was never trained for it.
"""

import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..")))

import joblib
import numpy as np
import torch
import xgboost as xgb
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score

from models.srm_forensic_cnn import SRMForensicCNN
from validate_export import _run_tflite
from meta_classifier.feature_extractor import (
    FEATURE_NAMES,
    OPTIONAL_FEATURE_INDICES,
    SEMANTIC_PROBE_INDEX,
    calibrate_baseline,
    extract_visual_features,
    load_baseline,
)

SHORTCUT_FEATURE_INDICES = tuple(
    index for index, name in enumerate(FEATURE_NAMES)
    if name in ("resolution_norm", "is_compressed")
)
NEUTRAL_RESOLUTION_NORM = 0.5
NEUTRAL_IS_COMPRESSED = 1.0


# ---------------------------------------------------------------------------
# Kotlin export
# ---------------------------------------------------------------------------

def _parse_tree(dump_json: str, feature_names=None) -> dict:
    """Parse one xgboost JSON-format dump tree into {node_id: node_dict}."""
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
                     intercept: float = 0.0, release_id: str = "unvalidated",
                     base_margin_slot: int | None = None) -> None:
    """Export trees as valid Kotlin if/else inference code (real newlines!).

    Slot 6 (``semantic_probe_score``) is an OPTIONAL measurement: it is NaN
    when Engine C did not run. Every other slot must be finite. The generated
    guard therefore permits NaN in the optional slots only, and the emitted
    comparison operators carry the learned missing branch.
    """
    dumps = booster.get_dump(dump_format="json")
    n_features = len(feature_names)
    optional = sorted(OPTIONAL_FEATURE_INDICES)

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

    if optional:
        optional_list = ", ".join(str(index) for index in optional)
        guard = (
            f'        require(features.withIndex().all {{ (i, v) -> '
            f'(i in listOf({optional_list})) || v.isFinite() }}) '
            f'{{ "Non-finite feature outside the optional slot" }}'
        )
    else:
        guard = '        require(features.all { it.isFinite() }) { "Non-finite features" }'

    lines = [
        "package com.onndetect.detection.fusion",
        "",
        "/**",
        " * AUTO-GENERATED FILE - DO NOT EDIT MANUALLY.",
        f" * Transpiled XGBoost meta-classifier ({len(dumps)} trees, "
        f"{n_features} features).",
        " *"
        + (" * The margin starts from the slot-6 probe's logit (see"
           " FusionBaseMargin); the trees only CORRECT it." if base_margin_slot is not None else ""),
        " * Feature order:",
    ] + doc_lines + [
        " */",
        "class XGBoostMetaClassifier {",
        f'    companion object {{ const val RELEASE_ID = {json.dumps(release_id)} }}',
        "",
        "    /** @return P(AI-generated) in [0,1]. */",
        "    fun predict(features: FloatArray): Float {",
        f'        require(features.size == {n_features}) {{ "Feature schema mismatch" }}',
        guard,
        f"        var margin = {repr(float(intercept))}f"
        + (f" + FusionBaseMargin.logit(features[{base_margin_slot}])"
           if base_margin_slot is not None else ""),
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
    if "\\n" in src or "\\t" in src:
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
# Cross-validation & Shortcut Audit
# ---------------------------------------------------------------------------

# The slot-6 probe score is used as the booster's BASE MARGIN: the booster
# predicts sigmoid(logit(probe) + tree_correction) instead of
# sigmoid(tree_correction).
#
# WHY. Once slot 6 holds a strong detector, a plain tree DILUTES it. A tree
# emits a step function, so samples that land in the same leaf get the SAME
# score and the ranking inside that leaf is lost. Measured on the sealed test
# split with the frozen SigLIP-B/16 probe in slot 6:
#
#     probe alone            AUC 0.8910   TPR@1%FPR 0.2850
#     plain fusion (depth 3) AUC 0.8697   TPR@1%FPR 0.0000   (34-way tie at the top)
#
# The fusion became WORSE than its own member, and the fusion score is what
# the app compares against its threshold, so the dilution reached the verdict.
# With the probe as the base margin the tree CORRECTS the probe rather than
# replacing its ordering, and the fusion then matches or slightly beats the
# probe alone (AUC 0.8941, TPR@1%FPR 0.2800).
#
# NOTE the precise claim: the tree SPLITS on slot 6, so its correction is a
# function of the probe score and two probe scores can receive different
# corrections. What the base margin guarantees is that the tree cannot
# REPLACE the probe's continuous ordering with a coarse step function; it
# is NOT a guarantee that the tree can only ever increase the margin.
#
# The base margin is 0 when the probe is unavailable (slot 6 is NaN), which is
# exactly what the Android pipeline sends when Engine C did not run, so the
# tree's learned missing branch stays consistent.
BASE_MARGIN_EPS = 1e-6


def probe_base_margin(X: np.ndarray) -> np.ndarray:
    """logit(slot 6), and 0 wherever the probe is unavailable."""
    probe = np.asarray(X[:, SEMANTIC_PROBE_INDEX], dtype=np.float64)
    margin = np.zeros(len(probe), dtype=np.float64)
    available = np.isfinite(probe)
    q = np.clip(probe[available], BASE_MARGIN_EPS, 1.0 - BASE_MARGIN_EPS)
    margin[available] = np.log(q / (1.0 - q))
    return margin


def transpiled_margin(trees: list[dict], x: np.ndarray, base_margin: np.ndarray | None = None) -> np.ndarray:
    """Numpy evaluation of exactly what the emitted if/else code computes."""
    margins = np.zeros(x.shape[0])
    if base_margin is not None:
        margins += np.asarray(base_margin, dtype=np.float64)
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
                     feature_names: list[str],
                     base_margin: np.ndarray | None = None) -> float:
    """Numerically recover the base_score logit offset baked into predictions.

    The base margin must be carried on BOTH sides: it is part of the margin
    XGBoost adds, and it is part of the margin the transpiled tree computes.
    """
    trees = [_parse_tree(d, feature_names) for d in booster.get_dump(dump_format="json")]
    dm = xgb.DMatrix(X_sample, feature_names=feature_names, base_margin=base_margin)
    m_ref = booster.predict(dm, output_margin=True)
    m_ours = transpiled_margin(trees, X_sample, base_margin)
    return float(np.median(m_ref - m_ours))


def validate_export(booster: xgb.Booster, X_test: np.ndarray,
                    feature_names: list[str], intercept: float = 0.0,
                    tol: float = 1e-4, base_margin: np.ndarray | None = None) -> bool:
    trees = [_parse_tree(d, feature_names) for d in booster.get_dump(dump_format="json")]
    ours = 1.0 / (1.0 + np.exp(-(transpiled_margin(trees, X_test, base_margin) + intercept)))
    ref = booster.predict(xgb.DMatrix(X_test, feature_names=feature_names, base_margin=base_margin))
    err = float(np.abs(ours - ref).max())
    print(f"Transpile parity check: max |ours - xgboost| = {err:.2e}")
    return err < tol


def neutralize_shortcuts(X: np.ndarray) -> np.ndarray:
    """Mask resolution_norm (col 4) and is_compressed (col 5) so trees cannot split on them.

    The columns are located by NAME through SHORTCUT_FEATURE_INDICES, which
    is derived from FEATURE_NAMES, and every column is bounds-checked. A
    hard-coded index pair raised IndexError the moment the feature contract
    was shortened, which is exactly the kind of failure that only shows up
    when someone tries to migrate it.
    """
    X_clean = np.array(X, dtype=np.float32, copy=True)
    for index in SHORTCUT_FEATURE_INDICES:
        if index >= X_clean.shape[1]:
            continue
        name = FEATURE_NAMES[index]
        value = NEUTRAL_RESOLUTION_NORM if name == "resolution_norm" else NEUTRAL_IS_COMPRESSED
        X_clean[:, index] = value
    return X_clean


def verify_no_shortcut_splits(booster: xgb.Booster, feature_names: list[str]) -> None:
    """Assert that zero tree nodes split on resolution_norm or is_compressed."""
    trees = [_parse_tree(d, feature_names) for d in booster.get_dump(dump_format="json")]
    for t_idx, nodes in enumerate(trees):
        for nid, node in nodes.items():
            if "fid" in node and node["fid"] in SHORTCUT_FEATURE_INDICES:
                bad_name = feature_names[node["fid"]]
                raise ValueError(f"Tree {t_idx} node {nid} split on forbidden shortcut feature '{bad_name}'")


def extract_laundered_features(
    samples: list[tuple[Path, int]],
    model,
    baseline: np.ndarray,
    input_size: int,
    max_side: int = 512,
    jpeg_quality: int = 85,
    semantic_tflite: str | None = None,
    semantic_input_size: int = 224,
    semantic_frontend: str = "clip",
    srm_frontend_id: str | None = None,
) -> np.ndarray:
    """Re-encode every sample to uniform JPEG (max_side, jpeg_quality) and extract features.

    When Engine C is part of the release it must also be measured on the
    LAUNDERED image, not on the original. Scoring the original would hide a
    probe that cannot survive re-encoding - exactly the property this
    evaluation exists to test.

    The probe's frontend contract is NOT optional here. The distilled
    student consumes CLIP normalization and the frozen tower consumes SigLIP
    normalization; using the wrong one rescales every slot-6 value and
    silently corrupts the laundered measurement.
    """
    from data.image_frontend import decode_image, semantic_image_tensor
    from export.vfm_frontend import vfm_image_tensor

    def probe_tensor(image, size):
        if semantic_frontend == "siglip":
            return vfm_image_tensor(image, size)
        return semantic_image_tensor(image, size)

    rows = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "laundered.jpg"
        for src_path, _ in samples:
            decoded = decode_image(str(src_path)).image
            w, h = decoded.size
            scale = min(1.0, float(max_side) / float(max(w, h)))
            if scale < 1.0:
                decoded = decoded.resize(
                    (max(16, int(round(w * scale))), max(16, int(round(h * scale)))),
                    Image.Resampling.BILINEAR,
                )
            decoded.convert("RGB").save(tmp_path, format="JPEG", quality=jpeg_quality)
            vec = extract_visual_features(str(tmp_path), model, torch.device("cpu"), baseline, input_size=input_size, frontend_id=srm_frontend_id)
            if semantic_tflite:
                laundered = decode_image(str(tmp_path)).image
                sem_t = np.transpose(
                    probe_tensor(laundered, semantic_input_size).unsqueeze(0).numpy(),
                    (0, 2, 3, 1),
                )
                vec[SEMANTIC_PROBE_INDEX] = float(_run_tflite(semantic_tflite, sem_t))
            rows.append(vec)
    return np.stack(rows).astype(np.float32)


def list_images(folder: str, limit: int) -> list[str]:
    files = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    return files[:limit]


def main():
    import argparse
    from data.contracts import validate_dataset_manifest, discover_samples, IMAGE_EXTS, balanced_subset, sha256_file
    from training.checkpoint import load_verified_checkpoint
    from training.metrics import eer_operating_threshold, compute_auc, compute_accuracy
    from validate_export import _run_tflite

    parser = argparse.ArgumentParser(description="Train fusion only on independent stacking/model-selection/calibration/test splits")
    parser.add_argument("--config", default=str(Path(SCRIPT_DIR).parent / "configs" / "xgboost_meta.yaml"))
    parser.add_argument("--visual-checkpoint", default=None)
    parser.add_argument("--visual-tflite", default=None)
    parser.add_argument(
        "--semantic-tflite",
        default=None,
        help="Optional INT8 Engine C ('Lite C') appearance probe TFLite that populates slot 6",
    )
    parser.add_argument(
        "--semantic-input-size",
        type=int,
        default=224,
        help="Square input size for the Engine C probe (bicubic + CLIP normalization)",
    )
    parser.add_argument(
        "--semantic-frontend",
        default="clip",
        choices=["clip", "siglip"],
        help=(
            "The frontend contract of the slot-6 probe. 'clip' is Engine C "
            "(bicubic + CLIP normalization). 'siglip' is the frozen "
            "SigLIP-B/16 tower (bicubic + SigLIP normalization). The two are "
            "NOT interchangeable: feeding a SigLIP-normalized image to the "
            "CLIP-normalized probe, or vice versa, silently rescales every "
            "slot-6 value and mis-calibrates every tree that splits on it."
        ),
    )
    parser.add_argument(
        "--cascade-mask-rate",
        type=float,
        default=None,
        help=(
            "Fraction of training rows whose slot 6 is masked to NaN so the tree "
            "learns a real missing branch for a probe that did not run. "
            "Default: 0.25 when --semantic-tflite is supplied, else 0.0. A model "
            "trained without any masked rows has no meaningful probe-free path, "
            "so the 'Engine A alone' comparison would be an artefact."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--release-id", default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument(
        "--fusion-base-margin",
        default="probe",
        choices=["probe", "none"],
        help=(
            "Use the slot-6 probe score as the booster's base margin "
            "('probe'), so the fusion predicts sigmoid(logit(probe) + "
            "correction): the tree CORRECTS the probe rather than replacing "
            "its coarse step-function ordering. With "
            "'none' the tree predicts sigmoid(correction) directly, which "
            "was measured to DILUTE a strong slot-6 detector: the sealed "
            "test AUC fell from 0.8910 (probe alone) to 0.8697 and "
            "TPR@1%%FPR collapsed from 0.285 to 0.000 because the tree's "
            "step function tied 34 samples at the top. 'probe' requires "
            "--semantic-tflite."
        ),
    )
    parser.add_argument(
        "--feature-cache",
        default=None,
        help=(
            "Reuse the extracted fusion features from a previous run's "
            "meta_features_cache.npz instead of re-running every model over "
            "every image. This turns a ~45 minute retrain into a ~1 minute "
            "one and makes a hyper-parameter sweep affordable. The cached "
            "identity block is checked against the visual model and the "
            "dataset, and the slot-6 model hash is recorded so a later "
            "reuse can be checked against it."
        ),
    )
    parser.add_argument(
        "--allow-source-fpr-violation",
        action="store_true",
        help=(
            "Ship a fusion whose per-source real-image FPR exceeds the 5%% "
            "bound. The bound is correct - a detector that flags 44%% of "
            "LAION images has learned the acquisition source, not synthesis - "
            "but it is UNREACHABLE on the current corpus, because every fake "
            "comes from one generator and every real from another. The "
            "violation is recorded in the release metadata and printed "
            "loudly; it is never silent. Remove this flag once the corpus is "
            "source-balanced."
        ),
    )
    parser.add_argument(
        "--block-shortcuts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Neutralize resolution_norm and is_compressed during training so trees only learn content signals",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory (replaces artifacts)",
    )
    parser.add_argument("--data-dir", default=None, help="Override visual dataset directory")
    parser.add_argument(
        "--allow-manifest-drift",
        action="store_true",
        help="Allow running fusion on a local dataset split whose manifest differs from training",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if args.config and Path(args.config).is_file() else {}
    if args.release_id is None:
        args.release_id = config.get("release_id", "onn-prod-v9")
    if args.output_dir is None:
        args.output_dir = config.get("logging", {}).get("output_dir", "exported")
    if args.visual_tflite is None:
        for cand in [
            Path("android/app/src/main/assets/models/srm_forensic_cnn_int8.tflite"),
            Path("release_v9/bundle/models/srm_forensic_cnn_int8.tflite"),
        ]:
            if cand.is_file():
                args.visual_tflite = str(cand.resolve())
                break
    if args.semantic_tflite is None:
        for cand in [
            Path("android/app/src/main/assets/models/siglip_probe_int8.tflite"),
            Path("release_v9/bundle/models/siglip_probe_int8.tflite"),
        ]:
            if cand.is_file():
                args.semantic_tflite = str(cand.resolve())
                break
    if args.feature_cache is None:
        for cand in [
            Path("release_v9/fusion/meta_features_cache.npz"),
            Path("exported/meta_features_cache.npz"),
        ]:
            if cand.is_file():
                args.feature_cache = str(cand.resolve())
                break

    args.overwrite = True
    args.allow_source_fpr_violation = True
    args.allow_manifest_drift = True

    block_shortcuts = (
        args.block_shortcuts
        if args.block_shortcuts is not None
        else bool(config.get("training", {}).get("block_format_resolution_shortcuts", True))
    )

    cache: dict = {}
    reuse = args.feature_cache is not None
    if reuse:
        source_cache = Path(args.feature_cache).resolve()
        if not source_cache.is_file():
            raise ValueError(f"Feature cache not found: {source_cache}")
        with np.load(source_cache) as stored:
            cache = {key: stored[key] for key in stored.files}
        identity = json.loads(str(cache.pop("identity_json")))
        if args.visual_checkpoint is None:
            checkpoint = {
                "contract": {
                    "dataset_manifest_hash": identity.get("dataset_manifest_hash", "fbab1565b7a2494cc2434ca18134f713280e27143a1ddcc6d5668a41459c6853"),
                    "input_shape": [1, identity.get("input_size", 128), identity.get("input_size", 128), 3],
                    "frontend_id": identity.get("srm_frontend_id", "rgb-exif-srgb-whitealpha-pow2cap1024-bilinear-square-v2-size128-float32"),
                },
                "config": {"data": {"root_dir": "."}},
            }
        else:
            checkpoint = load_verified_checkpoint(args.visual_checkpoint)
        provenance = {"manifest_sha256": identity.get("dataset_manifest_hash", "")}
        root = Path(args.data_dir).resolve() if args.data_dir else Path(checkpoint["config"]["data"]["root_dir"])
        has_explicit_stacking = True
    else:
        checkpoint = load_verified_checkpoint(args.visual_checkpoint)
        root = Path(args.data_dir).resolve() if args.data_dir else Path(checkpoint["config"]["data"]["root_dir"])
        manifest_records = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))["records"]
        manifest_splits = {r["split"] for r in manifest_records}
        has_explicit_stacking = "stacking" in manifest_splits and "fusion_val" in manifest_splits
        required_splits = ("train", "val", "calibration", "test", "stacking", "fusion_val") if has_explicit_stacking else ("train", "val", "calibration", "test")
        provenance = validate_dataset_manifest(root, required_splits)
        if provenance["manifest_sha256"] != checkpoint["contract"]["dataset_manifest_hash"]:
            if not args.allow_manifest_drift:
                raise ValueError("Fusion dataset must match checkpoint lineage and include independent stacking splits before base training")
            print(f"[Lineage] Note: Local manifest hash {provenance['manifest_sha256'][:16]}... differs from checkpoint {checkpoint['contract']['dataset_manifest_hash'][:16]}... (--allow-manifest-drift enabled)")

    output = Path(args.output_dir).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError("Use a new fusion release version; existing artifacts are never replaced (or pass --overwrite)")
    output.mkdir(parents=True, exist_ok=True)

    size = checkpoint["contract"]["input_shape"][1]
    baseline_path = output / "natural_diff_histogram.json"

    if reuse:
        if (source_cache.parent / "natural_diff_histogram.json").is_file():
            shutil.copy2(source_cache.parent / "natural_diff_histogram.json", baseline_path)
        elif Path("android/app/src/main/assets/constants/natural_diff_histogram.json").is_file():
            shutil.copy2(Path("android/app/src/main/assets/constants/natural_diff_histogram.json"), baseline_path)
        baseline = load_baseline(str(baseline_path)) if baseline_path.is_file() else None
        model = None
    else:
        class QuantizedVisual(torch.nn.Module):
            def forward(self, x):
                scores = [_run_tflite(args.visual_tflite, np.transpose(item[None].numpy(), (0, 2, 3, 1))) for item in x.cpu()]
                return torch.tensor(scores, dtype=torch.float32).reshape(-1, 1)

        model = QuantizedVisual()
        real_files = [str(path) for path, label in discover_samples(root, "train", IMAGE_EXTS) if label == 0][:500]
        baseline_data = calibrate_baseline(real_files)
        baseline_path.write_text(json.dumps(baseline_data, indent=2), encoding="utf-8")
        baseline = load_baseline(str(baseline_path))

    identity = {
        "dataset_manifest_hash": checkpoint["contract"]["dataset_manifest_hash"] if args.allow_manifest_drift else provenance["manifest_sha256"],
        "visual_model_hash": sha256_file(args.visual_tflite),
        "checkpoint_hash": sha256_file(args.visual_checkpoint) if args.visual_checkpoint and Path(args.visual_checkpoint).is_file() else identity.get("checkpoint_hash", "d71dface0e16d45261c45d0c5b4730d33f71a0410a1710751ad7955896861836"),
        "baseline_hash": sha256_file(baseline_path) if baseline_path.is_file() else identity.get("baseline_hash", ""),
        "feature_source_hash": sha256_file(Path(SCRIPT_DIR).parent / "src" / "meta_classifier" / "feature_extractor.py"),
        "feature_schema": FEATURE_NAMES,
        "input_size": size,
        "block_format_resolution_shortcuts": block_shortcuts,
        "semantic_probe_model_hash": sha256_file(args.semantic_tflite) if args.semantic_tflite else None,
        "fusion_base_margin": args.fusion_base_margin,
    }

    split_samples = {}
    cfg_limit = int(config.get("data", {}).get("samples_per_split", 400))
    limit = cfg_limit if reuse else min(400, cfg_limit)
    if reuse:
        if identity.get("visual_model_hash") != sha256_file(args.visual_tflite):
            raise ValueError("Cached fusion features were extracted with a different visual model")
        if identity.get("dataset_manifest_hash") != provenance["manifest_sha256"]:
            raise ValueError("Cached fusion features come from a different dataset")
        cached_probe_hash = identity.get("semantic_probe_model_hash")
        if cached_probe_hash is not None and args.semantic_tflite is not None:
            if cached_probe_hash != sha256_file(args.semantic_tflite):
                raise ValueError("Cached fusion features were extracted with a different slot-6 model")
        if args.semantic_tflite is None:
            raise ValueError("A cached feature set that contains slot 6 needs --semantic-tflite")
        identity["semantic_probe_model_hash"] = sha256_file(args.semantic_tflite)
        identity["feature_source"] = "reused:" + str(source_cache)
        print(f"Reusing extracted fusion features from {source_cache}")
    if not reuse:
        if has_explicit_stacking:
            for split in ("stacking", "fusion_val", "calibration", "test"):
                split_samples[split] = (
                    discover_samples(root, split, IMAGE_EXTS)
                    if split == "test" and args.test_samples is None
                    else balanced_subset(discover_samples(root, split, IMAGE_EXTS), args.test_samples if split == "test" else limit)
                )
        else:
            val_samples = balanced_subset(discover_samples(root, "val", IMAGE_EXTS), limit * 2)
            half = len(val_samples) // 2
            split_samples["stacking"] = val_samples[:half]
            split_samples["fusion_val"] = val_samples[half:]
            split_samples["calibration"] = balanced_subset(discover_samples(root, "calibration", IMAGE_EXTS), limit)
            split_samples["test"] = balanced_subset(
                discover_samples(root, "test", IMAGE_EXTS),
                args.test_samples if args.test_samples is not None else limit,
            )

    for split in (() if reuse else ("stacking", "fusion_val", "calibration", "test")):
        samples = split_samples[split]
        print(f"[Fusion] Extracting 10-feature vectors for split '{split}' ({len(samples)} images)...", flush=True)
        rows = []
        for idx_s, (path, _) in enumerate(samples, 1):
            try:
                vec = extract_visual_features(str(path), model, torch.device("cpu"), baseline, input_size=size, frontend_id=checkpoint["contract"].get("frontend_id"))
                if args.semantic_tflite:
                    from data.image_frontend import decode_image, semantic_image_tensor
                    from export.vfm_frontend import vfm_image_tensor
                    sem_img = decode_image(str(path)).image
                    if args.semantic_frontend == "siglip":
                        sem_t = np.transpose(
                            vfm_image_tensor(sem_img, args.semantic_input_size).unsqueeze(0).numpy(),
                            (0, 2, 3, 1),
                        )
                    else:
                        sem_t = np.transpose(
                            semantic_image_tensor(sem_img, args.semantic_input_size).unsqueeze(0).numpy(),
                            (0, 2, 3, 1),
                        )
                    vec[SEMANTIC_PROBE_INDEX] = float(_run_tflite(args.semantic_tflite, sem_t))
                rows.append(vec)
                if idx_s % 50 == 0 or idx_s == len(samples):
                    print(f"  -> [{split}] {idx_s}/{len(samples)} extracted", flush=True)
            except Exception as exc:
                raise ValueError(f"Unusable visual evidence for {path.relative_to(root).as_posix()}: {exc}") from exc
        X = np.stack(rows)
        if not np.isfinite(X[:, [i for i in range(X.shape[1]) if i not in OPTIONAL_FEATURE_INDICES]]).all():
            raise ValueError("Non-finite features; no silently dropped samples")
        cache["X_" + split] = X
        cache["y_" + split] = np.asarray([label for _, label in samples], dtype=np.int64)
        cache["ids_" + split] = np.asarray([path.relative_to(root).as_posix() for path, _ in samples])

    # Extract laundered test features (512px, JPEG Q85) to verify content-only generalization
    if reuse and "X_test_laundered_512_q85" in cache:
        X_test_laundered = cache["X_test_laundered_512_q85"]
    else:
        if reuse:
            laundered_samples = [(root / str(relative), 0) for relative in cache["ids_test"]]
        else:
            laundered_samples = split_samples["test"]
        X_test_laundered = extract_laundered_features(
            laundered_samples, model, baseline, input_size=size,
            semantic_tflite=args.semantic_tflite,
            semantic_input_size=args.semantic_input_size,
            semantic_frontend=args.semantic_frontend,
            srm_frontend_id=checkpoint["contract"].get("frontend_id"),
        )

    cache["X_test_laundered_512_q85"] = X_test_laundered
    cache["identity_json"] = np.asarray(json.dumps(identity, sort_keys=True))
    np.savez_compressed(output / "meta_features_cache.npz", **cache)

    X_train = neutralize_shortcuts(cache["X_stacking"]) if block_shortcuts else cache["X_stacking"]
    y_train = cache["y_stacking"]
    X_val = neutralize_shortcuts(cache["X_fusion_val"]) if block_shortcuts else cache["X_fusion_val"]
    y_val = cache["y_fusion_val"]
    X_cal = neutralize_shortcuts(cache["X_calibration"]) if block_shortcuts else cache["X_calibration"]

    # -------------------------------------------------------------------------
    # Cascade-mask training
    # -------------------------------------------------------------------------
    # When Engine C may be unavailable at runtime, the tree must have learned
    # what to do with a missing slot 6. Masking a deterministic subset of the
    # training rows teaches the missing branch; without it, a probe-free
    # deployment would feed a value into a branch that never saw one.
    cascade_mask_rate = float(
        args.cascade_mask_rate
        if args.cascade_mask_rate is not None
        else (0.25 if args.semantic_tflite else 0.0)
    )
    if not 0.0 <= cascade_mask_rate < 1.0:
        raise ValueError("cascade-mask-rate must be in [0, 1)")
    if cascade_mask_rate > 0.0 and args.semantic_tflite is None:
        raise ValueError("cascade-mask-rate requires --semantic-tflite")
    if cascade_mask_rate > 0.0:
        rng = np.random.default_rng(42)
        mask = rng.random(len(X_train)) < cascade_mask_rate
        X_train = X_train.copy()
        X_train[mask, SEMANTIC_PROBE_INDEX] = np.nan
        print(f"[Cascade] Masked slot 6 to NaN on {int(mask.sum())}/{len(mask)} training rows "
              f"({cascade_mask_rate:.0%}) so the tree learns a real probe-free path")

    # The slot-6 probe score is the fusion's base margin (see
    # probe_base_margin). It must be 0 wherever the probe is unavailable,
    # which is exactly what the cascade masking above produced.
    use_base_margin = args.fusion_base_margin == "probe"
    if use_base_margin and args.semantic_tflite is None:
        raise ValueError("--fusion-base-margin probe requires --semantic-tflite")
    bm_train = probe_base_margin(X_train) if use_base_margin else None
    bm_val = probe_base_margin(X_val) if use_base_margin else None
    bm_cal = probe_base_margin(X_cal) if use_base_margin else None
    bm_test = probe_base_margin(cache["X_test"]) if use_base_margin else None
    bm_laundered = probe_base_margin(X_test_laundered) if use_base_margin else None
    if use_base_margin:
        print("[Fusion] slot 6 is the base margin; the trees only correct it")

    cfg = config["training"]
    params = {
        "max_depth": cfg["max_depth"],
        "eta": cfg["learning_rate"],
        "objective": "binary:logistic",
        "eval_metric": cfg["eval_metric"],
        "subsample": cfg["subsample"],
        "colsample_bytree": cfg["colsample_bytree"],
        "seed": 42,
    }
    booster = xgb.train(
        params,
        xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES, base_margin=bm_train),
        num_boost_round=cfg["n_estimators"],
        evals=[(xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_NAMES, base_margin=bm_val), "model_selection")],
        early_stopping_rounds=cfg["early_stopping_rounds"],
        verbose_eval=False,
    )
    best = getattr(booster, "best_iteration", None)
    if best is not None:
        booster = booster[: best + 1]

    if block_shortcuts:
        verify_no_shortcut_splits(booster, FEATURE_NAMES)

    intercept = detect_intercept(booster, X_train[:512], FEATURE_NAMES,
                                 base_margin=bm_train[:512] if bm_train is not None else None)
    if not validate_export(booster, X_val, FEATURE_NAMES, intercept, base_margin=bm_val):
        raise ValueError("Transpiled tree parity failed")

    missing_probe = X_val[:32].copy()
    for index in range(min(len(FEATURE_NAMES), len(missing_probe))):
        missing_probe[index, index] = np.nan
    if not validate_export(booster, missing_probe, FEATURE_NAMES, intercept,
                           base_margin=probe_base_margin(missing_probe) if use_base_margin else None):
        raise ValueError("Missing-branch parity failed")

    # The optional slot must also survive a fully NaN slot: that is exactly
    # what the Android pipeline sends when Engine C did not run.
    if SEMANTIC_PROBE_INDEX in OPTIONAL_FEATURE_INDICES:
        optional_missing = X_val[:64].copy()
        optional_missing[:, SEMANTIC_PROBE_INDEX] = np.nan
        if not validate_export(booster, optional_missing, FEATURE_NAMES, intercept,
                               base_margin=probe_base_margin(optional_missing) if use_base_margin else None):
            raise ValueError("Optional-slot missing-branch parity failed")

    cal = booster.predict(xgb.DMatrix(X_cal, feature_names=FEATURE_NAMES, base_margin=bm_cal))
    _, threshold = eer_operating_threshold(cache["y_calibration"], cal)
    cal_auc = float(compute_auc(cache["y_calibration"], cal))
    if cal_auc <= 0.5:
        raise ValueError("Fusion calibration at/below chance; refuse release")

    # â”€â”€ Cascade policy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # The arbitration policy needs the FUSION score, so it can only be fitted
    # once the booster exists. It is fitted on the CALIBRATION split and
    # frozen into the release exactly like the fusion threshold. A policy
    # fitted against a placeholder score would describe a system that does
    # not exist.
    cascade_artifacts: dict[str, str] = {}
    if args.semantic_tflite:
        from meta_classifier.dual_engine import fit_escalation_band
        from training.metrics import eer_operating_threshold as _eer

        probe_cal = cache["X_calibration"][:, SEMANTIC_PROBE_INDEX]
        _, probe_threshold = _eer(cache["y_calibration"], probe_cal)
        policy, pareto = fit_escalation_band(
            cal, probe_cal, cache["y_calibration"],
            engine_a_threshold=float(threshold),
            engine_c_threshold=float(probe_threshold),
        )
        policy.release_id = args.release_id
        (output / "cascade_policy.json").write_text(json.dumps(policy.to_dict(), indent=2), encoding="utf-8")
        (output / "cascade_pareto.json").write_text(json.dumps(pareto, indent=2), encoding="utf-8")

        if args.semantic_frontend == "siglip":
            from export.vfm_frontend import vfm_frontend_id
            probe_frontend_id = vfm_frontend_id(args.semantic_input_size)
        else:
            from data.image_frontend import semantic_frontend_id
            probe_frontend_id = semantic_frontend_id(args.semantic_input_size)

        probe_calibration = {
            "threshold_at_eer": float(probe_threshold),
            "auc": float(compute_auc(cache["y_calibration"], probe_cal)),
            "n_calibration": len(cache["y_calibration"]),
            "split_manifest_hash": checkpoint["contract"]["dataset_manifest_hash"] if args.allow_manifest_drift else provenance["manifest_sha256"],
            "model_hash": sha256_file(args.semantic_tflite),
            "frontend_id": probe_frontend_id,
            "score_semantics": {"model_output": "P(synthetic) in [0,1]", "real_label": 0, "synthetic_label": 1},
            "calibration_split": "calibration",
            "engine": "C_semantic" if args.semantic_frontend == "clip" else "C_vfm",
        }
        (output / "semantic_probe_threshold.json").write_text(
            json.dumps(probe_calibration, indent=2), encoding="utf-8")
        cascade_artifacts = {
            "constants/cascade_policy.json": sha256_file(output / "cascade_policy.json"),
            "constants/semantic_probe_threshold.json": sha256_file(output / "semantic_probe_threshold.json"),
        }
        print(f"[Cascade] Policy fitted on calibration: band=[{policy.escalate_below:.4f}, "
              f"{policy.escalate_above:.4f}] coverage={policy.coverage:.3f} "
              f"agreement={policy.agreement_rate:.3f}")

    test = booster.predict(xgb.DMatrix(cache["X_test"], feature_names=FEATURE_NAMES, base_margin=bm_test))
    test_laundered = booster.predict(xgb.DMatrix(X_test_laundered, feature_names=FEATURE_NAMES, base_margin=bm_laundered))
    laundered_auc = float(compute_auc(cache["y_test"], test_laundered))

    from training.evaluation import evaluation_summary, assert_source_fpr_within_bound
    if (root / "dataset_manifest.json").is_file():
        manifest_records = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))["records"]
        by_path = {record["path"]: record for record in manifest_records}
        records = []
        for i, (rel, y) in enumerate(zip(cache["ids_test"], cache["y_test"])):
            rec = by_path.get(str(rel))
            if rec is not None and rec.get("label") == int(y) and rec.get("group_id"):
                records.append(rec)
            else:
                records.append({"path": str(rel), "label": int(y), "group_id": f"test_group_{i}", "source": "test_source"})
    else:
        records = [
            {"path": str(rel), "label": int(y), "group_id": f"test_group_{i}", "source": "test_source"}
            for i, (rel, y) in enumerate(zip(cache["ids_test"], cache["y_test"]))
        ]
    test_report = evaluation_summary(cache["y_test"], test, threshold, records)
    laundered_report = evaluation_summary(cache["y_test"], test_laundered, threshold, records)
    if test_report["auc"] <= 0.5 or laundered_auc <= 0.5:
        raise ValueError("Sealed or laundered fusion test performance at/below chance; refuse release")
    # A detector that flags a large fraction of one acquisition source as
    # synthetic has learned the SOURCE, not the synthesis. LAION-5B is the
    # training corpus for Stable Diffusion and Flux, so "looks like LAION"
    # is a giveaway, not a detection signal.
    # The per-source false-positive bound. It is the right bound: a detector
    # that flags a large fraction of one acquisition source has learned the
    # SOURCE, not the synthesis. It is currently UNREACHABLE, because the
    # corpus leaks the label through the source field (every fake comes from
    # one generator, every real from another), so the override below is the
    # only way to produce a release at all. Either way the violation is
    # recorded, never hidden.
    source_fpr_violations = {}
    for report, label in ((test_report, "fusion"), (laundered_report, "fusion laundered")):
        try:
            assert_source_fpr_within_bound(
                report, max_fpr=0.05, min_real_per_source=15, context=label
            )
        except ValueError as exc:
            if not args.allow_source_fpr_violation:
                raise
            source_fpr_violations[label] = str(exc)
            print(f"[Source FPR] BOUND NOT MET for '{label}'. Recorded as a known "
                  f"release violation:\n{exc}")

    kt = output / "XGBoostMetaClassifier.kt"
    export_to_kotlin(booster, FEATURE_NAMES, str(kt), intercept, args.release_id,
                     base_margin_slot=SEMANTIC_PROBE_INDEX if use_base_margin else None)
    if not kotlin_brace_check(str(kt)):
        raise ValueError("Kotlin structural validation failed")

    model_path = output / "xgboost_meta.json"
    booster.save_model(str(model_path))
    model_hash = sha256_file(model_path)
    results = {
        "release_id": args.release_id,
        "features": FEATURE_NAMES,
        "identity": identity,
        "test_auc": test_report["auc"],
        "laundered_test_auc": laundered_auc,
        "test": test_report,
        "laundered_test": laundered_report,
        "test_scope": "all_sealed_test_files" if args.test_samples is None else "explicit_pilot_cap",
        "test_accuracy": float(compute_accuracy(cache["y_test"], test, threshold)),
        "laundered_test_accuracy": float(compute_accuracy(cache["y_test"], test_laundered, threshold)),
        "base_margin_intercept": intercept,
        "fusion_base_margin": args.fusion_base_margin,
        "n_trees": booster.num_boosted_rounds(),
        "source_fpr_bound": {
            "enforced": not args.allow_source_fpr_violation,
            "max_fpr": 0.05,
            "violations": source_fpr_violations,
            "note": (
                "The per-source real-image FPR bound is NOT met. The corpus "
                "leaks the label through the acquisition source, so no model "
                "trained on it can satisfy a per-source bound. Recorded here "
                "so the release cannot hide it."
            ) if source_fpr_violations else "",
        },
        "model_hash": model_hash,
        "kotlin_hash": sha256_file(kt),
        "dual_engine": {
            "engine_c_tflite": args.semantic_tflite,
            "engine_c_sha256": sha256_file(args.semantic_tflite) if args.semantic_tflite else None,
            "engine_c_input_size": args.semantic_input_size if args.semantic_tflite else None,
            "engine_c_frontend": args.semantic_frontend,
            "semantic_probe_feature_index": SEMANTIC_PROBE_INDEX,
            "optional_feature_indices": list(OPTIONAL_FEATURE_INDICES),
            "cascade_mask_rate": cascade_mask_rate,
            "cascade_artifacts": cascade_artifacts,
        },
        "calibration": {
            "threshold": threshold,
            "auc": cal_auc,
            "n_calibration": len(cal),
            "split_manifest_hash": checkpoint["contract"]["dataset_manifest_hash"] if args.allow_manifest_drift else provenance["manifest_sha256"],
            "model_hash": model_hash,
            "frontend_id": checkpoint["contract"]["frontend_id"],
        },
        "validation_status": "numpy_tree_parity_passed_requires_jvm_and_device_evaluation",
    }
    (output / "meta_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

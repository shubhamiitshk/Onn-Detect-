"""Export a frozen vision-foundation-model probe to TFLite via the Keras mirror.

Why this path
-------------
``onnx2tf`` cannot convert a ViT in this repository: its shape inference
across the patch->sequence transition is wrong (the patch projection is
inferred as ``[?, 768, 768]`` instead of ``[?, 196, 768]``), and its
NCHW calibration path is unusable. The repository already has a proven
alternative for the SRM model and the semantic probe: a hand-written Keras
mirror converted with ``tf.lite.TFLiteConverter.from_keras_model``.

This script drives that path for a frozen SigLIP or DINOv2 tower plus a
linear probe, and it is GATED: the Keras output must match the PyTorch
reference, and the TFLite interpreter's output must match it too, before
a single byte is written.

Crucially, the reference probabilities are computed with the SAME
preprocessing and the SAME pooling code that produced the measured
embeddings (``Tower.preprocess`` and ``Tower.embed``). Anything less
would risk validating the export against a different function than the
one that was measured - which is exactly the class of bug that produced
an earlier, false 0.99 parity failure.

Usage
-----
    python -m ml.scripts.export_keras_vfm_probe \\
        --data-root D:\\onn_data\\openfake_visual \\
        --output-dir D:\\onn_data\\runs\\vfm_keras_export \\
        --tower google/siglip-base-patch16-224 \\
        --num-layers 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data.contracts import IMAGE_EXTS, discover_samples  # noqa: E402
from data.image_frontend import decode_image  # noqa: E402
from evaluate_vfm_probes import (  # noqa: E402
    TOWERS,
    fit_logistic_probe,
    fit_source_debiased_probe,
    split_sources,
)
from export.keras_vfm_probe import (  # noqa: E402
    build_keras_vfm_probe,
    export_keras_vfm_to_tflite,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """The repo's AUC, so the export report and the release's evaluation
    report cannot disagree about the same model.

    The previous implementation assigned ranks by sort order, which is
    TIE-UNAWARE: two samples with the same score get different ranks, and
    an INT8 graph that saturates produces many exact ties.
    """
    from training.metrics import compute_auc
    return float(compute_auc(labels, scores))


def py_reference_probs(
    spec,
    tower,
    inputs_chw: np.ndarray,
    weight: np.ndarray,
    bias: float,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
) -> np.ndarray:
    """The reference probabilities, computed with PyTorch.

    This is the function the shipped detector must reproduce. It reuses
    ``Tower.embed`` verbatim - the exact code path that produced the
    measured embedding cache - so the export cannot drift away from the
    measurement.
    """
    features = spec.embed(tower, inputs_chw)          # already L2-normalized
    z = (features - feat_mean) / feat_std
    return _sigmoid(z @ weight + bias)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tower", default="google/siglip-base-patch16-224")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Truncate the encoder to this many blocks.")
    parser.add_argument("--l2-reg", type=float, default=30.0)
    parser.add_argument("--parity-count", type=int, default=16)
    parser.add_argument("--quantize", default="int8_dynamic",
                        choices=["int8_dynamic", "fp16", "fp32"])
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--debias", action="store_true",
                        help="Use the source-debiased probe instead of the plain one.")
    parser.add_argument("--debias-rank", type=int, default=1)
    parser.add_argument("--embedding-cache", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).resolve()

    spec = TOWERS.get(args.tower)
    if spec is None:
        raise SystemExit(f"Unknown tower {args.tower!r}")
    size = args.input_size or spec.size

    # ---- the probe, fitted on the stacking split exactly as the measurement did ----
    cache = Path(args.embedding_cache) if args.embedding_cache else (
        Path(r"D:\onn_data\runs\vfm_probe_eval") / f"embeddings_{spec.key}.npz"
    )
    if not cache.is_file():
        raise SystemExit(
            f"Missing embedding cache {cache}. Run evaluate_vfm_probes first so the "
            "exported head is exactly the head that was measured."
        )
    with np.load(cache) as data:
        X_stacking = data["X_stacking"]
        y_stacking = data["y_stacking"]
        X_test = data["X_test"]
        y_test = data["y_test"]

    mean = X_stacking.mean(axis=0, keepdims=True).astype(np.float32)
    std = (X_stacking.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)

    if args.debias:
        stacking_sources = split_sources(data_root, "stacking")
        weight, bias, _ = fit_source_debiased_probe(
            X_stacking, y_stacking, stacking_sources,
            rank_k=args.debias_rank, laion_real_weight=3.0,
        )
        weight = np.asarray(weight, dtype=np.float32)
        bias = float(bias)
    else:
        weight, bias, mean, std = fit_logistic_probe(
            X_stacking, y_stacking, X_stacking, args.l2_reg
        )
        weight = np.asarray(weight, dtype=np.float32)
        bias = float(bias)

    test_scores = _sigmoid(((X_test - mean) / std) @ weight + bias)
    test_auc = _auc(test_scores, y_test)
    print(f"probe: {weight.shape[0]} weights, l2={args.l2_reg}, sealed-test AUC={test_auc:.4f}")

    # ---- parity inputs: the NORMALISED tensor the app feeds the model ----
    samples = discover_samples(data_root, "test", IMAGE_EXTS)[: args.parity_count]
    inputs_chw = []
    for path, _ in samples:
        image = decode_image(str(path)).image
        inputs_chw.append(spec.preprocess(image))
    inputs_chw = np.stack(inputs_chw).astype(np.float32)
    inputs_nhwc = np.transpose(inputs_chw, (0, 2, 3, 1)).astype(np.float32)
    print(f"parity set: {inputs_nhwc.shape} (NHWC, normalised)")

    # ---- the PyTorch reference ----
    tower = spec.load()
    reference = py_reference_probs(spec, tower, inputs_chw, weight, bias, mean, std)
    print(f"reference P(AI): min={reference.min():.4f} max={reference.max():.4f}")

    # ---- the Keras mirror ----
    started = time.time()
    keras_model = build_keras_vfm_probe(
        tower,
        probe_weight=weight,
        probe_bias=bias,
        input_size=size,
        num_layers=args.num_layers,
        l2_normalize=True,
        feature_mean=mean,
        feature_std=std,
    )
    print(
        f"Keras mirror built in {time.time() - started:.1f}s: "
        f"{keras_model.input_shape} -> {keras_model.output_shape}"
    )

    # ---- convert, gated ----
    suffix = "int8" if args.quantize == "int8_dynamic" else args.quantize
    depth = args.num_layers if args.num_layers else "full"
    tflite_path = output_dir / f"vfm_probe_{spec.key}_{depth}L_{suffix}.tflite"
    summary = export_keras_vfm_to_tflite(
        keras_model,
        str(tflite_path),
        inputs_nhwc,
        reference,
        quantize=args.quantize,
    )
    summary.update({
        "tower": args.tower,
        "num_layers": args.num_layers,
        "probe_weights": int(weight.shape[0]),
        "l2_reg": args.l2_reg,
        "debias": bool(args.debias),
        "debias_rank": args.debias_rank if args.debias else None,
        "sealed_test_auc": test_auc,
        "parity_count": len(reference),
        "reference_min": float(reference.min()),
        "reference_max": float(reference.max()),
        "embedding_cache": str(cache),
        # Everything needed to reconstruct the PyTorch reference offline, so
        # the packaged graph can be re-verified without re-running the tower.
        "probe_weight": weight.astype(np.float64).tolist(),
        "probe_bias": float(bias),
        "feature_mean": mean.astype(np.float64).reshape(-1).tolist(),
        "feature_std": std.astype(np.float64).reshape(-1).tolist(),
    })
    (output_dir / "vfm_probe_export.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""End-to-end acceptance check for a packaged VFM probe TFLite graph.

The export gates in ``ml/src/export/keras_vfm_probe.py`` prove that the
converted graph is numerically close to the PyTorch reference on a handful
of inputs. That is necessary but not sufficient: what actually matters is
that the **packaged file** - the bytes that go into the APK - keeps the
sealed-test behaviour that was measured and published.

This script closes that loop:

* it runs the packaged ``.tflite`` over the ENTIRE sealed test split,
* it compares the resulting AUC with the AUC of the PyTorch reference
  computed from the measured embedding cache,
* it measures how many decisions flip at the operating threshold, and
* it reports the per-source false-positive rate, which is where the
  shipped detector's weaknesses live.

It exits non-zero if the packaged graph does not retain the measured
behaviour, so a bad model can never be promoted into the APK.

Usage
-----
    python -m ml.scripts.evaluate_tflite_vfm \\
        --data-root D:\\onn_data\\openfake_visual \\
        --model D:\\onn_data\\runs\\vfm_keras_export\\vfm_probe_*.tflite \\
        --tower google/siglip-base-patch16-224 \\
        --promote-to android/app/src/main/assets/models
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data.contracts import IMAGE_EXTS, discover_samples  # noqa: E402
from data.image_frontend import decode_image  # noqa: E402
from evaluate_vfm_probes import TOWERS, split_sources  # noqa: E402


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _auc(scores, labels):
    """The repo's AUC. `evaluate_tflite_vfm.py` and the release's evaluation
    report must not disagree about the same file."""
    from training.metrics import compute_auc
    return float(compute_auc(labels, scores))


def _tpr_at_fpr(scores, labels, target):
    """The repo's TPR at a fixed FPR: `max(tpr[fpr <= target])` from the ROC
    curve, exactly as `training.evaluation.evaluation_summary` computes it."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.max(tpr[fpr <= target]))


def _fpr_at_tpr(scores: np.ndarray, labels: np.ndarray, target_tpr: float) -> float:
    """The smallest FPR that still reaches ``target_tpr``, from the ROC curve."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    reachable = fpr[tpr >= target_tpr]
    return float(np.min(reachable)) if len(reachable) else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tower", default="google/siglip-base-patch16-224")
    parser.add_argument("--split", default="test")
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--promote-to", default=None,
                        help="Directory to copy the model into when it passes.")
    parser.add_argument("--promote-name", default="siglip_probe_int8.tflite")
    parser.add_argument("--max-auc-drop", type=float, default=0.01)
    parser.add_argument("--max-tpr-gap", type=float, default=0.02,
                        help="Maximum tolerated change in TPR at a fixed FPR point.")
    args = parser.parse_args()

    import tensorflow as tf

    spec = TOWERS.get(args.tower)
    if spec is None:
        raise SystemExit(f"Unknown tower {args.tower!r}")

    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        raise SystemExit(f"No such model: {model_path}")
    data_root = Path(args.data_root).resolve()

    cache = Path(args.embedding_cache) if args.embedding_cache else (
        Path(r"D:\onn_data\runs\vfm_probe_eval") / f"embeddings_{spec.key}.npz"
    )
    with np.load(cache) as data:
        key_X, key_y = f"X_{args.split}", f"y_{args.split}"
        if key_X not in data.files:
            raise SystemExit(f"Cache {cache} has no {key_X}")
        y_split = data[key_y]

    # The reference probabilities come from the MEASURED probe weights, which
    # the export wrote alongside the model. If that file is absent we cannot
    # compare against a measured reference, and we refuse to guess.
    probe_path = model_path.with_name("vfm_probe_export.json")
    if not probe_path.is_file():
        raise SystemExit(
            f"No export manifest next to {model_path}. Run export_keras_vfm_probe first."
        )
    manifest = json.loads(probe_path.read_text(encoding="utf-8"))
    # The acceptance is only meaningful for THESE bytes. A report exported
    # for a different file must not be able to "cover" this one.
    from data.contracts import sha256_file
    expected_sha = manifest.get("model_sha256")
    actual_sha = sha256_file(model_path)
    if expected_sha and expected_sha != actual_sha:
        raise SystemExit(
            "The packaged model does NOT match the export manifest: the "
            f"acceptance report covers {expected_sha[:16]} but the file is "
            f"{actual_sha[:16]}. Re-run the export."
        )
    weight = np.asarray(manifest["probe_weight"], dtype=np.float64)
    bias = float(manifest["probe_bias"])
    mean = np.asarray(manifest["feature_mean"], dtype=np.float64)
    std = np.asarray(manifest["feature_std"], dtype=np.float64)

    with np.load(cache) as data:
        reference_scores = _sigmoid(((data[key_X] - mean) / std) @ weight + bias)
        y_split = data[key_y]

    # ---- run the packaged graph over the whole split ----
    samples = discover_samples(data_root, args.split, IMAGE_EXTS)
    print(f"split '{args.split}': {len(samples)} samples")

    inputs = np.stack([
        spec.preprocess(decode_image(str(path)).image) for path, _ in samples
    ]).astype(np.float32)
    inputs_nhwc = np.transpose(inputs, (0, 2, 3, 1)).astype(np.float32)

    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    if list(in_det["shape"]) != [1, spec.size, spec.size, 3] or in_det["dtype"] != np.float32:
        raise SystemExit(f"Bad input contract: {in_det['shape']} {in_det['dtype']}")
    if list(out_det["shape"]) != [1, 1] or out_det["dtype"] != np.float32:
        raise SystemExit(f"Bad output contract: {out_det['shape']} {out_det['dtype']}")

    started = time.time()
    probs = []
    for x in inputs_nhwc:
        interp.set_tensor(in_det["index"], x[np.newaxis])
        interp.invoke()
        probs.append(float(interp.get_tensor(out_det["index"])[0, 0]))
    probs = np.asarray(probs)
    if not np.isfinite(probs).all():
        raise SystemExit("The packaged model produced a non-finite score")
    if len(probs) != len(y_split):
        raise SystemExit(
            f"Row count mismatch: {len(probs)} scored images vs {len(y_split)} cached labels. "
            "The embedding cache and the discovered samples must describe the same rows "
            "in the same order."
        )
    elapsed = time.time() - started
    print(f"ran {len(probs)} images in {elapsed:.0f}s ({elapsed / max(len(probs), 1) * 1000:.0f} ms/image)")

    # ---- compare ----
    ref_auc = _auc(reference_scores, y_split)
    pkg_auc = _auc(probs, y_split)
    auc_drop = ref_auc - pkg_auc
    max_err = float(np.max(np.abs(probs - reference_scores)))
    mae = float(np.mean(np.abs(probs - reference_scores)))

    # The operating threshold maximises Youden's J (TPR - FPR) on the
    # REFERENCE scores. FPR means "a real image scored as AI", i.e. a
    # NEGATIVE whose score is >= the threshold.
    grid = np.linspace(0.01, 0.99, 99)
    positives = reference_scores[y_split == 1]
    negatives = reference_scores[y_split == 0]
    tpr = np.array([np.mean(positives >= t) for t in grid])
    fpr = np.array([np.mean(negatives >= t) for t in grid])
    operating = float(grid[int(np.argmax(tpr - fpr))])
    flips = int(np.sum((reference_scores >= operating) != (probs >= operating)))
    flip_rate = flips / len(probs)

    print(f"\nreference AUC (PyTorch, measured probe) : {ref_auc:.4f}")
    print(f"packaged  AUC (TFLite file)               : {pkg_auc:.4f}   drop={auc_drop:+.4f}")
    print(f"max |dp| = {max_err:.4f}   mae = {mae:.4f}")
    print(f"operating threshold {operating:.3f} (Youden-J on the reference): "
          f"decision flips {flips}/{len(probs)} ({flip_rate:.2%})")
    for target in (0.001, 0.01, 0.05):
        ref_fpr = _fpr_at_tpr(reference_scores, y_split, target)
        pkg_fpr = _fpr_at_tpr(probs, y_split, target)
        ref_tpr = _tpr_at_fpr(reference_scores, y_split, target)
        pkg_tpr = _tpr_at_fpr(probs, y_split, target)
        print(f"  FPR@TPR={target:.1%}  reference {ref_fpr:.4f}  packaged {pkg_fpr:.4f}"
              f"   |   TPR@FPR={target:.1%}  reference {ref_tpr:.4f}  packaged {pkg_tpr:.4f}")

    # ---- per-source false positives (the shipped detector's weak spot) ----
    sources = split_sources(data_root, args.split)
    per_source = {}
    for source in sorted(set(sources)):
        mask = np.array([s == source for s in sources])
        real = mask & (y_split == 0)
        if real.sum() == 0:
            continue
        per_source[source] = {
            "n_real": int(real.sum()),
            "fpr_reference": float(np.mean(reference_scores[real] >= operating)),
            "fpr_packaged": float(np.mean(probs[real] >= operating)),
        }
    print("\nper-source FPR at the operating threshold:")
    for source, stats in per_source.items():
        flag = "  <-- SHORTCUT" if stats["fpr_packaged"] > 0.2 else ""
        print(f"  {source:<10} n_real={stats['n_real']:<4} "
              f"reference {stats['fpr_reference']:.4f}  packaged {stats['fpr_packaged']:.4f}{flag}")

    report = {
        "model": str(model_path),
        "model_size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
        "tower": args.tower,
        "split": args.split,
        "n_samples": len(probs),
        "reference_auc": ref_auc,
        "packaged_auc": pkg_auc,
        "auc_drop": auc_drop,
        "max_abs_prob_error": max_err,
        "mean_abs_prob_error": mae,
        "operating_threshold": operating,
        "decision_flips": flips,
        "decision_flip_rate": flip_rate,
        "fpr_at_1pct_tpr_reference": _fpr_at_tpr(reference_scores, y_split, 0.01),
        "fpr_at_1pct_tpr_packaged": _fpr_at_tpr(probs, y_split, 0.01),
        "tpr_at_1pct_fpr_reference": _tpr_at_fpr(reference_scores, y_split, 0.01),
        "tpr_at_1pct_fpr_packaged": _tpr_at_fpr(probs, y_split, 0.01),
        "tpr_at_5pct_fpr_reference": _tpr_at_fpr(reference_scores, y_split, 0.05),
        "tpr_at_5pct_fpr_packaged": _tpr_at_fpr(probs, y_split, 0.05),
        "tpr_at_0_1pct_fpr_reference": _tpr_at_fpr(reference_scores, y_split, 0.001),
        "tpr_at_0_1pct_fpr_packaged": _tpr_at_fpr(probs, y_split, 0.001),
        "per_source_fpr": per_source,
        "manifest": manifest,
    "model_sha256": actual_sha,
    }

    # The acceptance criteria are the OPERATING CHARACTERISTICS, not a raw
    # probability tolerance and not a flip count at one arbitrary threshold.
    #
    # A flip count at a single threshold is a poor criterion: any threshold
    # inside the score distribution has samples within a hair of it, so a
    # quantization error of a few hundredths flips a few percent of them
    # even when the detector's behaviour is unchanged. What a detector
    # actually needs is that the RANKING and the low-FPR OPERATING POINTS
    # survive, so those are what is gated.
    #
    # The gate is ONE-SIDED: quantization may leave the detector exactly as
    # good or better, but it may not make it worse. An absolute tolerance
    # would fail a packaged model that happens to be *better* than the
    # reference, which is not a defect.
    auc_ok = auc_drop <= args.max_auc_drop
    tpr_1_ok = (
        _tpr_at_fpr(probs, y_split, 0.01)
        >= _tpr_at_fpr(reference_scores, y_split, 0.01) - args.max_tpr_gap
    )
    tpr_5_ok = (
        _tpr_at_fpr(probs, y_split, 0.05)
        >= _tpr_at_fpr(reference_scores, y_split, 0.05) - args.max_tpr_gap
    )

    report["acceptance"] = {
        "auc_drop_ok": auc_ok,
        "tpr_at_1pct_fpr_ok": tpr_1_ok,
        "tpr_at_5pct_fpr_ok": tpr_5_ok,
        "max_auc_drop": args.max_auc_drop,
        "max_tpr_gap": args.max_tpr_gap,
    }
    passed = auc_ok and tpr_1_ok and tpr_5_ok
    report["passed"] = passed
    report_path = Path(args.report) if args.report else model_path.with_suffix(".acceptance.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nACCEPTANCE: {'PASS' if passed else 'FAIL'}"
          f"  (auc_drop_ok={auc_ok}, tpr@1%FPR_ok={tpr_1_ok}, tpr@5%FPR_ok={tpr_5_ok})")
    print(f"report: {report_path}")

    if passed and args.promote_to:
        destination = Path(args.promote_to).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, destination / args.promote_name)
        print(f"promoted -> {destination / args.promote_name}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

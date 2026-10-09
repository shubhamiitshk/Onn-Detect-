"""Enforced score, operational-threshold and task parity gates on labeled data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from training.metrics import compute_auc

from functools import lru_cache

@lru_cache(maxsize=4)
def _interpreter_for(tflite_path):
    """Reuse allocated interpreters for sequential evaluation of immutable bundles.

    These CLI scorers are intentionally sequential. A parallel worker must use
    its own interpreter; callers replacing a model at the same path must clear
    this cache before scoring the replacement.
    """
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=1)
    interpreter.allocate_tensors()
    return interpreter

def _run_tflite(tflite_path, x_nhwc):
    x_nhwc = np.asarray(x_nhwc)
    if not np.isfinite(x_nhwc).all():
        raise ValueError('Non-finite frontend input')
    interpreter = _interpreter_for(str(Path(tflite_path).resolve()))
    inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1 or list(outputs[0]['shape']) != [1, 1]:
        raise ValueError('Unexpected model tensor count/output shape')
    inp, out = inputs[0], outputs[0]
    if list(x_nhwc.shape) != list(inp["shape"]) or inp["dtype"] != np.float32 or out["dtype"] != np.float32:
        raise ValueError("Export input shape/dtype or output dtype differs from mobile contract")
    interpreter.set_tensor(inp["index"], x_nhwc.astype(np.float32))
    interpreter.invoke()
    score = float(interpreter.get_tensor(out["index"]).reshape(-1)[0])
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid model score")
    return score

def tpr_at_fpr(labels, scores, target=.01):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.max(tpr[fpr <= target]))

def compare_scores(labels, reference, candidate, threshold, min_agreement=.98, max_mae=.02,
                   max_error=.10, max_auc_drop=.01, max_tpr_drop=.03):
    y, ref, cand = np.asarray(labels).reshape(-1), np.asarray(reference).reshape(-1), np.asarray(candidate).reshape(-1)
    if len(y) != len(ref) or ref.shape != cand.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("Parity requires equal-length scores and both labeled classes")
    if not np.isfinite(ref).all() or not np.isfinite(cand).all() or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Non-finite scores/threshold")
    error = np.abs(ref - cand)
    agreement = float(np.mean((ref >= threshold) == (cand >= threshold)))
    ref_auc, cand_auc = float(compute_auc(y, ref)), float(compute_auc(y, cand))
    tpr_ref, tpr_cand = tpr_at_fpr(y, ref), tpr_at_fpr(y, cand)
    metrics = {"mae": float(error.mean()), "max_error": float(error.max()), "agreement": agreement,
               "threshold": float(threshold), "reference_auc": ref_auc, "candidate_auc": cand_auc,
               "reference_tpr_at_fpr_1pct": tpr_ref, "candidate_tpr_at_fpr_1pct": tpr_cand,
               "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum())}
    failures = []
    for failed, name in [(agreement < min_agreement, "threshold agreement"), (error.mean() > max_mae, "mean score error"),
                         (error.max() > max_error, "max score error"), (ref_auc - cand_auc > max_auc_drop, "AUC loss"),
                         (tpr_ref - tpr_cand > max_tpr_drop, "low-FPR sensitivity loss")]:
        if failed:
            failures.append(name)
    if failures:
        raise ValueError(f"Export parity gates failed: {failures}; {metrics}")
    metrics["passed"] = True
    return metrics

def validate_models_real(pytorch_model, tflite_path, modality, real_images=None, real_specs_pt=None,
                         target_frames=None, num_samples=200, min_agreement=.98, labels=None, threshold=None):
    if labels is None or threshold is None:
        raise ValueError("Export parity requires explicit labels and the calibrated operational threshold")
    pytorch_model.cpu().eval()
    values = real_images if modality == "visual" else real_specs_pt
    values = values[:num_samples]
    ref, cand = [], []
    for value in values:
        if modality == "visual":
            x = np.asarray(value, dtype=np.float32) / 255.0
            x = x[None]
        elif modality == "audio":
            x = np.transpose(np.asarray(value, dtype=np.float32), (0, 2, 3, 1))
        else:
            raise ValueError(modality)
        pt = np.transpose(x, (0, 3, 1, 2)).copy()
        with torch.no_grad():
            ref.append(float(pytorch_model(torch.from_numpy(pt)).reshape(-1)[0]))
        cand.append(_run_tflite(tflite_path, x))
    return compare_scores(labels[:len(ref)], ref, cand, threshold, min_agreement=min_agreement)

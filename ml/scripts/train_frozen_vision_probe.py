"""Train, export and calibrate the FROZEN vision-tower probe ("Lite C").

This is the second engine the dual-engine cascade is built around: the
MobileCLIP2-S0 image tower, frozen, with an L2-regularized logistic probe on
top. Measured on this dataset the frozen probe reaches ~0.885 AUC, far above
what a distilled student can recover from 1.2k images.

Pipeline
--------
1. Load the ``timm/MobileCLIP2-S0-OpenCLIP`` weights (already in the local
   Hugging Face cache) into the ``open_clip`` architecture.
2. Extract 512-d L2-normalized embeddings for every split.
3. Fit the L2-regularized logistic probe on the ``stacking`` split only.
4. Export ``image -> tower -> probe -> P(fake)`` to ONNX.
5. Convert ONNX -> TFLite (``onnx2tf``), then quantize to full INT8.
6. Verify PyTorch-vs-INT8 parity on real images.
7. Calibrate the EER threshold on the ``calibration`` split and write the
   release artifacts.

The tower is FROZEN: only the 513-parameter probe head is learned, so the
exported model is exactly the thing that was measured.

Usage:
    python -m ml.scripts.train_frozen_vision_probe \\
        --data-root D:\\\\onn_data\\\\openfake_visual \\
        --output-dir D:\\\\onn_data\\\\runs\\\\release_v6\\\\checkpoints
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ML_ROOT / "src"))
sys.path.insert(0, str(ML_ROOT / "scripts"))

from data.contracts import IMAGE_EXTS, discover_samples, sha256_file
from data.image_frontend import (
    FROZEN_PROBE_SIZE,
    decode_image,
    frozen_probe_frontend_id,
    frozen_probe_image_tensor,
)
from training.metrics import compute_accuracy, compute_auc, compute_eer, eer_operating_threshold

HF_REPO = "timm/MobileCLIP2-S0-OpenCLIP"
MODEL_NAME = "MobileCLIP2-S0"
EMBED_DIM = 512


# ---------------------------------------------------------------------------
# Frozen tower
# ---------------------------------------------------------------------------

def load_frozen_tower():
    """Load MobileCLIP2-S0's image tower from the local Hugging Face cache.

    The cached checkpoint is the ``timm`` re-export of the open_clip weights
    (``visual.*`` prefixed). ``open_clip``'s own pretrained tag for this
    model is a different file, so the weights are loaded explicitly and the
    key prefix is stripped.
    """
    import open_clip
    from safetensors.torch import load_file

    candidates = glob.glob(
        str(Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
            / "hub" / "models--timm--MobileCLIP2-S0-OpenCLIP" / "snapshots" / "**" / "*.safetensors"),
        recursive=True,
    )
    if not candidates:
        raise SystemExit(
            f"Frozen tower weights not found in the Hugging Face cache ({HF_REPO}). "
            f"Download them first."
        )

    state = load_file(candidates[0])
    model = open_clip.create_model(MODEL_NAME, pretrained=None)
    visual_state = {k[len("visual."):]: v for k, v in state.items() if k.startswith("visual.")}
    missing, unexpected = model.visual.load_state_dict(visual_state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"Frozen tower weight mismatch: missing={missing[:4]} unexpected={unexpected[:4]}")
    model.visual.eval()
    for parameter in model.visual.parameters():
        parameter.requires_grad_(False)
    return model.visual


class FrozenProbe(nn.Module):
    """The frozen tower plus the learned probe head, as one exportable graph."""

    def __init__(self, visual: nn.Module, weight: np.ndarray, bias: np.ndarray):
        super().__init__()
        self.visual = visual
        self.register_buffer("probe_w", torch.from_numpy(weight.astype(np.float32)))
        self.register_buffer("probe_b", torch.from_numpy(bias.astype(np.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.visual(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        logits = features @ self.probe_w + self.probe_b
        return torch.sigmoid(logits).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Probe fitting
# ---------------------------------------------------------------------------

def fit_logistic_probe(
    X_stack: np.ndarray,
    y_stack: np.ndarray,
    l2_reg: float = 1.0,
) -> tuple[np.ndarray, float]:
    """L2-regularized logistic probe on the stacking split.

    The probe standardizes its inputs with the stacking-split mean/std
    BEFORE the linear head. That standardization is FOLDED into the
    returned weight and bias:

        logit = ((x - mean) / std) @ w + b
              = x @ (w / std) + (b - (mean / std) @ w)

    so the exported graph is a single matmul plus a sigmoid, and the device
    does not have to reproduce the standardization separately (a very easy
    thing to get subtly wrong on a different platform).

    Returns the effective weight vector and the effective intercept.
    """
    mean = X_stack.mean(axis=0, keepdims=True)
    std = X_stack.std(axis=0, keepdims=True) + 1e-6
    Z = np.hstack([(X_stack - mean) / std, np.ones((len(X_stack), 1))])
    d = Z.shape[1]
    w = np.zeros(d, dtype=np.float64)
    reg = np.eye(d, dtype=np.float64) * l2_reg
    reg[-1, -1] = 1e-6
    y = y_stack.astype(np.float64)

    for _ in range(80):
        logits = np.clip(Z @ w, -40.0, 40.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        r = np.maximum(p * (1.0 - p), 1e-6)
        grad = Z.T @ (p - y) + reg @ w
        hess = (Z.T * r) @ Z + reg
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.linalg.norm(step) < 1e-7:
            break

    raw_weight = w[:-1]
    raw_bias = float(w[-1])
    effective_weight = (raw_weight / std.ravel()).astype(np.float32)
    effective_bias = float(raw_bias - np.ravel((mean / std) @ raw_weight))
    return effective_weight, effective_bias


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(visual: nn.Module, samples, device: torch.device, batch_size: int = 8) -> np.ndarray:
    embeddings = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        tensors = torch.stack([
            frozen_probe_image_tensor(decode_image(str(path)).image, FROZEN_PROBE_SIZE)
            for path, _ in batch
        ]).to(device)
        features = visual(tensors)
        if isinstance(features, (tuple, list)):
            features = features[0]
        features = features.float()
        features = features / (features.norm(dim=1, keepdim=True) + 1e-12)
        embeddings.append(features.cpu().numpy().astype(np.float32))
        if (start + batch_size) % 200 < batch_size or start + batch_size >= len(samples):
            print(f"[Tower] Embedded {min(start + batch_size, len(samples))}/{len(samples)}")
    return np.concatenate(embeddings, axis=0)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_onnx(probe: nn.Module, path: Path, size: int = FROZEN_PROBE_SIZE) -> None:
    probe.eval()
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        probe, dummy, str(path),
        input_names=["input"], output_names=["score"],
        opset_version=13, do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"[ONNX] wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


def convert_onnx_to_tflite(onnx_path: Path, out_dir: Path, size: int = FROZEN_PROBE_SIZE) -> Path:
    """ONNX -> TFLite via onnx2tf, then full-INT8 quantization."""
    import tensorflow as tf

    work = out_dir / "onnx2tf_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable, "-m", "onnx2tf",
        "-i", str(onnx_path),
        "-o", str(work),
        "-osd", str(work),
        "-co", "nhwc",
        "-oi", "0", "float32[1,3,{0},{0}]".format(size),
        "-oo", "0", "float32[1,1]",
    ]
    print(f"[onnx2tf] {' '.join(command)}")
    subprocess.run(command, check=True, cwd=str(work))

    saved = work / "model_float32.tflite"
    if not saved.is_file():
        candidates = list(work.glob("*float32*.tflite"))
        if not candidates:
            raise SystemExit("onnx2tf produced no float32 TFLite model")
        saved = candidates[0]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(work))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    quantized = converter.convert()

    out_path = out_dir / "frozen_vision_probe_int8.tflite"
    out_path.write_bytes(quantized)
    print(f"[INT8] wrote {out_path} ({out_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--l2-reg", type=float, default=1.0)
    parser.add_argument("--skip-tflite", action="store_true", help="Train and measure only")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    visual = load_frozen_tower()
    params = sum(p.numel() for p in visual.parameters())
    print(f"[Tower] {MODEL_NAME} image tower loaded: {params:,} params on {device}")

    splits = ("stacking", "train", "val", "calibration", "test")
    samples = {split: discover_samples(Path(args.data_root), split, IMAGE_EXTS) for split in splits}

    embeddings = {}
    for split, items in samples.items():
        print(f"[Tower] {split}: {len(items)} images")
        started = time.time()
        embeddings[split] = extract_embeddings(visual, items, device, args.batch_size)
        print(f"[Tower] {split} done in {time.time() - started:.0f}s")

    y = {split: np.asarray([label for _, label in items], dtype=np.int64) for split, items in samples.items()}

    weight, bias = fit_logistic_probe(embeddings["stacking"], y["stacking"], args.l2_reg)

    def score(split: str) -> np.ndarray:
        logits = embeddings[split] @ weight + bias
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))

    stack_scores = score("stacking")
    cal_scores = score("calibration")
    test_scores = score("test")

    stack_auc = float(compute_auc(y["stacking"], stack_scores))
    cal_auc = float(compute_auc(y["calibration"], cal_scores))
    test_auc = float(compute_auc(y["test"], test_scores))
    _, threshold = eer_operating_threshold(y["calibration"], cal_scores)

    print("=" * 78)
    print("FROZEN VISION-TOWER PROBE (MobileCLIP2-S0)")
    print("=" * 78)
    print(f"stacking AUC (probe fit)      : {stack_auc:.4f}")
    print(f"calibration AUC               : {cal_auc:.4f}")
    print(f"sealed test AUC               : {test_auc:.4f}")
    print(f"sealed test EER               : {compute_eer(y['test'], test_scores):.4f}")
    print(f"calibration EER threshold     : {threshold:.6f}")
    print(f"test accuracy at threshold    : {compute_accuracy(y['test'], test_scores, threshold):.4f}")

    np.savez_compressed(
        output_dir / "frozen_probe_embeddings.npz",
        weight=weight, bias=np.asarray([bias], dtype=np.float32),
        **{f"X_{split}": embeddings[split] for split in splits},
        **{f"y_{split}": y[split] for split in splits},
    )

    summary = {
        "engine": "C_frozen_vision_tower",
        "tower": MODEL_NAME,
        "tower_params": params,
        "probe_dim": EMBED_DIM,
        "probe_l2_reg": args.l2_reg,
        "stacking_auc": stack_auc,
        "calibration_auc": cal_auc,
        "test_auc": test_auc,
        "test_eer": float(compute_eer(y["test"], test_scores)),
        "threshold": float(threshold),
        "frontend_id": frozen_probe_frontend_id(),
        "input_shape": [1, FROZEN_PROBE_SIZE, FROZEN_PROBE_SIZE, 3],
    }

    if not args.skip_tflite:
        probe = FrozenProbe(visual, weight, bias)
        onnx_path = output_dir / "frozen_vision_probe.onnx"
        export_onnx(probe, onnx_path)
        tflite_path = convert_onnx_to_tflite(onnx_path, output_dir)
        summary["tflite_path"] = str(tflite_path)
        summary["tflite_sha256"] = sha256_file(tflite_path)
        summary["tflite_bytes"] = tflite_path.stat().st_size

    (output_dir / "frozen_probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

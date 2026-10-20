"""Export a trained Engine C student to INT8 TFLite.

Separated from training so a checkpoint can be re-exported without repeating
the (long) teacher extraction.

Usage:
    python -m ml.scripts.export_semantic_probe \
        --checkpoint D:\\onn_data\\runs\\release_v6\\checkpoints\\semantic_probe_best.pth \
        --output D:\\onn_data\\runs\\release_v6\\checkpoints\\semantic_probe_int8.tflite
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ML_ROOT / "src"))
sys.path.insert(0, str(ML_ROOT / "scripts"))

from data.contracts import IMAGE_EXTS, balanced_subset, discover_samples
from data.image_frontend import semantic_image_tensor
from export.quantize_int8 import quantize_keras_model, semantic_calibration_dataset
from export.torch_to_tflite_keras import build_keras_semantic_probe, keras_parity_check
from training.metrics import compute_auc, eer_operating_threshold
from validate_export import _run_tflite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--calibration-samples", type=int, default=400)
    args = parser.parse_args()

    from train_semantic_probe import SemanticProbeCNN

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    width = int(state["config"]["model"]["width"])
    size = int(state["contract"]["input_shape"][1])
    model = SemanticProbeCNN(width=width)
    model.load_state_dict(state["state_dict"])
    model.cpu().eval()

    keras_model = build_keras_semantic_probe(model, size)

    root = Path(args.data_root)
    train_items = balanced_subset(discover_samples(root, "train", IMAGE_EXTS), 200)
    cal_items = balanced_subset(discover_samples(root, "calibration", IMAGE_EXTS), args.calibration_samples)

    representative = semantic_calibration_dataset([str(p) for p, _ in train_items], 200, size)
    from data.image_frontend import decode_image

    nhwc = []
    for path, _ in cal_items:
        tensor = semantic_image_tensor(decode_image(str(path)).image, size)
        # keras_parity_check adds the batch dimension itself.
        nhwc.append(np.transpose(tensor.numpy(), (1, 2, 0)).astype(np.float32))
    batched = [value[None] for value in nhwc]

    fp32_error = keras_parity_check(keras_model, model, nhwc, "visual")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    quantization = quantize_keras_model(keras_model, representative, str(output))

    labels = [label for _, label in cal_items]
    int8_scores = [_run_tflite(str(output), value) for value in batched]
    _, threshold = eer_operating_threshold(labels, int8_scores)
    auc = float(compute_auc(labels, int8_scores))
    if auc <= 0.5:
        raise SystemExit("INT8 probe is at/below chance; refuse export")

    summary = {
        "checkpoint": str(args.checkpoint),
        "tflite": str(output),
        "tflite_bytes": output.stat().st_size,
        "quantization": quantization,
        "fp32_parity_max_error": fp32_error,
        "input_shape": [1, size, size, 3],
        "frontend_id": f"clip-appearance-bicubic{size}-clipnorm-meanstd-float32",
        "calibration_auc": auc,
        "calibration_threshold": float(threshold),
        "n_calibration": len(labels),
    }
    print(json.dumps(summary, indent=2))
    output.with_suffix(".export.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

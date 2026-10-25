#!/usr/bin/env python3
"""Export trained SRMForensicCNN PyTorch checkpoint to INT8 TFLite for Android.

Usage:
    python ml/scripts/export_srm_to_tflite.py \
        --checkpoint ml/checkpoints/deep_run_40epochs/patchcraft/srm_dcpt_patchcraft_best.pth \
        --data-dir datasets/openfake_visual \
        --output-dir ml/export/srm_tflite
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPT_DIR.parent
if (ML_ROOT / "src").exists():
    sys.path.insert(0, str(ML_ROOT / "src"))

from data.contracts import sha256_file
from models.srm_forensic_cnn import SRMForensicCNN
from training.checkpoint import load_verified_checkpoint
from export.torch_to_tflite_keras import build_keras_srm
from export.quantize_int8 import quantize_keras_model, visual_calibration_dataset


def export_srm_to_tflite(
    checkpoint_path: str,
    output_dir: str,
    data_dir: str | None = None,
    limit: int = 150,
) -> Path:
    ckpt_path = Path(checkpoint_path).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tflite = out_dir / "srm_forensic_cnn_int8.tflite"

    print(f"Loading checkpoint: {ckpt_path}")
    state = load_verified_checkpoint(str(ckpt_path))
    contract = state.get("contract", {})
    input_shape = contract.get("input_shape", [1, 128, 128, 3])
    input_size = int(input_shape[1])
    srm_channels = contract.get("srm_channels", 20)
    include_npr = srm_channels in (16, 20)
    include_rgb = srm_channels == 20

    print(f"  - Input size   : {input_size}x{input_size}")
    print(f"  - SRM channels : {srm_channels} (include_npr={include_npr}, include_rgb={include_rgb})")

    # 1. Instantiate and load PyTorch model
    pt_model = SRMForensicCNN(include_npr=include_npr, include_rgb=include_rgb)
    pt_model.load_state_dict(state["state_dict"])
    pt_model.eval()

    # 2. Reconstruct in Keras & transfer weights
    print("Reconstructing architecture in Keras (channels-last)...")
    keras_model = build_keras_srm(pt_model, input_size=input_size)

    # 3. Prepare representative calibration dataset
    print("Preparing representative dataset for INT8 calibration...")
    image_paths = []
    if data_dir and Path(data_dir).is_dir():
        droot = Path(data_dir).resolve()
        for split in ("calibration", "val", "train"):
            if (droot / split).is_dir():
                for p in (droot / split).rglob("*.*"):
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                        image_paths.append(str(p))
                        if len(image_paths) >= limit:
                            break
            if len(image_paths) >= limit:
                break

    if len(image_paths) > 0:
        print(f"  - Using {len(image_paths)} real images from {data_dir}")
        rep_gen = visual_calibration_dataset(image_paths, limit=limit, input_size=input_size)
    else:
        print("  - Using synthetic representative dataset generator (0.0 .. 1.0 float32)")
        def rep_gen():
            rng = np.random.RandomState(42)
            for _ in range(limit):
                yield [rng.uniform(0.0, 1.0, size=(1, input_size, input_size, 3)).astype(np.float32)]

    # 4. Quantize to INT8
    print("Quantizing to INT8 TFLite flatbuffer...")
    kind = quantize_keras_model(keras_model, rep_gen, str(out_tflite), allow_dynamic_range=False)

    # 5. Check Parity on Real / Random Input
    test_arr = np.random.uniform(0.0, 1.0, size=(1, input_size, input_size, 3)).astype(np.float32)
    # PyTorch CHW
    pt_in = torch.from_numpy(test_arr.transpose(0, 3, 1, 2))
    with torch.no_grad():
        pt_out = float(torch.sigmoid(pt_model(pt_in)).numpy().flatten()[0])

    # Keras NHWC
    keras_out = float(keras_model(test_arr).numpy().flatten()[0])

    # TFLite
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(out_tflite))
        interpreter.allocate_tensors()
        inp_detail = interpreter.get_input_details()[0]
        out_detail = interpreter.get_output_details()[0]
        interpreter.set_tensor(inp_detail["index"], test_arr)
        interpreter.invoke()
        tflite_out = float(interpreter.get_tensor(out_detail["index"]).flatten()[0])
    except Exception as exc:
        tflite_out = None
        print(f"Warning: TFLite runtime check failed ({exc})")

    tflite_size_kb = out_tflite.stat().st_size / 1024
    tflite_sha256 = sha256_file(out_tflite)

    print("\n" + "=" * 60)
    print("🎉 EXPORT AND QUANTIZATION SUCCESSFUL!")
    print(f"  Output path    : {out_tflite}")
    print(f"  Model format   : {kind}")
    print(f"  Model size     : {tflite_size_kb:.1f} KB")
    print(f"  SHA-256        : {tflite_sha256}")
    print(f"  PyTorch Score  : {pt_out:.6f}")
    print(f"  Keras Score    : {keras_out:.6f} (PT vs Keras diff: {abs(pt_out - keras_out):.2e})")
    if tflite_out is not None:
        print(f"  TFLite Score   : {tflite_out:.6f} (Keras vs TFLite INT8 diff: {abs(keras_out - tflite_out):.4f})")
    print("=" * 60 + "\n")

    # Write export manifest
    info = {
        "source_checkpoint": str(ckpt_path),
        "source_checkpoint_sha256": sha256_file(ckpt_path),
        "tflite_path": str(out_tflite.name),
        "tflite_sha256": tflite_sha256,
        "tflite_size_bytes": out_tflite.stat().st_size,
        "input_shape": [1, input_size, input_size, 3],
        "input_type": "float32",
        "output_shape": [1, 1],
        "output_type": "float32",
        "kind": kind,
        "contract": contract,
    }
    (out_dir / "srm_export_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return out_tflite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to input .pth checkpoint")
    parser.add_argument("--output-dir", default=None, help="Directory to save exported TFLite")
    parser.add_argument("--output", default=None, help="Output path (directory or .tflite file)")
    parser.add_argument("--data-dir", default=None, help="Optional dataset directory for calibration images")
    parser.add_argument("--num-calibration-samples", "--limit", dest="limit", type=int, default=150, help="Number of calibration samples")
    args = parser.parse_args()

    out = args.output or args.output_dir or str(ML_ROOT / "export" / "srm_tflite")
    out_path = Path(out)
    if out_path.suffix.lower() == ".tflite":
        output_dir = str(out_path.parent)
    else:
        output_dir = str(out_path)

    export_srm_to_tflite(
        checkpoint_path=args.checkpoint,
        output_dir=output_dir,
        data_dir=args.data_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

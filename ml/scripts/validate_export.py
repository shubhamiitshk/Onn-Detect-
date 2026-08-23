"""Validate exported TFLite models against PyTorch on REAL inputs.

The previous validator compared both models on random Gaussian noise, which
can hide quantization damage that only appears on realistic inputs. Here the
TFLite model is fed NHWC tensors built exactly like Android will build them,
and the PyTorch reference scores the same tensor converted back to its own
layout.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import torch

from export.torch_to_tflite_keras import from_keras_input


def _run_tflite(tflite_path: str, x_nhwc: np.ndarray) -> float:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    expected = list(inp["shape"])
    if list(x_nhwc.shape) != expected:
        raise ValueError(f"input {x_nhwc.shape} != model input {expected}")
    interpreter.set_tensor(inp["index"], x_nhwc.astype(inp["dtype"], copy=False))
    interpreter.invoke()
    return float(interpreter.get_tensor(out["index"]).ravel()[0])


def validate_models_real(pytorch_model, tflite_path: str, modality: str,
                         real_images=None, real_specs_pt=None,
                         target_frames: int | None = None, num_samples=50):
    """Compare PyTorch and TFLite outputs on real images / spectrograms.

    modality="visual": real_images = PIL RGB images (any size 32x32 native).
    modality="audio" : real_specs_pt = list of PT-layout (1,1,T,F) LFCC arrays;
                       cropped/padded along time to target_frames.
    """
    print(f"Validating {tflite_path} on real inputs...")
    pytorch_model.eval()

    inputs_nhwc = []
    if modality == "visual":
        for img in (real_images or [])[:num_samples]:
            arr = np.asarray(img, dtype=np.float32) / 255.0
            if arr.ndim == 3:
                inputs_nhwc.append(arr)
    elif modality == "audio":
        for s in (real_specs_pt or [])[:num_samples]:
            arr = np.transpose(np.asarray(s, dtype=np.float32)[0], (1, 2, 0))  # (T,F,1)
            if target_frames:
                t = arr.shape[0]
                if t > target_frames:
                    off = (t - target_frames) // 2
                    arr = arr[off:off + target_frames]
                elif t < target_frames:
                    arr = np.pad(arr, ((0, target_frames - t), (0, 0), (0, 0)))
            inputs_nhwc.append(arr)
    else:
        raise ValueError(modality)

    inputs_nhwc = [x[np.newaxis] for x in inputs_nhwc]  # add batch dim
    if not inputs_nhwc:
        raise ValueError("No usable real inputs were provided")

    max_err, total_err = 0.0, 0.0
    pt_scores, tl_scores = [], []
    for x in inputs_nhwc:
        with torch.no_grad():
            pt_out = pytorch_model(torch.from_numpy(
                from_keras_input(x, modality))).numpy().ravel()[0]
        tl_out = _run_tflite(tflite_path, x)
        err = abs(pt_out - tl_out)
        max_err = max(max_err, err)
        total_err += err
        pt_scores.append(float(pt_out))
        tl_scores.append(tl_out)

    mae = total_err / len(inputs_nhwc)
    agree = sum(1 for a, b in zip(pt_scores, tl_scores)
                if (a > 0.5) == (b > 0.5)) / len(pt_scores)

    print("=" * 40)
    print(f"Validation Report ({len(inputs_nhwc)} real samples)")
    print("=" * 40)
    print(f"Mean |PyTorch - TFLite|: {mae:.6f}")
    print(f"Max  |PyTorch - TFLite|: {max_err:.6f}")
    print(f"Decision agreement @0.5: {agree:.2%}")
    print("=" * 40)
    return {"mae": mae, "max_error": max_err, "agreement": agree}

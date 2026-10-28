#!/usr/bin/env python3
"""Verify SRM convolutional stem weights directly against the frozen .tflite tensor buffer.

Verifiable Evidence Chain Gate:
1. Loads `android/app/src/main/assets/models/srm_forensic_cnn_int8.tflite`.
2. Locates the input convolutional stem tensor `(20, 5, 5, 3)` (20 filters, 5x5 spatial, 3 RGB channels).
3. Dequantizes INT8 weights to FP32 using per-channel scale and zero-point parameters.
4. Compares against the mathematical reference filters from `ml/src/models/srm_forensic_cnn.py`:
   - Channels 0-11: 12 Fixed SRM (Spatial Rich Model) high-pass & steganalysis filters (Fridrich & Kodovsky, 2012).
   - Channels 12-15: 4 Fixed NPR (Neighboring Pixel Relationships) 2x2 upsampling residual filters (Tan et al., CVPR 2024).
   - Channels 16-19: 4 Centered RGB pass-through and luminance low-pass filters.
5. Asserts numerical agreement within INT8 quantization tolerance (max absolute error < 0.005).

Usage:
    python scripts/verify_srm_stem_weights.py
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml" / "src"))

MODEL_PATH = ROOT / "android" / "app" / "src" / "main" / "assets" / "models" / "srm_forensic_cnn_int8.tflite"

def verify_srm_stem():
    if not MODEL_PATH.is_file():
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        sys.exit(1)

    import tensorflow as tf
    from models.srm_forensic_cnn import SRMLayer

    print(f"Loading TFLite model from {MODEL_PATH}...")
    interp = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interp.allocate_tensors()

    details = interp.get_tensor_details()
    stem_tensor_detail = None
    for d in details:
        if d["shape"].tolist() == [20, 5, 5, 3] and "conv" in d["name"].lower():
            stem_tensor_detail = d
            break

    if stem_tensor_detail is None:
        # Fallback to any tensor of shape [20, 5, 5, 3]
        for d in details:
            if d["shape"].tolist() == [20, 5, 5, 3]:
                stem_tensor_detail = d
                break

    if stem_tensor_detail is None:
        print("ERROR: Could not locate stem tensor with shape [20, 5, 5, 3] in TFLite model.")
        sys.exit(1)

    print(f"Found stem tensor: index={stem_tensor_detail['index']} name='{stem_tensor_detail['name']}' shape={stem_tensor_detail['shape']}")

    raw_int8 = interp.get_tensor(stem_tensor_detail["index"]).astype(np.float32)
    scales = stem_tensor_detail["quantization_parameters"]["scales"]
    zero_points = stem_tensor_detail["quantization_parameters"]["zero_points"]

    dequantized = np.zeros_like(raw_int8)
    for ch in range(20):
        s = scales[ch] if len(scales) > ch else scales[0]
        zp = zero_points[ch] if len(zero_points) > ch else zero_points[0]
        dequantized[ch] = (raw_int8[ch] - zp) * s

    # Reference SRMLayer (PyTorch: [20, 3, 5, 5])
    ref_layer = SRMLayer(include_npr=True, include_rgb=True)
    ref_weight = ref_layer.conv.weight.detach().cpu().numpy() # [20, 3, 5, 5]
    # Transpose to TFLite layout: [20, 5, 5, 3]
    ref_tflite_layout = np.transpose(ref_weight, (0, 2, 3, 1))

    channel_names = [
        "SRM 1: 5x5 Laplacian-like center-surround",
        "SRM 2: 5x5 Horizontal edge/noise detector",
        "SRM 3: 5x5 Vertical edge/noise detector",
        "SRM 4: 5x5 Main diagonal high-frequency detector",
        "SRM 5: 5x5 Anti-diagonal high-frequency detector",
        "SRM 6: 5x5 Fridrich KV 4th-order steganalysis filter",
        "SRM 7: 5x5 Nyquist checkerboard / VAE upsampling filter",
        "SRM 8: 3x3 Red Laplacian embedded in 5x5",
        "SRM 9: 3x3 Green Laplacian embedded in 5x5",
        "SRM 10: 3x3 Blue Laplacian embedded in 5x5",
        "SRM 11: 3x3 R-G color opponent high-pass filter",
        "SRM 12: 3x3 B-Y color opponent high-pass filter",
        "NPR 1: 2x2 Horizontal adjacent difference upsampling filter",
        "NPR 2: 2x2 Vertical adjacent difference upsampling filter",
        "NPR 3: 2x2 Main-diagonal difference upsampling filter",
        "NPR 4: 2x2 Anti-diagonal difference upsampling filter",
        "RGB 1: Centered Red pass-through impulse",
        "RGB 2: Centered Green pass-through impulse",
        "RGB 3: Centered Blue pass-through impulse",
        "Luma Low-Pass: 5x5 Gaussian/box luminance stem filter",
    ]

    all_passed = True
    max_error_overall = 0.0

    print("\n--- Stem Kernel Numerical Parity Verification ---")
    for ch in range(20):
        tflite_kernel = dequantized[ch]
        ref_kernel = ref_tflite_layout[ch]
        max_err = float(np.max(np.abs(tflite_kernel - ref_kernel)))
        max_error_overall = max(max_error_overall, max_err)
        passed = max_err < 0.005
        if not passed:
            all_passed = False
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] Ch {ch:02d} ({channel_names[ch]}): max_err = {max_err:.6f}")

    print(f"\nOverall Max Quantization Error: {max_error_overall:.6f}")
    if all_passed:
        print("SUCCESS: All 20 stem filter kernels match mathematical reference within INT8 tolerance (< 0.005)!")
        return True
    else:
        print("FAILURE: One or more stem filter kernels diverge from reference.")
        sys.exit(1)

if __name__ == "__main__":
    verify_srm_stem()

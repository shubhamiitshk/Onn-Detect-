"""Verify the PyTorch -> Keras export path end to end (random weights).

    python scripts/verify_export_path.py         # parity only (fast, CI-safe)
    python scripts/verify_export_path.py --full  # + Keras->TFLite conversion
                                                 #   and inference smoke test

Builds both Keras models from randomly-initialized PyTorch parents and checks
PyTorch-vs-Keras numerical parity on synthetic inputs. This exercises exactly
the code that ships models to the phone:

  - SRM: ZeroPadding grids, depthwise weight layout, BN epsilon, head dims
  - LCNN: the AdaptiveAvgPool2d((1,2)) time-pool + frequency-preserving
    flatten, the (B,C,W)->(B,C*W) order contract, and the Dropout-skipping
    classifier indices ([0] and [3], NOT [2])

Random weights are sufficient: a layout/order bug shifts outputs by O(1),
while float32 rounding is O(1e-6). Exit non-zero on failure, so this gates CI.
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "ml", "src"))

import torch

from models.lcnn_mfm_audio import LCNNMFMAudio
from models.srm_forensic_cnn import SRMForensicCNN

failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also run Keras->TFLite conversion + inference smoke")
    ap.add_argument("--allow-missing-dependencies", action="store_true",
                    help="local diagnostics only: explicitly allow skipped TensorFlow checks")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    try:
        from export.torch_to_tflite_keras import (
            build_keras_lcnn,
            build_keras_srm,
            keras_parity_check,
        )
    except ImportError as e:
        print(f"TensorFlow not installed, cannot verify export path: {e}")
        sys.exit(0 if args.allow_missing_dependencies else 1)

    print("=" * 72)
    print("Export-path parity (random weights - layout/order verification)")
    print("=" * 72)

    # ---- SRM ----
    print("\n1) SRMForensicCNN -> Keras")
    srm = SRMForensicCNN().eval()
    ksrm = build_keras_srm(srm)
    imgs = [np.random.rand(32, 32, 3).astype(np.float32) for _ in range(8)]
    err = keras_parity_check(ksrm, srm, imgs, "visual")
    check("SRM keras-vs-pytorch max err < 1e-4", err < 1e-4, f"max err {err:.2e}")
    check("SRM input shape is NHWC (1,32,32,3)",
          list(ksrm.input_shape) == [None, 32, 32, 3],
          f"got {ksrm.input_shape}")

    # ---- LCNN ----
    print("\n2) LCNNMFMAudio -> Keras (T=401, F=40)")
    lcnn = LCNNMFMAudio().eval()
    klcnn = build_keras_lcnn(lcnn, input_frames=401)
    # NHWC (1, T=401, F=40, C=1)
    specs = [np.random.rand(401, 40, 1).astype(np.float32) for _ in range(4)]
    err = keras_parity_check(klcnn, lcnn, specs, "audio")
    check("LCNN keras-vs-pytorch max err < 1e-4", err < 1e-4, f"max err {err:.2e}")
    check("LCNN input shape is NHWC (1,401,40,1)",
          list(klcnn.input_shape) == [None, 401, 40, 1],
          f"got {klcnn.input_shape}")
    # The (1,2)-pool contract: flatten must present 192 = 96ch x 2freq.
    flat = [l for l in klcnn.layers if "gap_flat" in l.name]
    check("gap_flat layer exists (frequency-preserving head)",
          len(flat) == 1, f"found {[l.name for l in flat]}")
    if flat:
        dim = int(np.prod(flat[0].output.shape[1:]))
        check("flatten dim is 192 (96 channels x 2 freq groups)",
              dim == 192, f"got {flat[0].output.shape}")

    print()
    print("=" * 72)
    if args.full:
        run_full_conversion(ksrm, klcnn)
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        sys.exit(1)
    print("Export path verified: layouts, pooling, and flatten order agree.")
    print("=" * 72)


def run_full_conversion(ksrm, klcnn) -> None:
    """Keras -> TFLite full-INT8 conversion + inference smoke (random weights).

    Uses synthetic calibration data, so this proves the CONVERSION CHAIN
    (ops supported, I/O dtypes, inference runs) - not quantization quality,
    which needs real calibration data. Verified once here at full-int8 for
    both models; do not regress to dynamic-range silently (export writes
    export_report.json with the kind per model).
    """
    import tempfile

    from export.quantize_int8 import quantize_keras_model

    print("\n3) Keras -> TFLite full-INT8 conversion (synthetic calibration)")

    def srm_cal():
        for _ in range(5):
            yield [np.random.rand(1, 32, 32, 3).astype(np.float32)]

    def lcnn_cal():
        for _ in range(5):
            yield [np.random.rand(1, 401, 40, 1).astype(np.float32)]

    with tempfile.TemporaryDirectory() as tmp:
        for name, kmodel, cal, shape in (
            ("srm", ksrm, srm_cal, (1, 32, 32, 3)),
            ("lcnn", klcnn, lcnn_cal, (1, 401, 40, 1)),
        ):
            out = os.path.join(tmp, f"{name}.tflite")
            kind = quantize_keras_model(kmodel, cal, out)
            check(f"{name} reaches full-int8 (not silent dynamic-range)",
                  kind == "full-int8", f"got {kind}")
            try:
                import tensorflow as tf

                it = tf.lite.Interpreter(model_path=out)
                it.allocate_tensors()
                inp, outd = it.get_input_details()[0], it.get_output_details()[0]
                it.set_tensor(inp["index"],
                              np.random.rand(*shape).astype(np.float32))
                it.invoke()
                v = float(it.get_tensor(outd["index"]).ravel()[0])
                check(f"{name} tflite inference finite", bool(np.isfinite(v)),
                      f"output {v:.6f}")
            except ImportError:
                check(f"{name} tflite inference", False,
                      "tensorflow not installed")


if __name__ == "__main__":
    main()

"""Export the real Meta AudioSeal watermark detector to ONNX and TFLite.

Why this exists
---------------
The first AudioSeal integration in this repository correlated the audio
against a hard-coded 64-element +/-1 sequence. That sequence is NOT
AudioSeal's carrier: AudioSeal's carrier is a pseudo-random sequence
derived from a secret key, and the detector only responds to a watermark
embedded with that key. A correlation against an arbitrary fixed pattern
therefore:

  * never detects a real AudioSeal watermark, and
  * produces FALSE POSITIVES on any audio that happens to correlate with
    that pattern - and the p-value was the max over ~100 frames with no
    correction for the multiple comparisons, so the false-positive rate
    was wildly overstated.

A false positive here is worse than a false negative: it would confirm
AI origin on authentic audio.

The real detector is a compact 1-D SEANet convolutional network with a
32-bit message decoder. It is small, it is 1-D (so there is no NHWC/NCHW
layout problem at all), and it exports cleanly.

Usage
-----
    python -m ml.scripts.export_audioseal_detector \\
        --output-dir D:\\onn_data\\runs\\audioseal_export
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# AudioSeal operates on 16 kHz mono audio.
SAMPLE_RATE = 16000
# The detector consumes a fixed window; 1 second is the smallest that
# carries a usable number of message bits.
WINDOW_SAMPLES = SAMPLE_RATE


def load_detector():
    """The REAL AudioSeal detector, with its published weights.

    ``AudioSeal.load_detector()`` is the public entry point; it resolves the
    config and the checkpoint together. Anything else would be a
    re-implementation with no guarantee of matching the trained carrier.
    """
    from audioseal import AudioSeal

    return AudioSeal.load_detector()


class AgreeingBitCounter(torch.nn.Module):
    """Wraps the detector so the graph emits a SCALAR: the agreeing-bit count.

    The app's inference engine asserts a scalar model output
    (`LiteRTInferenceEngine`: `require(output.numElements() == 1)`), and the
    raw AudioSeal detector emits one probability per message bit. Wrapping
    the count keeps the engine's contract intact and puts the statistic
    where it belongs: the sign test over the 32 message bits is computed
    inside the graph, and the app only has to turn a bit count into a
    p-value.
    """

    def __init__(self, detector: torch.nn.Module) -> None:
        super().__init__()
        self.detector = detector

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        message = self.detector(audio)
        if message.dim() == 3:
            message = message[:, 0, :]
        elif message.dim() == 1:
            message = message.unsqueeze(0)
        # A bit agrees when it is confidently 0 or confidently 1, so the
        # sign test is over |2p - 1| > 0.5, i.e. p < 0.25 or p > 0.75.
        agreeing = ((message > 0.75) | (message < 0.25)).sum(dim=-1, keepdim=True)
        return agreeing.to(torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-samples", type=int, default=WINDOW_SAMPLES)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = load_detector()
    detector.eval()
    total = sum(p.numel() for p in detector.parameters())
    print(f"AudioSeal detector: {total:,} parameters ({total * 4 / 1e6:.2f} MB fp32)")

    window = args.window_samples
    dummy = torch.rand(1, window)

    # --- ONNX ---
    onnx_path = output_dir / "audioseal_detector.onnx"
    with torch.no_grad():
        probe = detector(dummy)
    print(f"detector output shape: {tuple(probe.shape)}")

    model = AgreeingBitCounter(detector).eval()
    with torch.no_grad():
        counted = model(dummy)
    print(f"wrapped output shape: {tuple(counted.shape)}")

    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["audio"], output_names=["agreeing_bits"],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print(f"ONNX: {onnx_path.name} ({onnx_path.stat().st_size / 1e6:.2f} MB)")

    # --- parity ---
    from onnxruntime import InferenceSession

    session = InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    batch = np.random.RandomState(0).rand(4, window).astype(np.float32)
    with torch.no_grad():
        torch_out = model(torch.from_numpy(batch)).numpy()
    onnx_out = session.run(None, {"audio": batch})[0]
    error = float(np.abs(torch_out - onnx_out).max())
    print(f"ONNX parity: max |onnx - torch| = {error:.3e}")
    if error > 1e-4:
        raise SystemExit("ONNX parity failed")

    # --- TFLite ---
    tflite_path = output_dir / "audioseal_detector.tflite"
    converter = None
    try:
        import tensorflow as tf

        converter = tf.lite.TFLiteConverter.from_onnx_model(str(onnx_path))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: (
            [batch[index:index + 1].astype(np.float32)] for index in range(len(batch))
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_path.write_bytes(converter.convert())
        print(f"TFLite INT8: {tflite_path.stat().st_size / 1e6:.2f} MB")
    except Exception as exc:  # pragma: no cover - toolchain dependent
        print(f"TFLite conversion failed: {type(exc).__name__}: {exc}")

    summary = {
        "sample_rate": SAMPLE_RATE,
        "window_samples": window,
        "parameters": int(total),
        "onnx_bytes": onnx_path.stat().st_size,
        "tflite_bytes": tflite_path.stat().st_size if tflite_path.is_file() else None,
        "onnx_parity_max_error": error,
    }
    (output_dir / "audioseal_export.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

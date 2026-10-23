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


def _strip_weight_norm(module: torch.nn.Module) -> torch.nn.Module:
    """Fuses all WeightNorm / Parametrization hooks into plain weights before export."""
    for child in module.modules():
        try:
            torch.nn.utils.remove_weight_norm(child)
        except ValueError:
            pass
        if hasattr(torch.nn.utils, "parametrize") and torch.nn.utils.parametrize.is_parametrized(child):
            for tensor_name in list(child.parametrizations.keys()):
                torch.nn.utils.parametrize.remove_parametrizations(child, tensor_name, leave_parametrized=True)
    return module


def _tflite_bytes(model: torch.nn.Module, onnx_path: str, workdir: str, calib: np.ndarray, window: int) -> bytes:
    """Primary path: ai-edge-torch direct PyTorch -> LiteRT lowering.
    Fallback path: onnx2tf with weight_norm already fused."""
    try:
        import ai_edge_torch
        sample = (torch.from_numpy(calib[:1]).to(torch.float32),)
        edge_model = ai_edge_torch.convert(model.eval(), sample)
        out_file = Path(workdir) / "audioseal_ai_edge.tflite"
        edge_model.export(str(out_file))
        return out_file.read_bytes()
    except (ImportError, Exception):
        pass

    import glob
    import onnx2tf
    tf_dir = str(Path(workdir) / "tf_saved_model")
    try:
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=tf_dir,
            non_verbose=True,
            keep_ncw_or_nchw_or_ncdhw_input_names=["audio"],
            disable_group_convolution=True,
            output_dynamic_range_quantized_tflite=True,
        )
        candidates = sorted(glob.glob(str(Path(tf_dir) / "*float32.tflite"))) + \
                     sorted(glob.glob(str(Path(tf_dir) / "*dynamic_range_quantize*.tflite")))
        if candidates:
            return Path(candidates[0]).read_bytes()
    except Exception:
        pass
    raise RuntimeError("No TFLite candidate emitted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-samples", type=int, default=WINDOW_SAMPLES)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = load_detector()
    detector.eval()
    detector = _strip_weight_norm(detector)
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
    try:
        tflite_data = _tflite_bytes(detector, str(onnx_path), str(output_dir), batch, window)
        tflite_path.write_bytes(tflite_data)
        print(f"TFLite: {tflite_path.stat().st_size / 1e6:.2f} MB")
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

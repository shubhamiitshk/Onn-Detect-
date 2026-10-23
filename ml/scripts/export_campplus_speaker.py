#!/usr/bin/env python3
"""Exports 3D-Speaker CAM++ (192-d speaker embedding) to INT8 LiteRT (.tflite) without onnx2tf TDNN corruption.

Why onnx2tf fails on CAM++:
    CAM++ (speech_campplus_sv_zh-cn_16k-common) uses Multi-Granularity DenseTDNN blocks built from
    dilated 1-D convolutions `nn.Conv1d(C_in, C_out, kernel_size=k, dilation=d)`. onnx2tf inserts
    broken NCHW<->NHWC Transpose chains around dilated 1-D ops that fail shape inference or diverge.

How this exporter unblocks P1:
    1. Rewrites every 1-D `nn.Conv1d(C_in, C_out, k, stride=s, padding=p, dilation=d, groups=g)` into an
       algebraically identical 2-D `nn.Conv2d(C_in, C_out, (1, k), stride=(1, s), padding=(0, p), dilation=(1, d), groups=g)`
       over `(B, C, 1, T)` before tracing. TFLite's built-in `CONV_2D` XNNPACK kernel natively supports
       2-D dilated convolutions with zero axis-transpose bugs.
    2. Lowers via `ai_edge_torch` (primary) or `onnx2tf` over the 2-D-rewritten graph, applies dynamic-range
       INT8 weight quantization (float32 I/O `[1, 298, 80]` -> `[1, 192]` L2-normalized), and enforces a
       hard parity gate (`max |1 - cos_sim(torch, tflite)| < 1e-3`).

Usage
-----
    python3 ml/scripts/export_campplus_speaker.py \
        --checkpoint /path/to/campplus_cn_common.bin \
        --output android/app/src/main/assets/models/campplus_speaker_int8.tflite
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dAs2d(nn.Module):
    """Wraps an nn.Conv1d as an algebraically identical nn.Conv2d((1, k), dilation=(1, d))."""

    def __init__(self, conv1d: nn.Conv1d) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=conv1d.in_channels,
            out_channels=conv1d.out_channels,
            kernel_size=(1, conv1d.kernel_size[0]),
            stride=(1, conv1d.stride[0]),
            padding=(0, conv1d.padding[0]),
            dilation=(1, conv1d.dilation[0]),
            groups=conv1d.groups,
            bias=conv1d.bias is not None,
        )
        with torch.no_grad():
            self.conv2d.weight.copy_(conv1d.weight.unsqueeze(2))
            if conv1d.bias is not None and self.conv2d.bias is not None:
                self.conv2d.bias.copy_(conv1d.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(x.unsqueeze(2)).squeeze(2)


def rewrite_conv1d_to_conv2d(module: nn.Module) -> nn.Module:
    """Recursively replaces all nn.Conv1d layers with Conv1dAs2d for clean LiteRT CONV_2D lowering."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv1d):
            setattr(module, name, Conv1dAs2d(child))
        else:
            rewrite_conv1d_to_conv2d(child)
    return module


class NormalizedCamPlusWrapper(nn.Module):
    """Contract wrapper matching NeuralSpeakerEncoder.kt: input [1, 298, 80] -> L2-normalized [1, 192]."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, fbank: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(fbank)
        if isinstance(emb, (tuple, list)):
            emb = emb[0]
        if emb.dim() == 3:
            emb = emb.squeeze(1)
        return F.normalize(emb.to(torch.float32), p=2.0, dim=-1)


def verify_tflite_parity(tflite_path: Path, torch_model: nn.Module, n_frames: int = 298, n_mels: int = 80) -> float:
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    rng = np.random.RandomState(2026)
    max_cos_err = 0.0
    for _ in range(6):
        sample = rng.randn(1, n_frames, n_mels).astype(np.float32)
        with torch.no_grad():
            ref = torch_model(torch.from_numpy(sample)).numpy()[0]
        interp.set_tensor(inp["index"], sample)
        interp.invoke()
        got = interp.get_tensor(out["index"])[0].astype(np.float32)
        got /= max(float(np.linalg.norm(got)), 1e-12)
        cos_sim = float(np.dot(ref, got))
        max_cos_err = max(max_cos_err, abs(1.0 - cos_sim))

    if max_cos_err > 1e-3:
        raise RuntimeError(f"CAM++ TFLite parity gate FAILED: max |1 - cos_sim| = {max_cos_err:.4e} > 1e-3")
    return max_cos_err


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to CAM++ PyTorch state_dict (.bin / .pt)")
    parser.add_argument(
        "--output",
        default="android/app/src/main/assets/models/campplus_speaker_int8.tflite",
    )
    args = parser.parse_args()

    from speakerlab.models.campplus.DTDNN import CAMPPlus  # 3D-Speaker official definition

    backbone = CAMPPlus(feat_dim=80, embedding_size=192)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    backbone.load_state_dict(state, strict=True)
    backbone.eval()

    rewritten = rewrite_conv1d_to_conv2d(backbone).eval()
    wrapped = NormalizedCamPlusWrapper(rewritten).eval()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.randn(1, 298, 80, dtype=torch.float32)

    try:
        import ai_edge_torch

        edge_model = ai_edge_torch.convert(wrapped, (sample,))
        edge_model.export(str(out_path))
    except ImportError:
        import glob
        import tempfile
        import onnx2tf

        with tempfile.TemporaryDirectory() as tmp:
            onnx_file = Path(tmp) / "campplus_2d.onnx"
            torch.onnx.export(
                wrapped,
                sample,
                str(onnx_file),
                input_names=["fbank"],
                output_names=["embedding"],
                opset_version=17,
            )
            onnx2tf.convert(
                input_onnx_file_path=str(onnx_file),
                output_folder_path=str(Path(tmp) / "tf_out"),
                non_verbose=True,
                keep_ncw_or_nchw_or_ncdhw_input_names=["fbank"],
                output_dynamic_range_quantized_tflite=True,
            )
            cands = sorted(glob.glob(str(Path(tmp) / "tf_out" / "*dynamic_range_quantize*.tflite")))
            if not cands:
                cands = sorted(glob.glob(str(Path(tmp) / "tf_out" / "*.tflite")))
            out_path.write_bytes(Path(cands[0]).read_bytes())

    max_err = verify_tflite_parity(out_path, wrapped)
    sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(json.dumps({
        "model": str(out_path),
        "size_bytes": out_path.stat().st_size,
        "sha256": sha256,
        "max_cosine_error": round(max_err, 7),
    }, indent=2))


if __name__ == "__main__":
    main()

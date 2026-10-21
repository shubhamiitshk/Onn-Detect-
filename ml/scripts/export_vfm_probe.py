"""Export a frozen VFM + linear probe to ONNX and then to TFLite.

Why this exists
---------------
The measurement in ``evaluate_vfm_probes.py`` shows a frozen SigLIP-B/16
linear probe reaching **0.8919** on the sealed test split against the
shipped Engine C's **0.6871**. The probe head is 769 numbers; the rest is
the tower.

Whether that is deployable is a question about the toolchain, not about
the accuracy, so this script answers it end to end:

  1. compose ``image -> tower -> probe -> P(AI)`` into ONE graph;
  2. export it to ONNX;
  3. convert it to TFLite with onnx2tf;
  4. report the size and the numerical parity against the PyTorch graph.

The repository already solved the "small student" version of this problem
(``train_semantic_probe.py`` -> ``export_semantic_probe.py``). This is the
"ship the frozen tower" version, which the repo's own measurements say is
the stronger option.

Usage
-----
    python -m ml.scripts.export_vfm_probe \\
        --data-root D:\\onn_data\\openfake_visual \\
        --output-dir D:\\onn_data\\runs\\vfm_probe_export \\
        --tower google/siglip-base-patch16-224
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data.contracts import IMAGE_EXTS, discover_samples  # noqa: E402
from data.image_frontend import decode_image  # noqa: E402
from evaluate_vfm_probes import TOWERS, fit_logistic_probe  # noqa: E402
from training.metrics import compute_auc  # noqa: E402


class VfmProbe(nn.Module):
    """A frozen vision tower with a learned linear probe head, as one graph."""

    def __init__(self, tower: nn.Module, weight: np.ndarray, bias: np.ndarray,
                 feat_mean: np.ndarray, feat_std: np.ndarray,
                 pix_mean: np.ndarray, pix_std: np.ndarray, size: int):
        super().__init__()
        self.tower = tower
        self.size = size
        self.register_buffer("probe_w", torch.from_numpy(weight.astype(np.float32)))
        self.register_buffer("probe_b", torch.tensor(float(bias), dtype=torch.float32))
        self.register_buffer("feat_mean", torch.from_numpy(feat_mean.astype(np.float32)))
        self.register_buffer("feat_std", torch.from_numpy(feat_std.astype(np.float32)))
        # PIXEL normalisation is the tower's own; it is NOT the feature
        # statistics. Conflating the two is a silent numerical bug.
        self.register_buffer("pix_mean", torch.from_numpy(pix_mean.astype(np.float32)))
        self.register_buffer("pix_std", torch.from_numpy(pix_std.astype(np.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) float32 in [0, 1] -> (B, 1) probability.

        The ONNX graph is NCHW. onnx2tf consumes NCHW and emits an NHWC
        TFLite, which is exactly the layout the Android app uses for every
        other model - so the shipped graph needs no transpose on device.
        """
        rgb = (x - self.pix_mean.view(1, 3, 1, 1)) / self.pix_std.view(1, 3, 1, 1)
        features = self.tower(pixel_values=rgb)
        if hasattr(self.tower.config, "vision_config"):
            pooled = features.pooler_output
        else:
            pooled = features.last_hidden_state[:, 0]
        pooled = pooled.float()
        pooled = pooled / (pooled.norm(dim=1, keepdim=True) + 1e-12)
        logits = ((pooled - self.feat_mean) / self.feat_std) @ self.probe_w + self.probe_b
        return torch.sigmoid(logits).unsqueeze(1)


def load_tower(repo: str, model_cls: str):
    import transformers
    model = getattr(transformers, model_cls).from_pretrained(repo)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tower", default="google/siglip-base-patch16-224")
    parser.add_argument("--l2-reg", type=float, default=30.0)
    parser.add_argument("--split", default="stacking")
    args = parser.parse_args()

    tower_spec = TOWERS.get(args.tower)
    if tower_spec is None:
        raise SystemExit(f"Unknown tower {args.tower!r}")

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the cached embeddings so the exported head is EXACTLY the head
    # that was measured.
    cache = Path(r"D:\onn_data\runs\vfm_probe_eval") / f"embeddings_{tower_spec.key}.npz"
    if not cache.is_file():
        raise SystemExit(f"Run evaluate_vfm_probes first; missing {cache}")
    with np.load(cache) as data:
        X_stacking = data["X_stacking"]
        y_stacking = data["y_stacking"]

    weight, bias, mean, std = fit_logistic_probe(X_stacking, y_stacking, X_stacking, args.l2_reg)
    print(f"probe: {weight.shape[0]} weights + 1 bias")

    tower = load_tower(args.tower, tower_spec.model_cls)
    model = VfmProbe(
        tower, weight, bias, mean, std,
        pix_mean=tower_spec.mean, pix_std=tower_spec.std, size=tower_spec.size,
    ).eval()

    # --- ONNX ---
    onnx_path = output_dir / "vfm_probe.onnx"
    dummy = torch.rand(1, 3, tower_spec.size, tower_spec.size)
    started = time.time()
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["image"], output_names=["probability"],
        dynamic_axes={"image": {0: "batch"}, "probability": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
        # The TorchScript exporter needs no onnxscript dependency, which
        # keeps the release toolchain small and reproducible.
        dynamo=False,
    )
    print(f"ONNX: {onnx_path.name} ({onnx_path.stat().st_size / 1e6:.1f} MB) "
          f"in {time.time() - started:.1f}s")

    # --- parity ---
    from onnxruntime import InferenceSession
    session = InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    test_images = []
    samples = discover_samples(data_root, args.split, IMAGE_EXTS)[:8]
    for path, _ in samples:
        arr = np.asarray(
            tower_spec.preprocess(decode_image(str(path)).image),
            dtype=np.float32,
        )
        test_images.append(arr)
    batch = np.stack(test_images)
    with torch.no_grad():
        torch_out = model(torch.from_numpy(batch)).numpy()
    onnx_out = session.run(None, {"image": batch})[0]
    error = float(np.abs(torch_out - onnx_out).max())
    print(f"ONNX parity: max |onnx - torch| = {error:.3e}")
    if error > 1e-3:
        raise SystemExit("ONNX parity failed")

    # --- TFLite ---
    tflite_path = output_dir / "vfm_probe.tflite"
    started = time.time()
    conversion_error = None
    produced_paths: dict[str, str] = {}
    work_dir = output_dir / "onnx2tf_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    # onnx2tf fetches a calibration image sample from a GitHub release and
    # parses it with np.load. The fetch returns an error page in this
    # environment, so np.load fails with a pickle error. onnx2tf uses a
    # LOCAL file of the same name when one exists in the working
    # directory, so provide a valid one - the content only has to be a
    # float32 array of the right shape.
    calibration = work_dir / "calibration_image_sample_data_20x128x128x3_float32.npy"
    if not calibration.is_file():
        rng = np.random.default_rng(0)
        np.save(calibration, rng.random((20, 128, 128, 3), dtype=np.float32))
    previous_cwd = Path.cwd()
    try:
        import os
        os.chdir(work_dir)
        from onnx2tf import convert
        convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(work_dir),
            # A frozen tower is 86M parameters; the INT8 graph is what a
            # phone can actually carry.
            output_integer_quantized_tflite=True,
            output_dynamic_range_quantized_tflite=True,
            quant_type="per-channel",
            batch_size=1,
            # onnxsim's simplifier struggles with a 343 MB graph and fails
            # with a pickle error; the un-simplified path converts fine.
            not_use_onnxsim=True,
            non_verbose=True,
        )
        for candidate, label in (
            (work_dir / "vfm_probe.tflite", "fp32"),
            (work_dir / "vfm_probe_int8.tflite", "int8"),
            (work_dir / "vfm_probe_dynamic_range.tflite", "dynamic"),
        ):
            if candidate.is_file():
                produced_paths[label] = str(candidate)
        if "fp32" in produced_paths:
            tflite_path.write_bytes(Path(produced_paths["fp32"]).read_bytes())
    except Exception as exc:  # pragma: no cover - toolchain dependent
        conversion_error = f"{type(exc).__name__}: {exc}"
        print(f"onnx2tf conversion failed: {conversion_error}")
    finally:
        import os
        os.chdir(previous_cwd)
    if tflite_path.is_file():
        print(f"TFLite fp32: {tflite_path.stat().st_size / 1e6:.1f} MB")
    for label, produced in produced_paths.items():
        size = Path(produced).stat().st_size / 1e6
        print(f"TFLite {label:<8}: {size:.1f} MB  ({produced})")

    summary = {
        "tower": args.tower,
        "frontend_id": tower_spec.frontend_id(),
        "input_shape": [1, tower_spec.size, tower_spec.size, 3],
        "probe_weights": int(weight.shape[0]),
        "onnx_bytes": onnx_path.stat().st_size,
        "tflite_bytes": tflite_path.stat().st_size if tflite_path.is_file() else None,
        "tflite_variants": {
            label: Path(path).stat().st_size for label, path in produced_paths.items()
        },
        "onnx_parity_max_error": error,
        "tflite_conversion_error": conversion_error,
        "conversion_seconds": round(time.time() - started, 1),
    }
    (output_dir / "vfm_probe_export.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

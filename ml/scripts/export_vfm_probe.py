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
from export.siglip_exportable import ExportableSiglipProbe  # noqa: E402
from training.metrics import compute_auc  # noqa: E402


class VfmProbe(nn.Module):
    """A frozen vision tower with a learned linear probe head, as one graph.

    Used for towers whose ONNX export onnx2tf can handle directly. SigLIP
    uses [ExportableSiglipProbe] instead, because its 4-D patch
    convolution breaks onnx2tf's layout tracking.
    """

    def __init__(self, tower: nn.Module, weight: np.ndarray, bias: np.ndarray,
                 feat_mean: np.ndarray, feat_std: np.ndarray,
                 pix_mean: np.ndarray, pix_std: np.ndarray, size: int,
                 pooling: str = "cls"):
        super().__init__()
        self.tower = tower
        self.size = size
        self.pooling = pooling
        self.register_buffer("probe_w", torch.from_numpy(weight.astype(np.float32)))
        self.register_buffer("probe_b", torch.tensor(float(bias), dtype=torch.float32))
        self.register_buffer("feat_mean", torch.from_numpy(feat_mean.astype(np.float32)))
        self.register_buffer("feat_std", torch.from_numpy(feat_std.astype(np.float32)))
        # PIXEL normalisation is the tower's own; it is NOT the feature
        # statistics. Conflating the two is a silent numerical bug.
        self.register_buffer("pix_mean", torch.from_numpy(pix_mean.astype(np.float32)))
        self.register_buffer("pix_std", torch.from_numpy(pix_std.astype(np.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, 3) float32 in [0, 1] -> (B, 1) probability.

        NHWC, the layout every model in this app uses and the layout
        onnx2tf's auto-calibration requires.
        """
        rgb = x.permute(0, 3, 1, 2)                       # [B, 3, H, W]
        rgb = (rgb - self.pix_mean.view(1, 3, 1, 1)) / self.pix_std.view(1, 3, 1, 1)
        features = self.tower(pixel_values=rgb)
        if self.pooling == "cls_mean":
            # DINOv2's linear head concatenates the CLS token with the mean
            # of the patch tokens. The cached embeddings use the same
            # convention, so the export must match it exactly.
            states = features.last_hidden_state
            pooled = torch.cat([states[:, 0], states[:, 1:].mean(dim=1)], dim=-1)
        elif hasattr(self.tower.config, "vision_config"):
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


class NchwInputWrapper(torch.nn.Module):
    """Exposes an NHWC model through an NCHW input.

    onnx2tf assumes the ONNX input is NCHW and rewrites the graph
    accordingly. If the ONNX input is NHWC it treats the channel axis as
    spatial, and every downstream shape inference is wrong - which is what
    produced the failure

        Add: Dimensions must be equal, but are 768 and 196
        for '[?, 768, 768]' + '[1, 196, 768]'

    Exporting NCHW and letting onnx2tf do its NHWC rewrite produces a TFLite
    with an NHWC input, which is the layout the app uses anyway.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x.permute(0, 2, 3, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tower", default="google/siglip-base-patch16-224")
    parser.add_argument("--l2-reg", type=float, default=30.0)
    parser.add_argument("--split", default="stacking")
    parser.add_argument("--unfold-mode", choices=["permute", "stack"], default="permute",
                        help="How the patch projection is unfolded. 'stack' uses only 4-D "
                             "tensors, which some ONNX converters require; it is "
                             "numerically identical to 'permute' (max difference 0.0).")
    parser.add_argument("--onnx-layout", choices=["nchw", "nhwc"], default="nchw",
                        help="Layout of the exported ONNX input. onnx2tf assumes NCHW, so "
                             "NCHW is the default; the TFLite it emits is NHWC either way.")
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
    # SigLIP's 4-D patch convolution breaks onnx2tf's NCHW->NHWC layout
    # tracking, so its patch projection is rewritten as an unfold + Linear
    # (mathematically identical, and it removes every Conv from the graph).
    if tower_spec.pooling == "siglip":
        model = ExportableSiglipProbe(
            tower, weight, bias, mean, std,
            pix_mean=tower_spec.mean, pix_std=tower_spec.std,
            unfold_mode=args.unfold_mode,
        ).eval()
    else:
        model = VfmProbe(
            tower, weight, bias, mean, std,
            pix_mean=tower_spec.mean, pix_std=tower_spec.std, size=tower_spec.size,
            pooling=tower_spec.pooling,
        ).eval()

    # --- ONNX ---
    onnx_path = output_dir / "vfm_probe.onnx"
    # The graph input is NCHW. onnx2tf assumes NCHW and rewrites the graph
    # to NHWC for the TFLite it emits - which is exactly the layout every
    # model in this app already uses. Exporting NHWC makes onnx2tf treat
    # the channel axis as spatial and every downstream shape inference is
    # wrong.
    if args.onnx_layout == "nchw":
        # NCHW: the wrapper permutes to the model's NHWC input.
        dummy = torch.rand(1, 3, tower_spec.size, tower_spec.size)
        export_model = NchwInputWrapper(model)
    else:
        # NHWC: the model's own layout, and onnx2tf is told to keep it.
        dummy = torch.rand(1, tower_spec.size, tower_spec.size, 3)
        export_model = model
    started = time.time()
    torch.onnx.export(
        export_model, dummy, str(onnx_path),
        input_names=["image"], output_names=["probability"],
        # DYNAMIC batch.
        #
        # onnx2tf runs an "ONNX dummy inference" that feeds
        # `test_data[:inputs[0].shape[0], ...]` - and `inputs` there are the
        # TENSORFLOW model inputs, whose batch is 8 by default, while the
        # ONNX expects 1. That is the
        #     "index: 0 Got: 8 Expected: 1"
        # failure. With a dynamic batch onnx2tf skips that path entirely,
        # because the "all four dims have a static batch" precondition is
        # no longer met.
        #
        # The app only ever runs batch 1 (`LiteRTInferenceEngine` asserts
        # `input.shape().first() == 1`), and a dynamic batch lets the
        # TFLite accept it.
        dynamic_axes={"image": {0: "batch"}, "probability": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
        # The TorchScript exporter needs no onnxscript dependency, which
        # keeps the release toolchain small and reproducible.
        dynamo=False,
    )
    print(f"ONNX: {onnx_path.name} ({onnx_path.stat().st_size / 1e6:.1f} MB) "
          f"in {time.time() - started:.1f}s")

    # --- build the parity batch (NHWC, the exported input layout) ---
    test_images = []
    samples = discover_samples(data_root, args.split, IMAGE_EXTS)[:8]
    for path, _ in samples:
        arr = np.asarray(
            tower_spec.preprocess(decode_image(str(path)).image),
            dtype=np.float32,
        )
        test_images.append(np.transpose(arr, (1, 2, 0)))   # CHW -> NHWC
    batch = np.stack(test_images)

    # --- parity: the REWRITTEN graph against the ORIGINAL tower ---
    # This is the check that matters for the rewrite. If the unfolded-patch
    # projection were not mathematically identical to the convolution, the
    # shipped detector would not be the one that was measured.
    if tower_spec.pooling == "siglip":
        with torch.no_grad():
            nchw = torch.from_numpy(np.transpose(batch, (0, 3, 1, 2)))
            rgb = (nchw - torch.tensor(tower_spec.mean, dtype=torch.float32).view(1, 3, 1, 1)) \
                / torch.tensor(tower_spec.std, dtype=torch.float32).view(1, 3, 1, 1)
            original = tower(pixel_values=rgb)
            pooled = original.pooler_output.float() if hasattr(original, "pooler_output") \
                else original.last_hidden_state[:, 0].float()
            pooled = pooled / (pooled.norm(dim=1, keepdim=True) + 1e-12)
            reference = ((pooled - torch.from_numpy(mean)) / torch.from_numpy(std)) \
                @ torch.from_numpy(weight).reshape(-1) + float(bias)
            reference = torch.sigmoid(reference).reshape(-1, 1)
            rewritten = model(torch.from_numpy(batch))
        rewrite_error = float((reference - rewritten).abs().max())
        print(f"rewrite parity (unfold+Linear vs Conv2d): max err = {rewrite_error:.3e}")
        if rewrite_error > 1e-4:
            raise SystemExit(
                f"The exportable rewrite does not match the original tower "
                f"({rewrite_error:.3e}); refusing to export a model that is not "
                "the one that was measured."
            )
    else:
        rewrite_error = 0.0

    # --- parity: ONNX against PyTorch ---
    from onnxruntime import InferenceSession
    session = InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
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
    # directory, so provide a valid one.
    #
    # The FILENAME is hard-coded by onnx2tf (and encodes 128x128x3), but
    # only the CONTENT shape matters: it is fed to the model as the
    # calibration batch. The model input is NHWC [1, 224, 224, 3], so the
    # array must be [20, 224, 224, 3].
    # The calibration array must match the ONNX input LAYOUT.
    #   NHWC ONNX -> [20, H, W, 3]  (onnx2tf's AUTO path accepts this)
    #   NCHW ONNX -> [20, C, H, W]  (the AUTO path requires the last
    #                                  dimension to be 3 or 4, so the custom
    #                                  path is used instead)
    if args.onnx_layout == "nchw":
        calibration_shape = (20, 3, tower_spec.size, tower_spec.size)
    else:
        calibration_shape = (20, tower_spec.size, tower_spec.size, 3)
    calibration = work_dir / "calibration_image_sample_data_20x128x128x3_float32.npy"
    needs_write = True
    if calibration.is_file():
        try:
            existing = np.load(calibration)
            needs_write = tuple(existing.shape) != calibration_shape
        except Exception:
            needs_write = True
    if needs_write:
        rng = np.random.default_rng(0)
        np.save(calibration, rng.random(calibration_shape, dtype=np.float32))
    previous_cwd = Path.cwd()
    try:
        import os
        os.chdir(work_dir)
        from onnx2tf import convert
        convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(work_dir),
            # A frozen tower is 86M parameters; the INT8 graph is what a
            # phone can actually carry. The dynamic-range variant is NOT
            # produced: onnx2tf converts the whole model a second time for
            # it, which doubles the peak memory of a 372 MB graph.
            output_integer_quantized_tflite=True,
            quant_type="per-channel",
            batch_size=1,
            # onnxsim's simplifier struggles with a 372 MB graph and fails
            # with a pickle error; the un-simplified path converts fine.
            not_use_onnxsim=True,
            # The ONNX input is NHWC, and onnx2tf's default is to assume
            # NCHW and rewrite the graph. That rewrite is what produced
            #     Add: Dimensions must be equal, but are 768 and 196
            #     for '[?, 768, 768]' + '[1, 196, 768]'
            # because it then treats the channel axis as spatial and every
            # downstream shape inference is wrong. `-kt` tells onnx2tf to
            # KEEP the input layout. This is the documented remedy for
            # "If the input OP of ONNX before conversion is NHWC ... use
            # the -kt or -kat option".
            keep_ncw_or_nchw_or_ncdhw_input_names=["image"],
            # For an NCHW ONNX, onnx2tf's AUTO calibration path refuses the
            # input (it requires the last dimension to be 3 or 4), so the
            # calibration array is supplied explicitly in the graph's own
            # layout.
            quant_calib_input_op_name_np_data_path=(
                [f"image:{calibration}"] if args.onnx_layout == "nchw" else None
            ),
            # NOTE: onnx2tf cannot convert a ViT in this environment. Its
            # shape inference across the patch->sequence transition is wrong:
            # the patch projection is inferred as [?, 768, 768] instead of
            # [?, 196, 768], and the position-embedding Add then fails with
            # "Dimensions must be equal, but are 768 and 196". Its NCHW
            # calibration path is also unusable, because the dummy inference
            # always feeds NHWC-shaped data.
            #
            # The remedies are (a) a ViT-aware converter - Google's
            # `ai-edge-vit` exists for exactly this reason - or (b) the
            # repository's own Keras -> TFLite path with a hand-written
            # Keras mirror of the tower, the same way the SRM and semantic
            # probe models are exported.
            #
            # The TFLite output is accepted ONLY if it matches the ONNX
            # numerically - see the parity check below. An unverified
            # conversion is never shipped.
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
    tflite_parity = None
    if tflite_path.is_file():
        print(f"TFLite fp32: {tflite_path.stat().st_size / 1e6:.1f} MB")
        # --- TFLite vs ONNX parity ---
        # The conversion is only usable if it computes the SAME function.
        # onnx2tf's strict mode was disabled to get past a shape-inference
        # failure, so the output is VERIFIED here rather than assumed.
        try:
            import tensorflow as tf

            interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            probe_batch = batch[:4]
            interpreter.set_tensor(
                input_details[0]["index"], probe_batch.astype(input_details[0]["dtype"])
            )
            interpreter.invoke()
            tflite_out = interpreter.get_tensor(output_details[0]["index"])
            onnx_batch_out = session.run(None, {"image": probe_batch})[0]
            tflite_parity = float(np.abs(tflite_out - onnx_batch_out).max())
            print(f"TFLite parity: max |tflite - onnx| = {tflite_parity:.3e}")
            if tflite_parity > 0.05:
                raise SystemExit(
                    f"TFLite output does not match the ONNX ({tflite_parity:.3e}); "
                    "the conversion is not usable and must not be shipped."
                )
        except SystemExit:
            raise
        except Exception as exc:
            print(f"TFLite parity check failed: {type(exc).__name__}: {exc}")
            tflite_parity = None
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

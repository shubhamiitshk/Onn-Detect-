#!/usr/bin/env python3
"""Train Engine A (`SRMForensicCNN`) with DCPT Paired Consistency + Native PatchCraft Mosaics.

Drop-in script for `ml/scripts/train_srm_dcpt_patchcraft.py` in the ONN-DETECT repository.

What this script does
---------------------
1. Keeps 100% compatibility with the existing repository contracts:
   - Uses `StrictImageDataset` (`ml/src/data/image_dataset.py`), `SRMForensicCNN`
     (`ml/src/models/srm_forensic_cnn.py`), `Trainer` (`ml/src/training/trainer.py`),
     and `load_verified_checkpoint` (`ml/src/training/checkpoint.py`).
   - Emits checkpoints with `schema_version=1` and the exact `contract` dictionary
     expected by `ml/scripts/export_all_models.py` and `ml/scripts/release_bundle.py`.
   - Zero APK size increase and zero Android model-graph changes (`srm_forensic_cnn_int8.tflite`
     keeps its `[1, 128, 128, 3]` float32 input and `[1, 1]` probability output).

2. Implements DCPT (arXiv:2604.10102) Paired Consistency Training:
   - For each training image, generates `(clean_view, degraded_view)` from the SAME
     underlying photo using random JPEG compression (`QF=30..90`), bilinear resize
     round-trips (`160..320px`), Gaussian blur (`sigma=0.5..1.5`), and sensor noise.
   - Stacks `[clean_tensor, degraded_tensor]` into a single `(2, 3, H, W)` tensor inside
     `DCPTPatchCraftTrainTransform` so `StrictImageDataset.__getitem__`'s
     `isinstance(tensor, torch.Tensor)` and `torch.isfinite(tensor).all()` checks pass
     without modifying `StrictImageDataset`.
   - Unpacks both views inside `DCPTTrainer.train_epoch()` and optimizes:
       L = 0.5 * (BCE(logits_clean, y) + BCE(logits_deg, y))
           + lambda_pred * BernoulliSymmetricKL(logits_clean, logits_deg)
           + lambda_feat * (1 - cos_sim(z_clean_64d, z_deg_64d))

3. Implements Native-Resolution PatchCraft (arXiv:2311.12397) + NPR (CVPR 2024) Mosaics:
   - Extracts non-overlapping `32x32` tiles from the native-resolution (`pow2cap1024`)
     image BEFORE any bilinear downsampling, ranks tiles by integer BT.601 luma L1
     gradient energy, and assembles a `128x128` (`4x4` tile) high-texture mosaic with
     zero interpolation so the 2x2 generator upsampling grid survives into Engine A.
   - Supports `--patchcraft-mode {mosaic,crop,off}` so you can run clean A/B ablations
     between DCPT alone (`--patchcraft-mode off`) and DCPT + PatchCraft (`--patchcraft-mode mosaic`).

4. Evaluates the Full Sealed Degradation Matrix:
   - Evaluates the best checkpoint across `clean`, `whatsapp_512_q85`, `jpeg_qf90`,
     `jpeg_qf70`, `jpeg_qf50`, `jpeg_qf30`, `downscale_0_5x`, and `blur_sigma_1_0`,
     reporting `auc`, `eer`, `tpr_at_fpr_1pct`, and `tpr_at_fpr_5pct` for every slice.

Usage
-----
    # Full DCPT + PatchCraft 20-channel (SRM + NPR + RGB) training at 128x128:
    python3 ml/scripts/train_srm_dcpt_patchcraft.py \\
        --config ml/configs/srm_cnn_openfake.yaml \\
        --output-dir ml/checkpoints/srm_dcpt_patchcraft_v8 \\
        --patchcraft-mode mosaic \\
        --dcpt-logit-weight 0.5 \\
        --dcpt-feature-weight 0.5

    # Ablation: DCPT only (standard bilinear resize, no PatchCraft mosaic):
    python3 ml/scripts/train_srm_dcpt_patchcraft.py \\
        --config ml/configs/srm_cnn_openfake.yaml \\
        --output-dir ml/checkpoints/srm_dcpt_only_v8 \\
        --patchcraft-mode off

    # Run built-in self-test (no dataset or GPU required):
    python3 ml/scripts/train_srm_dcpt_patchcraft.py --self-test
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np
from PIL import Image, ImageFilter

SCRIPT_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPT_DIR.parent
if (ML_ROOT / "src").exists():
    sys.path.insert(0, str(ML_ROOT / "src"))
if (ML_ROOT / "scripts").exists():
    sys.path.insert(0, str(ML_ROOT / "scripts"))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.amp import autocast
    from torch.utils.data import DataLoader

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - lightweight CI self-test without PyTorch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    autocast = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


# ============================================================================
# Native-Resolution PatchCraft Mosaic & DCPT Paired Degradation Transforms
# ============================================================================


def extract_patchcraft_mosaic(
    img: Image.Image,
    mosaic_size: int = 128,
    tile_size: int = 32,
    select: str = "rich",
) -> Image.Image:
    """Assembles a `mosaic_size x mosaic_size` mosaic from native `tile_size x tile_size` tiles.

    Matches `PhaseSpectrumAndPatchCraftAndroid.extractPatchCraftMosaics` bit-for-bit:
      - Integer BT.601 luma `(299*R + 587*G + 114*B) // 1000`
      - L1 horizontal + vertical adjacent-pixel difference energy per `32x32` tile
      - Deterministic tie-breaking by `(energy, y0, x0)`
      - Zero interpolation when copying tiles into the `128x128` (`4x4` tile) canvas.
    """
    if mosaic_size % tile_size != 0:
        raise ValueError(f"mosaic_size ({mosaic_size}) must be divisible by tile_size ({tile_size})")
    grid_n = mosaic_size // tile_size
    needed = grid_n * grid_n

    rgb = img.convert("RGB")
    w, h = rgb.size
    if w < mosaic_size or h < mosaic_size:
        rgb = rgb.resize((max(w, mosaic_size), max(h, mosaic_size)), Image.Resampling.NEAREST)
        w, h = rgb.size

    arr = np.asarray(rgb, dtype=np.int32)
    luma = (arr[..., 0] * 299 + arr[..., 1] * 587 + arr[..., 2] * 114) // 1000

    tiles: list[tuple[int, int, int]] = []
    n_rows = h // tile_size
    n_cols = w // tile_size
    for ry in range(n_rows):
        y0 = ry * tile_size
        for cx in range(n_cols):
            x0 = cx * tile_size
            patch = luma[y0 : y0 + tile_size, x0 : x0 + tile_size]
            gx = int(np.abs(patch[:, 1:] - patch[:, :-1]).sum())
            gy = int(np.abs(patch[1:, :] - patch[:-1, :]).sum())
            tiles.append((gx + gy, y0, x0))

    if select == "poor":
        ordered = sorted(tiles, key=lambda item: (item[0], item[1], item[2]))
    else:
        ordered = sorted(tiles, key=lambda item: (-item[0], item[1], item[2]))

    selected = [ordered[i % len(ordered)] for i in range(needed)]
    rgb_u8 = arr.astype(np.uint8)
    canvas = np.zeros((mosaic_size, mosaic_size, 3), dtype=np.uint8)
    for idx in range(needed):
        dst_y = (idx // grid_n) * tile_size
        dst_x = (idx % grid_n) * tile_size
        _, sy, sx = selected[idx]
        canvas[dst_y : dst_y + tile_size, dst_x : dst_x + tile_size] = rgb_u8[
            sy : sy + tile_size, sx : sx + tile_size
        ]
    return Image.fromarray(canvas)


def extract_single_high_texture_crop(img: Image.Image, patch_size: int = 128, stride: int = 64) -> Image.Image:
    """Single highest-gradient `128x128` native crop (compatible with `srm_forensic_cnn.py`)."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    if w < patch_size or h < patch_size:
        return rgb.resize((patch_size, patch_size), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32)
    luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    gx = np.abs(np.diff(luma, axis=1, append=luma[:, -1:]))
    gy = np.abs(np.diff(luma, axis=0, append=luma[-1:, :]))
    energy = gx + gy
    best_hi = -1.0
    hi_box = (0, 0, patch_size, patch_size)
    ys = list(range(0, h - patch_size + 1, stride)) or [0]
    xs = list(range(0, w - patch_size + 1, stride)) or [0]
    for y in ys:
        for x in xs:
            score = float(energy[y : y + patch_size, x : x + patch_size].mean())
            if score > best_hi:
                best_hi = score
                hi_box = (x, y, x + patch_size, y + patch_size)
    return rgb.crop(hi_box)


def apply_spatial_frontend(img: Image.Image, size: int, patchcraft_mode: str = "mosaic") -> Image.Image:
    """Converts a native-resolution PIL image into the `size x size` input expected by Engine A."""
    if patchcraft_mode == "mosaic":
        tile = 32 if size >= 64 and size % 32 == 0 else max(8, size // 4)
        return extract_patchcraft_mosaic(img, mosaic_size=size, tile_size=tile, select="rich")
    if patchcraft_mode == "crop":
        return extract_single_high_texture_crop(img, patch_size=size, stride=max(16, size // 2))
    if patchcraft_mode == "off":
        return img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    raise ValueError(f"Unsupported patchcraft_mode: {patchcraft_mode!r}")


class PairedDegradationGenerator:
    """Produces a degraded view from the SAME underlying native-resolution image (DCPT)."""

    def __init__(
        self,
        p_degrade: float = 0.85,
        min_jpeg_q: int = 30,
        max_jpeg_q: int = 90,
        p_downscale: float = 0.60,
        p_blur: float = 0.25,
    ) -> None:
        self.p_degrade = float(p_degrade)
        self.min_jpeg_q = int(min_jpeg_q)
        self.max_jpeg_q = int(max_jpeg_q)
        self.p_downscale = float(p_downscale)
        self.p_blur = float(p_blur)

    def __call__(self, clean_img: Image.Image) -> Image.Image:
        view = clean_img.convert("RGB")
        if random.random() >= self.p_degrade:
            return view.copy()

        w, h = view.size
        # 1. Social-media / WhatsApp resolution scaling round-trip
        if random.random() < self.p_downscale and min(w, h) >= 64:
            scale = random.uniform(0.45, 0.85)
            dw = max(32, int(round(w * scale)))
            dh = max(32, int(round(h * scale)))
            view = view.resize((dw, dh), Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BILINEAR)

        # 2. Optional mild optical/motion blur before JPEG quantization
        if random.random() < self.p_blur:
            sigma = random.uniform(0.4, 1.2)
            view = view.filter(ImageFilter.GaussianBlur(radius=sigma))

        # 3. JPEG block DCT quantization (QF 30..90)
        q = random.randint(self.min_jpeg_q, self.max_jpeg_q)
        buf = io.BytesIO()
        view.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def _pil_to_chw_float(img: Image.Image) -> Any:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(np.ascontiguousarray(arr))


class DCPTPatchCraftTrainTransform:
    """Returns a stacked `(2, 3, size, size)` tensor `[clean_tensor, degraded_tensor]`.

    Returning a single `torch.Tensor` satisfies `StrictImageDataset.__getitem__`'s
    `isinstance(tensor, torch.Tensor)` and `torch.isfinite(tensor).all()` checks without
    needing any changes to `ml/src/data/image_dataset.py`.
    """

    def __init__(
        self,
        size: int = 128,
        patchcraft_mode: str = "mosaic",
        p_degrade: float = 0.85,
        noise_sigma_max: float = 0.03,
    ) -> None:
        self.size = int(size)
        self.patchcraft_mode = patchcraft_mode
        self.degrader = PairedDegradationGenerator(p_degrade=p_degrade)
        self.noise_sigma_max = float(noise_sigma_max)

    def __call__(self, image: Image.Image) -> Any:
        rgb = image.convert("RGB")
        if random.random() < 0.5:
            rgb = rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            rgb = rgb.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        # Degrade the SAME oriented image before applying the spatial frontend
        degraded_rgb = self.degrader(rgb)

        clean_patch = apply_spatial_frontend(rgb, self.size, self.patchcraft_mode)
        deg_patch = apply_spatial_frontend(degraded_rgb, self.size, self.patchcraft_mode)

        t_clean = _pil_to_chw_float(clean_patch)
        t_deg = _pil_to_chw_float(deg_patch)

        if self.noise_sigma_max > 0.0 and random.random() < 0.35:
            sigma = random.uniform(0.005, self.noise_sigma_max)
            t_deg = (t_deg + torch.randn_like(t_deg) * sigma).clamp(0.0, 1.0)

        return torch.stack([t_clean, t_deg], dim=0)


class DCPTPatchCraftEvalTransform:
    """Deterministic single-view `(3, size, size)` transform for validation and test splits."""

    def __init__(
        self,
        size: int = 128,
        patchcraft_mode: str = "mosaic",
        pre_degrade_fn: Callable[[Image.Image], Image.Image] | None = None,
    ) -> None:
        self.size = int(size)
        self.patchcraft_mode = patchcraft_mode
        self.pre_degrade_fn = pre_degrade_fn

    def __call__(self, image: Image.Image) -> Any:
        rgb = image.convert("RGB")
        if self.pre_degrade_fn is not None:
            rgb = self.pre_degrade_fn(rgb)
        patch = apply_spatial_frontend(rgb, self.size, self.patchcraft_mode)
        return _pil_to_chw_float(patch)


# ============================================================================
# DCPT Losses & Penultimate Feature Extraction on `SRMForensicCNN`
# ============================================================================


def forward_srm_logits_and_features(model: Any, x: Any) -> Tuple[Any, Any]:
    """Runs `SRMForensicCNN` and returns `(logits, penultimate_64d_features)`."""
    if hasattr(model, "srm") and hasattr(model, "backbone") and hasattr(model, "head"):
        srm_out = model.srm(x)
        feat_map = model.backbone(srm_out)
        # model.head: [0] AdaptiveAvgPool2d(1), [1] Flatten(), [2] Linear(256, 64),
        #             [3] ReLU(), [4] Dropout(0.3), [5] Linear(64, 1), [6] Sigmoid()
        pooled = model.head[1](model.head[0](feat_map))
        z_64 = model.head[3](model.head[2](pooled))
        logits = model.head[5](model.head[4](z_64))
        return logits, z_64
    logits = model.forward_logits(x)
    return logits, None


def dcpt_bernoulli_kl_loss(clean_logits: Any, degraded_logits: Any, temperature: float = 1.0) -> Any:
    """Symmetric KL divergence between two binary Bernoulli distributions `sigmoid(logits / T)`."""
    eps = 1e-6
    p = torch.sigmoid(clean_logits / temperature).clamp(eps, 1.0 - eps)
    q = torch.sigmoid(degraded_logits / temperature).clamp(eps, 1.0 - eps)
    kl_pq = p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))
    kl_qp = q * torch.log(q / p) + (1.0 - q) * torch.log((1.0 - q) / (1.0 - p))
    return 0.5 * (kl_pq + kl_qp).mean() * (temperature ** 2)


def dcpt_cosine_feature_loss(z_clean: Any, z_degraded: Any) -> Any:
    """Cosine distance `mean(1 - cos(z_clean, z_degraded))` on the 64-d penultimate embedding."""
    return (1.0 - F.cosine_similarity(z_clean, z_degraded, dim=-1, eps=1e-8)).mean()


# ============================================================================
# Degradation Matrix Builders (for Sealed Test Evaluation)
# ============================================================================


def _degrade_jpeg(quality: int) -> Callable[[Image.Image], Image.Image]:
    def _fn(img: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    return _fn


def _degrade_whatsapp_512_q85(img: Image.Image) -> Image.Image:
    rgb = img.convert("RGB")
    w, h = rgb.size
    if max(w, h) > 512:
        scale = 512.0 / float(max(w, h))
        rgb = rgb.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _degrade_downscale(factor: float = 0.5) -> Callable[[Image.Image], Image.Image]:
    def _fn(img: Image.Image) -> Image.Image:
        rgb = img.convert("RGB")
        w, h = rgb.size
        dw, dh = max(16, int(round(w * factor))), max(16, int(round(h * factor)))
        return rgb.resize((dw, dh), Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BILINEAR)

    return _fn


def _degrade_blur(sigma: float = 1.0) -> Callable[[Image.Image], Image.Image]:
    def _fn(img: Image.Image) -> Image.Image:
        return img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=float(sigma)))

    return _fn


DEGRADATION_CONDITIONS: dict[str, Callable[[Image.Image], Image.Image] | None] = {
    "clean": None,
    "whatsapp_512_q85": _degrade_whatsapp_512_q85,
    "jpeg_qf90": _degrade_jpeg(90),
    "jpeg_qf70": _degrade_jpeg(70),
    "jpeg_qf50": _degrade_jpeg(50),
    "jpeg_qf30": _degrade_jpeg(30),
    "downscale_0_5x": _degrade_downscale(0.5),
    "blur_sigma_1_0": _degrade_blur(1.0),
}


def compute_low_fpr_operating_points(labels: list[int], scores: list[float]) -> dict[str, float]:
    """Computes TPR at 1% FPR and TPR at 5% FPR using exact empirical quantiles on real samples."""
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    real_scores = s[y == 0]
    fake_scores = s[y == 1]
    if len(real_scores) == 0 or len(fake_scores) == 0:
        return {"tpr_at_fpr_1pct": 0.0, "tpr_at_fpr_5pct": 0.0, "threshold_at_fpr_1pct": 1.0}
    thr_1pct = float(np.quantile(real_scores, 0.99))
    thr_5pct = float(np.quantile(real_scores, 0.95))
    tpr_1pct = float(np.mean(fake_scores >= thr_1pct))
    tpr_5pct = float(np.mean(fake_scores >= thr_5pct))
    return {
        "tpr_at_fpr_1pct": round(tpr_1pct, 6),
        "tpr_at_fpr_5pct": round(tpr_5pct, 6),
        "threshold_at_fpr_1pct": round(thr_1pct, 6),
    }


# ============================================================================
# DCPTTrainer (Subclasses Repository `Trainer` When PyTorch Is Present)
# ============================================================================


def build_dcpt_trainer_class() -> Any:
    from training.trainer import Trainer

    class DCPTTrainer(Trainer):
        """Extends `Trainer` with paired DCPT logit KL + penultimate feature cosine losses."""

        def __init__(
            self,
            model: Any,
            train_loader: Any,
            val_loader: Any,
            config: dict,
            device: Any = None,
            dcpt_logit_weight: float = 0.5,
            dcpt_feature_weight: float = 0.5,
            dcpt_temperature: float = 1.0,
        ) -> None:
            super().__init__(model, train_loader, val_loader, config, device=device)
            self.dcpt_logit_weight = float(dcpt_logit_weight)
            self.dcpt_feature_weight = float(dcpt_feature_weight)
            self.dcpt_temperature = float(dcpt_temperature)
            self.history["dcpt_logit_kl"] = []
            self.history["dcpt_feature_cos"] = []

        def train_epoch(self) -> float:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            total, count, accumulated_samples = 0.0, 0, 0
            total_kl, total_cos = 0.0, 0.0
            length = len(self.train_loader)

            for index, (inputs, targets) in enumerate(self.train_loader):
                inputs = inputs.to(self.device, dtype=torch.float32)
                targets = targets.to(self.device, dtype=torch.float32).reshape(-1, 1)
                if not torch.isfinite(inputs).all():
                    raise ValueError("Non-finite training inputs")
                if not torch.isfinite(targets).all() or not ((targets == 0) | (targets == 1)).all():
                    raise ValueError("Training requires finite binary labels")

                with autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                    if inputs.ndim == 5 and inputs.shape[1] == 2:
                        clean_in = inputs[:, 0]
                        deg_in = inputs[:, 1]
                        clean_logits, z_clean = forward_srm_logits_and_features(self.model, clean_in)
                        deg_logits, z_deg = forward_srm_logits_and_features(self.model, deg_in)

                        bce_loss = 0.5 * (
                            self.criterion(clean_logits, targets) + self.criterion(deg_logits, targets)
                        )
                        kl_loss = dcpt_bernoulli_kl_loss(
                            clean_logits, deg_logits, temperature=self.dcpt_temperature
                        )
                        loss = bce_loss + self.dcpt_logit_weight * kl_loss
                        cos_val = 0.0
                        if z_clean is not None and z_deg is not None and self.dcpt_feature_weight > 0.0:
                            cos_loss = dcpt_cosine_feature_loss(z_clean, z_deg)
                            loss = loss + self.dcpt_feature_weight * cos_loss
                            cos_val = float(cos_loss.detach())
                        kl_val = float(kl_loss.detach())
                    else:
                        clean_logits = self.model.forward_logits(inputs)
                        loss = self.criterion(clean_logits, targets)
                        kl_val, cos_val = 0.0, 0.0

                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss; checkpoint will not be written")

                batch_samples = targets.shape[0]
                accumulated_samples += batch_samples
                self.scaler.scale(loss * batch_samples).backward()

                if (index + 1) % self.grad_accum_steps == 0 or index + 1 == length:
                    self.scaler.unscale_(self.optimizer)
                    for parameter in self.model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(accumulated_samples)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    accumulated_samples = 0

                total += float(loss.detach()) * batch_samples
                total_kl += kl_val * batch_samples
                total_cos += cos_val * batch_samples
                count += batch_samples

            self.history["dcpt_logit_kl"].append(total_kl / max(count, 1))
            self.history["dcpt_feature_cos"].append(total_cos / max(count, 1))
            return total / max(count, 1)

    return DCPTTrainer


# ============================================================================
# CLI Entrypoint & Degradation Matrix Evaluation
# ============================================================================


def patchcraft_frontend_id(size: int, patchcraft_mode: str) -> str:
    from data.image_frontend import frontend_id

    base = frontend_id(size)
    if patchcraft_mode == "off":
        return base
    return f"{base}-patchcraft-{patchcraft_mode}-tile32"


def run_training(args: argparse.Namespace) -> None:
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for full training; run with `--self-test` in lightweight CI.")

    import yaml
    from data.contracts import sha256_file, validate_dataset_manifest
    from data.image_dataset import StrictImageDataset
    from models.srm_forensic_cnn import SRMForensicCNN
    from training.checkpoint import load_verified_checkpoint
    from training.metrics import compute_accuracy, compute_auc, compute_eer
    from training.provenance import training_code_hash
    from utils.device import get_device
    from utils.seed import set_global_seed

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for flag in ("epochs", "batch_size", "num_workers", "amp"):
        val = getattr(args, flag, None)
        if val is not None:
            config["training"][flag] = val
    if args.lr is not None:
        config["optimizer"]["lr"] = args.lr
    if getattr(args, "patience", None) is not None:
        config["early_stopping"]["patience"] = args.patience
    if getattr(args, "data_dir", None):
        config["data"]["root_dir"] = str(Path(args.data_dir).resolve())

    data_cfg = config["data"]
    root = Path(data_cfg["root_dir"])
    root = root if root.is_absolute() else (ML_ROOT / root).resolve()
    provenance = validate_dataset_manifest(root)
    config["data"]["root_dir"] = str(root)
    config["checkpoint_dir"] = str(Path(args.output_dir).resolve())
    Path(config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    name = args.model_name
    config["model_name"] = name
    if (
        not args.resume
        and not args.eval_only
        and any((Path(config["checkpoint_dir"]) / f"{name}_{kind}.pth").exists() for kind in ("best", "last"))
    ):
        raise FileExistsError(
            "Existing model checkpoints are preserved; choose a new --output-dir or explicitly --resume"
        )

    device = get_device() if args.device == "auto" else torch.device(args.device)
    set_global_seed(config.get("seed", 42))

    size = int(data_cfg["img_size"])
    model = SRMForensicCNN(include_npr=args.include_npr, include_rgb=args.include_rgb)

    train_tf = DCPTPatchCraftTrainTransform(
        size=size,
        patchcraft_mode=args.patchcraft_mode,
        p_degrade=args.dcpt_degrade_prob,
    )
    eval_tf = DCPTPatchCraftEvalTransform(size=size, patchcraft_mode=args.patchcraft_mode)

    datasets = {
        "train": StrictImageDataset(
            root,
            "train",
            size,
            transform=train_tf,
            cache_in_memory=data_cfg.get("cache_in_memory", False),
            cache_bytes=data_cfg.get("cache_bytes", 32 * 1024 * 1024),
        ),
        "val": StrictImageDataset(
            root,
            "val",
            size,
            transform=eval_tf,
            cache_in_memory=data_cfg.get("cache_in_memory", False),
            cache_bytes=data_cfg.get("cache_bytes", 32 * 1024 * 1024),
        ),
    }

    shape = [1, size, size, 3]
    config["contract"] = {
        "frontend_id": patchcraft_frontend_id(size, args.patchcraft_mode),
        "input_shape": shape,
        "architecture": type(model).__name__,
        "srm_channels": int(model.srm.conv.out_channels),
        "patchcraft_mode": args.patchcraft_mode,
        "dcpt_logit_weight": float(args.dcpt_logit_weight),
        "dcpt_feature_weight": float(args.dcpt_feature_weight),
        "label_convention": "real=0_fake=1",
        "dataset_manifest_hash": provenance["manifest_sha256"],
        "source": provenance["source"],
        "revision": provenance["revision"],
        "training_code_hash": training_code_hash(),
        "audio_crop_policy": None,
    }

    tc = config["training"]
    workers = int(tc.get("num_workers", 0))
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(tc["batch_size"]),
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = int(tc.get("prefetch_factor", 2))
        loader_kwargs["persistent_workers"] = False

    train_loader = DataLoader(datasets["train"], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(datasets["val"], shuffle=False, **loader_kwargs)

    DCPTTrainerCls = build_dcpt_trainer_class()
    trainer = DCPTTrainerCls(
        model,
        train_loader,
        val_loader,
        config,
        device=device,
        dcpt_logit_weight=args.dcpt_logit_weight,
        dcpt_feature_weight=args.dcpt_feature_weight,
        dcpt_temperature=args.dcpt_temperature,
    )
    if args.resume:
        trainer.load_checkpoint(args.resume)
    if not args.eval_only:
        trainer.train()

    best_path = Path(config["checkpoint_dir"]) / f"{name}_best.pth"
    state = load_verified_checkpoint(best_path, config["contract"])
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()

    # Evaluate across the full degradation matrix on the sealed test split
    degradation_results: dict[str, dict[str, float]] = {}
    for cond_name, deg_fn in DEGRADATION_CONDITIONS.items():
        cond_tf = DCPTPatchCraftEvalTransform(
            size=size,
            patchcraft_mode=args.patchcraft_mode,
            pre_degrade_fn=deg_fn,
        )
        test_ds = StrictImageDataset(root, "test", size, transform=cond_tf, cache_in_memory=False)
        test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
        labels: list[int] = []
        scores: list[float] = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                probs = model(inputs.to(device, dtype=torch.float32)).cpu().flatten().tolist()
                scores.extend(probs)
                labels.extend(targets.tolist())

        low_fpr = compute_low_fpr_operating_points(labels, scores)
        degradation_results[cond_name] = {
            "auc": round(float(compute_auc(labels, scores)), 6),
            "eer": round(float(compute_eer(labels, scores)), 6),
            "accuracy_at_0_5": round(float(compute_accuracy(labels, scores)), 6),
            **low_fpr,
        }

    clean_metrics = degradation_results["clean"]
    deg_aucs = [v["auc"] for k, v in degradation_results.items() if k != "clean"]
    report = {
        "test_auc": clean_metrics["auc"],
        "test_eer": clean_metrics["eer"],
        "test_accuracy_at_0_5": clean_metrics["accuracy_at_0_5"],
        "test_tpr_at_fpr_1pct": clean_metrics["tpr_at_fpr_1pct"],
        "degraded_mean_auc": round(float(np.mean(deg_aucs)), 6),
        "degradation_matrix": degradation_results,
        "eval_split": "test",
        "contract": config["contract"],
        "checkpoint_sha256": sha256_file(best_path),
        "input_size": shape,
    }
    out_file = Path(config["checkpoint_dir"]) / f"{name}_results.json"
    out_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def _run_self_test() -> None:
    """Verifies transforms, degradation matrix, and low-FPR metrics without requiring a dataset or PyTorch."""
    random.seed(42)
    np.random.seed(42)
    rng = np.random.RandomState(42)
    sample_rgb = Image.fromarray(rng.randint(0, 256, size=(384, 512, 3), dtype=np.uint8))

    # 1. Verify all PatchCraft modes produce exact 128x128 RGB images
    for mode in ("mosaic", "crop", "off"):
        out_img = apply_spatial_frontend(sample_rgb, size=128, patchcraft_mode=mode)
        assert out_img.size == (128, 128) and out_img.mode == "RGB", f"Failed mode {mode}"

    # 2. Verify all 8 degradation conditions run cleanly on PIL images
    for cond_name, fn in DEGRADATION_CONDITIONS.items():
        deg_img = fn(sample_rgb) if fn is not None else sample_rgb
        patch = apply_spatial_frontend(deg_img, size=128, patchcraft_mode="mosaic")
        assert patch.size == (128, 128), f"Condition {cond_name} failed"

    # 3. Verify low-FPR operating point calculation
    y_test = [0] * 200 + [1] * 200
    s_test = list(rng.beta(2, 6, size=200)) + list(rng.beta(6, 2, size=200))
    op = compute_low_fpr_operating_points(y_test, s_test)
    assert 0.0 <= op["tpr_at_fpr_1pct"] <= op["tpr_at_fpr_5pct"] <= 1.0

    print("SELF-TEST PASSED (train_srm_dcpt_patchcraft.py):")
    print(f"  - Verified 3 PatchCraft modes ('mosaic', 'crop', 'off') at 128x128")
    print(f"  - Verified all {len(DEGRADATION_CONDITIONS)} degradation matrix conditions: {list(DEGRADATION_CONDITIONS.keys())}")
    print(f"  - Verified low-FPR operating point metrics: {op}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        default=str(ML_ROOT / "configs" / "srm_cnn_openfake.yaml"),
        help="Path to YAML training config (default: ml/configs/srm_cnn_openfake.yaml)",
    )
    parser.add_argument("--data-dir", default=None, help="Override dataset root directory from YAML config")
    parser.add_argument("--output-dir", default=str(ML_ROOT / "checkpoints" / "srm_dcpt_patchcraft_v8"))
    parser.add_argument("--model-name", default="srm_cnn_openfake")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--patchcraft-mode",
        choices=("mosaic", "crop", "off"),
        default="mosaic",
        help="Native-resolution tile preprocessing mode ('mosaic' = 4x4 32x32 tiles, 'crop' = 128x128 top tile, 'off' = bilinear resize)",
    )
    parser.add_argument("--include-npr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-rgb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dcpt-logit-weight", type=float, default=0.5, help="Weight lambda_p on DCPT Bernoulli KL loss")
    parser.add_argument("--dcpt-feature-weight", type=float, default=0.5, help="Weight lambda_f on DCPT cosine feature loss")
    parser.add_argument("--dcpt-temperature", type=float, default=1.0)
    parser.add_argument("--dcpt-degrade-prob", type=float, default=0.85)
    for flag, kind in [("epochs", int), ("batch-size", int), ("lr", float), ("num-workers", int), ("patience", int)]:
        parser.add_argument("--" + flag, type=kind)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", help="Resume from an existing checkpoint path")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run the degradation matrix on _best.pth")
    parser.add_argument("--self-test", action="store_true", help="Run built-in verification tests and exit")
    return parser


if __name__ == "__main__":
    parsed_args = build_arg_parser().parse_args()
    if parsed_args.self_test:
        _run_self_test()
    else:
        run_training(parsed_args)

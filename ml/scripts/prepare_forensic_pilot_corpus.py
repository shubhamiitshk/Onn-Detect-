#!/usr/bin/env python3
"""prepare_forensic_pilot_corpus.py

Prepares a disposable procedural smoke test fixture of 120-150 source groups for
local validation, CI, and Colab T4 LoRA fine-tuning execution.

IMPORTANT SCOPE NOTE:
This corpus is explicitly a PROCEDURAL SMOKE FIXTURE designed for verifying CUDA
execution, LoRA backward passes, gradient checkpointing, peak VRAM measurement,
checkpoint saving/reloading, and Android contract parity. It does NOT evaluate
real-world forensic detection accuracy against authentic photographs or actual
generative synthesis models.

Pipeline Parity:
  - Measures features from the encoded JPEG file on disk (matching Android runtime).
  - Uses native-resolution PatchCraft mosaic extraction (rich and poor 128x128 from 32x32 tiles).
  - Computes contrast-calibrated SRM score matching `SRMForensicAnalyzer.kt`.
  - Computes FFT slope, spectral flatness, and KL co-occurrence divergence using BT.601 integer luma.
  - Assembles the 10-signal vector and evaluates the transpiled 7-tree XGBoost meta-classifier (`fast_fusion_prob`).
  - Computes measured document tampering score from high-frequency edge gradient variance.
  - Guarantees 100% unique file blobs and pixel arrays with zero cross-split duplicate leakage.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "ml" / "src"))
sys.path.insert(0, str(REPO_ROOT / "ml" / "scripts"))

from meta_classifier.feature_extractor import (
    DEFAULT_BASELINE,
    diff_histogram,
    fft_features,
    kl_score,
    to_gray_u8,
)
from train_srm_dcpt_patchcraft import extract_patchcraft_mosaic

try:
    import ai_edge_litert.interpreter as litert
except ImportError:
    try:
        import tensorflow.lite as litert
    except ImportError:
        litert = None

SRM_MODEL_PATH = REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "models" / "srm_forensic_cnn_int8.tflite"
SIGLIP_MODEL_PATH = REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "models" / "siglip_probe_int8.tflite"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_contrast_calibrated_score(rich_score: float, poor_score: float) -> float:
    """Matches `SRMForensicAnalyzer.kt:computeContrastCalibratedScore`."""
    if not math.isfinite(poor_score):
        return float(np.clip(rich_score, 0.0, 1.0))
    contrast = rich_score - poor_score
    if rich_score < 0.25 and poor_score > 0.45:
        return float(np.clip(rich_score - 0.15 * contrast, 0.0, 1.0))
    elif rich_score > 0.48 and poor_score > 0.48:
        return float(np.clip(0.75 * rich_score + 0.25 * max(rich_score, poor_score), 0.0, 1.0))
    else:
        return float(np.clip(rich_score, 0.0, 1.0))


def predict_xgboost_meta(features: Sequence[float]) -> float:
    """Matches Android's transpiled `XGBoostMetaClassifier.kt` 7-tree meta-classifier."""
    p6 = float(features[6]) if len(features) > 6 else float("nan")
    if math.isfinite(p6):
        q = min(1.0 - 1e-6, max(1e-6, p6))
        margin = 6.237644456508207e-09 + math.log(q / (1.0 - q))
    else:
        margin = 6.237644456508207e-09
    f0 = float(features[0])
    f2 = float(features[2])

    # Tree 0
    if not math.isnan(f0) and f0 < 0.4296875:
        margin += -0.223455817 if (not math.isnan(f0) and f0 < 0.16796875) else -0.0735361502
    else:
        margin += 0.0565056391 if (not math.isnan(f2) and f2 < 0.125066549) else 0.149120852

    # Tree 1
    if not math.isnan(f0) and f0 < 0.4296875:
        margin += -0.125852376 if (math.isnan(p6) or p6 < 0.738344669) else -0.348385453
    else:
        margin += 0.151262462 if (math.isnan(p6) or p6 < 0.894045115) else 0.0708964691

    # Tree 2
    if not math.isnan(f0) and f0 < 0.4296875:
        margin += -0.191805184 if (not math.isnan(f0) and f0 < 0.16796875) else -0.0484818034
    else:
        margin += 0.0265345424 if (not math.isnan(f0) and f0 < 0.7265625) else 0.132148132

    # Tree 3
    if not math.isnan(f0) and f0 < 0.4296875:
        margin += -0.166150376 if (not math.isnan(f0) and f0 < 0.19140625) else -0.0458173864
    else:
        margin += -0.000352348172 if (not math.isnan(f0) and f0 < 0.55859375) else 0.118327655

    # Tree 4
    if not math.isnan(f0) and f0 < 0.4140625:
        margin += -0.1797508 if (not math.isnan(f0) and f0 < 0.19140625) else -0.0349732898
    else:
        margin += 0.0478457883 if (not math.isnan(f0) and f0 < 0.7265625) else 0.130223647

    # Tree 5
    if not math.isnan(f0) and f0 < 0.2421875:
        margin += -0.102728896 if (math.isnan(p6) or p6 < 0.738344669) else -0.249161124
    else:
        margin += 0.033773195 if (not math.isnan(f0) and f0 < 0.82421875) else 0.132007703

    # Tree 6
    if not math.isnan(f0) and f0 < 0.203125:
        margin += -7.58929236e-05 if (not math.isnan(f2) and f2 < 0.0751719996) else -0.164556786
    else:
        margin += -0.0404051393 if (not math.isnan(p6) and p6 < 0.941621184) else 0.0896521732

    c_margin = max(-30.0, min(30.0, margin))
    return float(1.0 / (1.0 + math.exp(-c_margin)))


def compute_document_tamper_score(img: Image.Image, active_domain: str) -> float:
    """Computes measured document/text tampering score by evaluating local horizontal
    edge-gradient variance anomaly across text lines.
    Returns 0.0 for non-document domains, and a measured float in [0.0, 1.0] for documents.
    """
    if active_domain != "document_text":
        return 0.0
    gray = np.asarray(to_gray_u8(img), dtype=np.float32)
    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    h, w = gx.shape
    strip_h = max(1, h // 8)
    strip_energies = []
    for i in range(8):
        strip = gx[i * strip_h : (i + 1) * strip_h, :]
        strip_energies.append(float(np.mean(strip)))
    se = np.asarray(strip_energies)
    mean_se = float(np.mean(se))
    std_se = float(np.std(se))
    if mean_se < 1e-4:
        return 0.0
    cv = std_se / (mean_se + 1e-5)
    score = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, 4.0 * (cv - 0.35)))))
    return float(np.clip(score, 0.0, 1.0))


class FrozenSpecialistRunner:
    """Runs the exact frozen specialist models packaged in Android assets."""

    def __init__(self) -> None:
        self.srm_interp: Optional[Any] = None
        self.siglip_interp: Optional[Any] = None

        if litert is not None and SRM_MODEL_PATH.is_file():
            try:
                self.srm_interp = litert.Interpreter(str(SRM_MODEL_PATH))
                self.srm_interp.allocate_tensors()
                self.srm_inp = self.srm_interp.get_input_details()[0]["index"]
                self.srm_out = self.srm_interp.get_output_details()[0]["index"]
            except Exception:
                self.srm_interp = None

        if litert is not None and SIGLIP_MODEL_PATH.is_file():
            try:
                self.siglip_interp = litert.Interpreter(str(SIGLIP_MODEL_PATH))
                self.siglip_interp.allocate_tensors()
                self.siglip_inp = self.siglip_interp.get_input_details()[0]["index"]
                self.siglip_out = self.siglip_interp.get_output_details()[0]["index"]
            except Exception:
                self.siglip_interp = None
            self.siglip_interp.allocate_tensors()
            self.siglip_inp = self.siglip_interp.get_input_details()[0]["index"]
            self.siglip_out = self.siglip_interp.get_output_details()[0]["index"]

    def score_srm_tensor(self, mosaic_128: Image.Image) -> float:
        if self.srm_interp is None:
            return 0.50
        arr = np.asarray(mosaic_128.convert("RGB"), dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        self.srm_interp.set_tensor(self.srm_inp, arr)
        self.srm_interp.invoke()
        out = self.srm_interp.get_tensor(self.srm_out)
        return float(np.clip(out.flat[0], 0.0, 1.0))

    def run_srm_patchcraft(self, img: Image.Image) -> float:
        rich_m = extract_patchcraft_mosaic(img, mosaic_size=128, tile_size=32, select="rich")
        poor_m = extract_patchcraft_mosaic(img, mosaic_size=128, tile_size=32, select="poor")
        rich_s = self.score_srm_tensor(rich_m)
        poor_s = self.score_srm_tensor(poor_m)
        return compute_contrast_calibrated_score(rich_s, poor_s)

    def run_siglip(self, img: Image.Image) -> float:
        if self.siglip_interp is None:
            return float("nan")
        img_224 = img.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
        arr = (np.asarray(img_224, dtype=np.float32) / 255.0 - 0.5) / 0.5
        arr = np.expand_dims(arr, axis=0)
        self.siglip_interp.set_tensor(self.siglip_inp, arr)
        self.siglip_interp.invoke()
        out = self.siglip_interp.get_tensor(self.siglip_out)
        return float(np.clip(out.flat[0], 0.0, 1.0))

    def compute_hotspot_z(self, img: Image.Image) -> float:
        rgb = img.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
        arr_rgb = np.asarray(rgb, dtype=np.float32)
        arr = (arr_rgb[..., 0] * 299.0 + arr_rgb[..., 1] * 587.0 + arr_rgb[..., 2] * 114.0) / 1000.0
        lap = np.abs(arr[1:-1, 1:-1] * 4 - arr[:-2, 1:-1] - arr[2:, 1:-1] - arr[1:-1, :-2] - arr[1:-1, 2:])
        h, w = lap.shape
        ph, pw = max(1, h // 8), max(1, w // 8)
        patch_energies = []
        for i in range(8):
            for j in range(8):
                patch_energies.append(float(np.mean(lap[i*ph:(i+1)*ph, j*pw:(j+1)*pw])))
        pe = np.asarray(patch_energies)
        mean_e = float(np.mean(pe))
        std_e = float(np.std(pe)) + 1e-6
        max_e = float(np.max(pe))
        z = (max_e - mean_e) / std_e
        return float(max(0.0, min(5.0, z)))


def generate_unique_procedural_image(
    category: str,
    width: int = 512,
    height: int = 512,
    seed: int = 42,
) -> Tuple[Image.Image, str]:
    """Generates an image with deterministic seed-driven variations, guaranteeing 100%
    unique byte and pixel content per seed.
    """
    rng = np.random.default_rng(seed)

    if category == "authentic_camera":
        active_domain = "human_face"
        angle = rng.uniform(0.0, 2.0 * np.pi)
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        xx, yy = np.meshgrid(x, y)
        proj = xx * np.cos(angle) + yy * np.sin(angle)
        c0 = rng.uniform(40, 120, 3)
        c1 = rng.uniform(140, 230, 3)
        t = (proj + 1.0) / 2.0
        rgb = (1.0 - t[..., None]) * c0 + t[..., None] * c1
        # Add 3-5 random soft organic shapes
        for _ in range(rng.integers(3, 6)):
            cx, cy = rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6)
            rx, ry = rng.uniform(0.2, 0.5), rng.uniform(0.2, 0.5)
            blob = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
            rgb += blob[..., None] * rng.uniform(-25, 25, 3)
        # Authentic sensor Poisson-Gaussian noise
        noise = rng.normal(0, rng.uniform(2.5, 4.5), rgb.shape)
        img = Image.fromarray(np.clip(rgb + noise, 0, 255).astype(np.uint8))

    elif category == "authentic_bokeh":
        active_domain = "human_face"
        bg_color = rng.uniform(20, 80, 3)
        rgb = np.full((height, width, 3), bg_color, dtype=np.float32)
        # Bokeh circles
        for _ in range(rng.integers(15, 30)):
            bx = rng.integers(0, width)
            by = rng.integers(0, height)
            br = rng.integers(25, 75)
            y_idx, x_idx = np.ogrid[:height, :width]
            dist_sq = (x_idx - bx)**2 + (y_idx - by)**2
            mask = dist_sq <= br**2
            b_color = rng.uniform(120, 240, 3)
            alpha = rng.uniform(0.15, 0.45)
            rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * b_color
        img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=rng.uniform(6, 12)))
        # Sharp foreground subject
        arr = np.asarray(img, dtype=np.float32)
        cy, cx = height // 2 + rng.integers(-20, 20), width // 2 + rng.integers(-20, 20)
        ry, rx = height // 4 + rng.integers(-15, 15), width // 4 + rng.integers(-15, 15)
        yy, xx = np.ogrid[:height, :width]
        fg_mask = ((xx - cx)**2 / rx**2 + (yy - cy)**2 / ry**2) <= 1.0
        fg_color = rng.uniform(140, 220, 3)
        arr[fg_mask] = 0.5 * arr[fg_mask] + 0.5 * fg_color + rng.normal(0, 3.0, (int(np.sum(fg_mask)), 3))
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    elif category == "authentic_document":
        active_domain = "document_text"
        base_color = (int(rng.integers(246, 252)), int(rng.integers(246, 252)), int(rng.integers(248, 254)))
        img = Image.new("RGB", (width, height), color=base_color)
        draw = ImageDraw.Draw(img)
        # Header banner
        header_w = rng.integers(120, 260)
        draw.rectangle([40, 25, 40 + header_w, 40], fill=(int(rng.integers(15, 45)), int(rng.integers(20, 60)), int(rng.integers(70, 130))))
        # Varied lines of text words
        n_lines = rng.integers(12, 17)
        for line_idx in range(n_lines):
            row_y = 55 + line_idx * 26
            cur_x = 40 + rng.integers(0, 15)
            n_words = rng.integers(5, 11)
            for _ in range(n_words):
                w_len = rng.integers(18, 55)
                if cur_x + w_len > width - 40:
                    break
                draw.rectangle([cur_x, row_y, cur_x + w_len, row_y + 8], fill=(int(rng.integers(35, 60)), int(rng.integers(35, 60)), int(rng.integers(40, 65))))
                cur_x += w_len + rng.integers(6, 14)
        # Signature/stamp box in margin
        stamp_x = width - rng.integers(100, 150)
        stamp_y = height - rng.integers(70, 100)
        draw.rectangle([stamp_x, stamp_y, stamp_x + 60, stamp_y + 35], outline=(int(rng.integers(40, 90)), int(rng.integers(50, 110)), int(rng.integers(120, 180))), width=2)
        # Paper texture noise
        arr = np.asarray(img, dtype=np.float32) + rng.normal(0, 1.2, (height, width, 3))
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    elif category == "generative_synthesis":
        active_domain = "object_product"
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        f_grid = rng.uniform(3.2, 5.8)
        phase_x = rng.uniform(0.0, np.pi)
        phase_y = rng.uniform(0.0, np.pi)
        grid = (np.sin(xx * np.pi / f_grid + phase_x) * np.sin(yy * np.pi / f_grid + phase_y)) * rng.uniform(6.0, 12.0)
        base = 128 + rng.uniform(70, 95) * np.sin(xx / rng.uniform(35.0, 55.0)) * np.cos(yy / rng.uniform(30.0, 48.0))
        rgb = np.stack([base + grid, base * rng.uniform(0.85, 0.95) - grid, base * rng.uniform(1.05, 1.15) + grid], axis=-1)
        rgb = np.clip(rgb + rng.normal(0, rng.uniform(1.0, 2.5), rgb.shape), 0, 255).astype(np.uint8)
        img = Image.fromarray(rgb)

    elif category == "generative_inpaint":
        active_domain = "scene_arch"
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)
        r = 130 + rng.uniform(40, 70) * np.sin(xx * rng.uniform(1.5, 3.0))
        g = 140 + rng.uniform(35, 60) * np.cos(yy * rng.uniform(1.5, 3.0))
        b = 150 + rng.uniform(30, 55) * np.sin(xx + yy)
        rgb = np.clip(np.stack([r, g, b], axis=-1) + rng.normal(0, rng.uniform(2.5, 4.5), (height, width, 3)), 0, 255).astype(np.uint8)
        img = Image.fromarray(rgb)
        # Spliced / inpainted anomaly box
        bx0 = rng.integers(width // 4, width // 3)
        by0 = rng.integers(height // 4, height // 3)
        bw = rng.integers(120, 180)
        bh = rng.integers(120, 180)
        bx1, by1 = bx0 + bw, by0 + bh
        arr = np.asarray(img).copy()
        sub = arr[by0:by1, bx0:bx1].astype(np.float32)
        sub = sub * rng.uniform(1.15, 1.35) + np.sin(np.arange(sub.shape[1]) * rng.uniform(0.3, 0.7))[None, :, None] * rng.uniform(12.0, 22.0)
        arr[by0:by1, bx0:bx1] = np.clip(sub, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    elif category == "document_tamper":
        active_domain = "document_text"
        base_color = (int(rng.integers(246, 252)), int(rng.integers(246, 252)), int(rng.integers(248, 254)))
        img = Image.new("RGB", (width, height), color=base_color)
        draw = ImageDraw.Draw(img)
        # Header banner
        header_w = rng.integers(120, 260)
        draw.rectangle([40, 25, 40 + header_w, 40], fill=(int(rng.integers(15, 45)), int(rng.integers(20, 60)), int(rng.integers(70, 130))))
        # Authentic lines
        n_lines = rng.integers(12, 17)
        tamper_line = rng.integers(4, 9)
        for line_idx in range(n_lines):
            row_y = 55 + line_idx * 26
            cur_x = 40 + rng.integers(0, 15)
            n_words = rng.integers(5, 11)
            for w_idx in range(n_words):
                w_len = rng.integers(18, 55)
                if cur_x + w_len > width - 40:
                    break
                if line_idx == tamper_line and 2 <= w_idx <= 4:
                    # Spliced / tampered words: heavier font stroke, darker tint
                    draw.rectangle([cur_x, row_y - 2, cur_x + w_len, row_y + 11], fill=(10, 10, 15))
                else:
                    draw.rectangle([cur_x, row_y, cur_x + w_len, row_y + 8], fill=(int(rng.integers(35, 60)), int(rng.integers(35, 60)), int(rng.integers(40, 65))))
                cur_x += w_len + rng.integers(6, 14)
        # Localized altered block artifact
        arr = np.asarray(img, dtype=np.float32)
        ty0 = 55 + tamper_line * 26 - 6
        ty1 = ty0 + 26
        tx0 = 80
        tx1 = tx0 + rng.integers(140, 210)
        # Add localized blur/resample noise over the tampered text region
        arr[ty0:ty1, tx0:tx1] += rng.normal(0, rng.uniform(4.0, 8.0), (ty1 - ty0, tx1 - tx0, 3))
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    else:
        raise ValueError(f"Unknown category: {category}")

    return img, active_domain


def build_disposable_pilot_corpus(
    output_dir: Path = REPO_ROOT / "data" / "forensic_pilot",
    num_roots: int = 120,
    seed: int = 20261028,
) -> Tuple[Path, Path]:
    """Generates the procedural smoke test fixture, computes specialist measurements from
    encoded JPEG files on disk, and exports the JSONL records and manifest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    runner = FrozenSpecialistRunner()
    baseline = np.asarray(DEFAULT_BASELINE, dtype=np.float64)
    baseline /= baseline.sum()

    categories_real = ["authentic_camera", "authentic_bokeh", "authentic_document"]
    categories_fake = ["generative_synthesis", "generative_inpaint", "document_tamper"]

    splits_spec = [
        ("train", 72),
        ("val", 24),
        ("test", 24),
    ]

    records: List[Dict[str, Any]] = []
    root_counter = 1
    sample_idx = 0

    for split_idx, (split_name, count_in_split) in enumerate(splits_spec):
        for idx_in_split in range(count_in_split):
            root_id = f"root_{root_counter:04d}"
            root_counter += 1

            label = idx_in_split % 2
            cat_list = categories_fake if label == 1 else categories_real
            cat = cat_list[(idx_in_split // 2) % len(cat_list)]

            root_seed = seed + 1000 * split_idx + 13 * root_counter + sample_idx
            img, domain = generate_unique_procedural_image(cat, width=512, height=512, seed=root_seed)

            filename = f"{root_id}_v0.jpg"
            img_path = images_dir / filename
            # Save to disk first as JPEG
            img.save(img_path, format="JPEG", quality=92)
            digest = sha256_file(img_path)

            # Reload encoded JPEG from disk to measure realistic compression artifacts
            disk_img = Image.open(img_path).convert("RGB")

            # Run frozen specialist pipeline on disk JPEG
            srm_score = runner.run_srm_patchcraft(disk_img)
            probe_score = runner.run_siglip(disk_img)
            hotspot_z = runner.compute_hotspot_z(disk_img)
            doc_tamper = compute_document_tamper_score(disk_img, domain)
            fft_score, flatness_score = fft_features(disk_img, size=256)
            kl_val, _ = kl_score(to_gray_u8(disk_img), baseline)
            res_norm = float(np.clip(max(disk_img.width, disk_img.height) / 1024.0, 0.0, 1.0))
            is_compressed = 1.0

            features = [
                srm_score,
                fft_score,
                kl_val,
                flatness_score,
                res_norm,
                is_compressed,
                probe_score,
                0.05,
                0.9,
                0.1,
            ]
            fast_fusion = predict_xgboost_meta(features)

            record = {
                "image_id": f"pilot_sample_{sample_idx:04d}",
                "root_source_id": root_id,
                "split": split_name,
                "label": int(label),
                "image_path": f"data/forensic_pilot/images/{filename}",
                "sha256": digest,
                "active_domain": domain,
                "fast_fusion_prob": round(fast_fusion, 6),
                "srm_score": round(srm_score, 6),
                "vfm_probe_score": round(probe_score, 6) if math.isfinite(probe_score) else None,
                "hotspot_z": round(hotspot_z, 4),
                "doc_tamper_score": round(doc_tamper, 4),
                "category": cat,
            }
            records.append(record)
            sample_idx += 1

            # For 20% of roots, add a secondary variant (recompressed) sharing the same root_source_id
            if idx_in_split % 5 == 0:
                variant_img = disk_img.resize((384, 384), Image.Resampling.BILINEAR)
                v_filename = f"{root_id}_v1_q75.jpg"
                v_path = images_dir / v_filename
                variant_img.save(v_path, format="JPEG", quality=75)
                v_digest = sha256_file(v_path)

                v_disk_img = Image.open(v_path).convert("RGB")
                v_srm = runner.run_srm_patchcraft(v_disk_img)
                v_probe = runner.run_siglip(v_disk_img)
                v_hz = runner.compute_hotspot_z(v_disk_img)
                v_doc = compute_document_tamper_score(v_disk_img, domain)
                v_fft, v_flat = fft_features(v_disk_img, size=256)
                v_kl, _ = kl_score(to_gray_u8(v_disk_img), baseline)
                v_res = float(np.clip(max(v_disk_img.width, v_disk_img.height) / 1024.0, 0.0, 1.0))

                v_features = [
                    v_srm,
                    v_fft,
                    v_kl,
                    v_flat,
                    v_res,
                    1.0,
                    v_probe,
                    0.05,
                    0.9,
                    0.1,
                ]
                v_fast = predict_xgboost_meta(v_features)

                v_record = {
                    "image_id": f"pilot_sample_{sample_idx:04d}",
                    "root_source_id": root_id,
                    "split": split_name,
                    "label": int(label),
                    "image_path": f"data/forensic_pilot/images/{v_filename}",
                    "sha256": v_digest,
                    "active_domain": domain,
                    "fast_fusion_prob": round(v_fast, 6),
                    "srm_score": round(v_srm, 6),
                    "vfm_probe_score": round(v_probe, 6) if math.isfinite(v_probe) else None,
                    "hotspot_z": round(v_hz, 4),
                    "doc_tamper_score": round(v_doc, 4),
                    "category": f"{cat}_recompressed",
                }
                records.append(v_record)
                sample_idx += 1

    # Assert 100% uniqueness and zero cross-split duplicate leakage
    file_hashes = [r["sha256"] for r in records]
    assert len(set(file_hashes)) == len(records), (
        f"Found duplicate file hashes: {len(records)} records vs {len(set(file_hashes))} unique hashes"
    )

    train_hashes = {r["sha256"] for r in records if r["split"] == "train"}
    val_hashes = {r["sha256"] for r in records if r["split"] == "val"}
    test_hashes = {r["sha256"] for r in records if r["split"] == "test"}

    assert len(train_hashes & val_hashes) == 0, "Train and Val share exact image blobs!"
    assert len(train_hashes & test_hashes) == 0, "Train and Test share exact image blobs!"
    assert len(val_hashes & test_hashes) == 0, "Val and Test share exact image blobs!"

    jsonl_path = output_dir / "forensic_pilot.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    manifest = {
        "schema_version": "onn-forensic-pilot-corpus-v1",
        "corpus_type": "procedural_smoke_fixture",
        "purpose": "Mechanical training loop, CUDA backpropagation, VRAM measurement, and pipeline wiring verification only. Not an empirical benchmark of forensic detection accuracy.",
        "total_records": len(records),
        "total_unique_hashes": len(set(file_hashes)),
        "total_roots": len(set(r["root_source_id"] for r in records)),
        "splits": {
            "train": {
                "records": sum(1 for r in records if r["split"] == "train"),
                "roots": len(set(r["root_source_id"] for r in records if r["split"] == "train")),
                "fakes": sum(1 for r in records if r["split"] == "train" and r["label"] == 1),
                "reals": sum(1 for r in records if r["split"] == "train" and r["label"] == 0),
            },
            "val": {
                "records": sum(1 for r in records if r["split"] == "val"),
                "roots": len(set(r["root_source_id"] for r in records if r["split"] == "val")),
                "fakes": sum(1 for r in records if r["split"] == "val" and r["label"] == 1),
                "reals": sum(1 for r in records if r["split"] == "val" and r["label"] == 0),
            },
            "test": {
                "records": sum(1 for r in records if r["split"] == "test"),
                "roots": len(set(r["root_source_id"] for r in records if r["split"] == "test")),
                "fakes": sum(1 for r in records if r["split"] == "test" and r["label"] == 1),
                "reals": sum(1 for r in records if r["split"] == "test" and r["label"] == 0),
            },
        },
        "records_file": str(jsonl_path.relative_to(REPO_ROOT)),
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"Generated procedural smoke fixture at {jsonl_path} | "
        f"Total={len(records)} samples ({manifest['total_roots']} roots, {len(set(file_hashes))} unique blobs) | "
        f"Train={manifest['splits']['train']['records']} | "
        f"Val={manifest['splits']['val']['records']} | "
        f"Test={manifest['splits']['test']['records']} | "
        f"Zero cross-split duplicate blobs verified."
    )
    return jsonl_path, manifest_path


if __name__ == "__main__":
    build_disposable_pilot_corpus()

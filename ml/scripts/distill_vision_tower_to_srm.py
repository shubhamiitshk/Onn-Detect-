"""Teacher-Student Distillation & Laundered-Baseline Calibration for Onn-Detect (v4 / v5).

Bridges the gap between the honest `onn-20261019-v3` baseline (0.714 raw / 0.693 laundered AUC)
and the frozen vision-tower probes (`probe_smolvlm500_vision` 0.869 AUC, `probe_mobileclip2_s0`
0.885 AUC) WITHOUT bloating the Android APK or changing `ModelIntegrityChecker.kt`:

1. Reuses cached teacher logits in `D:\\onn_data\\runs\\release_v4\\checkpoints\\teacher_distill_cache.npz`
   (or `output_dir / teacher_distill_cache.npz`) when present so v5 hybrid training starts
   immediately without re-running `SmolVLM-500M-Instruct` vision extraction.
2. Fits an L2-regularized logistic teacher probe on the stacking/train embeddings.
3. Trains the 20-channel `SRMForensicCNN(include_npr=True, include_rgb=True)` hybrid student
   (16 SRM+NPR high-pass filters + 4 RGB/luminance low-pass filters) at 128x128 using
   blended ground-truth BCE + teacher logit KL distillation on phone-laundered augmentations
   and native-resolution PatchCraft high-texture crops.
4. Saves checkpoints with the exact `schema_version=1` + `weights_only=True`-safe RNG structure
   required by `load_verified_checkpoint()` in `ml/src/training/checkpoint.py`.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ML_ROOT / "src"))
sys.path.insert(0, str(ML_ROOT / "scripts"))

from data.augmentations import PhonePipelineAugmentation
from data.contracts import IMAGE_EXTS, discover_samples, sha256_file, validate_dataset_manifest
from data.image_dataset import image_tensor
from data.image_frontend import decode_image, frontend_id
from meta_classifier.feature_extractor import KL_BINS, diff_histogram, to_gray_u8
from models.srm_forensic_cnn import SRMForensicCNN, extract_high_low_texture_patches
from training.checkpoint import atomic_torch_save, load_verified_checkpoint
from training.metrics import compute_accuracy, compute_auc, compute_eer
from training.provenance import training_code_hash
from utils.seed import set_global_seed


def _preprocess_siglip(img: Image.Image, size: int = 512) -> torch.Tensor:
    resized = img.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    arr = (np.asarray(resized, dtype=np.float32) / 255.0 - 0.5) / 0.5
    return torch.from_numpy(arr.transpose(2, 0, 1))


@torch.no_grad()
def extract_vision_tower_embeddings(
    samples: list[tuple[Path, int]],
    repo_id: str,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    """Extract L2-normalized pooled embeddings from SigLIP / SmolVLM vision tower."""
    from transformers import AutoModelForImageTextToText, SiglipVisionModel

    print(f"[Teacher] Loading vision tower from {repo_id} on {device}...")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if "SmolVLM" in repo_id:
        full_vlm = AutoModelForImageTextToText.from_pretrained(repo_id, torch_dtype=dtype).to(device).eval()
        vision_model = full_vlm.model.vision_model
    else:
        vision_model = SiglipVisionModel.from_pretrained(repo_id, torch_dtype=dtype).to(device).eval()

    embs = []
    for start in range(0, len(samples), batch_size):
        batch_paths = [p for p, _ in samples[start : start + batch_size]]
        tensors = []
        for p in batch_paths:
            img = decode_image(str(p)).image
            tensors.append(_preprocess_siglip(img, 512))
        pixel_values = torch.stack(tensors).to(device, dtype=dtype)
        out = vision_model(pixel_values=pixel_values)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            vecs = out.pooler_output.float().cpu().numpy()
        else:
            vecs = out.last_hidden_state.float().mean(dim=1).cpu().numpy()
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        embs.append(vecs.astype(np.float32))
        if (start + batch_size) % 200 < batch_size or (start + batch_size) >= len(samples):
            print(f"[Teacher] Embedded {min(start + batch_size, len(samples))}/{len(samples)}")

    del vision_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(embs, axis=0)


def fit_teacher_logistic_probe(
    X_stack: np.ndarray,
    y_stack: np.ndarray,
    X_target: np.ndarray,
    l2_reg: float = 1.0,
) -> np.ndarray:
    """Fit L2-regularized logistic probe on stacking split and return teacher logits on X_target."""
    mean = X_stack.mean(axis=0, keepdims=True)
    std = X_stack.std(axis=0, keepdims=True) + 1e-6
    Z_tr = np.hstack([(X_stack - mean) / std, np.ones((len(X_stack), 1))])
    Z_tg = np.hstack([(X_target - mean) / std, np.ones((len(X_target), 1))])
    d = Z_tr.shape[1]
    w = np.zeros(d, dtype=np.float64)
    reg = np.eye(d, dtype=np.float64) * l2_reg
    reg[-1, -1] = 1e-6
    y_f = y_stack.astype(np.float64)

    for _ in range(60):
        logits = np.clip(Z_tr @ w, -40.0, 40.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        r = np.maximum(p * (1.0 - p), 1e-6)
        grad = Z_tr.T @ (p - y_f) + reg @ w
        hess = (Z_tr.T * r) @ Z_tr + reg
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.linalg.norm(step) < 1e-6:
            break

    target_logits = np.clip(Z_tg @ w, -15.0, 15.0)
    return target_logits.astype(np.float32)


def calibrate_laundered_kl_baseline(real_paths: list[Path], sample_limit: int = 500) -> dict:
    """Calibrate natural_diff_histogram.json across both raw and 512px Q85 laundered real photos."""
    acc = np.zeros(KL_BINS, dtype=np.float64)
    used = 0
    for p in real_paths[:sample_limit]:
        img = decode_image(str(p)).image
        acc += diff_histogram(to_gray_u8(img))
        w, h = img.size
        scale = min(1.0, 512.0 / float(max(w, h)))
        resized = img.resize((max(16, int(round(w * scale))), max(16, int(round(h * scale)))), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        resized.convert("RGB").save(buf, format="JPEG", quality=85)
        buf.seek(0)
        laundered = Image.open(buf).convert("RGB")
        acc += diff_histogram(to_gray_u8(laundered))
        used += 2

    acc /= acc.sum()
    return {"bins": [float(x) for x in acc], "num_images": used}


class DistilledSRMDataset(Dataset):
    """Yields (image_tensor, hard_label, teacher_logit) with native-patch + phone augmentations."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        teacher_logits: np.ndarray | None,
        input_size: int = 128,
        is_train: bool = True,
        patch_prob: float = 0.35,
    ):
        self.samples = samples
        self.teacher_logits = teacher_logits
        self.input_size = input_size
        self.is_train = is_train
        self.patch_prob = patch_prob
        self.phone_aug = PhonePipelineAugmentation(p_jpeg=0.75, p_downscale=0.55, p_blur=0.15)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = decode_image(str(path)).image
        if self.is_train:
            if random.random() < self.patch_prob:
                hi_patch, _ = extract_high_low_texture_patches(img, patch_size=max(self.input_size, 128))
                img = hi_patch
            img = self.phone_aug(img)
            if random.random() < 0.5:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        tensor = image_tensor(img, self.input_size).float()
        t_logit = float(self.teacher_logits[idx]) if self.teacher_logits is not None else 0.0
        return tensor, float(label), float(t_logit)


def _weights_only_safe_state(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_auc: float,
    contract: dict,
    config_record: dict,
    metrics: dict,
) -> dict:
    """Construct checkpoint dictionary matching Trainer._state() for torch.load(weights_only=True)."""
    npstate = np.random.get_state()
    return {
        "schema_version": 1,
        "epoch": int(epoch),
        "state_dict": model.cpu().state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": {},
        "best_metric": float(best_val_auc),
        "metrics": metrics,
        "contract": contract,
        "config": config_record,
        "early_stopping": {"counter": 0, "best_score": float(best_val_auc), "early_stop": False},
        "history": {"train_loss": [], "validation": [metrics]},
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(),
        "numpy_rng": [npstate[0], npstate[1].tolist(), int(npstate[2]), int(npstate[3]), float(npstate[4])],
    }


def train_distilled_srm(
    data_root: Path,
    output_dir: Path,
    teacher_repo: str,
    input_size: int = 128,
    epochs: int = 18,
    batch_size: int = 32,
    lr: float = 1e-3,
    alpha_kd: float = 0.6,
    temperature: float = 2.0,
    include_rgb: bool = True,
    device_str: str = "auto",
) -> Path:
    set_global_seed(42)
    provenance = validate_dataset_manifest(data_root)
    # "auto" must resolve to a concrete device. With no CUDA present the
    # literal string "auto" is not a valid torch device.
    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available())
        else ("cpu" if device_str == "auto" else device_str)
    )

    train_samples = discover_samples(data_root, "train", IMAGE_EXTS)
    val_samples = discover_samples(data_root, "val", IMAGE_EXTS)
    test_samples = discover_samples(data_root, "test", IMAGE_EXTS)
    stack_samples = discover_samples(data_root, "stacking", IMAGE_EXTS)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_npz = output_dir / "teacher_distill_cache.npz"
    v4_cache_npz = Path(r"D:\onn_data\runs\release_v4\checkpoints\teacher_distill_cache.npz")
    if not cache_npz.exists() and v4_cache_npz.exists():
        print(f"[Teacher] Reusing cached SmolVLM-500M teacher logits from {v4_cache_npz}...")
        shutil.copy2(v4_cache_npz, cache_npz)

    if cache_npz.exists():
        print(f"[Teacher] Loading cached teacher logits from {cache_npz}...")
        with np.load(cache_npz) as d:
            train_teacher_logits = d["train_teacher_logits"]
            val_teacher_logits = d["val_teacher_logits"]
    else:
        X_stack = extract_vision_tower_embeddings(stack_samples, teacher_repo, device)
        y_stack = np.asarray([lbl for _, lbl in stack_samples], dtype=np.int64)
        X_train = extract_vision_tower_embeddings(train_samples, teacher_repo, device)
        X_val = extract_vision_tower_embeddings(val_samples, teacher_repo, device)
        train_teacher_logits = fit_teacher_logistic_probe(X_stack, y_stack, X_train)
        val_teacher_logits = fit_teacher_logistic_probe(X_stack, y_stack, X_val)
        np.savez_compressed(
            cache_npz,
            train_teacher_logits=train_teacher_logits,
            val_teacher_logits=val_teacher_logits,
        )

    real_train_paths = [p for p, lbl in train_samples if lbl == 0]
    kl_hist = calibrate_laundered_kl_baseline(real_train_paths)
    (output_dir / "natural_diff_histogram.json").write_text(json.dumps(kl_hist, indent=2), encoding="utf-8")

    train_ds = DistilledSRMDataset(train_samples, train_teacher_logits, input_size=input_size, is_train=True)
    val_ds = DistilledSRMDataset(val_samples, val_teacher_logits, input_size=input_size, is_train=False)
    test_ds = DistilledSRMDataset(test_samples, None, input_size=input_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SRMForensicCNN(include_npr=True, include_rgb=include_rgb).to(device)
    stem_ch = int(model.srm.conv.out_channels)
    print(f"[Student] Initialized SRMForensicCNN with {stem_ch}-channel stem (include_npr=True, include_rgb={include_rgb})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    contract = {
        "frontend_id": frontend_id(input_size),
        "input_shape": [1, input_size, input_size, 3],
        "architecture": "SRMForensicCNN",
        "label_convention": "real=0_fake=1",
        "dataset_manifest_hash": provenance["manifest_sha256"],
        "source": provenance["source"],
        "revision": provenance["revision"],
        "training_code_hash": training_code_hash(),
        "audio_crop_policy": None,
    }
    config_record = {
        "model": {
            "name": "SRMForensicCNN",
            "input_size": input_size,
            "include_npr": True,
            "include_rgb": include_rgb,
        },
        "data": {"root_dir": str(data_root.resolve()), "img_size": input_size},
        "training": {"epochs": epochs, "batch_size": batch_size, "alpha_kd": alpha_kd, "temperature": temperature},
        "contract": contract,
    }

    best_val_auc = -1.0
    best_ckpt_path = output_dir / "srm_cnn_genimage_best.pth"

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y_hard, t_logit in train_loader:
            x = x.to(device)
            y_hard = y_hard.to(device).unsqueeze(1)
            t_logit = t_logit.to(device).unsqueeze(1)

            optimizer.zero_grad()
            s_logit = model.forward_logits(x)

            loss_bce = F.binary_cross_entropy_with_logits(s_logit, y_hard)
            t_prob = torch.sigmoid(t_logit / temperature)
            loss_kd = F.binary_cross_entropy_with_logits(s_logit / temperature, t_prob) * (temperature ** 2)
            loss = (1.0 - alpha_kd) * loss_bce + alpha_kd * loss_kd

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(x)

        scheduler.step()

        model.eval()
        val_scores, val_labels = [], []
        with torch.no_grad():
            for x, y_hard, _ in val_loader:
                probs = model(x.to(device)).cpu().flatten().tolist()
                val_scores.extend(probs)
                val_labels.extend([int(v) for v in y_hard.tolist()])

        val_auc = float(compute_auc(val_labels, val_scores))
        val_eer = float(compute_eer(val_labels, val_scores))
        val_acc = float(compute_accuracy(val_labels, val_scores))
        print(f"Epoch {epoch + 1:02d}/{epochs:02d} | loss={total_loss / len(train_ds):.4f} | val_auc={val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            contract["training_code_hash"] = training_code_hash()
            metrics = {"auc": val_auc, "eer": val_eer, "accuracy": val_acc}
            state = _weights_only_safe_state(
                epoch, model, optimizer, best_val_auc, contract, config_record, metrics
            )
            atomic_torch_save(state, best_ckpt_path)
            model.to(device)

    verified = load_verified_checkpoint(best_ckpt_path)
    model.load_state_dict(verified["state_dict"])
    model.to(device).eval()

    test_scores, test_labels = [], []
    with torch.no_grad():
        for x, y_hard, _ in test_loader:
            test_scores.extend(model(x.to(device)).cpu().flatten().tolist())
            test_labels.extend([int(v) for v in y_hard.tolist()])

    summary = {
        "stem_channels": stem_ch,
        "include_rgb": include_rgb,
        "best_val_auc": best_val_auc,
        "test_auc": float(compute_auc(test_labels, test_scores)),
        "test_eer": float(compute_eer(test_labels, test_scores)),
        "test_accuracy_at_0_5": float(compute_accuracy(test_labels, test_scores)),
        "checkpoint_path": str(best_ckpt_path),
        "checkpoint_sha256": sha256_file(best_ckpt_path),
    }
    (output_dir / "distilled_srm_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Distillation complete:", json.dumps(summary, indent=2))
    return best_ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Path to openfake_visual dataset root")
    parser.add_argument("--output-dir", required=True, help="Output directory for distilled checkpoint")
    parser.add_argument(
        "--teacher-encoder",
        default="HuggingFaceTB/SmolVLM-500M-Instruct",
        help="Vision tower teacher repo (SmolVLM-500M-Instruct [Apache-2.0] or google/siglip-base-patch16-512)",
    )
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha-kd", type=float, default=0.6)
    parser.add_argument(
        "--include-rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include 4 RGB+Luminance low-pass channels alongside 16 SRM+NPR channels (20 channels total)",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    train_distilled_srm(
        data_root=Path(args.data_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        teacher_repo=args.teacher_encoder,
        input_size=args.input_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha_kd=args.alpha_kd,
        include_rgb=args.include_rgb,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
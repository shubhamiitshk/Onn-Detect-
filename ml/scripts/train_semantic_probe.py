"""Train and export Engine C - the "Lite C" semantic second engine.

Engine C is a small appearance-stem CNN that is *distilled* from a frozen
CLIP vision tower. It deliberately uses a different inductive bias from
Engine A (the SRM+NPR high-pass forensic CNN) so that the two engines fail
differently: Engine A reads high-frequency residual structure, Engine C reads
semantic appearance. Diverse engines are what make a fusion worth more than
its best member.

Why a student and not the raw tower
-----------------------------------
Exporting a raw MobileCLIP2-S0 tower to TFLite needs an ONNX -> TFLite
toolchain that this repository does not carry. The repo already solves the
same problem for Engine A in ``distill_vision_tower_to_srm.py``: distil the
frozen teacher into a small student, then export the student through the
existing Keras -> TFLite path. This script follows that recipe with an
appearance stem instead of an SRM stem.

If a frozen-tower TFLite is produced later, it drops straight into
``train_xgboost_meta.py --semantic-tflite`` unchanged - the contract is
"224x224x3 float RGB in, one scalar P(ai) out".

Usage:
    python -m ml.scripts.train_semantic_probe \
        --data-root D:\\onn_data\\openfake_visual \
        --output-dir D:\\onn_data\\runs\\release_v6\\checkpoints \
        --teacher timm/MobileCLIP2-S0-OpenCLIP
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional

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

from data.contracts import IMAGE_EXTS, discover_samples, sha256_file, validate_dataset_manifest
from data.image_frontend import decode_image
from training.metrics import compute_accuracy, compute_auc, compute_eer
from training.provenance import training_code_hash
from utils.seed import set_global_seed


# ---------------------------------------------------------------------------
# Model: appearance stem
# ---------------------------------------------------------------------------

class DepthwiseSeparableBlock(nn.Module):
    """Same block family as Engine A so the Keras exporter can mirror it."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Hardswish(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class SemanticProbeCNN(nn.Module):
    """Appearance-stem CNN for AI-generated image detection.

    No high-pass filters: the stem sees ordinary RGB, so the network is
    forced to learn appearance/semantics rather than the residual structure
    Engine A already covers.
    """

    def __init__(self, width: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.Hardswish(inplace=True),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableBlock(width, width, stride=1),
            DepthwiseSeparableBlock(width, width * 2, stride=2),
            DepthwiseSeparableBlock(width * 2, width * 2, stride=1),
            DepthwiseSeparableBlock(width * 2, width * 4, stride=2),
            DepthwiseSeparableBlock(width * 4, width * 4, stride=1),
            DepthwiseSeparableBlock(width * 4, width * 8, stride=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        if return_logits:
            return self.head[:-1](x)
        return self.head(x)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, return_logits=True)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Teacher: frozen CLIP vision tower
# ---------------------------------------------------------------------------

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


def _clip_preprocess(img: Image.Image, size: int = 224) -> torch.Tensor:
    """STUDENT frontend: 224x224 bicubic + CLIP mean/std normalization.

    Must match `data.image_frontend.semantic_image_tensor` and the Android
    `SemanticProbeAnalyzer` exactly.
    """
    resized = img.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.asarray(CLIP_MEAN, dtype=np.float32)
    std = np.asarray(CLIP_STD, dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1))


def _teacher_preprocess(img: Image.Image, teacher_repo: str) -> torch.Tensor:
    """TEACHER frontend: each tower's own native preprocessing.

    The teacher never ships, so it does not need to match the student
    frontend - but it does need to match what the tower was trained with, or
    the embeddings are noise and the whole distillation is worthless.
    """
    if "SmolVLM" in teacher_repo or "siglip" in teacher_repo.lower():
        # SigLIP family: 512x512 bicubic, scaled to [-1, 1].
        resized = img.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
        arr = (np.asarray(resized, dtype=np.float32) / 255.0 - 0.5) / 0.5
        return torch.from_numpy(arr.transpose(2, 0, 1))
    # CLIP family: 224x224 bicubic + CLIP mean/std.
    return _clip_preprocess(img, 224)



_TEACHER_CACHE: dict[str, Any] = {}


def _load_teacher(teacher_repo: str, device: torch.device):
    """Load the frozen vision tower once and reuse it across splits."""
    cached = _TEACHER_CACHE.get(teacher_repo)
    if cached is not None:
        return cached

    print(f"[Teacher] Loading vision tower from {teacher_repo} on {device}...")
    dtype = torch.float32

    if teacher_repo.startswith("timm/"):
        import timm
        from timm.models import create_model

        name = teacher_repo.split("/", 1)[1]
        # timm renamed the MobileCLIP2 entries across releases; try the
        # registered names before giving up.
        candidates = [name, name.replace("-OpenCLIP", ""), name.lower(),
                      name.lower().replace("-openclip", "")]
        full = None
        for candidate in candidates:
            try:
                full = create_model(candidate, pretrained=True)
                print(f"[Teacher] timm model resolved as '{candidate}'")
                break
            except Exception:
                continue
        if full is None:
            raise RuntimeError(
                f"timm cannot resolve '{name}'. Install a timm build that ships "
                f"MobileCLIP2, or use a transformers teacher."
            )
        vision_model = full.trunk if hasattr(full, "trunk") else full
        vision_model = vision_model.eval().to(device)
    else:
        from transformers import AutoModelForImageTextToText, CLIPVisionModel, SiglipVisionModel

        if "SmolVLM" in teacher_repo:
            full_vlm = AutoModelForImageTextToText.from_pretrained(
                teacher_repo, torch_dtype=dtype
            ).to(device).eval()
            vision_model = full_vlm.model.vision_model
        elif "clip" in teacher_repo.lower():
            vision_model = CLIPVisionModel.from_pretrained(
                teacher_repo, torch_dtype=dtype
            ).to(device).eval()
        else:
            vision_model = SiglipVisionModel.from_pretrained(
                teacher_repo, torch_dtype=dtype
            ).to(device).eval()

    _TEACHER_CACHE[teacher_repo] = vision_model
    return vision_model


@torch.no_grad()
def extract_teacher_embeddings(
    samples: list[tuple[Path, int]],
    teacher_repo: str,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    """L2-normalized pooled embeddings from a frozen CLIP-style vision tower.

    Supports three teacher families so the script works with whatever is
    available offline:
      * ``timm/MobileCLIP2-S0-OpenCLIP`` (timm)
      * ``openai/clip-vit-base-patch32`` (transformers CLIPVisionModel)
      * ``HuggingFaceTB/SmolVLM-500M-Instruct`` (transformers SiglipVisionModel)
    """
    vision_model = _load_teacher(teacher_repo, device)
    dtype = torch.float32

    embeddings: list[np.ndarray] = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        tensors = torch.stack([
            _teacher_preprocess(decode_image(str(path)).image, teacher_repo) for path, _ in batch
        ]).to(device, dtype=dtype)

        if hasattr(vision_model, "forward_features") and not hasattr(vision_model, "vision_model"):
            features = vision_model.forward_features(tensors)
            if features.ndim == 3:
                vecs = features.mean(dim=1)
            else:
                vecs = features
        else:
            out = vision_model(tensors)
            if getattr(out, "pooler_output", None) is not None:
                vecs = out.pooler_output
            else:
                vecs = out.last_hidden_state.mean(dim=1)

        vecs = vecs.float()
        vecs = vecs / (vecs.norm(dim=1, keepdim=True) + 1e-12)
        embeddings.append(vecs.cpu().numpy().astype(np.float32))

        if (start + batch_size) % 200 < batch_size or start + batch_size >= len(samples):
            print(f"[Teacher] Embedded {min(start + batch_size, len(samples))}/{len(samples)}")

    del vision_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(embeddings, axis=0)


def fit_teacher_logistic_probe(
    X_stack: np.ndarray,
    y_stack: np.ndarray,
    X_target: np.ndarray,
    l2_reg: float = 1.0,
) -> np.ndarray:
    """L2-regularized logistic probe on the stacking split -> teacher logits.

    Identical in spirit to ``distill_vision_tower_to_srm.fit_teacher_logistic_probe``
    so both engines are trained the same way.
    """
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

    return np.clip(Z_tg @ w, -15.0, 15.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ProbeDataset(Dataset):
    """Yields (tensor, hard label, teacher logit) with light augmentation."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        teacher_logits: Optional[np.ndarray],
        input_size: int = 224,
        is_train: bool = True,
    ):
        self.samples = samples
        self.teacher_logits = teacher_logits
        self.input_size = input_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = decode_image(str(path)).image.convert("RGB")
        if self.is_train:
            if random.random() < 0.5:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            # Mild JPEG/compression jitter so the student does not overfit to
            # the dataset's exact encoding.
            if random.random() < 0.4:
                import io
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=random.randint(70, 95))
                buf.seek(0)
                img = Image.open(buf).convert("RGB")
        tensor = _clip_preprocess(img, self.input_size)
        t_logit = float(self.teacher_logits[idx]) if self.teacher_logits is not None else 0.0
        return tensor, float(label), float(t_logit)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _weights_only_safe_state(epoch, model, optimizer, best_val_auc, contract, config_record, metrics):
    """Checkpoint structure matching ``training.checkpoint.load_verified_checkpoint``."""
    from training.checkpoint import atomic_torch_save

    npstate = np.random.get_state()
    state = {
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
        "cuda_rng": torch.cuda.get_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(),
        "numpy_rng": [npstate[0], npstate[1].tolist(), int(npstate[2]), int(npstate[3]), float(npstate[4])],
    }
    atomic_torch_save(state, contract["_checkpoint_path"])
    return state


def train_semantic_probe(
    data_root: Path,
    output_dir: Path,
    teacher_repo: str = "timm/MobileCLIP2-S0-OpenCLIP",
    input_size: int = 224,
    width: int = 32,
    epochs: int = 12,
    batch_size: int = 16,
    lr: float = 1e-3,
    alpha_kd: float = 0.6,
    temperature: float = 2.0,
    device_str: str = "auto",
    max_samples: int | None = None,
) -> Path:
    set_global_seed(42)
    provenance = validate_dataset_manifest(
        data_root, ("train", "val", "calibration", "test", "stacking")
    )
    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available())
        else ("cpu" if device_str == "auto" else device_str)
    )

    train_samples = discover_samples(data_root, "train", IMAGE_EXTS)
    val_samples = discover_samples(data_root, "val", IMAGE_EXTS)
    test_samples = discover_samples(data_root, "test", IMAGE_EXTS)
    stack_samples = discover_samples(data_root, "stacking", IMAGE_EXTS)

    if max_samples:
        # Smoke-test path only. A truncated run is NOT a valid release.
        from data.contracts import balanced_subset

        train_samples = balanced_subset(train_samples, max_samples)
        val_samples = balanced_subset(val_samples, max_samples)
        test_samples = balanced_subset(test_samples, max_samples)
        stack_samples = balanced_subset(stack_samples, max_samples)
        print(f"[Smoke] Truncated every split to {max_samples} samples - not a valid release")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_npz = output_dir / "semantic_teacher_cache.npz"

    if cache_npz.exists():
        print(f"[Teacher] Reusing cached embeddings from {cache_npz}")
        with np.load(cache_npz) as data:
            train_logits = data["train_teacher_logits"]
            val_logits = data["val_teacher_logits"]
    else:
        X_stack = extract_teacher_embeddings(stack_samples, teacher_repo, device)
        y_stack = np.asarray([label for _, label in stack_samples], dtype=np.int64)
        X_train = extract_teacher_embeddings(train_samples, teacher_repo, device)
        X_val = extract_teacher_embeddings(val_samples, teacher_repo, device)
        train_logits = fit_teacher_logistic_probe(X_stack, y_stack, X_train)
        val_logits = fit_teacher_logistic_probe(X_stack, y_stack, X_val)
        np.savez_compressed(
            cache_npz,
            train_teacher_logits=train_logits,
            val_teacher_logits=val_logits,
        )

    train_ds = ProbeDataset(train_samples, train_logits, input_size, is_train=True)
    val_ds = ProbeDataset(val_samples, val_logits, input_size, is_train=False)
    test_ds = ProbeDataset(test_samples, None, input_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SemanticProbeCNN(width=width).to(device)
    print(f"[Student] SemanticProbeCNN(width={width}) "
          f"params={model.count_parameters():,} stem=rgb-appearance")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    contract = {
        "frontend_id": f"clip-appearance-bicubic{input_size}-clipnorm-meanstd-float32",
        "input_shape": [1, input_size, input_size, 3],
        "architecture": "SemanticProbeCNN",
        "label_convention": "real=0_fake=1",
        "dataset_manifest_hash": provenance["manifest_sha256"],
        "source": provenance["source"],
        "revision": provenance["revision"],
        "training_code_hash": training_code_hash(),
        "audio_crop_policy": None,
        "engine": "C_semantic",
    }
    config_record = {
        "model": {"name": "SemanticProbeCNN", "input_size": input_size, "width": width},
        "data": {"root_dir": str(data_root.resolve()), "img_size": input_size},
        "training": {"epochs": epochs, "batch_size": batch_size,
                     "alpha_kd": alpha_kd, "temperature": temperature},
        "teacher": teacher_repo,
        "contract": contract,
    }

    best_val_auc = -1.0
    best_ckpt = output_dir / "semantic_probe_best.pth"
    contract["_checkpoint_path"] = str(best_ckpt)

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
            loss_kd = F.binary_cross_entropy_with_logits(
                s_logit / temperature, t_prob
            ) * (temperature ** 2)
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
        print(f"Epoch {epoch + 1:02d}/{epochs} | loss={total_loss / len(train_ds):.4f} | val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            contract.pop("_checkpoint_path", None)
            contract["training_code_hash"] = training_code_hash()
            metrics = {"auc": val_auc}
            _weights_only_safe_state(
                epoch, model, optimizer, best_val_auc, dict(contract, _checkpoint_path=str(best_ckpt)),
                config_record, metrics,
            )
            contract["_checkpoint_path"] = str(best_ckpt)
            model.to(device)

    from training.checkpoint import load_verified_checkpoint
    verified = load_verified_checkpoint(best_ckpt)
    model.load_state_dict(verified["state_dict"])
    model.to(device).eval()

    test_scores, test_labels = [], []
    with torch.no_grad():
        for x, y_hard, _ in test_loader:
            test_scores.extend(model(x.to(device)).cpu().flatten().tolist())
            test_labels.extend([int(v) for v in y_hard.tolist()])

    summary = {
        "engine": "C_semantic",
        "teacher": teacher_repo,
        "params": model.count_parameters(),
        "best_val_auc": best_val_auc,
        "test_auc": float(compute_auc(test_labels, test_scores)),
        "test_eer": float(compute_eer(test_labels, test_scores)),
        "test_accuracy_at_0_5": float(compute_accuracy(test_labels, test_scores)),
        "checkpoint_path": str(best_ckpt),
        "checkpoint_sha256": sha256_file(best_ckpt),
        "frontend_id": contract["frontend_id"],
        "input_shape": contract["input_shape"],
    }
    (output_dir / "semantic_probe_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return best_ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--teacher",
        default="timm/MobileCLIP2-S0-OpenCLIP",
        help="Frozen vision tower: timm/MobileCLIP2-S0-OpenCLIP, openai/clip-vit-base-patch32, or a SmolVLM repo id",
    )
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha-kd", type=float, default=0.6)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Truncate every split (smoke test only - a truncated run is not a valid release)",
    )
    args = parser.parse_args()

    train_semantic_probe(
        data_root=Path(args.data_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        teacher_repo=args.teacher,
        input_size=args.input_size,
        width=args.width,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha_kd=args.alpha_kd,
        device_str=args.device,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

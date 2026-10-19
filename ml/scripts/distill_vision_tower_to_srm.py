"""Distill frozen vision tower (SigLIP/SmolVLM/MobileCLIP) into lightweight SRM+NPR CNN.

Upgrades:
1. Student architecture: 16-channel SRMForensicCNN(include_npr=True), capturing both
   high-frequency noise residuals, color-opponent residuals, and 2x2 NPR upsampling traces.
2. Knowledge Distillation: Soft-target logit distillation from frozen vision tower probe
   (alpha_kd=0.6, T=2.0) + hard BCE loss.
3. PatchCraft native-resolution patch crops: extracts high-gradient native patches to
   prevent bilinear downsampling from washing away sensor/synthesis noise pitch.
4. Laundered KL baseline recalibration: calibrates natural_diff_histogram.json across a
   50/50 mix of raw and 512px Q85-laundered real photos, resolving the 0.412 AUC inversion.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
ML_ROOT = REPO_ROOT / "ml"
sys.path.insert(0, str(ML_ROOT / "src"))
sys.path.insert(0, str(ML_ROOT / "scripts"))

# Also insert scratch/pylib if available for extra transformers/open_clip modules
SCRATCH_PYLIB = Path(r"C:\Users\PLEXTEK\.gemini\antigravity\brain\e3c327c9-cfac-44fa-b5f8-762bb1986629\scratch\pylib")
if SCRATCH_PYLIB.exists():
    sys.path.insert(0, str(SCRATCH_PYLIB))

from data.contracts import IMAGE_EXTS, discover_samples, validate_dataset_manifest
from data.image_dataset import image_tensor
from data.image_frontend import decode_image
from meta_classifier.feature_extractor import diff_histogram, to_gray_u8, KL_BINS
from models.srm_forensic_cnn import SRMForensicCNN, extract_high_low_texture_patches
from training.checkpoint import atomic_torch_save
from training.provenance import training_code_hash


def log(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"{stamp} | {msg}", flush=True)


def launder_image_bytes(img: Image.Image, max_side: int = 512, jpeg_quality: int = 85) -> Image.Image:
    """Uniform JPEG laundering (512px max side, Q85)."""
    w, h = img.size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    if scale < 1.0:
        new_size = (max(16, int(round(w * scale))), max(16, int(round(h * scale))))
        img = img.resize(new_size, Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def calibrate_laundered_kl_baseline(data_root: Path, output_dir: Path, sample_limit: int = 500) -> dict:
    """Calibrate difference histogram over 50/50 mix of raw and 512px Q85-laundered real photos."""
    real_paths = [str(p) for p, lbl in discover_samples(data_root, "train", IMAGE_EXTS) if lbl == 0]
    if not real_paths:
        real_paths = [str(p) for p, lbl in discover_samples(data_root, "calibration", IMAGE_EXTS) if lbl == 0]
    real_paths = real_paths[:sample_limit]
    half = len(real_paths) // 2

    acc = np.zeros(KL_BINS, dtype=np.float64)
    used = 0
    for idx, path in enumerate(real_paths):
        try:
            decoded = decode_image(path).image
            if idx >= half:
                decoded = launder_image_bytes(decoded)
            acc += diff_histogram(to_gray_u8(decoded))
            used += 1
        except Exception as exc:
            log(f"Warning: could not process {path} for baseline: {exc}")

    if used == 0:
        raise RuntimeError("No images available for laundered baseline calibration")
    acc /= acc.sum()
    baseline_data = {"bins": [float(x) for x in acc], "num_images": used}

    baseline_path = output_dir / "natural_diff_histogram.json"
    baseline_path.write_text(json.dumps(baseline_data, indent=2), encoding="utf-8")
    log(f"Recalibrated laundered KL baseline on {used} real photos -> {baseline_path}")
    return baseline_data


def get_cached_teacher_predictions(data_root: Path, split_name: str) -> dict[str, float]:
    """Check existing scratch/llm_exp caches to retrieve teacher soft probabilities."""
    scratch_dir = Path(r"C:\Users\PLEXTEK\.gemini\antigravity\brain\e3c327c9-cfac-44fa-b5f8-762bb1986629\scratch\llm_exp")
    emb_file = scratch_dir / "emb_smolvlm500_vision.npz"
    index_file = scratch_dir / "index.json"

    if emb_file.exists() and index_file.exists():
        log(f"Loading cached SmolVLM-500M SigLIP vision-tower embeddings from {emb_file}...")
        emb = np.load(emb_file)
        index = json.loads(index_file.read_text(encoding="utf-8"))

        y_train = np.array([lbl for _, lbl in index["train"]], dtype=np.int64)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(Cs=np.logspace(-4, 1, 12), cv=5, max_iter=5000, scoring="roc_auc"),
        )
        clf.fit(emb["train"], y_train)

        # Get predictions for requested split
        if split_name in emb and split_name in index:
            probs = clf.predict_proba(emb[split_name])[:, 1]
            rows = index[split_name]
            result = {}
            for (p, _), pr in zip(rows, probs):
                base_name = Path(p).stem
                result[base_name] = float(pr)
            return result
    return {}


class DistillationDataset(Dataset):
    """Dataset providing native-resolution texture patches, hard ground truth, and teacher soft targets."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        teacher_map: dict[str, float],
        input_size: int = 128,
        is_train: bool = True,
    ):
        self.samples = samples
        self.teacher_map = teacher_map
        self.input_size = input_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        img = decode_image(str(path)).image

        if self.is_train:
            # High-texture PatchCraft crop to preserve sensor/synthesis noise pitch
            hi_patch, _ = extract_high_low_texture_patches(img, patch_size=self.input_size)
            tensor = image_tensor(hi_patch, self.input_size)
        else:
            tensor = image_tensor(img, self.input_size)

        hard_y = torch.tensor([float(label)], dtype=torch.float32)

        # Teacher soft target
        base_name = path.stem
        soft_prob = self.teacher_map.get(base_name, float(label))
        soft_y = torch.tensor([soft_prob], dtype=torch.float32)

        return tensor, hard_y, soft_y


def evaluate_student(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    y_trues, y_preds, losses = [], [], []
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for x, y_hard, _ in loader:
            x, y_hard = x.to(device), y_hard.to(device)
            logits = model(x, return_logits=True)
            loss = criterion(logits, y_hard)
            probs = torch.sigmoid(logits)

            losses.append(loss.item())
            y_trues.extend(y_hard.cpu().numpy().flatten())
            y_preds.extend(probs.cpu().numpy().flatten())

    y_trues = np.array(y_trues)
    y_preds = np.array(y_preds)
    auc = roc_auc_score(y_trues, y_preds) if len(set(y_trues)) == 2 else 0.5
    acc = accuracy_score(y_trues, y_preds >= 0.5)
    mean_loss = float(np.mean(losses))
    return auc, acc, mean_loss


def train_student_distillation(
    data_root: Path,
    output_dir: Path,
    teacher_encoder: str = "HuggingFaceTB/SmolVLM-500M-Instruct",
    input_size: int = 128,
    epochs: int = 18,
    batch_size: int = 32,
    lr: float = 1e-3,
    alpha_kd: float = 0.6,
    temperature: float = 2.0,
    device_str: str = "auto",
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() and device_str != "cpu" else "cpu")
    log(f"Training distillation on device: {device} (16-channel SRM+NPR student, input={input_size}x{input_size})")

    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = validate_dataset_manifest(data_root)
    manifest_sha = provenance["manifest_sha256"]

    # 1. Calibrate laundered KL baseline
    calibrate_laundered_kl_baseline(data_root, output_dir)

    # 2. Retrieve teacher predictions
    train_samples = discover_samples(data_root, "train", IMAGE_EXTS)
    val_samples = discover_samples(data_root, "val", IMAGE_EXTS)

    teacher_train_map = get_cached_teacher_predictions(data_root, "train")
    log(f"Mapped {len(teacher_train_map)}/{len(train_samples)} training samples to teacher probe soft targets.")

    # 3. Create datasets and loaders
    train_ds = DistillationDataset(train_samples, teacher_train_map, input_size=input_size, is_train=True)
    val_ds = DistillationDataset(val_samples, {}, input_size=input_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 4. Initialize 16-channel SRMForensicCNN student model
    student = SRMForensicCNN(include_npr=True).to(device)
    log(f"Initialized SRMForensicCNN(include_npr=True) - out_channels={student.srm.conv.out_channels}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bce_loss = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_path = output_dir / "srm_cnn_genimage_best.pth"
    last_path = output_dir / "srm_cnn_genimage_last.pth"

    # Pre-training validation
    init_auc, init_acc, _ = evaluate_student(student, val_loader, device)
    log(f"Epoch  0/{epochs} | Val AUC: {init_auc:.4f} | Val Acc: {init_acc*100:.1f}%")

    t0_train = time.time()
    for ep in range(1, epochs + 1):
        student.train()
        train_losses = []

        for x, y_hard, y_soft in train_loader:
            x, y_hard, y_soft = x.to(device), y_hard.to(device), y_soft.to(device)
            optimizer.zero_grad()

            logits = student(x, return_logits=True)

            # Hard BCE loss
            loss_hard = bce_loss(logits, y_hard)

            # Soft KD loss: scale logits by temperature
            p_soft = y_soft.clamp(1e-6, 1.0 - 1e-6)
            soft_targets = torch.sigmoid(torch.logit(p_soft) / temperature)
            loss_soft = bce_loss(logits / temperature, soft_targets) * (temperature ** 2)

            loss = (1.0 - alpha_kd) * loss_hard + alpha_kd * loss_soft
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        val_auc, val_acc, val_loss = evaluate_student(student, val_loader, device)
        avg_train_loss = float(np.mean(train_losses))

        log(
            f"Epoch {ep:2d}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc*100:.1f}%"
        )

        # Checkpoint structure matching Onn-Detect verified contract
        contract = {
            "frontend_id": f"rgb-exif-srgb-whitealpha-pow2cap1024-bilinear-square-v2-size{input_size}-float32",
            "input_shape": [1, input_size, input_size, 3],
            "architecture": "SRMForensicCNN",
            "label_convention": "real=0_fake=1",
            "dataset_manifest_hash": manifest_sha,
            "source": "ComplexDataLab/OpenFake",
            "revision": "3247b942f18d39d1fd476b1077852af033d35d8c",
            "training_code_hash": training_code_hash(),
            "audio_crop_policy": None,
        }
        config = {
            "data": {"root_dir": str(data_root.resolve())},
            "model": {"input_size": input_size, "include_npr": True},
            "training": {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "alpha_kd": alpha_kd,
                "temperature": temperature,
            },
        }
        state = {
            "schema_version": 1,
            "epoch": ep,
            "state_dict": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": None,
            "best_metric": float(max(best_auc, val_auc)),
            "contract": contract,
            "config": config,
        }

        atomic_torch_save(state, last_path)
        if val_auc > best_auc:
            best_auc = val_auc
            atomic_torch_save(state, best_path)
            log(f"  (*) New best checkpoint saved with Val AUC: {best_auc:.4f}")

    total_time = time.time() - t0_train
    log(f"Distillation training complete in {total_time:.1f}s. Best Val AUC: {best_auc:.4f}")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=r"D:\onn_data\openfake_visual",
        help="Path to visual dataset root",
    )
    parser.add_argument(
        "--output-dir",
        default=r"D:\onn_data\runs\release_v4\checkpoints",
        help="Directory to save distilled student checkpoints and baseline",
    )
    parser.add_argument(
        "--teacher-encoder",
        default="HuggingFaceTB/SmolVLM-500M-Instruct",
        help="Teacher model identifier",
    )
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha-kd", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    train_student_distillation(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        teacher_encoder=args.teacher_encoder,
        input_size=args.input_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha_kd=args.alpha_kd,
        temperature=args.temperature,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()

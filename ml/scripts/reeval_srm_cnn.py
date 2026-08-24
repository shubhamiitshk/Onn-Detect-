"""Re-evaluate trained SRM-CNN checkpoints on a held-out subset of genimage/val.

Uses the SAME forensics-preserving preprocessing as training and the Android
runtime: native 32x32 input, ToTensor only - NO resize, NO ImageNet
normalization. (The original 224x224 + Normalize pipeline interpolated away
the noise statistics and produced the historical chance-level AUC 0.5.)
"""
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from models.srm_forensic_cnn import SRMForensicCNN
from data.cifake_dataset import CIFAKEDataset
from training.metrics import compute_auc, compute_accuracy
from utils.seed import set_global_seed

CKPT = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'srm_cnn_genimage_best.pth')
OUT = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'srm_cnn_genimage_results.json')
N_PER_CLASS = 2000  # 4000 total eval images


def main():
    set_global_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Native resolution, no normalization - identical to train/eval pipeline.
    val_transform = transforms.Compose([transforms.ToTensor()])

    dataset = CIFAKEDataset(
        root_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'genimage')),
        split='val',
        img_size=32,
        transform=val_transform,
        cache_in_memory=False,
    )
    if len(dataset) == 0:
        print("Val split empty - nothing to evaluate.")
        sys.exit(1)

    # Balanced subset by label (dataset was shuffled at init)
    real_idx = [i for i, y in enumerate(dataset.labels) if y == 0][:N_PER_CLASS]
    fake_idx = [i for i, y in enumerate(dataset.labels) if y == 1][:N_PER_CLASS]
    subset = Subset(dataset, real_idx + fake_idx)

    loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    model = SRMForensicCNN()
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=True)
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    best_auc_seen = ckpt.get('best_metric', None)
    if best_auc_seen is not None:
        print(f"Checkpoint best val AUC recorded during training: {float(best_auc_seen):.4f}")

    preds, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            out = model(images).squeeze(1).float().cpu().numpy()
            preds.extend(out.tolist())
            labels.extend(targets.numpy().tolist())

    auc = compute_auc(labels, preds)
    acc = compute_accuracy(labels, preds)

    print("=" * 40)
    print("SRM-CNN honest re-evaluation (genimage/val subset, native 32x32)")
    print(f"Samples: {len(labels)} | AUC: {auc:.4f} | Acc: {acc:.4f}")
    print("=" * 40)

    with open(OUT, 'w') as f:
        json.dump({
            'test_auc': float(auc),
            'test_accuracy': float(acc),
            'eval_split': 'val_subset_native_res_32',
            'eval_samples': len(labels),
            'input_size': 32,
            'checkpoint_val_auc': float(best_auc_seen) if best_auc_seen is not None else None,
            'model_name': 'srm_cnn_genimage',
        }, f, indent=4)
    print(f"Saved -> {OUT}")


if __name__ == '__main__':
    main()

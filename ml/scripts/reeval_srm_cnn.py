"""Re-evaluate trained SRM-CNN checkpoints on a held-out subset of genimage/val.

The previous results JSON reported AUC 0.5 because datasets/genimage/test is
EMPTY (0 files). This script evaluates honestly on the val split and writes
a fresh results file.
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

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = CIFAKEDataset(
        root_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'genimage')),
        split='val',
        img_size=224,
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
    print("SRM-CNN honest re-evaluation (genimage/val subset)")
    print(f"Samples: {len(labels)} | AUC: {auc:.4f} | Acc: {acc:.4f}")
    print("=" * 40)

    with open(OUT, 'w') as f:
        json.dump({
            'test_auc': float(auc),
            'test_accuracy': float(acc),
            'eval_split': 'val',
            'eval_samples': len(labels),
            'checkpoint_val_auc': float(best_auc_seen) if best_auc_seen is not None else None,
            'model_name': 'srm_cnn_genimage',
        }, f, indent=4)
    print(f"Saved -> {OUT}")


if __name__ == '__main__':
    main()

"""Retrain SRM-CNN on CIFAKE/GenImage thumbnails at NATIVE resolution (32x32).

Root cause of previous chance-level results (AUC 0.5000):
1. 32x32 source images were bilinear-upscaled to 224x224 before SRM filtering,
   interpolating away exactly the high-frequency noise statistics the model
   must learn from.
2. ColorJitter augmentation corrupted the forensic noise statistics.

This script uses a forensics-preserving pipeline:
- Native 32x32 input, no resize (SRMForensicCNN is fully convolutional up to
  an AdaptiveAvgPool, so it accepts any spatial size).
- Only geometry-preserving flips as augmentation.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from models.srm_forensic_cnn import SRMForensicCNN
from data.cifake_dataset import CIFAKEDataset
from training.trainer import Trainer
from training.metrics import compute_auc, compute_accuracy
from utils.seed import set_global_seed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'genimage'))
CKPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))
EVAL_PER_CLASS = 2500


def main():
    set_global_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Forensics-preserving transforms: NO resize, NO color jitter
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),  # [0,1]; SRM filters expect un-normalized pixel stats
    ])
    eval_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = CIFAKEDataset(ROOT, split='train', img_size=32,
                                  transform=train_transform, cache_in_memory=False)
    val_dataset = CIFAKEDataset(ROOT, split='val', img_size=32,
                                transform=eval_transform, cache_in_memory=False)

    # Balanced held-out eval subset from val
    real_idx = [i for i, y in enumerate(val_dataset.labels) if y == 0][:EVAL_PER_CLASS]
    fake_idx = [i for i, y in enumerate(val_dataset.labels) if y == 1][:EVAL_PER_CLASS]
    test_subset = Subset(val_dataset, real_idx + fake_idx)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(test_subset, batch_size=512, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = SRMForensicCNN()
    print(f"Trainable params: {model.count_parameters():,}")

    config = {
        'training': {'epochs': 12, 'amp': True, 'patience': 5},
        'optimizer': {'lr': 8e-4},
        'checkpoint_dir': CKPT_DIR,
        'model_name': 'srm_cnn_genimage',
    }
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, config=config)
    trainer.train()

    # Honest final evaluation with best checkpoint
    ckpt_path = os.path.join(CKPT_DIR, 'srm_cnn_genimage_best.pth')
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()

    preds, labels = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            out = model(images.to(device)).squeeze(1).float().cpu().numpy()
            preds.extend(out.tolist())
            labels.extend(targets.numpy().tolist())

    auc = compute_auc(labels, preds)
    acc = compute_accuracy(labels, preds)
    print("=" * 40)
    print(f"Native-res Test (val subset): AUC {auc:.4f} | Acc {acc:.4f}")
    print("=" * 40)

    with open(os.path.join(CKPT_DIR, 'srm_cnn_genimage_results.json'), 'w') as f:
        np.save(os.path.join(CKPT_DIR, 'srm_val_preds.npy'), np.array(preds))
        json_dump = {
            'test_auc': float(auc),
            'test_accuracy': float(acc),
            'eval_split': 'val_subset_native_res_32',
            'eval_samples': len(labels),
            'input_size': 32,
            'checkpoint_val_auc': float(checkpoint.get('best_metric', 0.0)),
            'model_name': 'srm_cnn_genimage',
        }
        import json
        json.dump(json_dump, f, indent=4)


if __name__ == '__main__':
    main()

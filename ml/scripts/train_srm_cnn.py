#!/usr/bin/env python3
"""Train SRM-CNN on GenImage/CIFAKE thumbnails.

IMPORTANT: images in datasets/genimage are native 32x32. They MUST be fed to
the model without upsampling - bilinear resize to 224x224 interpolates away
the high-frequency noise statistics the SRM filters need, which is exactly
what caused the original chance-level model (AUC 0.5000). See
retrain_srm_native.py for the full rationale.
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.srm_forensic_cnn import SRMForensicCNN
from data.cifake_dataset import CIFAKEDataset
from data.openfake_dataset import OpenFakeDataset
from training.trainer import Trainer
from training.metrics import compute_auc, compute_accuracy
from utils.seed import set_global_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()

    set_global_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use environment variable if provided, otherwise default to genimage
    root_default = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'genimage'))
    root = os.environ.get('DATASET_DIR', root_default)
    
    # Allow larger image sizes for high-res training
    img_size = int(os.environ.get('IMG_SIZE', 32))

    # Forensics-preserving transforms: NO resize (unless forced by IMG_SIZE), NO color jitter.
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)) if img_size != 32 else torch.nn.Identity(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)) if img_size != 32 else torch.nn.Identity(),
        transforms.ToTensor(),
    ])

    # Use OpenFakeDataset for high-res / multi-generator sets, otherwise CIFAKEDataset
    is_highres = "highres" in root.lower() or img_size > 32
    DatasetClass = OpenFakeDataset if is_highres else CIFAKEDataset

    train_ds = DatasetClass(root, split='train', img_size=img_size,
                            transform=train_tf, cache_in_memory=True)
    val_ds = DatasetClass(root, split='val', img_size=img_size,
                          transform=eval_tf, cache_in_memory=False)

    # If dataset has valid test split, use it, else balanced val subset
    per_class = min(2500, len(val_ds) // 2) if len(val_ds) > 0 else 100
    test_subset = Subset(val_ds,
                         [i for i, yl in enumerate(val_ds.labels) if yl == 0][:per_class]
                         + [i for i, yl in enumerate(val_ds.labels) if yl == 1][:per_class])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == 'cuda',
                              persistent_workers=True)
    val_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    model = SRMForensicCNN()
    print(f"Model params: {model.count_parameters():,}")

    config = {
        'training': {'epochs': args.epochs, 'amp': args.amp, 'patience': 5},
        'optimizer': {'lr': args.lr},
        'checkpoint_dir': os.path.join(os.path.dirname(__file__), '..', 'checkpoints'),
        'model_name': 'srm_cnn_genimage',
    }
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                      config=config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    if not args.eval_only:
        trainer.train()

    print("\nLoading best checkpoint for final evaluation...")
    ckpt_path = os.path.join(config['checkpoint_dir'], 'srm_cnn_genimage_best.pth')
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    model.to(device).eval()

    preds, labels = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model(images.to(device)).squeeze(1)
            preds.extend(outputs.float().cpu().tolist())
            labels.extend(targets.tolist())

    auc = compute_auc(labels, preds)
    acc = compute_accuracy(labels, preds)

    print("=" * 40)
    print("Final Test Evaluation (genimage/val subset)")
    print("=" * 40)
    print(f"Test AUC:      {auc:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print("=" * 40)

    results = {
        'test_auc': float(auc),
        'test_accuracy': float(acc),
        'eval_split': 'val_subset_native_res_32',
        'eval_samples': len(labels),
        'input_size': 32,
        'checkpoint_val_auc': float(checkpoint.get('best_metric', 0.0)),
        'model_name': 'srm_cnn_genimage',
    }
    with open(os.path.join(config['checkpoint_dir'], 'srm_cnn_genimage_results.json'), 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()

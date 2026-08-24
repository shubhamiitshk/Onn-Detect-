import argparse
import os
import yaml
import json
import time
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Dict, Any

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.lcnn_mfm_audio import LCNNMFMAudio
from data.asvspoof_dataset import ASVspoofDataset
from data.inthewild_audio import InTheWildAudioDataset
from training.trainer import Trainer
from training.metrics import compute_eer, compute_auc
from utils.seed import set_global_seed
from utils.device import get_device

def parse_args():
    parser = argparse.ArgumentParser(description="Train LCNN-MFM on ASVspoof")
    parser.add_argument("--config", type=str, default="configs/lcnn_asvspoof.yaml")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=str)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    return parser.parse_args()

def load_config(args) -> Dict[str, Any]:
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    if args.batch_size: config['training']['batch_size'] = args.batch_size
    if args.epochs: config['training']['epochs'] = args.epochs
    if args.lr: config['optimizer']['lr'] = args.lr
    if args.num_workers: config['training']['num_workers'] = args.num_workers
    if args.amp: config['training']['amp'] = True
        
    config['checkpoint_dir'] = args.output_dir
    config['model_name'] = 'lcnn_audio_asvspoof'
    return config

def plot_det_curve(y_true, y_pred, save_path):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    fnr = 1 - tpr
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, fnr, color='blue', lw=2, label='DET curve')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title('Detection Error Tradeoff (DET) Curve')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def main():
    args = parse_args()
    config = load_config(args)
    
    set_global_seed(42)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    print(f"[DEVICE] Using device: {device}")
    
    root_dir = config['data']['root_dir']
    max_duration = config['data']['max_duration']
    sample_rate = config['data']['sample_rate']
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    
    dataset_type = config['data'].get('dataset', 'asvspoof2019')
    DatasetClass = InTheWildAudioDataset if dataset_type == 'inthewild_audio' else ASVspoofDataset
    
    print(f"Loading {dataset_type} datasets...")
    train_dataset = DatasetClass(root_dir=root_dir, split='train', max_duration=max_duration, sample_rate=sample_rate)
    val_dataset = DatasetClass(root_dir=root_dir, split='val', max_duration=max_duration, sample_rate=sample_rate)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    model = LCNNMFMAudio()
    print(f"Model params: {model.count_parameters():,}")
    
    # Customize early stopping and checkpointing for EER (min is better)
    config['early_stopping']['mode'] = 'min'
    config['early_stopping']['metric'] = 'val_eer'
    
    trainer = Trainer(model=model, train_loader=train_loader,
                      val_loader=val_loader, config=config, device=device)
    
    # Overwrite trainer callbacks to use EER minimization
    from training.callbacks import EarlyStopping, ModelCheckpoint
    trainer.early_stopping = EarlyStopping(patience=config['early_stopping']['patience'], mode='min')
    trainer.checkpoint = ModelCheckpoint(save_dir=config['checkpoint_dir'], model_name=config['model_name'], mode='min')
    
    # Patch train loop to track EER instead of AUC
    def patched_train():
        history = {'train_loss': [], 'val_loss': [], 'val_eer': []}
        for epoch in range(trainer.start_epoch, trainer.max_epochs):
            print(f"\\nEpoch {epoch+1}/{trainer.max_epochs}")
            current_lr = trainer.scheduler.step(epoch)
            train_loss = trainer.train_epoch()
            val_metrics = trainer.validate()
            
            val_eer = val_metrics.get('eer', 1.0)
            val_loss = val_metrics.get('loss', 0.0)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_eer'].append(val_eer)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val EER: {val_eer:.4f} | Val AUC: {val_metrics.get('auc', 0.0):.4f}")
            trainer.checkpoint(val_eer, trainer.model, trainer.optimizer, epoch)
            if trainer.early_stopping(val_eer):
                print("[STOP] Early stopping triggered!")
                break
        return history
    
    trainer.train = patched_train
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
        
    if not args.eval_only:
        history = trainer.train()
    
    best_ckpt_path = os.path.join(config['checkpoint_dir'], f"{config['model_name']}_best.pth")
    print(f"\n[EVAL] Loading best model from {best_ckpt_path} for Test evaluation...")
    
    model = LCNNMFMAudio.from_pretrained(best_ckpt_path).to(device).eval()
    
    test_dataset = DatasetClass(root_dir=root_dir, split='test', max_duration=max_duration, sample_rate=sample_rate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().squeeze(-1).numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            
    test_eer = compute_eer(all_targets, all_preds)
    test_auc = compute_auc(all_targets, all_preds)
    
    print("="*40)
    print("Final Audio Test Evaluation")
    print(f"Test EER:           {test_eer:.4f}")
    print(f"Test AUC:           {test_auc:.4f}")
    print("="*40)
    
    results = {"test_eer": test_eer, "test_auc": test_auc, "model_name": config['model_name']}
    with open(os.path.join(config['checkpoint_dir'], f"{config['model_name']}_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
        
    plot_det_curve(all_targets, all_preds, os.path.join(config['checkpoint_dir'], "lcnn_det_curve.png"))

if __name__ == "__main__":
    main()

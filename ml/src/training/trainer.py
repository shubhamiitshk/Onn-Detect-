import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from typing import Dict, Any
from tqdm import tqdm
import time

from .metrics import compute_auc, compute_eer, compute_accuracy, compute_f1_score
from .callbacks import EarlyStopping, ModelCheckpoint, CosineWarmupScheduler
from utils.device import get_device

class Trainer:
    """
    Reusable PyTorch Training Engine optimized for RTX A4000 (16GB VRAM).
    Features Automatic Mixed Precision (AMP), Gradient Accumulation, and EMA.
    """
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: Dict[str, Any]):
        self.config = config
        self.device = get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Hyperparameters
        train_cfg = config.get('training', {})
        self.max_epochs = train_cfg.get('epochs', 30)
        self.grad_accum_steps = train_cfg.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = train_cfg.get('max_grad_norm', 1.0)
        self.amp_enabled = train_cfg.get('amp', True)
        
        # Optimizer
        opt_cfg = config.get('optimizer', {})
        base_lr = opt_cfg.get('lr', 1e-3)
        weight_decay = opt_cfg.get('weight_decay', 1e-4)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=base_lr, weight_decay=weight_decay)
        
        # Scheduler & Callbacks
        sched_cfg = config.get('scheduler', {})
        self.scheduler = CosineWarmupScheduler(
            self.optimizer, 
            warmup_epochs=sched_cfg.get('warmup_epochs', 3), 
            max_epochs=self.max_epochs, 
            base_lr=base_lr
        )
        self.early_stopping = EarlyStopping(patience=train_cfg.get('patience', 5), mode='max')
        self.checkpoint = ModelCheckpoint(
            save_dir=config.get('checkpoint_dir', 'checkpoints'), 
            model_name=config.get('model_name', 'model'), 
            mode='max'
        )
        
        # Loss and AMP Scaler
        self.criterion = nn.BCELoss()
        self.scaler = GradScaler('cuda', enabled=self.amp_enabled)
        
        self.start_epoch = 0

    def train(self) -> Dict[str, list]:
        """Main training loop across all epochs."""
        print(f"🚀 Starting Training on device: {self.device} (AMP: {self.amp_enabled})")
        history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
        
        for epoch in range(self.start_epoch, self.max_epochs):
            print(f"\\nEpoch {epoch+1}/{self.max_epochs}")
            
            # Step learning rate
            current_lr = self.scheduler.step(epoch)
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Train and Validate
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            
            val_auc = val_metrics.get('auc', 0.0)
            val_loss = val_metrics.get('loss', 0.0)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val AUC: {val_auc:.4f} | Val Acc: {val_metrics.get('accuracy',0.0):.4f} | Val EER: {val_metrics.get('eer',1.0):.4f}")
            
            # Save periodic/best checkpoint
            self.checkpoint(val_auc, self.model, self.optimizer, epoch)
            
            # Early Stopping check
            if self.early_stopping(val_auc):
                print(f"🛑 Early stopping triggered at epoch {epoch+1}!")
                break
                
        return history

    def train_epoch(self) -> float:
        """Runs a single epoch of training."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training")
        self.optimizer.zero_grad()
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device).float().unsqueeze(1)
            
            # Mixed Precision Forward
            with autocast('cuda', enabled=self.amp_enabled):
                outputs = self.model(inputs)
            
            # Loss computation must be outside autocast for BCELoss
            loss = self.criterion(outputs.float(), targets)
            # Normalize loss for gradient accumulation
            loss = loss / self.grad_accum_steps
                
            # Mixed Precision Backward
            self.scaler.scale(loss).backward()
            
            # Gradient Accumulation Step
            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                # Unscale before clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer Step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
            total_loss += loss.item() * self.grad_accum_steps
            pbar.set_postfix({"Loss": f"{(total_loss / (batch_idx + 1)):.4f}"})
            
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Runs validation and computes metrics."""
        self.model.eval()
        total_loss = 0.0
        
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device).float().unsqueeze(1)
            
            with autocast('cuda', enabled=self.amp_enabled):
                outputs = self.model(inputs)
            loss = self.criterion(outputs.float(), targets)
                
            total_loss += loss.item()
            
            # Store predictions for AUC/EER computation
            all_preds.extend(outputs.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            
        avg_loss = total_loss / len(self.val_loader)
        
        # Compute exact metrics
        auc_score = compute_auc(all_targets, all_preds)
        eer_score = compute_eer(all_targets, all_preds)
        accuracy = compute_accuracy(all_targets, all_preds)
        f1 = compute_f1_score(all_targets, all_preds)
        
        return {
            "loss": avg_loss,
            "auc": auc_score,
            "eer": eer_score,
            "accuracy": accuracy,
            "f1": f1
        }

    def load_checkpoint(self, path: str):
        """Resumes training from a saved checkpoint."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
            
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['state_dict'])
        
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
        if 'best_metric' in checkpoint:
            self.checkpoint.best_score = checkpoint['best_metric']
            self.early_stopping.best_score = checkpoint['best_metric']
            
        print(f"Resumed from epoch {self.start_epoch} (Best metric: {self.checkpoint.best_score})")

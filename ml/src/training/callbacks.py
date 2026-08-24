import torch
import os
import math
from typing import Dict, Any

class EarlyStopping:
    """
    Early stopping to terminate training if a validation metric stops improving.
    """
    def __init__(self, patience: int = 5, mode: str = 'max', min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.is_better = lambda new, best: new > best + min_delta if mode == 'max' else new < best - min_delta

    def __call__(self, metric_value: float) -> bool:
        if self.best_score is None:
            self.best_score = metric_value
        elif not self.is_better(metric_value, self.best_score):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = metric_value
            self.counter = 0
            
        return self.early_stop


class ModelCheckpoint:
    """
    Saves the best model based on a monitored metric.
    """
    def __init__(self, save_dir: str, model_name: str, mode: str = 'max'):
        self.save_dir = save_dir
        self.model_name = model_name
        self.mode = mode
        self.best_score = None
        
        os.makedirs(save_dir, exist_ok=True)
        self.is_better = lambda new, best: new > best if mode == 'max' else new < best

    def __call__(self, metric_value: float, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
        if self.best_score is None or self.is_better(metric_value, self.best_score):
            print(f"\n🌟 Metric improved from {self.best_score} to {metric_value:.5f}. Saving checkpoint...")
            self.best_score = metric_value
            
            save_path = os.path.join(self.save_dir, f"{self.model_name}_best.pth")
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_metric': self.best_score
            }
            torch.save(state, save_path)


class CosineWarmupScheduler:
    """
    Learning Rate Scheduler: Linear Warmup followed by Cosine Annealing.
    Optimized for stable training on large batches (e.g. 128 on RTX A4000).
    """
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int, max_epochs: int, base_lr: float, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * ((epoch + 1) / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(1, (self.max_epochs - self.warmup_epochs))
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

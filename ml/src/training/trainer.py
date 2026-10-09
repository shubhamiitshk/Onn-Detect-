"""Reusable training engine with explicit precision, metrics and resumable state."""
import copy
import random
import numpy as np
import torch
from torch.amp import autocast, GradScaler
from .metrics import compute_auc, compute_eer, compute_accuracy, compute_f1_score
from .callbacks import EarlyStopping, CosineWarmupScheduler
from .checkpoint import atomic_torch_save, load_verified_checkpoint
from utils.device import get_device
from pathlib import Path

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device=None):
        self.config = config
        self.device = torch.device(device) if device is not None else get_device()
        self.model = model.to(self.device)
        if not hasattr(model, "forward_logits"):
            raise ValueError("Model must expose forward_logits for numerically stable training")
        self.train_loader, self.val_loader = train_loader, val_loader
        if not len(train_loader) or not len(val_loader):
            raise ValueError("Training and model-selection validation loaders must be nonempty")
        cfg = config.get("training", {})
        self.max_epochs = int(cfg.get("epochs", 30))
        if self.max_epochs < 1:
            raise ValueError("Training epochs must be positive")
        self.grad_accum_steps = int(cfg.get("gradient_accumulation_steps", 1))
        if self.grad_accum_steps < 1:
            raise ValueError("Gradient accumulation must be positive")
        self.max_grad_norm = cfg.get("gradient_clip", cfg.get("max_grad_norm", 1.0))
        precision = cfg.get("precision", "fp16" if cfg.get("amp", True) else "fp32")
        if precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("Unsupported training precision")
        self.amp_enabled = bool(cfg.get("amp", True)) and precision != "fp32" and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        opt = config.get("optimizer", {})
        if opt.get("name", "AdamW") != "AdamW":
            raise ValueError("Only AdamW is implemented")
        lr = opt.get("lr", 1e-3)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=opt.get("weight_decay", 1e-4))
        sched = config.get("scheduler", {})
        self.scheduler = CosineWarmupScheduler(self.optimizer, sched.get("warmup_epochs", 3), self.max_epochs, lr, sched.get("eta_min", 1e-6))
        early = config.get("early_stopping", {})
        self.monitor = early.get("metric", "val_auc").removeprefix("val_")
        if self.monitor not in ("auc", "eer", "loss", "accuracy", "f1"):
            raise ValueError("Unsupported monitored validation metric")
        self.mode = early.get("mode", "min" if self.monitor in ("loss", "eer") else "max")
        self.early_stopping = EarlyStopping(early.get("patience", cfg.get("patience", 5)), self.mode)
        self.best_score = None
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.scaler = GradScaler("cuda", enabled=self.amp_enabled and self.amp_dtype == torch.float16)
        self.start_epoch = 0
        self.history = {"train_loss": [], "validation": []}
        self.contract = config.get("contract")
        if not self.contract:
            raise ValueError("Training requires a versioned dataset/frontend contract")
        self.save_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.model_name = config.get("model_name", "model")
        if cfg.get("ema", False):
            raise ValueError("EMA is not implemented; do not advertise an unused setting")

    def _state(self, epoch):
        npstate = np.random.get_state()
        return {"schema_version": 1, "epoch": epoch, "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict(),
                "best_metric": self.best_score, "contract": self.contract, "config": self.config,
                "early_stopping": {"counter": self.early_stopping.counter, "best_score": self.early_stopping.best_score, "early_stop": self.early_stopping.early_stop},
                "history": self.history, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "python_rng": random.getstate(),
                "numpy_rng": [npstate[0], npstate[1].tolist(), int(npstate[2]), int(npstate[3]), float(npstate[4])]}

    def train(self):
        for epoch in range(self.start_epoch, self.max_epochs):
            self.scheduler.step(epoch)
            loss = self.train_epoch()
            metrics = self.validate()
            monitored = metrics[self.monitor]
            better = self.best_score is None or (monitored > self.best_score if self.mode == "max" else monitored < self.best_score)
            if better:
                self.best_score = monitored
            stopped = self.early_stopping(monitored)
            self.history["train_loss"].append(loss)
            self.history["validation"].append(metrics)
            state = self._state(epoch)
            atomic_torch_save(state, self.save_dir / f"{self.model_name}_last.pth")
            if better:
                atomic_torch_save(state, self.save_dir / f"{self.model_name}_best.pth")
            print(f"Epoch {epoch + 1}: train loss={loss:.6f}, validation={metrics}")
            if stopped:
                break
        return self.history

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total, count, accumulated_samples = 0.0, 0, 0
        length = len(self.train_loader)
        for index, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device, dtype=torch.float32)
            targets = targets.to(self.device, dtype=torch.float32).reshape(-1, 1)
            if not torch.isfinite(inputs).all():
                raise ValueError("Non-finite training inputs")
            if not torch.isfinite(targets).all() or not ((targets == 0) | (targets == 1)).all():
                raise ValueError("Training requires finite binary labels")
            with autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                loss = self.criterion(self.model.forward_logits(inputs), targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; checkpoint will not be written")
            # Accumulate summed sample losses, then normalize by the actual
            # group size. Unequal final minibatches must not get equal weight.
            batch_samples = targets.shape[0]
            accumulated_samples += batch_samples
            self.scaler.scale(loss * batch_samples).backward()
            if (index + 1) % self.grad_accum_steps == 0 or index + 1 == length:
                self.scaler.unscale_(self.optimizer)
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_samples)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                accumulated_samples = 0
            total += float(loss.detach()) * targets.shape[0]
            count += targets.shape[0]
        return total / count

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total, count, scores, labels = 0.0, 0, [], []
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device, dtype=torch.float32)
            targets = targets.to(self.device, dtype=torch.float32).reshape(-1, 1)
            if not torch.isfinite(inputs).all() or not torch.isfinite(targets).all() or not ((targets == 0) | (targets == 1)).all():
                raise ValueError("Validation requires finite inputs and binary labels")
            logits = self.model.forward_logits(inputs)
            loss = self.criterion(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite validation loss")
            total += float(loss) * targets.shape[0]
            count += targets.shape[0]
            scores.extend(torch.sigmoid(logits).float().cpu().flatten().tolist())
            labels.extend(targets.cpu().flatten().tolist())
        return {"loss": total / count, "auc": float(compute_auc(labels, scores)),
                "eer": float(compute_eer(labels, scores)), "accuracy": float(compute_accuracy(labels, scores)),
                "f1": float(compute_f1_score(labels, scores))}

    def load_checkpoint(self, path):
        state = load_verified_checkpoint(path, self.contract)
        # Resume changes to optimization/data order are not faithful resumes.
        for field in ("training", "optimizer", "scheduler", "early_stopping"):
            if state["config"].get(field) != self.config.get(field):
                raise ValueError(f"Resume {field} configuration differs")
        self.model.load_state_dict(state["state_dict"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        self.start_epoch = state["epoch"] + 1
        self.best_score = state["best_metric"]
        for key, value in state["early_stopping"].items():
            setattr(self.early_stopping, key, value)
        self.history = state["history"]
        torch.set_rng_state(state["torch_rng"])
        if self.device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        ns = state["numpy_rng"]
        np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))

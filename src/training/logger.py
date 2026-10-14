"""
src/training/logger.py
Bộ ghi chép Nhật ký Huấn luyện (Logger) và Lưu trữ Mô hình (Checkpoint).
"""
import os
from uuid import uuid4
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

class TrainerLogger:
    def __init__(
        self,
        log_dir: str = "logs/tensorboard",
        checkpoint_dir: str = "models/checkpoints",
        best_checkpoint_name: str = "best_model.pth",
    ):
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.best_checkpoint_name = best_checkpoint_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.best_reward = -float('inf')
        
    def log_metrics(self, step: int, metrics: dict):
        for tag, value in metrics.items():
            self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    @staticmethod
    def _atomic_save(checkpoint, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{uuid4().hex}.tmp")
        torch.save(checkpoint, tmp)
        tmp.replace(path)

    def save_latest(self, step, model, optimizer, metadata):
        checkpoint = {
            "step": step, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), **metadata,
        }
        self._atomic_save(checkpoint, self.checkpoint_dir / "last.pth")
            
    def save_checkpoint(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        current_reward: float,
        selection_metric: str = "reward",
        metadata: dict | None = None,
    ):
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'reward': current_reward,
            'selection_score': current_reward,
            'selection_metric': selection_metric,
        }
        if metadata:
            checkpoint.update(metadata)
        self._atomic_save(checkpoint, self.checkpoint_dir / f"checkpoint_step_{step}.pth")
        
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            self._atomic_save(checkpoint, self.checkpoint_dir / self.best_checkpoint_name)
            
    def close(self):
        self.writer.close()

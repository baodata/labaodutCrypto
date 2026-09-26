"""
src/training/logger.py
Bộ ghi chép Nhật ký Huấn luyện (Logger) và Lưu trữ Mô hình (Checkpoint).
"""
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

class TrainerLogger:
    def __init__(self, log_dir: str = "logs/tensorboard", checkpoint_dir: str = "models/checkpoints"):
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.best_reward = -float('inf')
        
    def log_metrics(self, step: int, metrics: dict):
        for tag, value in metrics.items():
            self.writer.add_scalar(tag, value, step)
            
    def save_checkpoint(self, step: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, current_reward: float):
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'reward': current_reward
        }
        torch.save(checkpoint, self.checkpoint_dir / f"checkpoint_step_{step}.pth")
        
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            torch.save(checkpoint, self.checkpoint_dir / "best_model.pth")
            
    def close(self):
        self.writer.close()

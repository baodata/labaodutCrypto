"""
src/train/logger.py
Bộ ghi chép Nhật ký Huấn luyện (Logger) và Lưu trữ Mô hình (Checkpoint).

Ticket: TRAIN-001 (Sprint 5)
Mục đích:
1. Vẽ biểu đồ quá trình học của AI (Loss, Reward, Lợi nhuận) lên TensorBoard.
2. Lưu lại "Bộ não" (.pth) của AI sau mỗi chu kỳ hoặc khi có kỷ lục lợi nhuận mới,
   tránh việc cúp điện làm mất trắng công sức huấn luyện.
"""
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

class TrainerLogger:
    def __init__(self, log_dir: str = "logs/tensorboard", checkpoint_dir: str = "models/checkpoints"):
        """
        Khởi tạo Bộ ghi chép.
        """
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Tự động tạo thư mục nếu chưa có
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Khởi tạo TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        self.best_reward = -float('inf')
        
    def log_metrics(self, step: int, metrics: dict):
        """
        Ghi lại các thông số vào TensorBoard.
        Args:
            step: Vòng lặp thứ bao nhiêu (Epoch/Step)
            metrics: Từ điển chứa các chỉ số (VD: {'Loss/Actor': 0.5, 'Reward/Total': 100})
        """
        for tag, value in metrics.items():
            self.writer.add_scalar(tag, value, step)
            
    def save_checkpoint(self, step: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, current_reward: float):
        """
        Lưu lại trạng thái mạng Nơ-ron. Tự động đánh dấu nếu đây là kỷ lục mới.
        """
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'reward': current_reward
        }
        
        # Lưu file checkpoint định kỳ
        regular_path = self.checkpoint_dir / f"checkpoint_step_{step}.pth"
        torch.save(checkpoint, regular_path)
        
        # Nếu AI vượt kỷ lục lợi nhuận cũ -> Lưu vào một file Tốt Nhất (Best)
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"🔥 [Kỷ lục mới!] AI vừa đạt Reward {current_reward:.2f}. Đã lưu vào {best_path}")
            
    def load_checkpoint(self, path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None) -> int:
        """
        Khôi phục trí nhớ cho AI từ file đã lưu.
        Trở về đúng điểm lưu (step) để tiếp tục Train hoặc chạy Test.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy file não bộ tại {path}")
            
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        print(f"✅ Đã khôi phục thành công não bộ từ {path} (Step {checkpoint.get('step', 'Unknown')})")
        return checkpoint.get('step', 0)
        
    def close(self):
        """Đóng kết nối TensorBoard"""
        self.writer.close()

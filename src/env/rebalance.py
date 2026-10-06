"""
src/env/rebalance.py
Ticket: ENV-003 (Sprint 4 - Thành viên A)

Bộ máy xử lý ràng buộc tỷ trọng (Rebalance Engine).
Đảm bảo các hành động từ Agent không vi phạm nguyên tắc toán học của danh mục đầu tư.
"""
import numpy as np

class RebalanceEngine:
    """Bộ xử lý ràng buộc và tái cân bằng danh mục."""
    
    @staticmethod
    def project_weights(weights: np.ndarray) -> np.ndarray:
        """
        Chiếu (Project) mảng tỷ trọng về không gian hợp lệ:
        1. Long-only: Mọi tỷ trọng đều phải >= 0.
        2. Fully-invested (kể cả cash): Tổng tỷ trọng = 1.0.
        
        Nếu mô hình xuất ra toàn số âm (hoặc 0), 
        hệ thống tự động chuyển 100% tỷ trọng vào Tiền mặt (Cash - node cuối).
        
        Args:
            weights: Mảng tỷ trọng thô do AI sinh ra (kích thước N+1).
            
        Returns:
            Mảng tỷ trọng hợp lệ, tổng đúng bằng 1.0.
        """
        raw_weights = np.asarray(weights, dtype=np.float64)
        if raw_weights.ndim != 1 or raw_weights.size < 2:
            raise ValueError("weights phải là vector chứa ít nhất một tài sản và tiền mặt.")
        if not np.all(np.isfinite(raw_weights)):
            raise ValueError("weights phải chứa các giá trị hữu hạn.")

        # Normalize after scaling by the maximum to avoid overflow for large finite inputs.
        nonnegative = np.maximum(raw_weights, 0.0)
        scale = float(np.max(nonnegative))
        if scale > 0.0:
            scaled = nonnegative / scale
            projected = scaled / np.sum(scaled)
        else:
            projected = np.zeros_like(nonnegative)
            projected[-1] = 1.0

        return projected.astype(np.float32)

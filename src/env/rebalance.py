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
        # Bước 1: Long-only (chặn các giá trị âm)
        w_proj = np.maximum(weights, 0.0)
        
        w_sum = np.sum(w_proj)
        
        # Bước 2: Chuẩn hóa để tổng = 1
        if w_sum > 1e-8:
            w_proj = w_proj / w_sum
        else:
            # Trường hợp fallback an toàn: Tự động phòng thủ bằng 100% tiền mặt
            w_proj = np.zeros_like(w_proj)
            w_proj[-1] = 1.0
            
        return w_proj

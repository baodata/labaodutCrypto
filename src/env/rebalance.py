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
    def project_weights(
        weights: np.ndarray,
        max_asset_weight: float | None = None,
    ) -> np.ndarray:
        """
        Chiếu (Project) mảng tỷ trọng về không gian hợp lệ:
        1. Long-only: Mọi tỷ trọng đều phải >= 0.
        2. Mỗi tài sản rủi ro không vượt max_asset_weight (nếu được cấu hình).
        3. Fully-invested (kể cả cash): Tổng tỷ trọng = 1.0.
        
        Nếu mô hình xuất ra toàn số âm (hoặc 0), 
        hệ thống tự động chuyển 100% tỷ trọng vào Tiền mặt (Cash - node cuối).
        
        Args:
            weights: Mảng tỷ trọng thô do AI sinh ra (kích thước N+1).
            max_asset_weight: Trần tỷ trọng cho mỗi cổ phiếu; tiền mặt không bị giới hạn.
            
        Returns:
            Mảng tỷ trọng hợp lệ, tổng đúng bằng 1.0.
        """
        raw_weights = np.asarray(weights, dtype=np.float64)
        if raw_weights.ndim != 1 or raw_weights.size < 2:
            raise ValueError("weights phải là vector chứa ít nhất một tài sản và tiền mặt.")
        if not np.all(np.isfinite(raw_weights)):
            raise ValueError("weights phải chứa các giá trị hữu hạn.")
        if max_asset_weight is not None and (
            not np.isfinite(max_asset_weight) or not 0.0 < max_asset_weight <= 1.0
        ):
            raise ValueError("max_asset_weight phải hữu hạn và thuộc (0, 1].")

        # Normalize after scaling by the maximum to avoid overflow for large finite inputs.
        nonnegative = np.maximum(raw_weights, 0.0)
        scale = float(np.max(nonnegative))
        if scale > 0.0:
            scaled = nonnegative / scale
            projected = scaled / np.sum(scaled)
        else:
            projected = np.zeros_like(nonnegative)
            projected[-1] = 1.0

        if max_asset_weight is not None:
            assets = np.minimum(projected[:-1], float(max_asset_weight))
            excess = float(np.sum(projected[:-1] - assets))
            projected = np.concatenate((assets, [projected[-1] + excess]))

        return projected.astype(np.float32)

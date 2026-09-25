"""
src/env/cost.py
Ticket: ENV-004 (Sprint 4 - Thành viên A)

Bộ máy Mô hình hóa chi phí giao dịch (Transaction Cost Engine).
Xử lý sự kiện "Trôi tỷ trọng" (Drifted Weights) và tính toán phần trăm phí bị trừ
do hành động tái cân bằng danh mục của AI.
"""
import numpy as np

class TransactionCostEngine:
    """Động cơ tính toán chi phí giao dịch dựa trên mức độ Turnover."""
    
    def __init__(self, cost_rate: float = 0.001):
        """
        Khởi tạo bộ máy tính phí.
        
        Args:
            cost_rate (c): Tỷ lệ phí giao dịch mặc định (ví dụ 0.001 = 0.1%).
        """
        self.c = cost_rate
        
    def compute_drifted_weights(
        self, 
        old_weights: np.ndarray, 
        asset_returns: np.ndarray, 
        cash_return: float, 
        portfolio_gross_return: float
    ) -> np.ndarray:
        """
        Tính tỷ trọng đã trôi (Drifted weight).
        Khi giá cổ phiếu tăng/giảm, tỷ trọng danh mục sẽ tự động thay đổi dù AI không làm gì.
        
        Công thức: w̃_i = w_{i,t-1} * (1 + r_i) / (1 + R_p)
        
        Args:
            old_weights: Tỷ trọng kỳ trước (t-1) có kích thước [N+1] (cash ở cuối).
            asset_returns: Lợi nhuận của nhóm cổ phiếu kỳ t [N].
            cash_return: Lợi nhuận sinh ra từ tiền mặt (r_f) kỳ t.
            portfolio_gross_return: Lợi nhuận gộp toàn danh mục (R_p).
            
        Returns:
            Mảng tỷ trọng bị trôi [N+1].
        """
        w_assets = old_weights[:-1]
        w_cash = old_weights[-1]
        
        # Mẫu số chung
        denominator = 1.0 + portfolio_gross_return
        
        # Trôi Cổ phiếu
        drifted_assets = w_assets * (1.0 + asset_returns) / denominator
        
        # Trôi Tiền mặt
        drifted_cash = w_cash * (1.0 + cash_return) / denominator
        
        drifted_weights = np.append(drifted_assets, drifted_cash)
        return drifted_weights
        
    def compute_cost_rate(self, target_weights: np.ndarray, drifted_weights: np.ndarray) -> float:
        """
        Tính Tỷ lệ Phí giao dịch (TC_t) cho lệnh tái cân bằng.
        
        Công thức: TC_t = c * sum(|w_{i,t} - w̃_i|)
        
        Args:
            target_weights: Tỷ trọng đích do AI quyết định (w_{t}).
            drifted_weights: Tỷ trọng do thị trường tự trôi về (w̃_i).
            
        Returns:
            TC_t (Phần trăm tài sản bị mất vì phí giao dịch).
        """
        # Tổng mức độ xáo trộn danh mục (Turnover)
        turnover = np.sum(np.abs(target_weights - drifted_weights))
        
        # Phí giao dịch tỷ lệ thuận với Turnover
        return float(self.c * turnover)

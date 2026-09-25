"""
src/env/returns.py
Ticket: ENV-002 (Sprint 4 - Thành viên A)

Bộ máy tính toán Lợi nhuận danh mục (Portfolio Return Engine).
Tính toán Lợi nhuận gộp (Gross Return) chưa bao gồm phí giao dịch.
"""
import numpy as np

class PortfolioReturnEngine:
    """Bộ máy xử lý Lợi nhuận cho danh mục đa tài sản (có tính lãi suất tiền mặt)."""
    
    def __init__(self, risk_free_rate_annual: float = 0.02, trading_days_per_year: int = 252):
        """
        Khởi tạo bộ máy.
        
        Args:
            risk_free_rate_annual: Lãi suất phi rủi ro theo năm (ví dụ 0.02 cho 2%).
            trading_days_per_year: Số ngày giao dịch trong năm (thường là 252).
        """
        self.daily_rf = risk_free_rate_annual / trading_days_per_year

    def compute_gross_return(self, weights: np.ndarray, asset_returns: np.ndarray) -> float:
        """
        Tính Lợi nhuận gộp (Gross Return) cho kỳ t+1.
        Công thức: R_p(t+1) = sum(w_i * r_i(t+1)) + w_cash * r_f
        
        Args:
            weights: Tỷ trọng danh mục [N+1] (node Tiền mặt ở cuối).
            asset_returns: Lợi nhuận từng cổ phiếu [N] tại kỳ t+1.
            
        Returns:
            Tổng lợi nhuận gộp của danh mục.
        """
        if len(weights) != len(asset_returns) + 1:
            raise ValueError(f"Kích thước weights ({len(weights)}) phải lớn hơn asset_returns ({len(asset_returns)}) đúng 1 đơn vị (Tiền mặt).")
            
        w_assets = weights[:-1]
        w_cash = weights[-1]
        
        # Tính Lãi/Lỗ từ cổ phiếu
        asset_gross_return = np.sum(w_assets * asset_returns)
        
        # Tính Lãi tiền mặt an toàn (Risk-free)
        cash_return = w_cash * self.daily_rf
        
        return float(asset_gross_return + cash_return)

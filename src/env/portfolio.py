"""
src/env/portfolio.py
Ticket: ENV-001 (Sprint 4 - Thành viên A)

Mô đun quản lý Trạng thái Danh mục Đầu tư (Portfolio State).
Đóng vai trò như một "sổ cái" lưu trữ và theo dõi tài sản, tiền mặt, và tỷ trọng hiện tại.
"""
import numpy as np

class PortfolioState:
    """Quản lý trạng thái danh mục đầu tư tại thời điểm hiện tại."""
    
    def __init__(self, initial_capital: float, num_assets: int):
        """
        Khởi tạo trạng thái danh mục.
        
        Args:
            initial_capital: Số vốn ban đầu (ví dụ: 100,000$).
            num_assets: Số lượng tài sản cổ phiếu (N). Không tính tiền mặt.
        """
        self.initial_capital = initial_capital
        self.num_assets = num_assets
        self.reset()
        
    def reset(self):
        """Khôi phục danh mục về trạng thái ban đầu (100% Tiền mặt)."""
        self.portfolio_value = float(self.initial_capital)
        self.peak_value = float(self.initial_capital)
        
        # weights có kích thước N + 1. Vị trí cuối cùng [-1] là Tiền mặt (Cash).
        # Khởi tạo: 0% cổ phiếu, 100% tiền mặt.
        self.weights = np.zeros(self.num_assets + 1, dtype=np.float32)
        self.weights[-1] = 1.0

    def update(self, new_value: float, new_weights: np.ndarray):
        """
        Cập nhật trạng thái danh mục sau một biến động thị trường hoặc giao dịch.
        
        Args:
            new_value: Tổng giá trị danh mục mới.
            new_weights: Tỷ trọng mới của danh mục (kích thước N+1).
        """
        if new_value < 0:
            raise ValueError("Giá trị danh mục không thể âm.")
            
        self.portfolio_value = float(new_value)
        self.weights = np.copy(new_weights)
        
        # Cập nhật giá trị đỉnh (để sau này tính Max Drawdown)
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value

    @property
    def cash_weight(self) -> float:
        """Lấy tỷ trọng tiền mặt hiện tại."""
        return float(self.weights[-1])
        
    @property
    def asset_weights(self) -> np.ndarray:
        """Lấy tỷ trọng của các cổ phiếu (không gồm tiền mặt)."""
        return self.weights[:-1]
        
    @property
    def cash_value(self) -> float:
        """Giá trị tuyệt đối của tiền mặt (bằng Đô la)."""
        return self.portfolio_value * self.cash_weight
        
    @property
    def assets_value(self) -> float:
        """Tổng giá trị tuyệt đối của cổ phiếu (bằng Đô la)."""
        return self.portfolio_value * (1.0 - self.cash_weight)

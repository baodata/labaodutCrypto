"""
src/env/drawdown.py
Ticket: ENV-005 (Sprint 4 - Thành viên A)

Bộ theo dõi Sụt giảm vốn (Drawdown Tracker).
Giám sát rủi ro lớn nhất của danh mục (Maximum Drawdown - MDD).
ứng dụng trong môi trường RL (Reinforcement Learning) để tính toán phần thưởng (Reward) dựa trên rủi ro.
Công thức:
    Drawdown_t = (V_t - Peak_t) / Peak_t
    Maximum Drawdown (MDD) = min(Drawdown_t)
"""

class DrawdownTracker:
    """Công cụ theo dõi sụt giảm giá trị tài sản so với đỉnh lịch sử."""
    
    def __init__(self):
        self.peak_value = 0.0
        self.max_drawdown = 0.0
        
    def reset(self, initial_value: float):
        """Khởi tạo lại trạng thái (thường dùng khi Env reset)."""
        if initial_value <= 0:
            raise ValueError("Vốn khởi điểm phải lớn hơn 0.")
            
        self.peak_value = float(initial_value)
        self.max_drawdown = 0.0
        
    def update(self, current_value: float) -> float:
        """
        Cập nhật giá trị danh mục và tính toán Drawdown hiện tại.
        
        Công thức: DD_t = (V_t - Peak_t) / Peak_t
        
        Args:
            current_value (V_t): Tổng giá trị danh mục tại bước t.
            
        Returns:
            current_drawdown: Tỷ lệ sụt giảm hiện tại (<= 0).
        """
        # Nếu danh mục lập đỉnh mới, cập nhật Peak
        if current_value > self.peak_value:
            self.peak_value = float(current_value)
            
        # Tính Drawdown hiện tại (Sẽ là 0 nếu đang ở đỉnh, hoặc số âm nếu sụt giảm)
        current_drawdown = (current_value - self.peak_value) / self.peak_value
        
        # Cập nhật Maximum Drawdown (MDD) nếu sụt giảm sâu hơn kỷ lục cũ
        if current_drawdown < self.max_drawdown:
            self.max_drawdown = current_drawdown
            
        return current_drawdown

"""
src/env/risk.py
Ticket: ENV-006 (Sprint 4 - Thành viên A)

Tính toán độ biến động rủi ro danh mục (Portfolio Risk Calculator).
Áp dụng công thức: sigma_p = sqrt(w^T * Sigma * w).
Mở rộng cửa sổ (window) lên 60 ngày để tránh suy biến ma trận hiệp phương sai.
"""
import numpy as np

class RiskCalculator:
    """Bộ máy tính toán rủi ro danh mục."""
    
    def __init__(self, window: int = 60):
        """
        Khởi tạo RiskCalculator.
        
        Args:
            window: Số ngày lấy dữ liệu quá khứ để tính Hiệp phương sai (Mặc định 60).
        """
        self.window = window
        
    def compute_covariance_matrix(self, returns_history: np.ndarray) -> np.ndarray:
        """
        Tính ma trận hiệp phương sai (Covariance Matrix - Sigma) cho các tài sản.
        
        Args:
            returns_history: Ma trận lịch sử lợi nhuận [T, N].
            
        Returns:
            Ma trận hiệp phương sai [N, N].
        """
        T, N = returns_history.shape
        if T < 2:
            return np.zeros((N, N))
            
        # Lấy tối đa 'window' ngày gần nhất
        recent_returns = returns_history[-self.window:]
        
        # np.cov yêu cầu rowvar=False nếu cột là variables (tài sản) và dòng là observations (ngày)
        cov_matrix = np.cov(recent_returns, rowvar=False)
        return cov_matrix
        
    def compute_portfolio_volatility(self, weights: np.ndarray, returns_history: np.ndarray) -> float:
        """
        Tính độ biến động danh mục (Portfolio Volatility).
        Công thức: sigma_p = sqrt(w_assets^T * Sigma * w_assets)
        (Tiền mặt giả định phi rủi ro, phương sai = 0).
        
        Args:
            weights: Tỷ trọng danh mục [N+1] (node Tiền mặt ở cuối).
            returns_history: Lịch sử lợi nhuận cổ phiếu [T, N].
            
        Returns:
            Độ biến động của danh mục (sigma_p).
        """
        w_assets = weights[:-1]
        
        cov_matrix = self.compute_covariance_matrix(returns_history)
        
        # Nếu N=1 (chỉ có 1 tài sản), cov_matrix là scalar, cần reshape thành ma trận
        if cov_matrix.ndim == 0:
            cov_matrix = np.array([[cov_matrix]])
            
        # Tính phương sai: Var = w^T * Sigma * w
        variance = np.dot(w_assets.T, np.dot(cov_matrix, w_assets))
        
        # Tránh lỗi sai số dấu phẩy động làm variance âm siêu nhỏ
        variance = max(float(variance), 0.0)
        
        return float(np.sqrt(variance))

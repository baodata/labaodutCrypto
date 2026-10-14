import numpy as np

class CashBaseline:
    """
    Chiến lược cơ sở: Giữ 100% tiền mặt trong suốt T ngày.
    Tiền mặt nhận lãi suất phi rủi ro hàng ngày.
    """
    def __init__(self, num_days: int, risk_free_rate_annual: float = 0.02):
        """
        Khởi tạo CashBaseline.
        
        Args:
            num_days: Số ngày giao dịch (T).
            risk_free_rate_annual: Lãi suất phi rủi ro hàng năm.
        """
        if not isinstance(num_days, (int, np.integer)) or num_days < 0:
            raise ValueError("num_days phải là số nguyên không âm.")
        if not np.isfinite(risk_free_rate_annual) or risk_free_rate_annual < 0:
            raise ValueError("risk_free_rate_annual phải hữu hạn và không âm.")
        self.num_days = num_days
        self.risk_free_rate_annual = risk_free_rate_annual
        self.daily_rf = self.risk_free_rate_annual / 252

    def run(self) -> np.ndarray:
        """Accrue cash via the same TradingEnv engine as the other baselines."""
        from src.evaluation.baselines.simulation import simulate_baseline
        if self.num_days == 0:
            return np.empty(0, dtype=np.float64)
        prices = np.ones((self.num_days + 1, 1), dtype=np.float64)
        return simulate_baseline(self, prices, prices, 0.0, self.risk_free_rate_annual)

    def get_weights(self, num_assets: int) -> np.ndarray:
        """
        Lấy tỷ trọng danh mục (100% tiền mặt).
        
        Args:
            num_assets: Số lượng tài sản rủi ro (N).
            
        Returns:
            np.ndarray: Mảng tỷ trọng shape [N+1] với phần tử cuối = 1.0.
        """
        weights = np.zeros(num_assets + 1)
        weights[-1] = 1.0
        return weights

    def __call__(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Strategy interface shared with TradingEnv backtests."""
        return self.get_weights(len(obs["portfolio_weights"]))

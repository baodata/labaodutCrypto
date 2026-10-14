import numpy as np

class EqualWeightBaseline:
    """
    Chiến lược Tỷ trọng đều (Equal Weight / 1/N).
    Tái cân bằng hàng ngày để duy trì tỷ trọng đều cho các tài sản rủi ro, 0 cho tiền mặt.
    """
    def __init__(
        self,
        open_prices: np.ndarray,
        num_assets: int,
        transaction_cost_rate: float = 0.001,
        risk_free_rate_annual: float = 0.02,
        cash_ratio: float = 0.0,
    ):
        """
        Khởi tạo chiến lược.
        
        Args:
            open_prices: Mảng giá mở cửa shape [T, N].
            num_assets: Số lượng tài sản (N).
            transaction_cost_rate: Tỷ lệ chi phí giao dịch (c).
            risk_free_rate_annual: Lãi suất phi rủi ro hàng năm.
            cash_ratio: Tỷ trọng tiền mặt cố định (mặc định 0.0 là 100% cổ phiếu).
        """
        self.open_prices = open_prices
        self.num_assets = num_assets
        self.c = transaction_cost_rate
        self.rf_daily = risk_free_rate_annual / 252
        if not (0.0 <= cash_ratio < 1.0):
            raise ValueError(f"cash_ratio phải thuộc [0, 1), nhận: {cash_ratio}")
        self.cash_ratio = float(cash_ratio)

    def get_weights(self, num_assets: int) -> np.ndarray:
        """
        Lấy tỷ trọng mục tiêu ((1 - cash_ratio)/N cho mỗi tài sản rủi ro, cash_ratio cho tiền mặt).
        
        Args:
            num_assets: Số lượng tài sản rủi ro (N).
            
        Returns:
            np.ndarray: Mảng tỷ trọng shape [N+1].
        """
        weights = np.zeros(num_assets + 1)
        weights[:-1] = (1.0 - self.cash_ratio) / num_assets
        weights[-1] = self.cash_ratio
        return weights

    def __call__(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Strategy interface shared with TradingEnv backtests."""
        return self.get_weights(len(obs["portfolio_weights"]))

    def run(self, *, close_prices: np.ndarray, action_alpha: float = 1.0,
            max_asset_weight: float = 1.0, dates=None) -> np.ndarray:
        """Use TradingEnv's causal T-1-step simulation, including execution costs."""
        from src.evaluation.baselines.simulation import simulate_baseline
        return simulate_baseline(
            self, self.open_prices, close_prices, self.c, self.rf_daily * 252,
            action_alpha=action_alpha, max_asset_weight=max_asset_weight, dates=dates,
        )

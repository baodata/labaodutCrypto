import numpy as np

from src.env.orders import PortfolioOrder

class BuyAndHoldBaseline:
    """
    Chiến lược Mua và Giữ (Buy and Hold).
    Bắt đầu với tỷ trọng đều, sau đó để tỷ trọng trôi dạt tự nhiên (không tái cân bằng).
    """
    def __init__(
        self,
        open_prices: np.ndarray,
        num_assets: int,
        transaction_cost_rate: float = 0.001,
        risk_free_rate_annual: float = 0.02
    ):
        """
        Khởi tạo chiến lược.
        
        Args:
            open_prices: Mảng giá mở cửa shape [T, N].
            num_assets: Số lượng tài sản (N).
            transaction_cost_rate: Tỷ lệ chi phí giao dịch (c).
            risk_free_rate_annual: Lãi suất phi rủi ro hàng năm.
        """
        self.open_prices = open_prices
        self.num_assets = num_assets
        self.c = transaction_cost_rate
        self.rf_daily = risk_free_rate_annual / 252

    def get_initial_weights(self, num_assets: int) -> np.ndarray:
        """
        Lấy tỷ trọng ban đầu (1/N cho mỗi tài sản, 0 cho tiền mặt).
        
        Args:
            num_assets: Số lượng tài sản rủi ro (N).
            
        Returns:
            np.ndarray: Mảng tỷ trọng shape [N+1].
        """
        weights = np.zeros(num_assets + 1)
        weights[:-1] = 1.0 / num_assets
        return weights

    def __call__(self, obs: dict, info: dict | None = None) -> PortfolioOrder:
        """Allocate fully at the first open, then keep share quantities unchanged."""
        if info is not None and info.get("step", 0) == 0:
            return PortfolioOrder.rebalance(
                self.get_initial_weights(len(obs["portfolio_weights"])),
                action_alpha=1.0,
            )
        return PortfolioOrder.hold()

    def run(self, *, close_prices: np.ndarray, action_alpha: float = 1.0,
            max_asset_weight: float = 1.0, dates=None) -> np.ndarray:
        """Use TradingEnv's causal T-1-step simulation, including execution costs."""
        from src.evaluation.baselines.simulation import simulate_baseline
        return simulate_baseline(
            self, self.open_prices, close_prices, self.c, self.rf_daily * 252,
            action_alpha=action_alpha, max_asset_weight=max_asset_weight, dates=dates,
        )

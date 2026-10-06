import numpy as np

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

    def __call__(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Buy once, then retain the environment's naturally drifted weights."""
        if info is not None and info.get("step", 0) == 0:
            return self.get_initial_weights(len(obs["portfolio_weights"]))
        return np.concatenate((obs["portfolio_weights"], obs["cash_ratio"])).astype(np.float32)

    def run(self) -> np.ndarray:
        """
        Mô phỏng chiến lược, chỉ trả chi phí giao dịch vào ngày 0.
        Sử dụng Open-to-Open returns, trả về mảng độ dài T-2.
        
        Returns:
            np.ndarray: Mảng lợi nhuận ròng hàng ngày.
        """
        T = self.open_prices.shape[0]
        net_returns = []
        
        # Bắt đầu với 100% tiền mặt
        current_weights = np.zeros(self.num_assets + 1)
        current_weights[-1] = 1.0
        
        # Mua vào ngày 0
        target_weights = self.get_initial_weights(self.num_assets)
        tc_initial = self.c * np.sum(
            np.abs(target_weights[:-1] - current_weights[:-1])
        )
        
        current_weights = target_weights
        
        for t in range(T - 2):
            p0 = self.open_prices[t+1]
            p1 = self.open_prices[t+2]
            asset_returns = np.where(p0 > 0, (p1 - p0) / p0, 0.0)
            
            gross_return = np.sum(current_weights[:-1] * asset_returns) + current_weights[-1] * self.rf_daily
            
            if t == 0:
                net_return = gross_return - tc_initial
            else:
                net_return = gross_return # tc = 0
                
            net_returns.append(net_return)
            
            # Cập nhật tỷ trọng trôi dạt
            new_weights = np.zeros(self.num_assets + 1)
            new_weights[:-1] = current_weights[:-1] * (1 + asset_returns)
            new_weights[-1] = current_weights[-1] * (1 + self.rf_daily)
            
            total_value = np.sum(new_weights)
            if total_value > 0:
                current_weights = new_weights / total_value
            else:
                current_weights = np.zeros(self.num_assets + 1)
                current_weights[-1] = 1.0
                
        return np.array(net_returns)

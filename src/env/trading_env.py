"""
src/env/trading_env.py
Ticket: ENV-008 (Sprint 4 - Thành viên A)

Môi trường Giao dịch chuẩn Gymnasium (Gymnasium-compatible Trading Environment).
Lắp ráp tất cả 7 module con của Sprint 4 vào một Environment thống nhất.
Tuân thủ chặt chẽ quy ước khớp lệnh tại ENV-009.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml

from src.env.portfolio import PortfolioState
from src.env.returns import PortfolioReturnEngine
from src.env.rebalance import RebalanceEngine
from src.env.cost import TransactionCostEngine
from src.env.drawdown import DrawdownTracker
from src.env.risk import RiskCalculator
from src.env.reward import RewardEngine
from src.utils.data_types import MarketDataTensor

class TradingEnv(gym.Env):
    """Môi trường giao dịch tài chính Đa tài sản, tương thích chuẩn Gymnasium."""
    
    def __init__(self, market_tensor: MarketDataTensor, open_prices: np.ndarray, config_path: str = "configs/env.yaml"):
        """
        Khởi tạo môi trường.
        
        Args:
            market_tensor: Dữ liệu tính năng (Features) của A.
            open_prices: Ma trận giá Open [T, N] dùng để khớp lệnh (ENV-009).
            config_path: Đường dẫn tới env.yaml.
        """
        super(TradingEnv, self).__init__()
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)["environment"]
            
        self.market_tensor = market_tensor
        self.open_prices = open_prices
        
        self.T, self.N, self.F = self.market_tensor.tensor.shape
        
        if self.open_prices.shape != (self.T, self.N):
            raise ValueError(f"open_prices shape {self.open_prices.shape} không khớp với T, N = {self.T}, {self.N}")
            
        # Không gian hành động: Tỷ trọng cho N cổ phiếu + 1 tiền mặt
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.N + 1,), dtype=np.float32)
        
        # Không gian trạng thái: Lấy toàn bộ đặc trưng của N cổ phiếu tại bước t
        self.observation_space = spaces.Dict({
            "market_state": spaces.Box(low=-np.inf, high=np.inf, shape=(self.N, self.F), dtype=np.float32),
            "portfolio_weights": spaces.Box(low=0.0, high=1.0, shape=(self.N,), dtype=np.float32),
            "cash_ratio": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })
        
        # Lắp ráp 7 động cơ lõi
        self.portfolio = PortfolioState(initial_capital=self.config["initial_balance"], num_assets=self.N)
        self.returns_engine = PortfolioReturnEngine(risk_free_rate_annual=self.config["risk_free_rate"])
        self.rebalance_engine = RebalanceEngine()
        self.cost_engine = TransactionCostEngine(cost_rate=self.config["transaction_cost_rate"])
        self.drawdown_tracker = DrawdownTracker()
        self.risk_calc = RiskCalculator(window=self.config["covariance_window"])
        self.reward_engine = RewardEngine(beta=self.config["reward_beta"], scaling_factor=self.config["reward_scaling"])
        
        self.current_step = 0
        self.last_drifted_weights = None
        
    def reset(self, seed=None, options=None):
        """Khôi phục môi trường về trạng thái nguyên thủy."""
        super().reset(seed=seed)
        self.current_step = 0
        
        self.portfolio.reset()
        self.drawdown_tracker.reset(self.config["initial_balance"])
        self.last_drifted_weights = np.copy(self.portfolio.weights)
        
        obs = self._get_obs()
        info = self._get_info()
        return obs, info
        
    def _get_obs(self):
        """Lấy quan sát tại thời điểm t (Market State + Portfolio State)."""
        return {
            "market_state": self.market_tensor.tensor[self.current_step].astype(np.float32),
            "portfolio_weights": self.portfolio.asset_weights.astype(np.float32),
            "cash_ratio": np.array([self.portfolio.cash_weight], dtype=np.float32)
        }
        
    def _get_info(self):
        return {
            "step": self.current_step,
            "portfolio_value": self.portfolio.portfolio_value,
            "cash_weight": self.portfolio.cash_weight,
            "max_drawdown": self.drawdown_tracker.max_drawdown
        }
        
    def step(self, action: np.ndarray):
        """
        Thực thi 1 bước giao dịch theo quy ước ENV-009:
        - Tín hiệu mua bán phát ra ở cuối ngày t (current_step).
        - Lệnh được khớp ở Open ngày t+1.
        - Lợi nhuận sinh ra từ Open t+1 đến Open t+2.
        """
        # Nếu đã đi đến ngày cuối cùng có thể dự phóng (cần t+2 để tính lợi nhuận Open-to-Open)
        if self.current_step >= self.T - 2:
            return self._get_obs(), 0.0, True, False, self._get_info()
            
        # 1. Ép tỷ trọng Action (Long-only, sum=1)
        target_weights = self.rebalance_engine.project_weights(action)
        
        # 2. Tính phí giao dịch cho lần đảo danh mục này (So với tỷ trọng trôi của kỳ trước)
        cost_rate = self.cost_engine.compute_cost_rate(target_weights, self.last_drifted_weights)
        
        # 3. Mua bán tại Open(t+1) và nắm giữ đến Open(t+2)
        t = self.current_step
        open_t1 = self.open_prices[t + 1]
        open_t2 = self.open_prices[t + 2]
        
        # Tính tỷ suất lợi nhuận từng mã
        safe_open_t1 = np.where(open_t1 == 0, 1e-8, open_t1)
        asset_returns = (open_t2 - safe_open_t1) / safe_open_t1
        cash_return = self.returns_engine.daily_rf
        
        # 4. Tính lợi nhuận gộp (Gross Return)
        gross_return = self.returns_engine.compute_gross_return(target_weights, asset_returns)
        
        # 5. Tính lợi nhuận ròng (Net Return = Gross - Cost)
        net_return = self.reward_engine.compute_net_return(gross_return, cost_rate)
        
        # Cập nhật giá trị danh mục
        new_value = self.portfolio.portfolio_value * (1.0 + net_return)
        
        # 6. Tính tỷ trọng trôi sau chu kỳ nắm giữ
        drifted_weights = self.cost_engine.compute_drifted_weights(
            target_weights, asset_returns, cash_return, gross_return
        )
        
        # Lưu vào State
        self.portfolio.update(new_value, drifted_weights)
        self.last_drifted_weights = drifted_weights
        
        # Cập nhật rủi ro
        self.drawdown_tracker.update(new_value)
        
        # 7. Tính Reward thưởng cho Agent (có shaping)
        reward = self.reward_engine.compute_reward(gross_return, cost_rate)
        
        # Tiến lên ngày mới
        self.current_step += 1
        
        obs = self._get_obs()
        info = self._get_info()
        info["net_return"] = net_return
        info["cost_rate"] = cost_rate
        info["reward"] = reward
        
        terminated = (self.current_step >= self.T - 2)
        truncated = False
        
        return obs, reward, terminated, truncated, info

"""
src/env/trading_env.py
Ticket: ENV-008 (Sprint 4 - Thành viên A)

Môi trường Giao dịch chuẩn Gymnasium (Gymnasium-compatible Trading Environment).
Kết hợp portfolio, return, cost, drawdown và reward cho mô phỏng giao dịch.
Tuân thủ chặt chẽ quy ước khớp lệnh tại ENV-009.
"""
from functools import lru_cache
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml

from src.env.portfolio import PortfolioState
from src.env.returns import PortfolioReturnEngine
from src.env.rebalance import RebalanceEngine
from src.env.cost import TransactionCostEngine
from src.env.drawdown import DrawdownTracker
from src.env.reward import RewardEngine
from src.utils.data_types import MarketDataTensor


@lru_cache(maxsize=16)
def _load_environment_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)["environment"]

class TradingEnv(gym.Env):
    """Môi trường giao dịch tài chính Đa tài sản, tương thích chuẩn Gymnasium."""
    
    def __init__(
        self,
        market_tensor: MarketDataTensor,
        open_prices: np.ndarray,
        config_path: str = "configs/env.yaml",
        reward_beta: float | None = None,
        reward_baseline: str = "none",
        history_tensor: MarketDataTensor | None = None,
        history_start_index: int = 0,
        lookback_window: int = 1,
        action_alpha: float = 1.0,
    ):
        """
        Khởi tạo môi trường.
        
        Args:
            market_tensor: Dữ liệu tính năng (Features) của A.
            open_prices: Ma trận giá Open [T, N] dùng để khớp lệnh (ENV-009).
            config_path: Đường dẫn tới env.yaml.
            reward_beta: Ghi đè hệ số phạt phí trong reward mà không đổi phí thật.
        """
        super(TradingEnv, self).__init__()
        
        config_key = str(Path(config_path).resolve())
        self.config = dict(_load_environment_config(config_key))
        if reward_beta is not None:
            if not np.isfinite(reward_beta) or reward_beta < 0:
                raise ValueError("reward_beta phải là số hữu hạn không âm.")
            self.config["reward_beta"] = float(reward_beta)
        if reward_baseline not in {"none", "equal_weight"}:
            raise ValueError("reward_baseline phải là 'none' hoặc 'equal_weight'.")
        self.reward_baseline = reward_baseline
            
        self.market_tensor = market_tensor
        self.market_tensor.validate()
        self.open_prices = np.asarray(open_prices)
        
        self.T, self.N, self.F = self.market_tensor.tensor.shape

        if self.T < 3:
            raise ValueError("TradingEnv cần ít nhất 3 mốc thời gian để tính Open-to-Open return.")
        
        if self.open_prices.shape != (self.T, self.N):
            raise ValueError(f"open_prices shape {self.open_prices.shape} không khớp với T, N = {self.T}, {self.N}")
        if not np.all(np.isfinite(self.open_prices)) or np.any(self.open_prices <= 0):
            raise ValueError("open_prices phải hữu hạn và lớn hơn 0.")

        self.history_tensor = history_tensor if history_tensor is not None else market_tensor
        self.history_tensor.validate()
        if self.history_tensor.tensor.shape[1:] != (self.N, self.F):
            raise ValueError("history_tensor phải có cùng số tài sản và feature với market_tensor.")
        if self.history_tensor.tickers != self.market_tensor.tickers:
            raise ValueError("history_tensor phải dùng cùng thứ tự tickers với market_tensor.")
        if self.history_tensor.feature_names != self.market_tensor.feature_names:
            raise ValueError("history_tensor phải dùng cùng thứ tự feature với market_tensor.")
        if not isinstance(history_start_index, (int, np.integer)) or history_start_index < 0:
            raise ValueError("history_start_index phải là số nguyên không âm.")
        if history_start_index + self.T > len(self.history_tensor.dates):
            raise ValueError("market_tensor vượt quá vùng chỉ số của history_tensor.")
        self.history_start_index = int(history_start_index)
        if lookback_window <= 0:
            raise ValueError("lookback_window phải lớn hơn 0.")
        self.lookback_window = int(lookback_window)
        if not np.isfinite(action_alpha) or not 0.0 < action_alpha <= 1.0:
            raise ValueError("action_alpha phải hữu hạn và thuộc (0, 1].")
        self.action_alpha = float(action_alpha)
            
        # Không gian hành động: Tỷ trọng cho N cổ phiếu + 1 tiền mặt
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.N + 1,), dtype=np.float32)
        
        # Keep current-day features and expose the exact policy lookback tensor.
        self.observation_space = spaces.Dict({
            "market_state": spaces.Box(low=-np.inf, high=np.inf, shape=(self.N, self.F), dtype=np.float32),
            "market_history": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.lookback_window, self.N, self.F),
                dtype=np.float32,
            ),
            "portfolio_weights": spaces.Box(low=0.0, high=1.0, shape=(self.N,), dtype=np.float32),
            "cash_ratio": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })
        
        # Lắp ráp các động cơ được dùng bởi bước giao dịch.
        self.portfolio = PortfolioState(initial_capital=self.config["initial_balance"], num_assets=self.N)
        self.returns_engine = PortfolioReturnEngine(risk_free_rate_annual=self.config["risk_free_rate"])
        self.rebalance_engine = RebalanceEngine()
        self.cost_engine = TransactionCostEngine(cost_rate=self.config["transaction_cost_rate"])
        self.drawdown_tracker = DrawdownTracker()
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
        history_index = self.history_start_index + self.current_step
        history_start = max(0, history_index - self.lookback_window + 1)
        history = self.history_tensor.tensor[history_start : history_index + 1]
        if len(history) < self.lookback_window:
            padding = np.zeros(
                (self.lookback_window - len(history), self.N, self.F),
                dtype=np.float32,
            )
            history = np.concatenate((padding, history), axis=0)
        return {
            "market_state": np.asarray(
                self.market_tensor.tensor[self.current_step], dtype=np.float32
            ),
            "market_history": np.asarray(history, dtype=np.float32),
            "portfolio_weights": np.asarray(
                self.portfolio.asset_weights, dtype=np.float32
            ),
            "cash_ratio": np.asarray([self.portfolio.cash_weight], dtype=np.float32),
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
            return self._get_obs(), 0.0, False, True, self._get_info()

        action = np.asarray(action)
        if not np.all(np.isfinite(action)):
            raise ValueError("Action phải chứa các giá trị hữu hạn.")
            
        # Project the policy target, then apply smoothing inside the environment
        # so every PPO rollout and deterministic PPO backtest uses the same rule.
        policy_target = self.rebalance_engine.project_weights(action)
        current_weights = self.portfolio.weights
        target_weights = (
            (1.0 - self.action_alpha) * current_weights
            + self.action_alpha * policy_target
        ).astype(np.float32)
        
        # 2. Tính phí giao dịch cho lần đảo danh mục này (So với tỷ trọng trôi của kỳ trước)
        previous_value = self.portfolio.portfolio_value
        turnover = float(
            np.sum(np.abs(target_weights[:-1] - self.last_drifted_weights[:-1]))
        )
        cost_rate = self.cost_engine.compute_cost_rate(target_weights, self.last_drifted_weights)
        transaction_cost = previous_value * cost_rate
        
        # 3. Mua bán tại Open(t+1) và nắm giữ đến Open(t+2)
        t = self.current_step
        open_t1 = self.open_prices[t + 1]
        open_t2 = self.open_prices[t + 2]
        
        # Tính tỷ suất lợi nhuận từng mã
        asset_returns = (open_t2 - open_t1) / open_t1
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
        self.last_drifted_weights = self.portfolio.weights.copy()
        
        # Cập nhật rủi ro
        self.drawdown_tracker.update(new_value)
        
        # 7. Tính Reward thưởng cho Agent (có shaping)
        benchmark_return = (
            float(np.mean(asset_returns))
            if self.reward_baseline == "equal_weight"
            else 0.0
        )
        reward = self.reward_engine.compute_reward(
            gross_return, cost_rate, benchmark_return=benchmark_return
        )
        
        # Tiến lên ngày mới
        self.current_step += 1
        
        obs = self._get_obs()
        info = self._get_info()
        info["net_return"] = net_return
        info["cost_rate"] = cost_rate
        info["turnover"] = turnover
        info["transaction_cost"] = transaction_cost
        info["reward"] = reward
        info["benchmark_return"] = benchmark_return
        
        terminated = False
        truncated = self.current_step >= self.T - 2
        
        return obs, reward, terminated, truncated, info

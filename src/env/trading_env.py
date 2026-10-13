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
from src.env.cost import TransactionCostEngine, resolve_transaction_cost_components
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
        reward_scaling: float | None = None,
        reward_baseline: str = "none",
        history_tensor: MarketDataTensor | None = None,
        history_start_index: int = 0,
        lookback_window: int = 1,
        action_alpha: float = 1.0,
        graph_adjacency_sequence: np.ndarray | None = None,
        graph_edge_mask_sequence: np.ndarray | None = None,
        graph_start_index: int = 0,
        close_prices: np.ndarray | None = None,
    ):
        """
        Khởi tạo môi trường.
        
        Args:
            market_tensor: Dữ liệu tính năng (Features) của A.
            open_prices: Ma trận giá Open [T, N] dùng để khớp lệnh.
            close_prices: Ma trận giá Close [T, N]. Cần truyền trong backtest
                thực tế để observation và NAV cùng được đánh dấu tại Close.
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
        if reward_scaling is not None:
            if not np.isfinite(reward_scaling) or reward_scaling <= 0:
                raise ValueError("reward_scaling phải là số hữu hạn lớn hơn 0.")
            self.config["reward_scaling"] = float(reward_scaling)
        if reward_baseline not in {"none", "equal_weight"}:
            raise ValueError("reward_baseline phải là 'none' hoặc 'equal_weight'.")
        self.reward_baseline = reward_baseline
        (
            self.transaction_fee_pct,
            self.slippage_pct,
            transaction_cost_rate,
        ) = resolve_transaction_cost_components(self.config)
        self.config["transaction_cost_rate"] = transaction_cost_rate
        self.config["transaction_fee_pct"] = self.transaction_fee_pct
        self.config["slippage_pct"] = self.slippage_pct
        self.max_asset_weight = float(self.config.get("max_asset_weight", 1.0))
        if not np.isfinite(self.max_asset_weight) or not 0.0 < self.max_asset_weight <= 1.0:
            raise ValueError("max_asset_weight phải hữu hạn và thuộc (0, 1].")
            
        self.market_tensor = market_tensor
        self.market_tensor.validate()
        self.open_prices = np.asarray(open_prices, dtype=np.float64)
        # Compatibility for callers that only have one price matrix. The
        # training/evaluation pipeline always supplies real close prices.
        self.close_prices = (
            self.open_prices.copy()
            if close_prices is None
            else np.asarray(close_prices, dtype=np.float64)
        )
        
        self.T, self.N, self.F = self.market_tensor.tensor.shape

        if self.T < 2:
            raise ValueError("TradingEnv cần ít nhất 2 mốc thời gian.")
        
        if self.open_prices.shape != (self.T, self.N):
            raise ValueError(f"open_prices shape {self.open_prices.shape} không khớp với T, N = {self.T}, {self.N}")
        if not np.all(np.isfinite(self.open_prices)) or np.any(self.open_prices <= 0):
            raise ValueError("open_prices phải hữu hạn và lớn hơn 0.")
        if self.close_prices.shape != (self.T, self.N):
            raise ValueError(
                f"close_prices shape {self.close_prices.shape} không khớp với T, N = {self.T}, {self.N}"
            )
        if not np.all(np.isfinite(self.close_prices)) or np.any(self.close_prices <= 0):
            raise ValueError("close_prices phải hữu hạn và lớn hơn 0.")

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

        if (graph_adjacency_sequence is None) != (graph_edge_mask_sequence is None):
            raise ValueError("Cần truyền đồng thời graph adjacency và edge mask.")
        self.graph_adjacency_sequence = None
        self.graph_edge_mask_sequence = None
        self.graph_relation_count = 0
        self.graph_start_index = 0
        if graph_adjacency_sequence is not None:
            graph_adjacency_sequence = np.asarray(graph_adjacency_sequence, dtype=np.float32)
            graph_edge_mask_sequence = np.asarray(graph_edge_mask_sequence, dtype=np.float32)
            # Accept the old single-relation form while storing all graph
            # histories with an explicit relation axis [T, R, N, N].
            if graph_adjacency_sequence.ndim == 3:
                graph_adjacency_sequence = graph_adjacency_sequence[:, None, :, :]
                graph_edge_mask_sequence = graph_edge_mask_sequence[:, None, :, :]
            if (
                graph_adjacency_sequence.ndim != 4
                or graph_adjacency_sequence.shape[2:] != (self.N, self.N)
                or graph_adjacency_sequence.shape[1] <= 0
            ):
                raise ValueError(
                    "graph_adjacency_sequence phải có shape [T, R, N, N]."
                )
            expected_graph_shape = (
                len(graph_adjacency_sequence),
                graph_adjacency_sequence.shape[1],
                self.N,
                self.N,
            )
            if graph_edge_mask_sequence.shape != expected_graph_shape:
                raise ValueError("graph_edge_mask_sequence phải cùng shape với adjacency.")
            if not np.all(np.isfinite(graph_adjacency_sequence)):
                raise ValueError("Graph adjacency phải chứa giá trị hữu hạn.")
            if not np.all(np.isfinite(graph_edge_mask_sequence)) or np.any(
                (graph_edge_mask_sequence != 0.0) & (graph_edge_mask_sequence != 1.0)
            ):
                raise ValueError("Graph edge mask chỉ được chứa 0 hoặc 1 hữu hạn.")
            if not isinstance(graph_start_index, (int, np.integer)) or graph_start_index < 0:
                raise ValueError("graph_start_index phải là số nguyên không âm.")
            if graph_start_index + self.T > len(graph_adjacency_sequence):
                raise ValueError("market_tensor vượt quá graph sequence.")
            self.graph_adjacency_sequence = graph_adjacency_sequence
            self.graph_edge_mask_sequence = graph_edge_mask_sequence
            self.graph_relation_count = int(graph_adjacency_sequence.shape[1])
            self.graph_start_index = int(graph_start_index)
            
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
        if self.graph_adjacency_sequence is not None:
            graph_history_shape = (
                self.lookback_window,
                self.graph_relation_count,
                self.N,
                self.N,
            )
            self.observation_space.spaces["graph_adjacency_history"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=graph_history_shape,
                dtype=np.float32,
            )
            self.observation_space.spaces["graph_edge_mask_history"] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=graph_history_shape,
                dtype=np.float32,
            )
        
        # Lắp ráp các động cơ được dùng bởi bước giao dịch.
        self.portfolio = PortfolioState(initial_capital=self.config["initial_balance"], num_assets=self.N)
        self.returns_engine = PortfolioReturnEngine(risk_free_rate_annual=self.config["risk_free_rate"])
        self.rebalance_engine = RebalanceEngine()
        self.cost_engine = TransactionCostEngine(cost_rate=transaction_cost_rate)
        self.drawdown_tracker = DrawdownTracker()
        self.reward_engine = RewardEngine(beta=self.config["reward_beta"], scaling_factor=self.config["reward_scaling"])
        
        self.current_step = 0
        
    def reset(self, seed=None, options=None):
        """Khôi phục môi trường về trạng thái nguyên thủy."""
        super().reset(seed=seed)
        self.current_step = 0
        
        self.portfolio.reset()
        self.drawdown_tracker.reset(self.config["initial_balance"])
        
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
        observation = {
            "market_state": np.asarray(
                self.market_tensor.tensor[self.current_step], dtype=np.float32
            ),
            "market_history": np.asarray(history, dtype=np.float32),
            "portfolio_weights": np.asarray(
                self.portfolio.asset_weights, dtype=np.float32
            ),
            "cash_ratio": np.asarray([self.portfolio.cash_weight], dtype=np.float32),
        }
        if self.graph_adjacency_sequence is not None:
            graph_index = self.graph_start_index + self.current_step
            graph_start = max(0, graph_index - self.lookback_window + 1)
            adjacency_history = self.graph_adjacency_sequence[graph_start : graph_index + 1]
            edge_mask_history = self.graph_edge_mask_sequence[graph_start : graph_index + 1]
            if len(adjacency_history) < self.lookback_window:
                padding_shape = (
                    self.lookback_window - len(adjacency_history),
                    self.graph_relation_count,
                    self.N,
                    self.N,
                )
                adjacency_history = np.concatenate(
                    (np.zeros(padding_shape, dtype=np.float32), adjacency_history), axis=0
                )
                edge_mask_history = np.concatenate(
                    (np.zeros(padding_shape, dtype=np.float32), edge_mask_history), axis=0
                )
            observation["graph_adjacency_history"] = np.asarray(
                adjacency_history, dtype=np.float32
            )
            observation["graph_edge_mask_history"] = np.asarray(
                edge_mask_history, dtype=np.float32
            )
        return observation
        
    def _get_info(self):
        return {
            "step": self.current_step,
            "observation_date": str(self.market_tensor.dates[self.current_step]),
            "portfolio_value": self.portfolio.portfolio_value,
            "cash_weight": self.portfolio.cash_weight,
            "max_drawdown": self.drawdown_tracker.max_drawdown
        }

    def project_action(
        self,
        action: np.ndarray,
        current_weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the capped policy target and the final blended execution target."""
        policy_target = self.rebalance_engine.project_weights(
            action, max_asset_weight=self.max_asset_weight
        )
        if current_weights is None:
            current_weights = self.portfolio.weights
        blended_target = (
            (1.0 - self.action_alpha) * np.asarray(current_weights, dtype=np.float64)
            + self.action_alpha * policy_target
        )
        execution_target = self.rebalance_engine.project_weights(
            blended_target, max_asset_weight=self.max_asset_weight
        )
        return policy_target, execution_target
        
    def step(self, action: np.ndarray):
        """
        Timeline nhân quả: quyết định tại Close[t], khớp tại Open[t+1],
        đánh dấu danh mục tại Close[t+1]. Danh mục cũ chịu return qua đêm;
        danh mục mới chịu return trong phiên. Observation tiếp theo và NAV
        vì vậy cùng thuộc Close[t+1].
        """
        if self.current_step >= self.T - 1:
            return self._get_obs(), 0.0, False, True, self._get_info()

        action = np.asarray(action)
        if not np.all(np.isfinite(action)):
            raise ValueError("Action phải chứa các giá trị hữu hạn.")
            
        # Blend from the portfolio state observed at Close[t], before any
        # unobserved Close[t] -> Open[t+1] move.
        current_weights = self.portfolio.weights.copy()
        _, target_weights = self.project_action(
            action, current_weights=current_weights
        )

        t = self.current_step
        previous_value = self.portfolio.portfolio_value
        daily_rf = self.returns_engine.daily_rf
        subperiod_rf = float(np.sqrt(1.0 + daily_rf) - 1.0)

        # Existing holdings earn the close-to-next-open return before the
        # rebalance. This drifted portfolio is the actual state at execution.
        overnight_returns = self.open_prices[t + 1] / self.close_prices[t] - 1.0
        overnight_gross = self._compute_gross_return(
            current_weights, overnight_returns, subperiod_rf
        )
        open_weights = self.cost_engine.compute_drifted_weights(
            current_weights, overnight_returns, subperiod_rf, overnight_gross
        )

        # Execute at Open[t+1]; turnover/cost are measured against the
        # overnight-drifted holdings and charged on the then-current NAV.
        turnover = float(np.sum(np.abs(target_weights[:-1] - open_weights[:-1])))
        cost_rate = self.cost_engine.compute_cost_rate(target_weights, open_weights)
        value_at_open = previous_value * (1.0 + overnight_gross)
        transaction_cost = value_at_open * cost_rate
        commission_cost = value_at_open * self.transaction_fee_pct * turnover
        slippage_cost = value_at_open * self.slippage_pct * turnover
        value_after_cost = value_at_open - transaction_cost

        # New holdings earn only the Open[t+1] -> Close[t+1] intraday return.
        intraday_returns = self.close_prices[t + 1] / self.open_prices[t + 1] - 1.0
        intraday_gross = self._compute_gross_return(
            target_weights, intraday_returns, subperiod_rf
        )
        gross_return = (1.0 + overnight_gross) * (1.0 + intraday_gross) - 1.0
        new_value = value_after_cost * (1.0 + intraday_gross)
        net_return = new_value / previous_value - 1.0

        # Observation t+1 is based on Close[t+1], so expose the portfolio
        # weights drifted to that same mark.
        drifted_weights = self.cost_engine.compute_drifted_weights(
            target_weights, intraday_returns, subperiod_rf, intraday_gross
        )
        self.portfolio.update(new_value, drifted_weights)
        self.drawdown_tracker.update(new_value)

        benchmark_return = (
            float(np.mean(self.close_prices[t + 1] / self.close_prices[t] - 1.0))
            if self.reward_baseline == "equal_weight"
            else 0.0
        )
        # Preserve the configured cost shaping while anchoring beta=1 reward
        # to the exact realized net return after the mid-period execution cost.
        cost_fraction = transaction_cost / previous_value
        reward = self.reward_engine.compute_reward(
            net_return + cost_fraction,
            cost_fraction,
            benchmark_return=benchmark_return,
        )
        self.current_step += 1
        
        obs = self._get_obs()
        info = self._get_info()
        info["execution_date"] = str(self.market_tensor.dates[t + 1])
        info["overnight_return"] = float(overnight_gross)
        info["intraday_return"] = float(intraday_gross)
        info["gross_return"] = float(gross_return)
        info["net_return"] = net_return
        info["cost_rate"] = cost_rate
        info["turnover"] = turnover
        info["transaction_cost"] = transaction_cost
        info["commission_cost"] = commission_cost
        info["slippage_cost"] = slippage_cost
        info["reward"] = reward
        info["benchmark_return"] = benchmark_return
        
        terminated = False
        truncated = self.current_step >= self.T - 1
        
        return obs, reward, terminated, truncated, info

    def _compute_gross_return(
        self, weights: np.ndarray, asset_returns: np.ndarray, cash_return: float
    ) -> float:
        """Portfolio return for one subperiod, including its cash yield."""
        return float(
            np.dot(weights[:-1], asset_returns) + weights[-1] * cash_return
        )

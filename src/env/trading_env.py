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
from src.env.orders import PortfolioOrder
from src.env.returns import PortfolioReturnEngine
from src.env.rebalance import RebalanceEngine
from src.env.cost import TransactionCostEngine, resolve_transaction_cost_components
from src.env.drawdown import DrawdownTracker
from src.env.reward import RewardEngine, DifferentialSharpeReward
from src.utils.data_types import MarketDataTensor


@lru_cache(maxsize=16)
def _load_environment_config(config_path: str, modified_ns: int, file_size: int) -> dict:
    with open(config_path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)["environment"]

class TradingEnv(gym.Env):
    """Môi trường giao dịch tài chính Đa tài sản, tương thích chuẩn Gymnasium."""
    
    def __init__(
        self,
        market_tensor: MarketDataTensor,
        open_prices: np.ndarray,
        config_path: str = "configs/core/env.yaml",
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
        environment_config: dict | None = None,
        quant_context: dict | None = None,
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
        
        path_obj = Path(config_path)
        if not path_obj.exists():
            from src.utils.config import resolve_config_path
            path_obj = resolve_config_path(config_path)
        config_key = str(path_obj.resolve())
        if environment_config is None:
            stat = Path(config_key).stat()
            self.config = dict(_load_environment_config(config_key, stat.st_mtime_ns, stat.st_size))
        else:
            self.config = dict(environment_config)
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
        if not np.array_equal(
            np.asarray(self.market_tensor.dates, dtype="datetime64[ns]"),
            np.asarray(self.history_tensor.dates[history_start_index:history_start_index + self.T], dtype="datetime64[ns]"),
        ):
            raise ValueError("Dates của market_tensor không khớp history slice; kiểm tra history_start_index.")
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
            "history_mask": spaces.MultiBinary(self.lookback_window),
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
        
        self.quant_context = quant_context if quant_context is not None else (getattr(self.history_tensor, "quant_context", None) or {})
        self.quant_config = self.quant_context.get("config", {})
        if self.quant_context and self.quant_context['tickers'] != list(self.market_tensor.tickers):
            raise ValueError("Quant bank ticker order mismatch.")
        self.bank_indices = None
        if self.quant_context:
            import pandas as pd
            self.bank_indices = pd.DatetimeIndex(self.quant_context['dates']).get_indexer(pd.DatetimeIndex(self.market_tensor.dates))
            if np.any(self.bank_indices < 0):
                raise ValueError("Quant bank missing episode dates.")
        from src.env.rebalance import ExecutionController
        if 'market_context' in self.quant_context:
            dimension = self.quant_context['market_context'].shape[-1]
            self.observation_space.spaces['market_context'] = spaces.Box(low=-np.inf,high=np.inf,shape=(dimension,),dtype=np.float32)
        self.execution_controller = ExecutionController(self.quant_config.get('execution', {}))
        self.impact_eta = float(self.quant_config.get('impact', {}).get('eta', 0.))
        if not np.isfinite(self.impact_eta) or self.impact_eta < 0:
            raise ValueError("Impact eta must be finite and nonnegative.")
        self.cash_funded_execution = self.execution_controller.enabled or self.impact_eta > 0 or self.quant_config.get('impact', {}).get('accounting') == 'cash_funded'
        if self.cash_funded_execution and 'liquidity' not in self.quant_context:
            raise ValueError("Impact/execution require a causal liquidity bank.")
        if self.execution_controller.enabled and 'risk' not in self.quant_context:
            raise ValueError("Adaptive execution requires causal full risk covariance.")
        if self.execution_controller.enabled and self.execution_controller.speed_mode == 'efficiency_ratio':
            trend = self.quant_context.get('trend')
            if (trend is None or trend.tickers != list(self.market_tensor.tickers)
                    or not pd.DatetimeIndex(trend.dates).equals(pd.DatetimeIndex(self.quant_context['dates']))
                    or trend.metadata['window'] != self.execution_controller.er_window):
                raise ValueError("ER execution requires an aligned trend bank with the configured window.")
        reward_config = self.quant_config.get('reward', {})
        self.dsr = None
        if reward_config.get('mode') == 'differential_sharpe':
            if reward_config.get('reference', 'risk_free') not in {'risk_free','equal_weight'}:
                raise ValueError("DSR reference must be risk_free or equal_weight.")
            if self.config['reward_beta'] != 1.:
                raise ValueError("DSR uses realized net returns; reward_beta must be 1.")
            self.dsr = DifferentialSharpeReward(**{k:v for k,v in reward_config.items() if k not in {'mode','reference'}})
            self.observation_space.spaces['reward_state'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.current_step = 0

    def reset(self, seed=None, options=None):
        """Khôi phục môi trường về trạng thái nguyên thủy."""
        super().reset(seed=seed)
        self.current_step = 0
        
        self.portfolio.reset()
        if self.dsr is not None:
            self.dsr.reset()
            if options and "reward_state" in options:
                self.dsr.load_state_dict(options["reward_state"])
        self.drawdown_tracker.reset(self.config["initial_balance"])
        
        obs = self._get_obs()
        info = self._get_info()
        return obs, info
        
    def _get_obs(self):
        """Lấy quan sát tại thời điểm t (Market State + Portfolio State)."""
        history_index = self.history_start_index + self.current_step
        history_start = max(0, history_index - self.lookback_window + 1)
        history = self.history_tensor.tensor[history_start : history_index + 1]
        history_mask = np.arange(self.lookback_window) >= self.lookback_window - len(history)
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
            "history_mask": history_mask,
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
        if 'market_context' in self.quant_context:
            observation['market_context'] = self.quant_context['market_context'][self.bank_indices[self.current_step]].astype(np.float32)
        if self.dsr is not None:
            observation['reward_state'] = np.asarray([self.dsr.a, self.dsr.b, self.dsr.count], dtype=np.float32)
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
        action_alpha: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return policy and Close execution proposals (before Open caps/costs)."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.N + 1,):
            raise ValueError("Action phải có shape [N + 1], gồm cổ phiếu và tiền mặt.")
        alpha = self.action_alpha if action_alpha is None else float(action_alpha)
        if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError("action_alpha phải hữu hạn và thuộc (0, 1].")
        policy_target = self.rebalance_engine.project_weights(
            action, max_asset_weight=self.max_asset_weight
        )
        if current_weights is None:
            current_weights = self.portfolio.weights
        if self.execution_controller.replaces_action_alpha:
            proposed, _, _, _ = self._execution_plan(current_weights, policy_target, action_alpha)
            return policy_target, np.append(proposed, max(0., 1-proposed.sum()))
        blended_target = (
            (1.0 - alpha) * np.asarray(current_weights, dtype=np.float64)
            + alpha * policy_target
        )
        execution_target = self.rebalance_engine.project_weights(
            blended_target, max_asset_weight=self.max_asset_weight
        )
        return policy_target, execution_target

    def _execution_plan(self, current_weights, target_weights, alpha_override=None):
        """Pure Close-only plan; safe to call for diagnostics before step()."""
        bank_index = self.bank_indices[self.current_step]
        liquidity, risk = self.quant_context['liquidity'], self.quant_context['risk']
        trend = self.quant_context.get('trend')
        return self.execution_controller.decide(
            current_weights, target_weights, self.portfolio.portfolio_value,
            liquidity.dollar_adv[bank_index], liquidity.volatility[bank_index],
            risk.covariance[bank_index], liquidity.ready[bank_index] & risk.ready[bank_index],
            self.transaction_fee_pct+self.slippage_pct, base_alpha=self.action_alpha,
            efficiency_ratio=trend.efficiency_ratio[bank_index] if trend is not None else None,
            trend_ready=trend.ready[bank_index] if trend is not None else None,
            return_diagnostics=True, alpha_override=alpha_override)
        
    def step(self, action: np.ndarray | PortfolioOrder):
        """
        Timeline nhân quả: quyết định tại Close[t], khớp tại Open[t+1],
        đánh dấu danh mục tại Close[t+1]. Danh mục cũ chịu return qua đêm;
        danh mục mới chịu return trong phiên. Observation tiếp theo và NAV
        vì vậy cùng thuộc Close[t+1].

        PortfolioOrder.hold() giữ nguyên số cổ phiếu qua cả hai phân kỳ,
        không tái cân bằng hoặc tính phí. Trần vị thế chỉ áp dụng khi đặt lệnh
        rebalance, không ép giao dịch khi trọng số tự trôi trong một bước hold.
        """
        if self.current_step >= self.T - 1:
            return self._get_obs(), 0.0, False, True, self._get_info()

        hold = isinstance(action, PortfolioOrder) and action.target_weights is None
        final_order = isinstance(action, PortfolioOrder) and action.final_execution
        current_weights = self.portfolio.weights.copy()
        if not hold:
            order_alpha = action.action_alpha if isinstance(action, PortfolioOrder) else None
            target = action.target_weights if isinstance(action, PortfolioOrder) else action
            # Rebalance targets are fixed at Close[t], before the unseen gap.
            if final_order:
                if not self.cash_funded_execution or self.execution_controller.enabled:
                    raise ValueError('Final proposals require cash-funded execution and the scalar controller disabled.')
                target_weights = np.asarray(target, dtype=np.float64).copy()
                if (target_weights.shape != (self.N+1,)
                        or np.any(target_weights[:-1] > np.maximum(current_weights[:-1], self.max_asset_weight)+1e-8)):
                    raise ValueError('Final Close proposal violates asset shape/cap.')
            else:
                policy_target, target_weights = self.project_action(
                    target, current_weights=current_weights, action_alpha=order_alpha
                )

        t = self.current_step
        controlled_hold = None
        execution_diagnostics = None
        if not hold and self.cash_funded_execution:
            bank_index = self.bank_indices[t]
            liquidity = self.quant_context['liquidity']
            adv, sigma = liquidity.dollar_adv[bank_index], liquidity.volatility[bank_index]
            ready = liquidity.ready[bank_index]
            if final_order:
                if not ready.all():
                    raise ValueError('Final execution requires ready causal liquidity.')
                proposed = target_weights[:-1].copy()
                controlled_hold = action.hold_mask.copy()
                notional_caps = action.notional_caps.copy()
                execution_diagnostics = dict(speed_mode='final_close_proposal_no_second_blend',
                    close_proposed_weights=target_weights.copy(), hold_mask=controlled_hold.copy())
                if controlled_hold.all():
                    hold = True
            elif self.execution_controller.enabled:
                controller_target = policy_target if self.execution_controller.replaces_action_alpha else target_weights
                proposed, controlled_hold, notional_caps, execution_diagnostics = self._execution_plan(
                    current_weights, controller_target, order_alpha)
                if controlled_hold.all():
                    hold = True
            elif not ready.all():
                raise ValueError("Liquidity not ready for impact execution; warm up before trading.")
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

        # A hold keeps the overnight-drifted share quantities. Returning the
        # previous close weights as a rebalance target would create a trade.
        if not hold and controlled_hold is not None:
            target_weights = self.execution_controller.fill(proposed, controlled_hold,
                notional_caps, open_weights, previous_value*(1+overnight_gross), self.max_asset_weight)
            if np.allclose(target_weights, open_weights, atol=1e-12, rtol=0):
                hold = True
        if hold:
            target_weights = open_weights
            turnover = 0.0
            cost_rate = 0.0
        else:
            turnover = float(np.sum(np.abs(target_weights[:-1] - open_weights[:-1])))
            cost_rate = self.cost_engine.compute_cost_rate(target_weights, open_weights)
        value_at_open = previous_value * (1.0 + overnight_gross)
        transaction_cost = value_at_open * cost_rate
        commission_cost = value_at_open * self.transaction_fee_pct * turnover
        slippage_cost = value_at_open * self.slippage_pct * turnover
        stock_costs = value_at_open * (self.transaction_fee_pct+self.slippage_pct) * np.abs(target_weights[:-1]-open_weights[:-1]) if not hold else np.zeros(self.N)
        impact_cost = 0.
        participation = np.zeros(self.N)
        if not hold and self.cash_funded_execution:
            from src.env.cost import execute_cash_funded_trade
            target_weights, costs = execute_cash_funded_trade(open_weights, target_weights,
                value_at_open, self.transaction_fee_pct, self.slippage_pct, adv, sigma,
                self.impact_eta, self.max_asset_weight)
            transaction_cost = costs['transaction_cost']
            commission_cost, slippage_cost = costs['commission_cost'], costs['slippage_cost']
            impact_cost, participation = costs['impact_cost'], costs['participation']
            turnover, cost_rate = costs['turnover'], costs['cost_rate']
            stock_costs = costs['stock_costs']
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
        if self.dsr is not None:
            reference = (float(np.mean(self.close_prices[t+1]/self.close_prices[t]-1))
                if self.quant_config['reward'].get('reference', 'risk_free') == 'equal_weight' else daily_rf)
            reward = self.dsr.update(net_return-reference)
        self.current_step += 1
        
        obs = self._get_obs()
        info = self._get_info()
        info["execution_date"] = str(self.market_tensor.dates[t + 1])
        info["order_type"] = "hold" if hold else "rebalance"
        info["execution_weights"] = np.asarray(target_weights, dtype=np.float64).copy()
        if execution_diagnostics is not None:
            info['execution_controller'] = execution_diagnostics
        info["overnight_return"] = float(overnight_gross)
        info["intraday_return"] = float(intraday_gross)
        info["gross_return"] = float(gross_return)
        info["net_return"] = net_return
        info["cost_rate"] = cost_rate
        info["turnover"] = turnover
        info["transaction_cost"] = transaction_cost
        info["commission_cost"] = commission_cost
        info["slippage_cost"] = slippage_cost
        info['impact_cost'] = impact_cost
        info['participation_max'] = float(participation.max())
        info['participation_warning'] = bool(participation.max() > self.quant_config.get('impact', {}).get('participation_warning', .01))
        if self.dsr is not None:
            info['reward_state'] = self.dsr.state_dict()
            info['dsr_ready'] = self.dsr.ready
        info['asset_net_pnl'] = previous_value*current_weights[:-1]*overnight_returns + value_after_cost*target_weights[:-1]*intraday_returns - stock_costs
        info['cash_pnl'] = previous_value*current_weights[-1]*subperiod_rf + value_after_cost*target_weights[-1]*subperiod_rf
        info['previous_value'] = previous_value
        info['asset_costs'] = stock_costs
        info["reward"] = reward
        info["benchmark_return"] = benchmark_return
        
        terminated = False
        truncated = self.current_step >= self.T - 1
        
        return obs, reward, terminated, truncated, info

    def reward_state_dict(self):
        return self.dsr.state_dict() if self.dsr is not None else None

    def load_reward_state_dict(self, state):
        if self.dsr is None:
            if state is not None:
                raise ValueError("Cannot load DSR state into return reward environment.")
        else:
            self.dsr.load_state_dict(state)

    def _compute_gross_return(
        self, weights: np.ndarray, asset_returns: np.ndarray, cash_return: float
    ) -> float:
        """Portfolio return for one subperiod, including its cash yield."""
        weights = np.asarray(weights, dtype=np.float64)
        asset_returns = np.asarray(asset_returns, dtype=np.float64)
        return float(
            np.dot(weights[:-1], asset_returns) + weights[-1] * cash_return
        )

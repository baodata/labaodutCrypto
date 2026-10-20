"""
src/evaluation/backtester.py
Ticket: BACKTEST-001 (Sprint 5 - Thành viên A)

Bộ máy Backtest xác định (Deterministic Backtester).
Lặp qua từng ngày trong tập Test, ghi nhận Portfolio Value, Return, Phí, Drawdown.
"""

from dataclasses import dataclass
from typing import Callable, Dict
import numpy as np
import pandas as pd

from src.env.trading_env import TradingEnv
from src.evaluation.metrics import FinancialMetrics

@dataclass
class BacktestResult:
    """Lưu trữ kết quả backtest."""
    portfolio_values: np.ndarray
    daily_net_returns: np.ndarray
    daily_cost_rates: np.ndarray
    daily_turnover: np.ndarray
    daily_costs: np.ndarray
    daily_rewards: np.ndarray
    weights_history: np.ndarray
    total_steps: int
    daily_commission: np.ndarray | None = None
    daily_slippage: np.ndarray | None = None
    daily_impact: np.ndarray | None = None
    daily_participation_max: np.ndarray | None = None
    daily_hold: np.ndarray | None = None
    daily_execution_weights: np.ndarray | None = None
    extended_diagnostics: bool = False


class DeterministicBacktester:
    """Trình chạy kiểm thử (Backtester) tất định cho TradingEnv."""
    
    def __init__(self, env: TradingEnv):
        self.env = env
        
    def run_strategy(self, strategy_fn: Callable, *, on_step: Callable | None = None) -> BacktestResult:
        """
        Chạy một chiến lược cố định qua toàn bộ episode.
        
        Args:
            strategy_fn: Hàm nhận (obs, info), trả vector [N+1] hoặc PortfolioOrder
                để giữ cổ phiếu hoặc ghi đè alpha cho riêng một lệnh rebalance.
            on_step: Optional diagnostic callback receiving every execution info,
                including the final step; it must not change the environment.
        """
        obs, info = self.env.reset()
        
        portfolio_values = [self.env.portfolio.portfolio_value]
        daily_net_returns = []
        daily_cost_rates = []
        daily_turnover = []
        daily_costs = []
        daily_rewards = []
        breakdown = {key: [] for key in ("commission_cost", "slippage_cost", "impact_cost", "participation_max", "hold", "execution_weights")}
        weights_history = [self.env.portfolio.weights.copy()]
        
        done = False
        steps = 0
        
        while not done:
            action = strategy_fn(obs, info)
            obs, reward, terminated, truncated, info = self.env.step(action)
            if on_step is not None:
                on_step(info)
            done = terminated or truncated
            
            portfolio_values.append(self.env.portfolio.portfolio_value)
            daily_net_returns.append(info.get('net_return', 0.0))
            daily_cost_rates.append(info.get('cost_rate', 0.0))
            daily_turnover.append(info.get('turnover', 0.0))
            daily_costs.append(info.get('transaction_cost', 0.0))
            daily_rewards.append(reward)
            for key in breakdown:
                breakdown[key].append(info.get(key, info.get("order_type") == "hold" if key == "hold" else 0.))
            weights_history.append(self.env.portfolio.weights.copy())
            steps += 1
            
        return BacktestResult(
            portfolio_values=np.array(portfolio_values),
            daily_net_returns=np.array(daily_net_returns),
            daily_cost_rates=np.array(daily_cost_rates),
            daily_turnover=np.array(daily_turnover),
            daily_costs=np.array(daily_costs),
            daily_rewards=np.array(daily_rewards),
            weights_history=np.array(weights_history),
            total_steps=steps,
            daily_commission=np.asarray(breakdown["commission_cost"]),
            daily_slippage=np.asarray(breakdown["slippage_cost"]),
            daily_impact=np.asarray(breakdown["impact_cost"]),
            daily_participation_max=np.asarray(breakdown["participation_max"]),
            daily_hold=np.asarray(breakdown["hold"], dtype=bool),
            daily_execution_weights=np.asarray(breakdown["execution_weights"]),
            extended_diagnostics=bool(self.env.quant_context),
        )
        
    def run_agent(self, agent_fn: Callable) -> BacktestResult:
        """
        Chạy một Agent RL qua toàn bộ episode.
        
        Args:
            agent_fn: Hàm nhận obs (dict) và trả về action [N+1].
        """
        def wrapper(obs, info):
            return agent_fn(obs)
        return self.run_strategy(wrapper)

    def compare(
        self,
        results: Dict[str, BacktestResult],
        risk_free_rate_annual: float = 0.02,
    ) -> pd.DataFrame:
        """
        So sánh kết quả của nhiều chiến lược bằng bảng Financial Metrics.
        
        Args:
            results: Dictionary ánh xạ tên chiến lược -> BacktestResult.
            
        Returns:
            pd.DataFrame với rows là chiến lược, columns là metrics.
        """
        records = []
        for name, result in results.items():
            fm = FinancialMetrics(
                daily_net_returns=result.daily_net_returns,
                risk_free_rate_annual=risk_free_rate_annual,
                daily_turnover=result.daily_turnover,
                daily_costs=result.daily_costs,
            )
            row = fm.summary_dict()
            if self.env.quant_context:
                for field in ('daily_commission','daily_slippage','daily_impact'):
                    values = getattr(result, field, None)
                    row['total_' + field.removeprefix('daily_')] = float(np.sum(values)) if values is not None else 0.
                row['hold_fraction'] = float(np.mean(result.daily_hold)) if result.daily_hold is not None else 0.
                row['participation_max'] = float(np.max(result.daily_participation_max)) if result.daily_participation_max is not None else 0.
            row['Strategy'] = name
            records.append(row)
            
        df = pd.DataFrame(records)
        df.set_index('Strategy', inplace=True)
        return df

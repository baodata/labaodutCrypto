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


class DeterministicBacktester:
    """Trình chạy kiểm thử (Backtester) tất định cho TradingEnv."""
    
    def __init__(self, env: TradingEnv):
        self.env = env
        
    def run_strategy(self, strategy_fn: Callable) -> BacktestResult:
        """
        Chạy một chiến lược cố định qua toàn bộ episode.
        
        Args:
            strategy_fn: Hàm nhận (obs, info) và trả về action [N+1].
        """
        obs, info = self.env.reset()
        
        portfolio_values = [self.env.portfolio.portfolio_value]
        daily_net_returns = []
        daily_cost_rates = []
        daily_turnover = []
        daily_costs = []
        daily_rewards = []
        weights_history = [self.env.portfolio.weights.copy()]
        
        done = False
        steps = 0
        
        while not done:
            action = strategy_fn(obs, info)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            portfolio_values.append(self.env.portfolio.portfolio_value)
            daily_net_returns.append(info.get('net_return', 0.0))
            daily_cost_rates.append(info.get('cost_rate', 0.0))
            daily_turnover.append(info.get('turnover', 0.0))
            daily_costs.append(info.get('transaction_cost', 0.0))
            daily_rewards.append(reward)
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
            total_steps=steps
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
            row['Strategy'] = name
            records.append(row)
            
        df = pd.DataFrame(records)
        df.set_index('Strategy', inplace=True)
        return df

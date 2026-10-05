"""
tests/env/test_backtester.py
Ticket: BACKTEST-001 (Sprint 5 - Thành viên A)

Kiểm thử bộ máy Backtest tất định.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtester import DeterministicBacktester, BacktestResult
from src.env.trading_env import TradingEnv
from src.utils.data_types import MarketDataTensor


class TestBacktester:
    @pytest.fixture
    def env(self):
        T, N, F = 20, 3, 5
        tensor = np.random.randn(T, N, F).astype(np.float32)
        tickers = ["A", "B", "C"]
        features = [f"f{i}" for i in range(F)]
        dates = [f"2024-01-{i+1:02d}" for i in range(T)]
        market_tensor = MarketDataTensor(tensor, tickers, features, dates)
        
        # Open prices: tất cả tăng nhẹ 0.5%/ngày
        open_prices = np.ones((T, N), dtype=np.float32) * 100.0
        for t in range(1, T):
            open_prices[t] = open_prices[t-1] * 1.005
            
        return TradingEnv(market_tensor, open_prices)
    
    def test_cash_only_strategy(self, env):
        """Chiến lược 100% tiền mặt."""
        bt = DeterministicBacktester(env)
        
        def cash_fn(obs, info):
            return np.array([0.0, 0.0, 0.0, 1.0])
        
        result = bt.run_strategy(cash_fn)
        
        assert isinstance(result, BacktestResult)
        assert result.total_steps == env.T - 2  # 18 bước
        assert len(result.portfolio_values) == result.total_steps + 1
        # Cash sinh lãi rất nhỏ (risk-free rate)
        assert result.portfolio_values[-1] >= 100000.0
        
    def test_equal_weight_strategy(self, env):
        """Chiến lược phân bổ đều."""
        bt = DeterministicBacktester(env)
        
        def eq_fn(obs, info):
            N = 3
            w = np.zeros(N + 1)
            w[:N] = 1.0 / N
            return w
        
        result = bt.run_strategy(eq_fn)
        
        assert result.total_steps == env.T - 2
        # Giá tăng 0.5%/ngày => danh mục phải tăng trưởng
        assert result.portfolio_values[-1] > 100000.0
        
    def test_compare_strategies(self, env):
        """So sánh 2 chiến lược bằng DataFrame."""
        bt = DeterministicBacktester(env)
        
        def cash_fn(obs, info):
            return np.array([0.0, 0.0, 0.0, 1.0])
            
        def eq_fn(obs, info):
            return np.array([1/3, 1/3, 1/3, 0.0])
        
        res_cash = bt.run_strategy(cash_fn)
        res_eq = bt.run_strategy(eq_fn)
        
        df = bt.compare({"Cash": res_cash, "EqualWeight": res_eq})
        
        assert isinstance(df, pd.DataFrame)
        assert "Cash" in df.index
        assert "EqualWeight" in df.index
        assert "cumulative_return" in df.columns
        assert "sharpe_ratio" in df.columns
        
    def test_run_agent(self, env):
        """Test chạy Agent RL (wrapper)."""
        bt = DeterministicBacktester(env)
        
        def agent_fn(obs):
            return np.array([0.25, 0.25, 0.25, 0.25])
        
        result = bt.run_agent(agent_fn)
        assert result.total_steps == env.T - 2

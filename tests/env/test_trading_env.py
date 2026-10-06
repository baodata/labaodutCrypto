import pytest
import numpy as np
from src.env.trading_env import TradingEnv
from src.utils.data_types import MarketDataTensor

class TestTradingEnv:
    @pytest.fixture
    def mock_data(self):
        T, N, F = 10, 3, 5
        tensor = np.random.randn(T, N, F).astype(np.float32)
        tickers = ["A", "B", "C"]
        features = ["f1", "f2", "f3", "f4", "f5"]
        dates = [f"2024-01-{i+1:02d}" for i in range(T)]
        
        market_tensor = MarketDataTensor(tensor, tickers, features, dates)
        
        # Open prices monotonically increasing for easy math
        open_prices = np.ones((T, N), dtype=np.float32) * 100.0
        # Make asset 0 increase by 1% daily, asset 1 flat, asset 2 decrease 1%
        for i in range(1, T):
            open_prices[i, 0] = open_prices[i-1, 0] * 1.01
            open_prices[i, 1] = open_prices[i-1, 1] * 1.00
            open_prices[i, 2] = open_prices[i-1, 2] * 0.99
            
        return market_tensor, open_prices

    def test_env_init_and_reset(self, mock_data):
        market_tensor, open_prices = mock_data
        env = TradingEnv(market_tensor, open_prices)
        
        obs, info = env.reset()
        
        assert obs["market_state"].shape == (3, 5)
        assert obs["portfolio_weights"].shape == (3,)
        assert obs["cash_ratio"].shape == (1,)
        assert env.portfolio.portfolio_value == 100000.0
        assert info["cash_weight"] == 1.0
        assert info["step"] == 0

    def test_env_step(self, mock_data):
        market_tensor, open_prices = mock_data
        env = TradingEnv(market_tensor, open_prices)
        env.reset()
        
        # Action dồn hết 100% vào tài sản 0 (tăng 1% mỗi ngày)
        action = np.array([1.0, 0.0, 0.0, 0.0]) # N+1=4 (Asset A, B, C, Cash)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Lệnh mua từ 100% cash -> 100% Asset 0 => Bị trừ phí (Turnover = 2.0)
        # Cost = 2.0 * 0.001 = 0.002
        assert np.isclose(info["cost_rate"], 0.002)
        
        # Gross Return của Asset 0 = 1% = 0.01
        # Net Return = 0.01 - 0.002 = 0.008
        assert np.isclose(info["net_return"], 0.008)
        
        # Vốn = 100000 * 1.008 = 100800
        assert np.isclose(env.portfolio.portfolio_value, 100800.0)
        
        # Bước 2: AI quyết định giữ nguyên (Hold) Asset 0
        obs, reward, terminated, truncated, info = env.step(action)
        
        # AI hold => Không mất phí (Turnover = 0)
        assert np.isclose(info["cost_rate"], 0.0)
        
        # Lợi nhuận ròng = Lợi nhuận gộp = 1%
        assert np.isclose(info["net_return"], 0.01)
        
        # Vốn mới = 100800 * 1.01 = 101808
        assert np.isclose(env.portfolio.portfolio_value, 101808.0)

    def test_equal_weight_reward_baseline_only_changes_training_reward(self, mock_data):
        market_tensor, open_prices = mock_data
        for step in range(1, len(open_prices)):
            open_prices[step] = open_prices[step - 1] * 1.01

        raw_env = TradingEnv(market_tensor, open_prices, reward_baseline="none")
        excess_env = TradingEnv(
            market_tensor, open_prices, reward_baseline="equal_weight"
        )
        raw_env.reset(seed=3)
        excess_env.reset(seed=3)
        cash_action = np.array([0.0, 0.0, 0.0, 1.0])

        _, raw_reward, _, _, raw_info = raw_env.step(cash_action)
        _, excess_reward, _, _, excess_info = excess_env.step(cash_action)

        assert np.isclose(excess_info["benchmark_return"], 0.01)
        assert np.isclose(raw_info["net_return"], excess_info["net_return"])
        assert np.isclose(raw_reward - excess_reward, 1.0)

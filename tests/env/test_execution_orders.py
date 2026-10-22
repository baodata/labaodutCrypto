import numpy as np
import pandas as pd
import pytest

from scripts.baselines.evaluate_equal_weight_tilt import make_tilt_policy
from src.env.orders import PortfolioOrder
from src.env.trading_env import TradingEnv
from src.evaluation.backtester import DeterministicBacktester
from src.evaluation.baselines import BuyAndHoldBaseline
from src.utils.data_types import MarketDataTensor


def market_tensor(length, n_assets=2):
    return MarketDataTensor(
        np.zeros((length, n_assets, 2), dtype=np.float32),
        [f"asset_{i}" for i in range(n_assets)],
        ["rsi_14", "mom_12_1"],
        pd.date_range("2024-01-01", periods=length, freq="B").strftime("%Y-%m-%d").tolist(),
    )


def test_hold_preserves_shares_across_overnight_gap_and_intraday_move(trading_env_config_path):
    open_prices = np.array([[100., 100.], [100., 100.], [110., 90.]])
    close_prices = np.array([[100., 100.], [100., 100.], [121., 81.]])
    env = TradingEnv(
        market_tensor(3), open_prices, close_prices=close_prices,
        config_path=str(trading_env_config_path),
    )
    env.reset()
    env.step(np.array([0.5, 0.5, 0.0]))
    initial_shares = env.portfolio.portfolio_value * env.portfolio.asset_weights / close_prices[1]

    _, _, terminated, truncated, info = env.step(PortfolioOrder.hold())

    assert info["order_type"] == "hold"
    for key in ("turnover", "cost_rate", "transaction_cost", "commission_cost", "slippage_cost"):
        assert info[key] == 0.0
    final_shares = env.portfolio.portfolio_value * env.portfolio.asset_weights / close_prices[2]
    np.testing.assert_allclose(final_shares, initial_shares, rtol=1e-6)
    np.testing.assert_allclose(info["execution_weights"], [0.55, 0.45, 0.0], atol=1e-7)
    assert np.isclose(env.portfolio.portfolio_value, np.dot(initial_shares, close_prices[2]))
    assert not terminated and truncated


def test_hold_in_cash_accrues_interest_and_advances_date(trading_env_config_path):
    prices = np.full((2, 2), 100.0)
    env = TradingEnv(market_tensor(2), prices, close_prices=prices, config_path=str(trading_env_config_path))
    env.reset()
    obs, _, _, truncated, info = env.step(PortfolioOrder.hold())
    assert np.isclose(env.portfolio.portfolio_value, 100000 * (1.0 + 0.02 / 252))
    assert info["turnover"] == 0.0
    assert obs["cash_ratio"][0] == 1.0
    assert info["step"] == 1 and truncated


def test_buy_and_hold_buys_once_even_with_environment_smoothing(trading_env_config_path):
    prices = np.array([[100., 100.], [100., 100.], [110., 90.], [115., 85.]])
    env = TradingEnv(
        market_tensor(4), prices, close_prices=prices,
        config_path=str(trading_env_config_path), action_alpha=0.25,
    )
    result = DeterministicBacktester(env).run_strategy(BuyAndHoldBaseline(prices, 2))
    np.testing.assert_allclose(result.daily_turnover, [1.0, 0.0, 0.0])
    assert result.daily_costs[0] > 0.0
    np.testing.assert_array_equal(result.daily_costs[1:], 0.0)
    assert result.weights_history[1, -1] == 0.0
    assert env.action_alpha == 0.25


@pytest.mark.parametrize("interval", [1, 5, 21])
def test_scheduled_policy_enters_fully_and_does_not_trade_off_schedule(interval, trading_env_config_path):
    tensor = market_tensor(45)
    prices = np.column_stack((100 * 1.01 ** np.arange(45), 100 * 0.99 ** np.arange(45)))
    env = TradingEnv(
        tensor, prices, close_prices=prices,
        config_path=str(trading_env_config_path), action_alpha=0.25,
    )
    policy = make_tilt_policy(
        tensor.tensor, {"rsi_14": 0, "mom_12_1": 1}, 2,
        cash_ratio=0.04, tilt_fraction=0.0, rebalance_every=interval,
        no_trade_l1=0.0, rsi_orientation=1,
    )
    executions = []

    def record_policy(obs, info):
        order = policy(obs, info)
        executions.append(order)
        return order

    result = DeterministicBacktester(env).run_strategy(record_policy)
    assert executions[0].action_alpha == 1.0
    assert np.isclose(result.daily_turnover[0], 0.96)
    assert result.daily_costs[0] > 0.0
    assert np.isclose(result.weights_history[1, -1], 0.04, atol=1e-5)
    hold_dates = np.arange(result.total_steps) % interval != 0
    np.testing.assert_array_equal(result.daily_turnover[hold_dates], 0.0)
    np.testing.assert_array_equal(result.daily_costs[hold_dates], 0.0)
    assert env.action_alpha == 0.25


def test_no_trade_band_returns_hold_on_a_scheduled_date():
    tensor = market_tensor(3)
    policy = make_tilt_policy(
        tensor.tensor, {"rsi_14": 0, "mom_12_1": 1}, 2,
        cash_ratio=0.04, tilt_fraction=0.0, rebalance_every=1,
        no_trade_l1=0.03, rsi_orientation=1,
    )
    order = policy({"portfolio_weights": np.array([0.48, 0.48]), "cash_ratio": np.array([0.04])}, {"step": 1})
    assert isinstance(order, PortfolioOrder) and order.target_weights is None


def test_full_initial_order_does_not_change_alpha_for_later_vector_actions(trading_env_config_path):
    prices = np.full((3, 2), 100.0)
    env = TradingEnv(
        market_tensor(3), prices, close_prices=prices,
        config_path=str(trading_env_config_path), action_alpha=0.25,
    )
    env.reset()
    env.step(PortfolioOrder.rebalance(np.array([0.5, 0.5, 0.0]), action_alpha=1.0))
    _, _, _, _, info = env.step(np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(info["execution_weights"], [0.375, 0.375, 0.25])
    assert env.action_alpha == 0.25

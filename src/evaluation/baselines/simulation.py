"""One trading engine for the callable and standalone baseline APIs."""

import numpy as np
import pandas as pd

from src.env.trading_env import TradingEnv
from src.evaluation.backtester import DeterministicBacktester
from src.utils.data_types import MarketDataTensor


def simulate_baseline(strategy, open_prices, close_prices, cost_rate, risk_free_rate,
                      *, action_alpha=1.0, max_asset_weight=1.0, dates=None):
    if close_prices is None:
        raise ValueError("Baseline.run cần close_prices để dùng đúng timeline Close→Open→Close.")
    prices = np.asarray(open_prices)
    if prices.ndim != 2:
        raise ValueError("open_prices phải có shape [T,N].")
    t, n = prices.shape
    calendar = dates if dates is not None else pd.bdate_range("2000-01-03", periods=t).strftime("%Y-%m-%d").tolist()
    tensor = MarketDataTensor(np.zeros((t, n, 1), dtype=np.float32), [f"asset_{i}" for i in range(n)], ["baseline"], list(calendar))
    env = TradingEnv(tensor, prices, close_prices=close_prices, action_alpha=action_alpha,
                     environment_config={
                         "initial_balance": 100000., "transaction_fee_pct": cost_rate,
                         "slippage_pct": 0., "risk_free_rate": risk_free_rate,
                         "reward_beta": 1., "reward_scaling": 1., "max_asset_weight": max_asset_weight,
                     })
    return DeterministicBacktester(env).run_strategy(strategy).daily_net_returns

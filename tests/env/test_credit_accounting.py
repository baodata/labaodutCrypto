"""Credit PnL must reconcile to NAV without float32 accounting roundoff."""
import numpy as np
import pandas as pd
import pytest

from src.env.orders import PortfolioOrder
from src.env.trading_env import TradingEnv
from src.features.liquidity import LiquidityBank
from src.risk.covariance import CovarianceBank
from src.utils.data_types import MarketDataTensor


@pytest.mark.parametrize('mode,eta,controller', [
    ('legacy', 0., False), ('cash_funded', 0., False), ('impact', .2, False), ('adaptive', .2, True),
])
def test_credit_nav_cost_and_hold_share_reconciliation(mode, eta, controller):
    dates = pd.bdate_range('2020-01-01', periods=5)
    market = MarketDataTensor(np.zeros((5, 2, 1), np.float32), ['a', 'b'], ['return'], list(dates))
    opening = np.array([[100.,100.], [100.,100.], [110.,90.], [108.,93.], [112.,95.]])
    closing = np.array([[100.,100.], [103.,97.], [108.,93.], [111.,94.], [113.,96.]])
    config = dict(initial_balance=100000.,risk_free_rate=.02,reward_beta=1.,reward_scaling=100.,
                  transaction_fee_pct=.001,slippage_pct=.0005,max_asset_weight=.6)
    liq = LiquidityBank(list(dates), ['a','b'], np.full((5,2), 1e6), np.full((5,2), .02), np.ones((5,2), bool), {})
    risk = CovarianceBank(list(dates), ['a','b'], np.tile(np.eye(2)*.0004,(5,1,1)), np.ones(5,bool), pd.DataFrame(), {})
    context = None if mode == 'legacy' else dict(dates=list(dates), tickers=['a','b'], liquidity=liq, risk=risk,
        config=dict(impact=dict(eta=eta,accounting='cash_funded'),
                    execution=dict(enabled=controller,no_trade_band=.002,participation_cap=.01)))
    env = TradingEnv(market, opening, close_prices=closing, environment_config=config,
                     quant_context=context, action_alpha=.25)
    env.reset()
    actions = [np.array([.45,.45,.1]), PortfolioOrder.hold(), np.array([.2,.6,.2]), PortfolioOrder.hold()]
    for t, action in enumerate(actions):
        previous = env.portfolio.portfolio_value
        shares = previous * env.portfolio.asset_weights / closing[t]
        obs, _, _, _, info = env.step(action)
        nav = env.portfolio.portfolio_value
        np.testing.assert_allclose(info['asset_net_pnl'].sum()+info['cash_pnl'], nav-previous, rtol=0, atol=1e-8)
        expected = (previous*(1+info['overnight_return'])-info['transaction_cost'])*(1+info['intraday_return'])
        np.testing.assert_allclose(nav, expected, rtol=0, atol=1e-8)
        np.testing.assert_allclose(info['commission_cost']+info['slippage_cost']+info['impact_cost'], info['transaction_cost'], rtol=0, atol=1e-8)
        assert env.portfolio.weights.dtype == np.float64
        assert info['execution_weights'].dtype == np.float64
        assert obs['portfolio_weights'].dtype == np.float32
        assert obs['cash_ratio'].dtype == np.float32
        if isinstance(action, PortfolioOrder):
            assert info['transaction_cost'] == 0
            np.testing.assert_allclose(nav*env.portfolio.asset_weights/closing[t+1], shares, rtol=1e-12, atol=1e-10)
        elif mode != 'legacy':
            open_nav = previous*(1+info['overnight_return'])
            q = np.abs((open_nav-info['transaction_cost'])*info['execution_weights'][:-1]-shares*opening[t+1])
            np.testing.assert_allclose(info['commission_cost'], q.sum()*.001, rtol=0, atol=1e-8)

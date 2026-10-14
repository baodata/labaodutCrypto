import numpy as np
import pytest

from src.evaluation.metrics import FinancialMetrics


def test_sharpe_uses_arithmetic_daily_excess_and_keeps_cagr_separate():
    returns = np.tile([0.10, -0.08, 0.04, -0.02], 63)
    rf = 0.02
    metrics = FinancialMetrics(returns, risk_free_rate_annual=rf)
    expected = np.sqrt(252) * np.mean(returns - rf / 252) / np.std(returns, ddof=1)
    assert metrics.sharpe_ratio == pytest.approx(expected)
    assert metrics.annualized_volatility == pytest.approx(np.std(returns, ddof=1) * np.sqrt(252))
    assert metrics.annualized_return == pytest.approx(np.prod(1.0 + returns) - 1.0)
    assert not np.isclose(metrics.sharpe_ratio, (metrics.annualized_return - rf) / metrics.annualized_volatility)


@pytest.mark.parametrize("returns", [[], [0.01], [0.001] * 10])
def test_sharpe_is_finite_zero_when_variance_cannot_be_estimated(returns):
    metrics = FinancialMetrics(np.asarray(returns))
    assert metrics.sharpe_ratio == 0.0
    assert np.isfinite(metrics.annualized_volatility)


def test_sharpe_respects_configured_trading_days():
    returns = np.array([0.02, -0.01, 0.0, 0.015])
    metrics = FinancialMetrics(returns, risk_free_rate_annual=0.03, trading_days_per_year=52)
    expected = np.sqrt(52) * np.mean(returns - 0.03 / 52) / np.std(returns, ddof=1)
    assert metrics.sharpe_ratio == pytest.approx(expected)

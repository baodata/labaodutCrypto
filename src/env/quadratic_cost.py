"""Declared quadratic surrogate for temporary impact; never an additional fee."""
import numpy as np


def fit_quadratic_cost(dollar_adv, daily_volatility, *, nav, eta, weight_grid,
                       multiplier=1., minimum_eigenvalue=1e-8):
    """Fit nonnegative diagonal cost/NAV = .5 lambda * delta_weight**2.

    All inputs must come from a known trailing training block. OHLCV-derived
    square-root scenarios are not execution-data calibration. Commission and
    slippage remain separate linear planning/realized costs.
    """
    adv, sigma, grid = map(lambda x: np.asarray(x, dtype=np.float64),
                          (dollar_adv, daily_volatility, weight_grid))
    if (adv.ndim != 2 or sigma.shape != adv.shape or not len(adv) or grid.ndim != 1 or not len(grid)
            or not np.isfinite([nav, eta, multiplier, minimum_eigenvalue]).all()
            or min(nav, multiplier, minimum_eigenvalue) <= 0 or eta < 0
            or not all(np.isfinite(x).all() for x in (adv, sigma, grid))
            or np.any(adv <= 0) or np.any(sigma < 0) or np.any(grid <= 0) or np.any(grid > 1)):
        raise ValueError('Invalid quadratic impact scenario inputs.')
    design = .5*grid**2
    target = (grid[None, :, None]*eta*sigma[:, None, :]
              *np.sqrt(nav*grid[None, :, None]/adv[:, None, :]))
    coefficient = np.sum(target*design[None, :, None], axis=(0, 1))/(len(adv)*np.sum(design**2))
    coefficients = np.maximum(multiplier*coefficient, minimum_eigenvalue)
    predicted = design[None, :, None]*coefficients[None, None, :]
    return np.diag(coefficients), dict(source='OHLCV_square_root_scenario_not_observed_fills',
        units='cost_fraction_NAV_per_squared_weight_change', nav=nav, eta=eta,
        multiplier=multiplier, order_weight_grid=grid.tolist(), samples=len(adv),
        minimum_eigenvalue=minimum_eigenvalue, eigenvalue_floors=int(np.sum(multiplier*coefficient < minimum_eigenvalue)),
        rmse_cost_fraction=float(np.sqrt(np.mean((predicted-target)**2))),
        coefficients=coefficients.tolist(), commission_included=False,
        realized_fee_policy='existing_cash_funded_cost_engine_no_quadratic_double_charge')


def weight_to_holdings_cost(lambda_weights, prices, nav):
    """Lambda_x = diag(price) Lambda_w diag(price) / NAV; fees in money."""
    matrix, prices = np.asarray(lambda_weights, float), np.asarray(prices, float)
    if (prices.ndim != 1 or matrix.shape != (len(prices), len(prices))
            or not np.isfinite(prices).all() or np.any(prices <= 0) or not np.isfinite(nav) or nav <= 0):
        raise ValueError('Invalid weight-to-holdings units.')
    return prices[:, None]*matrix*prices[None, :]/nav

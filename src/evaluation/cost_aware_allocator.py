"""Bounded 1/N tilts and a convex active-risk/linear-cost allocation benchmark.

Only Close-time holdings, forecasts and causal covariance are inputs. Future
Open drift is unknown here; actual turnover/fees are charged by TradingEnv.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize


def asset_bounds(n_assets, cash_ratio, tilt_fraction, max_asset_weight):
    if (n_assets <= 0 or not np.isfinite([cash_ratio, tilt_fraction, max_asset_weight]).all()
            or not 0 <= cash_ratio < 1 or not 0 <= tilt_fraction <= 1 or not 0 < max_asset_weight <= 1):
        raise ValueError("Invalid allocator cash/tilt/cap constraints.")
    core = np.full(n_assets, (1 - cash_ratio) / n_assets)
    lower = core * (1 - tilt_fraction)
    upper = np.minimum(core * (1 + tilt_fraction), max_asset_weight)
    if np.any(core > upper + 1e-12):
        raise ValueError("Equal-weight core is infeasible under this asset cap/cash ratio.")
    return core, lower, upper


def bounded_tilt(prediction, prediction_scale, *, cash_ratio, tilt_fraction, max_asset_weight):
    """Directional baseline: normalize only by the predictor's historical fit scale."""
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.ndim != 1 or not np.isfinite(prediction).all() or not np.isfinite(prediction_scale) or prediction_scale <= 0:
        raise ValueError("Finite predictions and a positive past-fitted scale are required.")
    core, lower, upper = asset_bounds(len(prediction), cash_ratio, tilt_fraction, max_asset_weight)
    score = np.clip(prediction / prediction_scale, -1., 1.)
    score -= score.mean()
    score /= max(float(np.abs(score).max()), 1.)
    proposed = core * (1 + tilt_fraction * score)
    lo, hi = float(np.min(proposed - upper)), float(np.max(proposed - lower))
    for _ in range(60):
        middle = (lo + hi) / 2
        if np.clip(proposed - middle, lower, upper).sum() > 1 - cash_ratio:
            lo = middle
        else:
            hi = middle
    return np.append(np.clip(proposed - (lo + hi) / 2, lower, upper), cash_ratio)


def cost_aware_target(prediction, daily_covariance, current_weights, *, horizon, action_alpha,
                      cash_ratio, tilt_fraction, max_asset_weight, risk_aversion, cost_rate,
                      cost_multiplier=1., max_iterations=200):
    """Solve over policy targets u, accounting for execution blending x=(1-a)w+a*u.

    Minimize lambda/2 * (x-x_core)' (h*Sigma) (x-x_core)
             - mu_h' x + multiplier * fee * sum(abs(x_stock-w_stock)).
    Cash target, tilt bounds and cap match the directional baseline. Risk is
    tracking risk; h*Sigma is an iid planning approximation, not an estimated
    multi-horizon covariance. This is a receding-horizon single-step benchmark.
    """
    mu, covariance, current = map(lambda x: np.asarray(x, dtype=np.float64),
                                  (prediction, daily_covariance, current_weights))
    n = len(mu)
    core, lower, upper = asset_bounds(n, cash_ratio, tilt_fraction, max_asset_weight)
    if mu.shape != (n,) or covariance.shape != (n, n) or current.shape != (n + 1,):
        raise ValueError("Allocator inputs have inconsistent asset dimensions.")
    if (not all(np.isfinite(x).all() for x in (mu, covariance, current))
            or not np.isclose(current.sum(), 1.) or np.any(current < -1e-10)
            or type(horizon) is not int or horizon <= 0 or not 0 < action_alpha <= 1
            or not np.isfinite([risk_aversion, cost_rate, cost_multiplier]).all()
            or min(risk_aversion, cost_rate, cost_multiplier) < 0):
        raise ValueError("Invalid causal allocation inputs/parameters.")
    covariance = (covariance + covariance.T) / 2
    if np.linalg.eigvalsh(covariance)[0] < -1e-10:
        raise ValueError("Convex allocator requires a PSD covariance matrix.")
    if tilt_fraction == 0:
        return np.append(core, cash_ratio), {"status": "fixed_core", "iterations": 0}
    alpha = action_alpha
    risk = covariance * horizon
    penalty = cost_rate * cost_multiplier
    # Epigraph variables z make the absolute turnover penalty differentiable.
    def objective(vector):
        u, z = vector[:n], vector[n:]
        active = alpha * (u - core)
        return 1000 * (.5 * risk_aversion * active @ risk @ active - alpha * mu @ u + penalty * z.sum())

    def gradient(vector):
        active = alpha * (vector[:n] - core)
        return 1000 * np.r_[alpha * (risk_aversion * risk @ active - mu), np.full(n, penalty)]

    identity = np.eye(n)
    inequalities = np.block([[-alpha * identity, identity], [alpha * identity, identity]])
    rhs = np.r_[-alpha * current[:-1], alpha * current[:-1]]
    constraints = [LinearConstraint(np.r_[np.ones(n), np.zeros(n)][None], [1-cash_ratio], [1-cash_ratio]),
                   LinearConstraint(inequalities, rhs, np.full(2*n, np.inf))]
    initial = np.r_[core, np.abs(alpha * (core - current[:-1]))]
    result = minimize(objective, initial, jac=gradient, method="SLSQP", constraints=constraints,
        bounds=Bounds(np.r_[lower, np.zeros(n)], np.r_[upper, np.ones(n)]),
        options={"maxiter": max_iterations, "ftol": 1e-10})
    if (not result.success or not np.isfinite(result.x).all()
            or abs(result.x[:n].sum() - (1-cash_ratio)) > 1e-7
            or np.min(inequalities @ result.x - rhs) < -1e-7):
        raise RuntimeError(f"Cost-aware allocator failed; no silent fallback: {result.message}")
    target = np.append(result.x[:n], cash_ratio)
    blended = (1-alpha) * current + alpha * target
    return target, {"status": "optimal", "iterations": int(result.nit),
        "estimated_cost_rate_at_close": float(cost_rate * np.abs(blended[:-1]-current[:-1]).sum()),
        "planned_active_variance": float((alpha*(target[:-1]-core)) @ risk @ (alpha*(target[:-1]-core)))}

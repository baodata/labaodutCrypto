"""Forecast-only GP proposals; final orders are filled once by TradingEnv."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.env.orders import PortfolioOrder
from src.evaluation.cost_aware_allocator import cost_aware_target
from src.evaluation.gp_allocator import solve_gp_reference, constrained_gp
from src.models.signal_dynamics import SignalDynamics
from src.utils.artifacts import atomic_json


GP_METHODS = ('matched_1n', 'convex', 'aim_fixed', 'aim_gp', 'gp_constrained', 'gp_mpc', 'gp_zero_forecast')


@dataclass
class GPAdapter:
    dynamics: SignalDynamics
    cost: np.ndarray
    config: dict
    core: np.ndarray
    fit_id: str

    def __post_init__(self):
        solver = self.config['solver']
        self.reference = solve_gp_reference(self.dynamics.covariance, self.cost,
            self.dynamics.transition, self.dynamics.loading,
            risk_aversion=self.config['risk_aversion'], discount=self.config['discount_factor'],
            tolerance=solver['bellman_tolerance'], max_iterations=solver['bellman_max_iterations'])

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.dynamics.save(path/'signal_dynamics.npz')
        r = self.reference
        np.savez_compressed(path/'reference.npz', cost=self.cost, core=self.core,
            P=r.P, Q=r.Q, M=r.M, D=r.D, hold_gain=r.hold_gain, signal_gain=r.signal_gain)
        atomic_json(path/'adapter.json', dict(config=self.config, fit_id=self.fit_id,
            reference=r.diagnostics, coordinate_system='active_NAV_fraction_frozen_planning_approximation'))

    @classmethod
    def load(cls, path):
        import json
        path = Path(path)
        metadata = json.loads((path/'adapter.json').read_text())
        with np.load(path/'reference.npz', allow_pickle=False) as a:
            return cls(SignalDynamics.load(path/'signal_dynamics.npz'), a['cost'].copy(),
                metadata['config'], a['core'].copy(), metadata['fit_id'])

    def budgets(self, step, horizon):
        execution = self.config['execution']
        return (1-execution['cash_ratio'])*(1-(1-execution['ramp_alpha'])**np.arange(step+1, step+horizon+1))

    def propose(self, method, forecast, current, *, step, adv, nav, max_asset_weight, horizon):
        if method not in GP_METHODS:
            raise ValueError('Unknown GP comparison method.')
        current = np.asarray(current, float)
        n = len(self.core)
        f = self.dynamics.factors(forecast)
        if method == 'gp_zero_forecast':
            f = np.zeros_like(f)  # includes the intercept: zero expected alpha, not just zero raw forecasts
        r, settings = self.reference, self.config
        caps = settings['execution']['participation_cap']*np.asarray(adv, float)
        H = settings['mpc_horizon'] if method == 'gp_mpc' else 1
        budgets = self.budgets(step, H)
        lower = budgets[:, None]/n*(1-settings['tilt_fraction'])*np.ones((1, n))
        upper = np.minimum(budgets[:, None]/n*(1+settings['tilt_fraction'])*np.ones((1, n)), max_asset_weight)
        active = current[:-1]-self.core
        aim = self.core+r.aim(f)
        unconstrained = self.core+r.proposal(active, f)
        solver_info = dict(status='explicit_control', iterations=0, primal_residual=0., planning_horizon=1)
        if method == 'matched_1n':
            assets = np.full(n, budgets[0]/n)
        elif method == 'convex':
            target, info = cost_aware_target(forecast, self.dynamics.covariance, current,
                horizon=horizon, action_alpha=1., cash_ratio=float(1-budgets[0]),
                tilt_fraction=settings['tilt_fraction'], max_asset_weight=max_asset_weight,
                risk_aversion=settings['risk_aversion'], cost_rate=settings['_linear_cost_rate'])
            assets, solver_info = target[:-1], dict(solver_info, **info)
        else:
            desired = None
            if method == 'aim_fixed':
                desired = current[:-1]+settings['execution']['ramp_alpha']*(aim-current[:-1])
            elif method == 'aim_gp':
                desired = unconstrained
            assets, solver_info = constrained_gp(r, current[:-1], f, self.core,
                budgets, lower, upper, caps/nav, linear_cost=settings['_linear_cost_rate'],
                max_iterations=settings['solver']['qp_max_iterations'],
                tolerance=settings['solver']['qp_tolerance'],
                gap_tolerance=settings['solver']['qp_gap_tolerance'], desired=desired)
        target = np.r_[assets, 1-assets.sum()]
        hold_mask = abs(assets-current[:-1]) <= settings['execution']['no_trade_band']
        order = PortfolioOrder.execution(target, hold_mask=hold_mask, notional_caps=caps)
        delta = assets-current[:-1]
        return order, dict(solver_info, fit_id=self.fit_id, budget=float(budgets[0]),
            expected_step_active_payoff=float((self.dynamics.loading@f)@assets),
            planned_quadratic_cost=float(.5*delta@self.cost@delta),
            planned_linear_cost=float(settings['_linear_cost_rate']*abs(delta).sum()),
            conditional_active_variance=float((assets-self.core)@self.dynamics.covariance@(assets-self.core)),
            mean_abs_tilt=float(np.mean(abs(assets-budgets[0]/n))),
            aim=aim.tolist(), unconstrained=unconstrained.tolist(), factors=f.tolist(), target=target.tolist(),
            bellman_residual=r.diagnostics['bellman_residual'])

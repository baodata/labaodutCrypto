"""Past-only forecast-to-step calibration and diagonal signal dynamics for GP."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.risk.covariance import estimate_covariance
from src.utils.artifacts import atomic_json


@dataclass
class SignalDynamics:
    mean: np.ndarray
    scale: np.ndarray
    transition: np.ndarray
    loading: np.ndarray
    covariance: np.ndarray
    metadata: dict

    def factors(self, forecast):
        forecast = np.asarray(forecast, dtype=np.float64)
        if forecast.shape != self.mean.shape or not np.isfinite(forecast).all():
            raise ValueError('GP forecast shape/order/values mismatch.')
        return np.r_[(forecast-self.mean)/self.scale, 1.]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, mean=self.mean, scale=self.scale, transition=self.transition,
            loading=self.loading, covariance=self.covariance)
        atomic_json(path.with_suffix('.json'), self.metadata)

    @classmethod
    def load(cls, path):
        import json
        path = Path(path)
        with np.load(path, allow_pickle=False) as arrays:
            return cls(**{k: arrays[k].copy() for k in ('mean', 'scale', 'transition', 'loading', 'covariance')},
                metadata=json.loads(path.with_suffix('.json').read_text()))


def fit_signal_dynamics(forecasts, intraday_returns, total_returns, positions, fit_ids,
                        calendar, *, cutoff, tickers, min_samples=126, ridge=5.,
                        max_persistence=.995, covariance_method='ledoit_wolf', jitter=1e-10):
    """Caller supplies only OOF rows; select outcomes ending no later than cutoff.

    Factors are centered/scaled on this past fit and augmented by a deterministic
    constant. F's stochastic block is stable; its constant block is exactly one.
    Calibration predicts cross-sectional intraday excess, expected overnight
    alpha is zero. Residual *total* returns retain overnight risk in Sigma.
    """
    mu, intra, total = [np.asarray(v, dtype=np.float64) for v in
                        (forecasts, intraday_returns, total_returns)]
    positions, fit_ids = np.asarray(positions), np.asarray(fit_ids)
    if (mu.ndim != 2 or intra.shape != mu.shape or total.shape != mu.shape
            or positions.shape != (len(mu),) or fit_ids.shape != positions.shape
            or mu.shape[1] != len(tickers) or len(set(tickers)) != len(tickers)
            or not np.issubdtype(positions.dtype, np.integer) or np.any(np.diff(positions) <= 0)
            or not all(np.isfinite(x).all() for x in (mu, intra, total))
            or type(min_samples) is not int or min_samples < 3
            or not np.isfinite([ridge, max_persistence, jitter]).all()
            or ridge < 0 or not 0 < max_persistence < 1 or jitter <= 0):
        raise ValueError('Invalid GP calibration panel/configuration.')
    index = pd.DatetimeIndex(calendar)
    cutoff = int(cutoff)
    if cutoff < 0 or cutoff >= len(index) or np.any(positions < 0) or np.any(positions+1 >= len(index)):
        raise ValueError('Calibration dates fall outside calendar.')
    good = positions+1 <= cutoff
    x, y, r, p, ids = mu[good], intra[good], total[good], positions[good], fit_ids[good]
    if len(x) < min_samples:
        raise ValueError(f'GP calibration needs {min_samples} matured OOF outcomes; got {len(x)}.')
    n = x.shape[1]
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-8)
    z = (x-mean)/scale
    y = y-y.mean(1, keepdims=True)
    intercept = y.mean(0)
    pooled = float(np.sum(z*(y-intercept))/max(float(np.sum(z*z)), 1e-12))
    slope = (np.sum(z*(y-intercept), axis=0)+ridge*pooled)/(np.sum(z*z, axis=0)+ridge)
    center = np.eye(n)-np.ones((n, n))/n
    loading = center @ np.column_stack((np.diag(slope), intercept))
    pairs = (np.diff(p) == 1) & (ids[:-1] == ids[1:])
    if np.count_nonzero(pairs) < max(10, min_samples//4):
        raise ValueError('Not enough adjacent same-fit forecast pairs for GP dynamics.')
    before, after = z[:-1][pairs], z[1:][pairs]
    raw = np.sum(before*after, axis=0)/(np.sum(before*before, axis=0)+ridge)
    persistence = np.clip(raw, -max_persistence, max_persistence)
    transition = np.diag(np.r_[persistence, 1.])
    factors = np.column_stack((z, np.ones(len(z))))
    expected = factors @ loading.T
    residual = r-expected
    covariance, risk_metadata = estimate_covariance(residual, method=covariance_method, jitter=jitter)
    # SPD is required by the reference; record any additional zero-risk floor.
    floor = max(0., jitter-float(np.linalg.eigvalsh(covariance)[0]))
    covariance = covariance+floor*np.eye(n)
    halves = [float(np.log(.5)/np.log(d)) if 0 < d < 1 else None for d in persistence]
    lag_rows = []
    for lag in (0, 1, 5, 20):
        left, right = np.where(p[:, None]+lag == p[None, :])
        if len(left) < 3:
            continue
        a, b = expected[left].ravel(), y[right].ravel()
        lag_rows.append(dict(lag=lag, observations=len(left),
            predictive_correlation=float(np.corrcoef(a, b)[0, 1]) if min(a.std(), b.std()) > 1e-12 else 0.))
    metadata = dict(tickers=list(tickers), samples=len(x), same_fit_pairs=int(pairs.sum()),
        first_decision=str(index[p[0]]), last_decision=str(index[p[-1]]),
        maximum_label_end_used=str(index[p[-1]+1]), fit_cutoff=str(index[cutoff]),
        minimum_samples=min_samples, ridge=ridge, stochastic_spectral_radius=float(np.abs(persistence).max()),
        persistence_raw=raw.tolist(), persistence=persistence.tolist(), half_life_sessions=halves,
        clipped_count=int(np.count_nonzero(raw != persistence)), max_persistence=max_persistence,
        constant_factor='deterministic_one_not_a_stochastic_unit_root',
        payoff='Open_tplus1_to_Close_tplus1_cross_sectional_excess',
        overnight_expected_active_return='zero_assumption_not_an_estimated_forecast',
        risk='Close_to_Close_total_return_minus_intraday_expected_active_payoff',
        risk_diagnostics=dict(risk_metadata, additional_spd_floor=floor), lag_efficacy=lag_rows,
        mapping_mse=float(np.mean((expected-y)**2)),
        zero_mapping_mse=float(np.mean(y*y)), condition_number=float(np.linalg.cond(covariance)),
        return_units='decimal_per_session', forecast_horizon_conversion='OOF_calibration_not_division_by_h')
    return SignalDynamics(mean, scale, transition, loading, covariance, metadata)

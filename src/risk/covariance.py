"""QNT-006: daily covariance, never annualized inside the bank.

Linear Ledoit-Wolf uses sklearn's centered estimator; sample/EWMA use the
same complete trailing observations. Missing rows are excluded, never imputed
with future data. An unready date produces zeros and readiness=False.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from io import StringIO
import json
from pathlib import Path
import numpy as np
import pandas as pd


@lru_cache(maxsize=1)
def _ledoit_wolf_backend():
    """Load sklearn only for Ledoit-Wolf; other estimators need only NumPy."""
    try:
        import sklearn
        from sklearn.covariance import ledoit_wolf
    except ModuleNotFoundError as exc:
        if exc.name != "sklearn" and not (exc.name or "").startswith("sklearn."):
            raise
        import shlex
        import sys
        command = shlex.join([sys.executable, "-m", "pip", "install", "scikit-learn>=1.3.0"])
        raise ModuleNotFoundError(
            "Covariance method 'ledoit_wolf' requires scikit-learn "
            f"(Python import: sklearn). Install it in this environment:\n{command}"
        ) from exc
    return sklearn.__version__, ledoit_wolf


def ensure_covariance_dependencies(method):
    """Fail before creating artifacts or preprocessing data if a backend is absent."""
    if method not in {"sample", "ewma", "ledoit_wolf"}:
        raise ValueError("Unknown covariance estimator.")
    return _ledoit_wolf_backend()[0] if method == "ledoit_wolf" else None


def covariance_to_correlation(covariance, eps=1e-12):
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or not np.isfinite(covariance).all():
        raise ValueError("Covariance must be a finite square matrix.")
    std = np.sqrt(np.maximum(np.diag(covariance), 0))
    denominator = np.outer(std, std)
    corr = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > eps)
    np.fill_diagonal(corr, 0)  # self loops belong to the GNN
    return np.clip((corr + corr.T) / 2, -1, 1)


def estimate_covariance(observations, method="ledoit_wolf", decay=.94, jitter=1e-12):
    x = np.asarray(observations, dtype=float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Need at least two complete daily observations [T,N].")
    if method not in {"sample", "ewma", "ledoit_wolf"} or not 0 < decay < 1 or jitter < 0 or not np.isfinite([decay,jitter]).all():
        raise ValueError("Invalid covariance configuration.")
    shrinkage = 0.
    if method == "ledoit_wolf":
        _, ledoit_wolf = _ledoit_wolf_backend()
        cov, shrinkage = ledoit_wolf(x, assume_centered=False)
    elif method == "sample":
        centered = x - x.mean(0)
        cov = centered.T @ centered / (len(x) - 1)
    else:
        weights = decay ** np.arange(len(x)-1, -1, -1)
        weights /= weights.sum()
        centered = x - weights @ x
        cov = (centered * weights[:, None]).T @ centered
    cov = np.atleast_2d(cov)
    cov = (cov + cov.T) / 2
    minimum = float(np.linalg.eigvalsh(cov)[0])
    correction = max(0., jitter - minimum)
    # A constant asset must retain zero variance, not gain artificial graph edges.
    if np.max(np.abs(cov)) > 0:
        cov += np.eye(x.shape[1]) * correction
    else:
        correction = 0.
    eigenvalues = np.linalg.eigvalsh(cov)
    condition = float(eigenvalues[-1] / max(eigenvalues[0], 1e-18))
    return cov, {"shrinkage": float(shrinkage), "jitter": correction,
                 "min_eigenvalue": float(eigenvalues[0]), "condition_number": condition}


@dataclass
class CovarianceBank:
    dates: list
    tickers: list
    covariance: np.ndarray
    ready: np.ndarray
    diagnostics: pd.DataFrame
    metadata: dict

    def at(self, date, tickers=None):
        if tickers is not None and list(tickers) != self.tickers:
            raise ValueError("Risk bank ticker order mismatch.")
        index = pd.DatetimeIndex(self.dates).get_loc(pd.Timestamp(date))
        if not self.ready[index]:
            raise ValueError("Covariance not ready on requested date.")
        return self.covariance[index].copy()


def build_covariance_bank(returns, *, window=60, min_periods=None, method="ledoit_wolf",
                          decay=.94, jitter=1e-12, cache_dir=None):
    if not isinstance(returns, pd.DataFrame) or not returns.index.is_unique or not returns.index.is_monotonic_increasing or not returns.columns.is_unique:
        raise ValueError("Ordered unique dates and tickers required.")
    min_periods = window if min_periods is None else int(min_periods)
    if method not in {"sample","ewma","ledoit_wolf"}:
        raise ValueError("Unknown covariance estimator.")
    sklearn_version = ensure_covariance_dependencies(method)
    if window < 2 or not 2 <= min_periods <= window or np.isinf(returns.to_numpy()).any():
        raise ValueError("Invalid covariance window/data.")
    from src.utils.artifacts import file_hashes
    config = {"window": window, "min_periods": min_periods, "method": method,
              "decay": decay, "jitter": jitter, "units": "daily_simple_return_squared",
              "alignment": "through_close_t", "numpy_version": np.__version__, "sklearn_version": sklearn_version, "tickers": list(returns.columns),
              "dates": [str(d) for d in returns.index],
              "implementation_hash": file_hashes([Path(__file__)], Path(__file__).resolve().parents[2])}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode() +
                           np.asarray(returns, dtype='<f8').tobytes()).hexdigest()
    path = Path(cache_dir) / f"covariance_{digest}.npz" if cache_dir else None
    if path and path.exists():
        with np.load(path, allow_pickle=False) as stored:
            return CovarianceBank(list(returns.index), list(returns.columns), stored['covariance'],
                stored['ready'], pd.read_json(StringIO(stored['diagnostics'].item()), orient='table'), dict(config, cache_key=digest))
    count, assets = returns.shape
    sequence = np.zeros((count, assets, assets))
    ready = np.zeros(count, dtype=bool)
    rows = []
    for t in range(count):
        observations = returns.iloc[max(0, t-window+1):t+1].dropna().to_numpy()
        row = {"date": str(returns.index[t]), "observations": len(observations),
               "fallback": "insufficient_history", "shrinkage": 0., "jitter": 0.,
               "min_eigenvalue": 0., "condition_number": 0.}
        if len(observations) >= min_periods:
            sequence[t], diagnostics = estimate_covariance(observations, method, decay, jitter)
            ready[t] = True
            row.update(diagnostics, fallback="none")
        rows.append(row)
    diagnostics = pd.DataFrame(rows)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, covariance=sequence, ready=ready,
                            diagnostics=np.asarray(diagnostics.to_json(orient='table', index=False)))
        temporary.replace(path)
    return CovarianceBank(list(returns.index), list(returns.columns), sequence, ready,
                          diagnostics, dict(config, cache_key=digest))


def causal_residual_returns(returns, benchmark, window=60, min_periods=None):
    """Fit alpha/beta on [t-window:t), then compute residual for date t.

    Today's stock/benchmark returns are available at Close[t]. Future residuals
    never refit past ones. Missing benchmark is an explicit error at the caller.
    """
    min_periods = window if min_periods is None else int(min_periods)
    if window < 2 or not 2 <= min_periods <= window:
        raise ValueError("Invalid residual regression window.")
    market = benchmark.reindex(returns.index)
    output = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    for t in range(min_periods, len(returns)):
        b = market.iloc[max(0, t-window):t]
        for column in returns:
            y = returns[column].iloc[max(0, t-window):t]
            valid = np.isfinite(y) & np.isfinite(b)
            if valid.sum() < min_periods or not np.isfinite(market.iloc[t]) or not np.isfinite(returns[column].iloc[t]):
                continue
            fit_x = np.column_stack([np.ones(valid.sum()), b[valid]])
            alpha, beta = np.linalg.lstsq(fit_x, y[valid], rcond=None)[0]
            output.iloc[t, output.columns.get_loc(column)] = returns[column].iloc[t] - alpha - beta * market.iloc[t]
    return output

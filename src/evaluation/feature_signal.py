"""Shared rank-IC and train-only ridge helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.evaluation.metrics import block_bootstrap_excess_returns


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)



def rank_ic_table(
    features: np.ndarray,
    feature_names: list[str],
    open_prices: np.ndarray,
    close_prices: np.ndarray,
    dates: list[str],
    start: int,
    end: int,
    horizons: tuple[int, ...] = (1, 5, 20),
    block_size: int = 20,
    bootstrap_samples: int = 2000,
) -> pd.DataFrame:
    """Cross-sectional Spearman IC; label endpoints stay inside their time split."""
    rows: list[dict[str, float | int | str]] = []
    label_specs = [("next_session_intraday", 1)] + [
        ("forward_close_to_close", horizon) for horizon in horizons
    ] + [("forward_open_to_close", horizon) for horizon in horizons if horizon > 1]
    for label_name, horizon in label_specs:
        daily_indices = range(start, max(start, end - horizon))
        for feature_index, feature_name in enumerate(feature_names):
            ics: list[float] = []
            for t in daily_indices:
                if label_name == "next_session_intraday":
                    asset_returns = close_prices[t + 1] / open_prices[t + 1] - 1.0
                elif label_name == "forward_open_to_close":
                    asset_returns = close_prices[t + horizon] / open_prices[t + 1] - 1.0
                else:
                    asset_returns = close_prices[t + horizon] / close_prices[t] - 1.0
                x = features[t, :, feature_index]
                if not np.isfinite(x).all() or not np.isfinite(asset_returns).all():
                    continue
                x_rank = _rank(x)
                y_rank = _rank(asset_returns)
                if np.std(x_rank) <= 1e-12 or np.std(y_rank) <= 1e-12:
                    continue
                ic = float(np.corrcoef(x_rank, y_rank)[0, 1])
                if np.isfinite(ic):
                    ics.append(ic)
            values = np.asarray(ics, dtype=np.float64)
            std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            ci = block_bootstrap_excess_returns(
                values, np.zeros_like(values), block_size=max(block_size, horizon),
                num_samples=bootstrap_samples, seed=1729 + feature_index + horizon,
            ) if len(values) >= max(block_size, horizon) else None
            rows.append(
                {
                    "horizon_days": horizon,
                    "target": label_name,
                    "feature": feature_name,
                    "n_dates": len(values),
                    "mean_rank_ic": float(values.mean()) if len(values) else float("nan"),
                    "std_rank_ic": std,
                    "block_ci_lower_95": ci["ci_lower_95"] / 252.0 if ci else float("nan"),
                    "block_ci_upper_95": ci["ci_upper_95"] / 252.0 if ci else float("nan"),
                    "block_size_sessions": max(block_size, horizon),
                    "label_purge_sessions": horizon,
                    "naive_t_stat": (
                        float(values.mean() / (std / np.sqrt(len(values))))
                        if len(values) > 1 and std > 0
                        else float("nan")
                    ),
                    "positive_ic_fraction": (
                        float(np.mean(values > 0)) if len(values) else float("nan")
                    ),
                    "start_date": dates[start],
                    "last_label_date": dates[end - 1],
                }
            )
    return pd.DataFrame(rows)



def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, float]:
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    x_centered = x - x_mean
    y_centered = y - y_mean
    covariance = x_centered.T @ x_centered / len(x_centered)
    cross_covariance = x_centered.T @ y_centered / len(x_centered)
    coefficients = np.linalg.solve(
        covariance + alpha * np.eye(x.shape[1], dtype=np.float64),
        cross_covariance,
    )
    return coefficients, x_mean, y_mean



def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    weights = np.exp(np.clip(shifted, -50.0, 50.0))
    return weights / weights.sum()


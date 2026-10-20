"""Fixed-direction cross-sectional signals calibrated using past labels only."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rank_signal(values: np.ndarray, direction: int) -> np.ndarray:
    """Tie-aware daily ranks in [-1,1], centered; N=1/constant rows yield 0."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or direction not in {-1, 1}:
        raise ValueError("Rank signals need finite [T,N] features and a fixed direction +/-1.")
    if values.shape[1] == 1:
        return np.zeros_like(values)
    ranks = pd.DataFrame(values).rank(axis=1, method="average").to_numpy()
    return direction * 2 * (ranks - (values.shape[1] + 1) / 2) / (values.shape[1] - 1)


def calibrate_rank_signal(scores: np.ndarray, excess_returns: np.ndarray, ridge_alpha: float) -> float:
    """Nonnegative no-intercept slope: never flip a failed economic hypothesis."""
    scores, excess_returns = np.asarray(scores), np.asarray(excess_returns)
    if scores.shape != excess_returns.shape or not scores.size or not np.isfinite(excess_returns).all():
        raise ValueError("Calibration scores and historical labels must be aligned and finite.")
    numerator = float(np.mean(scores * excess_returns))
    denominator = float(np.mean(scores ** 2) + ridge_alpha)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Calibration regularization must give a positive finite denominator.")
    return max(0., numerator / denominator)

"""Causal Kaufman efficiency ratios for execution research, not alpha forecasts."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TrendEfficiencyBank:
    dates: list
    tickers: list
    efficiency_ratio: np.ndarray
    ready: np.ndarray
    metadata: dict


def build_trend_efficiency_bank(close_prices, dates, tickers, *, window=10):
    """ER[t] uses exactly window increments ending at Close[t].

    Supply the same adjusted Close basis as the return engine. A flat, valid
    window has ER=0; missing/nonpositive prices make that asset unready. No
    filling or future observations enter the calculation.
    """
    index = pd.DatetimeIndex(dates)
    prices = np.asarray(close_prices, dtype=np.float64)
    if (type(window) is not int or window < 1
            or prices.shape != (len(index), len(tickers)) or not len(tickers)
            or len(set(tickers)) != len(tickers) or index.hasnans
            or not index.is_monotonic_increasing or not index.is_unique):
        raise ValueError("ER requires [dates, tickers] prices, ordered unique dates/tickers and a positive window.")
    frame = pd.DataFrame(prices, index=index, columns=tickers)
    valid = np.isfinite(frame) & (frame > 0)
    ready = valid.rolling(window+1, min_periods=window+1).sum().eq(window+1)
    path = frame.diff().abs().rolling(window, min_periods=window).sum()
    distance = (frame-frame.shift(window)).abs()
    ratio = np.divide(distance.to_numpy(), path.to_numpy(),
                      out=np.zeros_like(prices), where=ready.to_numpy() & (path.to_numpy() > 0))
    ratio = np.where(ready, np.clip(ratio, 0., 1.), 0.)
    return TrendEfficiencyBank(list(index), list(tickers), ratio, ready.to_numpy(),
        dict(window=window, price_basis="same_adjusted_Close_as_portfolio_returns",
             alignment="through_Close_t_for_next_Open_execution", flat_window_er=0.,
             meaning="directionless_path_efficiency_not_prediction_confidence"))

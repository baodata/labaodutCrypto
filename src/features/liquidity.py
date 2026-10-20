"""QNT-011: trailing raw-dollar ADV and daily total-return volatility."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class LiquidityBank:
    dates: list
    tickers: list
    dollar_adv: np.ndarray
    volatility: np.ndarray
    ready: np.ndarray
    metadata: dict


def build_liquidity_bank(processed_data, dates, tickers, *, window=20, min_periods=None):
    min_periods = window if min_periods is None else int(min_periods)
    if window < 2 or not 2 <= min_periods <= window:
        raise ValueError("Invalid liquidity window.")
    index = pd.DatetimeIndex(dates)
    if not index.is_monotonic_increasing or not index.is_unique or len(set(tickers)) != len(tickers):
        raise ValueError("Liquidity requires unique ordered dates/tickers.")
    adv, sigma, readiness = [], [], []
    for ticker in tickers:
        frame = processed_data[ticker].reindex(index)
        lookup = {str(c).lower().replace(' ', '_'): c for c in frame}
        raw_close = frame[lookup['close']].astype(float)
        volume = frame[lookup['volume']].astype(float)
        if (raw_close.dropna() <= 0).any() or (volume.dropna() < 0).any() or np.isinf(raw_close).any() or np.isinf(volume).any():
            raise ValueError(f"Invalid raw liquidity data: {ticker}")
        # Yahoo Close is split-adjusted (not dividend-adjusted), Volume uses
        # matching split-adjusted shares. Dividend factors must not enter ADV.
        dollars = raw_close * volume
        total_close = frame[lookup.get('adj_close', lookup['close'])].astype(float)
        returns = total_close.pct_change(fill_method=None)
        a = dollars.rolling(window, min_periods=min_periods).mean()
        s = returns.rolling(window, min_periods=min_periods).std(ddof=1)
        good = np.isfinite(a) & np.isfinite(s) & (a > 0) & (volume > 0)
        adv.append(a.fillna(0).to_numpy())
        sigma.append(s.fillna(0).to_numpy())
        readiness.append(good.to_numpy())
    return LiquidityBank(list(index), list(tickers), np.column_stack(adv), np.column_stack(sigma),
        np.column_stack(readiness), {'window': window, 'min_periods': min_periods,
        'adv_units': 'USD/day', 'volatility_units': 'daily_return_std',
        'adv_basis': 'Yahoo split-adjusted Close times matching split-adjusted Volume; no dividend factor',
        'alignment': 'through_Close_t_for_next_Open_execution'})

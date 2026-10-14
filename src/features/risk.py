"""Range-based and return-based risk feature calculations for one asset."""

import numpy as np
import pandas as pd


RISK_FEATURE_NAMES = (
    "vol_60d",
    "vol_ratio",
    "parkinson_vol_20d",
    "downside_vol_20d",
    "atr_14_close",
)


def calculate_risk_features(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
) -> pd.DataFrame:
    """Calculate realized, Parkinson, downside, and ATR/close risk measures."""
    close = pd.to_numeric(close, errors="coerce")
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    for name, series in (("Close", close), ("High", high), ("Low", low)):
        values = series.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"{name} phải hữu hạn và dương để tính risk features.")
    if (high < low).any():
        raise ValueError("High phải lớn hơn hoặc bằng Low.")

    daily_returns = close.pct_change(fill_method=None)
    vol20 = daily_returns.rolling(window=20, min_periods=2).std(ddof=1).fillna(0.0)
    vol60 = daily_returns.rolling(window=60, min_periods=20).std(ddof=1).fillna(0.0)
    vol60_values = vol60.to_numpy(dtype=np.float64)
    vol_ratio = np.divide(
        vol20.to_numpy(dtype=np.float64),
        vol60_values,
        out=np.zeros(len(close), dtype=np.float64),
        where=vol60_values > 1e-12,
    )

    parkinson_variance = np.log(high / low).pow(2) / (4.0 * np.log(2.0))
    parkinson_vol = np.sqrt(
        parkinson_variance.rolling(window=20, min_periods=1).mean()
    )
    # Downside semideviation around zero, including non-negative days as zeros.
    downside_vol = np.sqrt(
        daily_returns.clip(upper=0.0).pow(2).rolling(window=20, min_periods=2).mean()
    )

    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_14_close = true_range.rolling(window=14, min_periods=1).mean().div(close)

    features = pd.DataFrame(
        {
            "vol_60d": vol60,
            "vol_ratio": vol_ratio,
            "parkinson_vol_20d": parkinson_vol,
            "downside_vol_20d": downside_vol,
            "atr_14_close": atr_14_close,
        },
        index=close.index,
    )
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

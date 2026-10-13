"""Price momentum feature calculations for one asset."""

import pandas as pd


MOMENTUM_FEATURE_NAMES = (
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "mom_12_1",
    "dist_52w_high",
)


def calculate_momentum_features(close: pd.Series) -> pd.DataFrame:
    """Calculate trailing returns and price position using data through date t."""
    close = pd.to_numeric(close, errors="coerce")
    features = pd.DataFrame(index=close.index)
    for window in (5, 20, 60):
        features[f"ret_{window}d"] = close.div(close.shift(window)).sub(1.0)

    # 12-1 momentum excludes the latest 21 sessions from the 252-session return.
    features["mom_12_1"] = close.shift(21).div(close.shift(252)).sub(1.0)
    rolling_high = close.rolling(window=252, min_periods=1).max()
    features["dist_52w_high"] = close.div(rolling_high).sub(1.0)
    return features.replace([float("inf"), -float("inf")], float("nan")).fillna(0.0)

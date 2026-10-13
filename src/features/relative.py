"""Cross-sectional and benchmark-relative feature calculations."""

import numpy as np
import pandas as pd


RELATIVE_FEATURE_NAMES = (
    "xs_rank_ret_20d",
    "xs_rank_ret_60d",
    "excess_ret_20d",
    "beta_spy_60d",
)


def calculate_relative_features(
    processed_data: dict[str, pd.DataFrame],
    benchmark_ticker: str,
) -> dict[str, pd.DataFrame]:
    """Calculate cross-sectional ranks, excess return, and rolling SPY beta.

    If the benchmark is absent (for isolated synthetic use), excess return falls
    back to the cross-sectional mean and beta remains neutral at zero.
    """
    benchmark = benchmark_ticker.upper()
    benchmark_key = next(
        (ticker for ticker in processed_data if str(ticker).upper() == benchmark),
        None,
    )
    tradable_tickers = [
        ticker for ticker in processed_data if str(ticker).upper() != benchmark
    ]
    if not tradable_tickers:
        raise ValueError("Không có tài sản giao dịch để tính feature cắt ngang.")

    returns20 = pd.DataFrame(
        {ticker: processed_data[ticker]["ret_20d"] for ticker in tradable_tickers}
    )
    returns60 = pd.DataFrame(
        {ticker: processed_data[ticker]["ret_60d"] for ticker in tradable_tickers}
    )
    rank20 = returns20.rank(axis=1, method="average", pct=True)
    rank60 = returns60.rank(axis=1, method="average", pct=True)

    benchmark_returns = None
    if benchmark_key is not None:
        benchmark_frame = processed_data[benchmark_key]
        benchmark_returns = benchmark_frame["return"].astype(np.float64)
        benchmark_ret20 = benchmark_frame["ret_20d"].astype(np.float64)
        benchmark_variance = benchmark_returns.rolling(
            window=60, min_periods=20
        ).var(ddof=1)
    else:
        benchmark_ret20 = returns20.mean(axis=1)
        benchmark_variance = None

    results: dict[str, pd.DataFrame] = {}
    for ticker in tradable_tickers:
        asset = processed_data[ticker]
        features = pd.DataFrame(
            {
                "xs_rank_ret_20d": rank20[ticker],
                "xs_rank_ret_60d": rank60[ticker],
                "excess_ret_20d": asset["ret_20d"] - benchmark_ret20,
            },
            index=asset.index,
        )
        if benchmark_returns is not None and benchmark_variance is not None:
            covariance = asset["return"].astype(np.float64).rolling(
                window=60, min_periods=20
            ).cov(benchmark_returns)
            beta = covariance.div(benchmark_variance.where(benchmark_variance > 1e-12))
            features["beta_spy_60d"] = beta
        else:
            features["beta_spy_60d"] = 0.0
        results[ticker] = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return results

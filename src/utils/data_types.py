"""Data structures for unified market tensor representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class MarketDataTensor:
    """Standardized 3D Market Data Container.

    Represents market data aligned across multiple assets over time.
    Dimensions:
        - T: Number of time steps (trading days)
        - N: Number of assets (e.g. 24 stocks)
        - F: Number of features per asset (e.g. open, high, low, close, volume)

    Attributes:
        dates: 1D array of trading dates (datetime64), shape (T,).
        tickers: List of asset ticker symbols, length N.
        feature_names: List of feature names, length F.
        values: 3D float32 numpy array, shape (T, N, F).
        benchmark_ticker: Symbol for the benchmark asset (e.g. 'SPY').
        benchmark_values: 2D float32 numpy array, shape (T, F).
    """

    dates: np.ndarray
    tickers: List[str]
    feature_names: List[str]
    values: np.ndarray
    benchmark_ticker: Optional[str] = None
    benchmark_values: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Validate shapes and types upon initialization."""
        if self.values.ndim != 3:
            raise ValueError(f"values must be 3D [T, N, F], got shape {self.values.shape}")

        t, n, f = self.values.shape
        if len(self.dates) != t:
            raise ValueError(f"Length of dates ({len(self.dates)}) does not match T ({t})")
        if len(self.tickers) != n:
            raise ValueError(f"Length of tickers ({len(self.tickers)}) does not match N ({n})")
        if len(self.feature_names) != f:
            raise ValueError(f"Length of feature_names ({len(self.feature_names)}) does not match F ({f})")

        if self.benchmark_values is not None:
            if self.benchmark_values.ndim != 2:
                raise ValueError(
                    f"benchmark_values must be 2D [T, F], got shape {self.benchmark_values.shape}"
                )
            if self.benchmark_values.shape != (t, f):
                raise ValueError(
                    f"benchmark_values shape {self.benchmark_values.shape} must match (T={t}, F={f})"
                )

    @property
    def T(self) -> int:
        """Number of time steps."""
        return int(self.values.shape[0])

    @property
    def N(self) -> int:
        """Number of assets."""
        return int(self.values.shape[1])

    @property
    def F(self) -> int:
        """Number of features."""
        return int(self.values.shape[2])

    def get_feature_index(self, feature_name: str) -> int:
        """Return the index of a feature name."""
        if feature_name not in self.feature_names:
            raise KeyError(f"Feature '{feature_name}' not found in {self.feature_names}")
        return self.feature_names.index(feature_name)

    def get_ticker_index(self, ticker: str) -> int:
        """Return the index of a ticker symbol."""
        if ticker not in self.tickers:
            raise KeyError(f"Ticker '{ticker}' not found in {self.tickers}")
        return self.tickers.index(ticker)

    def get_feature(self, feature_name: str) -> np.ndarray:
        """Extract a single feature across all timesteps and tickers.

        Returns:
            np.ndarray of shape (T, N)
        """
        idx = self.get_feature_index(feature_name)
        return self.values[:, :, idx]

    def get_ticker_data(self, ticker: str) -> np.ndarray:
        """Extract all features for a single ticker over time.

        Returns:
            np.ndarray of shape (T, F)
        """
        idx = self.get_ticker_index(ticker)
        return self.values[:, idx, :]

    def slice_by_date(self, start_date: str, end_date: str) -> MarketDataTensor:
        """Filter the dataset chronologically between start_date and end_date (inclusive).

        Args:
            start_date: Start date string (e.g. '2015-01-01').
            end_date: End date string (e.g. '2021-12-31').

        Returns:
            A new MarketDataTensor containing only records in the date range.
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        dt_series = pd.to_datetime(self.dates)

        mask = (dt_series >= start_dt) & (dt_series <= end_dt)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            raise ValueError(
                f"No records found for date range [{start_date}, {end_date}]. "
                f"Available range: [{self.dates[0]}, {self.dates[-1]}]"
            )

        sliced_bench = (
            self.benchmark_values[indices] if self.benchmark_values is not None else None
        )

        return MarketDataTensor(
            dates=self.dates[indices],
            tickers=list(self.tickers),
            feature_names=list(self.feature_names),
            values=self.values[indices],
            benchmark_ticker=self.benchmark_ticker,
            benchmark_values=sliced_bench,
        )

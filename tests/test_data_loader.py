"""Unit tests for MarketDataTensor and the data loading pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import get_all_tickers, load_config, load_market_data
from src.utils.data_types import MarketDataTensor


def test_load_config():
    """Verify that assets.yaml loads correctly with all required sections."""
    cfg = load_config("configs/assets.yaml")
    assert "sectors" in cfg
    assert "benchmark" in cfg
    assert cfg["benchmark"] == "SPY"
    assert "features" in cfg
    assert "splits" in cfg

    tickers = get_all_tickers(cfg)
    assert len(tickers) == 24
    assert "AAPL" in tickers
    assert "SPY" not in tickers  # SPY is benchmark, not in asset universe


def test_market_data_tensor_validation():
    """Test MarketDataTensor post-init validation checks."""
    dates = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    tickers = ["AAPL", "MSFT"]
    features = ["close", "volume"]
    values = np.zeros((2, 2, 2), dtype=np.float32)

    # Valid initialization
    tensor = MarketDataTensor(
        dates=dates,
        tickers=tickers,
        feature_names=features,
        values=values,
    )
    assert tensor.T == 2
    assert tensor.N == 2
    assert tensor.F == 2

    # Invalid dimension
    with pytest.raises(ValueError, match="values must be 3D"):
        MarketDataTensor(
            dates=dates,
            tickers=tickers,
            feature_names=features,
            values=np.zeros((2, 2)),
        )

    # Mismatched dates
    with pytest.raises(ValueError, match="Length of dates"):
        MarketDataTensor(
            dates=dates[:1],
            tickers=tickers,
            feature_names=features,
            values=values,
        )


def test_load_market_data_full_and_splits():
    """Test full dataset loading and chronological splits."""
    full_data, splits = load_market_data("configs/assets.yaml")

    # 1. Check Full Dataset Dimensions
    assert full_data.T == 2765
    assert full_data.N == 24
    assert full_data.F == 6
    assert full_data.benchmark_ticker == "SPY"
    assert full_data.benchmark_values is not None
    assert full_data.benchmark_values.shape == (2765, 6)

    # 2. Check Data Integrity (No NaNs, No Infs)
    assert not np.isnan(full_data.values).any()
    assert not np.isinf(full_data.values).any()
    assert not np.isnan(full_data.benchmark_values).any()
    assert not np.isinf(full_data.benchmark_values).any()

    # 3. Check Monotonic Date Ordering
    dates_s = pd.Series(full_data.dates)
    assert dates_s.is_monotonic_increasing

    # 4. Check Splits
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits

    train_data = splits["train"]
    val_data = splits["val"]
    test_data = splits["test"]

    # Ensure chronological order with zero overlap
    assert train_data.dates[-1] < val_data.dates[0]
    assert val_data.dates[-1] < test_data.dates[0]

    # Verify expected split lengths (2015-2021: 1763, 2022-2023: 501, 2024-2025: 501)
    assert train_data.T == 1763
    assert val_data.T == 501
    assert test_data.T == 501
    assert train_data.T + val_data.T + test_data.T == 2765


def test_feature_and_ticker_helpers():
    """Test feature and ticker extraction helper methods."""
    full_data, _ = load_market_data("configs/assets.yaml")

    # Feature extraction across all stocks
    close_prices = full_data.get_feature("close")
    assert close_prices.shape == (2765, 24)
    assert (close_prices > 0).all()

    # Single ticker extraction
    aapl_data = full_data.get_ticker_data("AAPL")
    assert aapl_data.shape == (2765, 6)

    # Invalid feature / ticker raises KeyError
    with pytest.raises(KeyError):
        full_data.get_feature("non_existent_feature")

    with pytest.raises(KeyError):
        full_data.get_ticker_data("UNKNOWN_TICKER")

"""Unit tests for DataValidator."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.data.validator import DataValidator, ValidationResult


@pytest.fixture
def valid_ohlcv_df() -> pd.DataFrame:
    """Generate a clean, valid 10-day OHLCV DataFrame."""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    return pd.DataFrame(
        {
            "Open": [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 107.0, 108.0, 110.0],
            "High": [103.0, 104.0, 103.0, 106.0, 107.0, 106.0, 108.0, 109.0, 111.0, 112.0],
            "Low": [99.0, 100.0, 99.5, 101.0, 103.0, 102.5, 104.0, 105.0, 106.5, 108.0],
            "Close": [102.0, 101.0, 102.5, 105.0, 104.0, 105.5, 107.0, 108.0, 110.0, 111.5],
            "Volume": [1000, 1500, 1200, 1800, 2000, 1400, 1600, 1700, 1900, 2200],
        },
        index=dates,
    )


def test_valid_dataframe_passes(valid_ohlcv_df: pd.DataFrame):
    """Test that a well-formed OHLCV DataFrame passes all validation rules."""
    validator = DataValidator()
    result = validator.validate_dataframe(valid_ohlcv_df, ticker="TEST")

    assert result.is_valid is True
    assert len(result.errors) == 0
    assert result.stats["num_rows"] == 10
    assert result.stats["start_date"] == "2024-01-01"
    # Should not raise exception
    result.raise_if_invalid()


def test_empty_dataframe_fails():
    """Test that an empty DataFrame fails validation."""
    validator = DataValidator()
    result = validator.validate_dataframe(pd.DataFrame(), ticker="EMPTY")

    assert result.is_valid is False
    assert any("empty" in e.lower() for e in result.errors)
    with pytest.raises(ValueError, match="Validation failed"):
        result.raise_if_invalid()


def test_missing_required_column(valid_ohlcv_df: pd.DataFrame):
    """Test that missing any of OHLCV columns is caught."""
    df_missing = valid_ohlcv_df.drop(columns=["Volume"])
    validator = DataValidator()
    result = validator.validate_dataframe(df_missing, ticker="TEST")

    assert result.is_valid is False
    assert any("Missing required columns" in e for e in result.errors)


def test_invalid_index_types(valid_ohlcv_df: pd.DataFrame):
    """Test that non-DatetimeIndex is rejected."""
    df_bad_index = valid_ohlcv_df.reset_index(drop=True)
    validator = DataValidator()
    result = validator.validate_dataframe(df_bad_index, ticker="TEST")

    assert result.is_valid is False
    assert any("DatetimeIndex" in e for e in result.errors)


def test_unsorted_dates(valid_ohlcv_df: pd.DataFrame):
    """Test that non-chronological dates are rejected."""
    # Reverse index order
    df_unsorted = valid_ohlcv_df.iloc[::-1]
    validator = DataValidator()
    result = validator.validate_dataframe(df_unsorted, ticker="TEST")

    assert result.is_valid is False
    assert any("sorted chronologically" in e for e in result.errors)


def test_duplicate_dates(valid_ohlcv_df: pd.DataFrame):
    """Test that duplicate timestamps are flagged as errors."""
    df_dups = valid_ohlcv_df.copy()
    # Replace last date with first date
    df_dups.index = valid_ohlcv_df.index[:9].append(pd.DatetimeIndex([valid_ohlcv_df.index[0]]))
    validator = DataValidator()
    result = validator.validate_dataframe(df_dups, ticker="TEST")

    assert result.is_valid is False
    assert any("duplicate timestamp" in e for e in result.errors)


def test_negative_or_zero_prices(valid_ohlcv_df: pd.DataFrame):
    """Test that prices <= 0 are rejected."""
    validator = DataValidator()

    # Zero price
    df_zero = valid_ohlcv_df.copy()
    df_zero.loc[df_zero.index[0], "Close"] = 0.0
    res_zero = validator.validate_dataframe(df_zero, ticker="TEST")
    assert res_zero.is_valid is False
    assert any("non-positive" in e for e in res_zero.errors)

    # Negative price
    df_neg = valid_ohlcv_df.copy()
    df_neg.loc[df_neg.index[0], "Low"] = -5.0
    res_neg = validator.validate_dataframe(df_neg, ticker="TEST")
    assert res_neg.is_valid is False
    assert any("non-positive" in e for e in res_neg.errors)


def test_negative_volume(valid_ohlcv_df: pd.DataFrame):
    """Test that negative volume is rejected."""
    df_neg_vol = valid_ohlcv_df.copy()
    df_neg_vol.loc[df_neg_vol.index[0], "Volume"] = -100
    validator = DataValidator()
    result = validator.validate_dataframe(df_neg_vol, ticker="TEST")

    assert result.is_valid is False
    assert any("negative value" in e for e in result.errors)


def test_ohlc_inconsistencies(valid_ohlcv_df: pd.DataFrame):
    """Test that physical candlestick violations are caught."""
    validator = DataValidator()

    # High < Low
    df_hl = valid_ohlcv_df.copy()
    df_hl.loc[df_hl.index[0], "High"] = 90.0  # Low is 99.0
    assert validator.validate_dataframe(df_hl).is_valid is False

    # High < Open
    df_ho = valid_ohlcv_df.copy()
    df_ho.loc[df_ho.index[0], "High"] = 99.5  # Open is 100.0
    assert validator.validate_dataframe(df_ho).is_valid is False

    # Low > Close
    df_lc = valid_ohlcv_df.copy()
    df_lc.loc[df_lc.index[0], "Low"] = 103.0  # Close is 102.0
    assert validator.validate_dataframe(df_lc).is_valid is False


def test_nan_and_inf_detection(valid_ohlcv_df: pd.DataFrame):
    """Test that NaN and Inf are detected."""
    validator = DataValidator()

    # NaN in Close
    df_nan = valid_ohlcv_df.copy()
    df_nan.loc[df_nan.index[1], "Close"] = np.nan
    res_nan = validator.validate_dataframe(df_nan)
    assert res_nan.is_valid is False
    assert any("NaN/null" in e for e in res_nan.errors)

    # Inf in Open
    df_inf = valid_ohlcv_df.copy()
    df_inf.loc[df_inf.index[1], "Open"] = np.inf
    res_inf = validator.validate_dataframe(df_inf)
    assert res_inf.is_valid is False
    assert any("infinite" in e for e in res_inf.errors)


def test_allow_zero_volume_toggle(valid_ohlcv_df: pd.DataFrame):
    """Test that Volume == 0 can be warning or error depending on flag."""
    df_zero_vol = valid_ohlcv_df.copy()
    df_zero_vol.loc[df_zero_vol.index[0], "Volume"] = 0

    # Warning mode (default)
    validator_warn = DataValidator(allow_zero_volume=True)
    res_warn = validator_warn.validate_dataframe(df_zero_vol)
    assert res_warn.is_valid is True
    assert len(res_warn.warnings) > 0

    # Strict mode
    validator_strict = DataValidator(allow_zero_volume=False)
    res_strict = validator_strict.validate_dataframe(df_zero_vol)
    assert res_strict.is_valid is False
    assert any("Volume == 0" in e for e in res_strict.errors)


def test_validate_real_parquet_files():
    """Test validator against real downloaded parquet files in data/raw."""
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        pytest.skip("data/raw directory not found, skipping real file validation.")

    validator = DataValidator()
    results = validator.validate_raw_dir(raw_dir)

    assert len(results) > 0, "Expected parquet files in data/raw"
    for ticker, res in results.items():
        assert res.is_valid is True, f"Asset {ticker} failed validation: {res.errors}"
        assert res.stats["num_rows"] > 2500


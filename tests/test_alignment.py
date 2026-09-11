"""Unit tests for CalendarAligner and Trading Calendar Alignment."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.alignment import AlignmentResult, CalendarAligner


@pytest.fixture
def sample_assets_data() -> dict[str, pd.DataFrame]:
    """Generate sample OHLCV data for 3 assets with slight date misalignments."""
    # Asset A: Jan 1 to Jan 5 (5 business days: Mon Jan 1 -> Fri Jan 5)
    dates_a = pd.bdate_range("2024-01-01", periods=5)
    df_a = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [102.0, 103.0, 104.0, 105.0, 106.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        },
        index=dates_a,
    )

    # Asset B: Jan 2 to Jan 8 (5 business days: Tue Jan 2 -> Mon Jan 8)
    dates_b = pd.bdate_range("2024-01-02", periods=5)
    df_b = pd.DataFrame(
        {
            "Open": [50.0, 51.0, 52.0, 53.0, 54.0],
            "High": [52.0, 53.0, 54.0, 55.0, 56.0],
            "Low": [49.0, 50.0, 51.0, 52.0, 53.0],
            "Close": [51.0, 52.0, 53.0, 54.0, 55.0],
            "Volume": [2000, 2100, 2200, 2300, 2400],
        },
        index=dates_b,
    )

    # Asset C: Jan 1 to Jan 5, but missing Jan 3 (Trading halt)
    dates_c = dates_a.drop([pd.Timestamp("2024-01-03")])
    df_c = pd.DataFrame(
        {
            "Open": [200.0, 201.0, 203.0, 204.0],
            "High": [202.0, 203.0, 205.0, 206.0],
            "Low": [199.0, 200.0, 202.0, 203.0],
            "Close": [201.0, 202.0, 204.0, 205.0],
            "Volume": [500, 510, 530, 540],
        },
        index=dates_c,
    )

    return {"ASSET_A": df_a, "ASSET_B": df_b, "ASSET_C": df_c}


def test_align_intersection_method(sample_assets_data: dict[str, pd.DataFrame]):
    """Test alignment using intersection: keeps only dates common to ALL assets."""
    aligner = CalendarAligner(method="intersection")
    result = aligner.align(sample_assets_data)

    assert isinstance(result, AlignmentResult)
    assert result.num_assets == 3
    # Common dates: Jan 2, Jan 4, Jan 5 (Jan 1 missing from B, Jan 3 missing from C, Jan 6 missing from A/C)
    expected_dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05")]
    assert len(result.common_dates) == 3
    assert list(result.common_dates) == expected_dates

    # Check that all DataFrames have exactly the same length and index
    for ticker, df in result.aligned_data.items():
        assert len(df) == 3
        assert list(df.index) == expected_dates

    # Check dropped dates tracking
    assert pd.Timestamp("2024-01-01") in result.dropped_dates["ASSET_A"]
    assert pd.Timestamp("2024-01-08") in result.dropped_dates["ASSET_B"]


def test_align_union_ffill_method(sample_assets_data: dict[str, pd.DataFrame]):
    """Test alignment using union_ffill: forward-fills prices and fills volume with 0."""
    aligner = CalendarAligner(method="union_ffill")
    result = aligner.align(sample_assets_data)

    assert result.num_assets == 3
    # Asset C was missing Jan 3: check that Jan 3 is forward-filled from Jan 2
    df_c_aligned = result.aligned_data["ASSET_C"]
    assert pd.Timestamp("2024-01-03") in df_c_aligned.index
    # Price on Jan 3 should equal price on Jan 2 (Close=202.0)
    assert df_c_aligned.loc["2024-01-03", "Close"] == 202.0
    # Volume on non-trading day should be 0
    assert df_c_aligned.loc["2024-01-03", "Volume"] == 0.0


def test_to_price_panel(sample_assets_data: dict[str, pd.DataFrame]):
    """Test extracting [T, N] price panel from aligned data."""
    aligner = CalendarAligner(method="intersection")
    result = aligner.align(sample_assets_data)

    panel = CalendarAligner.to_price_panel(result.aligned_data, price_col="Close")
    assert panel.shape == (3, 3)  # [T=3, N=3]
    assert list(panel.columns) == ["ASSET_A", "ASSET_B", "ASSET_C"]
    assert panel.index.name == "Date"
    assert not panel.isna().any().any()


def test_invalid_method_raises():
    """Test that unsupported method raises ValueError."""
    with pytest.raises(ValueError, match="Invalid method"):
        CalendarAligner(method="invalid_method")  # type: ignore


def test_empty_dict_raises():
    """Test that aligning an empty dictionary raises ValueError."""
    aligner = CalendarAligner()
    with pytest.raises(ValueError, match="empty"):
        aligner.align({})


def test_align_real_parquet_dataset():
    """Test alignment on all real downloaded assets in data/raw/."""
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        pytest.skip("data/raw directory not found, skipping real dataset alignment test.")

    aligner = CalendarAligner(method="intersection")
    result = aligner.align_from_directory(raw_dir)

    # 24 stocks + SPY benchmark = 25 assets
    assert result.num_assets >= 25
    assert result.num_timesteps > 2500

    # Ensure 100% synchronization across all assets
    first_ticker = next(iter(result.aligned_data.keys()))
    reference_index = result.aligned_data[first_ticker].index

    for ticker, df in result.aligned_data.items():
        assert len(df) == result.num_timesteps, f"{ticker} has {len(df)} rows, expected {result.num_timesteps}"
        assert df.index.equals(reference_index), f"{ticker} index does not match reference"

    # Extract price panel and verify no NaNs
    close_panel = CalendarAligner.to_price_panel(result.aligned_data, price_col="close")
    assert close_panel.shape == (result.num_timesteps, result.num_assets)
    assert close_panel.isna().sum().sum() == 0, "Price panel must not contain NaNs"

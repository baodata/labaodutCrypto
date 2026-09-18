"""
tests/test_macd.py
Bộ kiểm thử tính toán chỉ báo MACD (Moving Average Convergence Divergence Tests).

Ticket: FEAT-004 (P1 - Thành viên A)
Sprint: 3

Mục đích kiểm thử:
1. Kiểm tra tính đúng đắn của MACD Line (EMA12 - EMA26), Signal Line (EMA9 của MACD) và Histogram.
2. Kiểm tra tính chất khi giá tăng liên tục: EMA nhanh > EMA chậm => MACD line > 0.
3. Kiểm tra tính chất khi giá giảm liên tục: EMA nhanh < EMA chậm => MACD line < 0.
4. Bắt lỗi khi fast_period >= slow_period hoặc period <= 0.
5. Kiểm tra MACDEngine trên dictionary đa tài sản.
6. Kiểm tra tính toán trên dữ liệu thật của 25 mã cổ phiếu.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.alignment import CalendarAligner
from src.features.macd import MACDEngine, compute_macd


def test_macd_bullish_trend():
    """Kiểm tra giá tăng liên tục thì EMA nhanh > EMA chậm => MACD Line > 0."""
    dates = pd.bdate_range("2024-01-01", periods=40)
    prices = pd.Series([100.0 + i * 2.0 for i in range(40)], index=dates)

    macd, signal, hist = compute_macd(prices, fast_period=12, slow_period=26, signal_period=9)

    # Sau chu kỳ khởi động, MACD Line phải dương
    assert (macd.iloc[26:] > 0).all(), "Trong xu hướng tăng mạnh, MACD Line phải luôn dương (> 0)"
    assert not macd.isna().any(), "MACD không được chứa NaN"
    assert not signal.isna().any(), "Signal không được chứa NaN"
    assert not hist.isna().any(), "Histogram không được chứa NaN"


def test_macd_bearish_trend():
    """Kiểm tra giá giảm liên tục thì EMA nhanh < EMA chậm => MACD Line < 0."""
    dates = pd.bdate_range("2024-01-01", periods=40)
    prices = pd.Series([200.0 - i * 2.0 for i in range(40)], index=dates)

    macd, signal, hist = compute_macd(prices, fast_period=12, slow_period=26, signal_period=9)

    # Sau chu kỳ khởi động, MACD Line phải âm
    assert (macd.iloc[26:] < 0).all(), "Trong xu hướng giảm mạnh, MACD Line phải luôn âm (< 0)"


def test_macd_histogram_identity():
    """Kiểm tra đồng nhất thức: Histogram = MACD Line - Signal Line."""
    dates = pd.bdate_range("2024-01-01", periods=50)
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(50)), index=dates)

    macd, signal, hist = compute_macd(prices)
    diff = hist - (macd - signal)
    assert (diff.abs() < 1e-10).all(), "Histogram phải bằng chính xác macd_line - signal_line"


def test_invalid_parameters_raise_error():
    """Kiểm tra ném lỗi khi tham số period không hợp lệ."""
    prices = pd.Series([100.0, 101.0, 102.0])

    with pytest.raises(ValueError, match="nhỏ hơn"):
        compute_macd(prices, fast_period=26, slow_period=12)

    with pytest.raises(ValueError, match="nhỏ hơn"):
        compute_macd(prices, fast_period=12, slow_period=12)

    with pytest.raises(ValueError, match="số nguyên dương"):
        compute_macd(prices, fast_period=0, slow_period=26)


def test_macd_engine_multi_asset_dict():
    """Kiểm tra MACDEngine trên dictionary đa tài sản."""
    dates = pd.bdate_range("2024-01-01", periods=30)
    data_dict = {
        "AAPL": pd.DataFrame({"close": [100.0 + i for i in range(30)]}, index=dates),
        "MSFT": pd.DataFrame({"close": [200.0 - i for i in range(30)]}, index=dates),
    }

    engine = MACDEngine()

    # Chế độ 1: Trả về 3 ma trận [T, N]
    matrices = engine.calculate_for_dict(data_dict, add_column=False)
    assert set(matrices.keys()) == {"macd", "signal", "hist"}
    assert matrices["macd"].shape == (30, 2)
    assert list(matrices["macd"].columns) == ["AAPL", "MSFT"]

    # Chế độ 2: Gắn thêm 3 cột vào từng DataFrame
    updated_dict = engine.calculate_for_dict(data_dict, add_column=True)
    for ticker in ["AAPL", "MSFT"]:
        assert "macd" in updated_dict[ticker].columns
        assert "macd_signal" in updated_dict[ticker].columns
        assert "macd_hist" in updated_dict[ticker].columns


def test_macd_on_real_market_data():
    """Kiểm tra MACD tính toán trên toàn bộ 25 mã cổ phiếu thật."""
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        pytest.skip("data/raw directory not found, skipping real market test.")

    aligner = CalendarAligner(method="intersection")
    result = aligner.align_from_directory(raw_dir)

    engine = MACDEngine()
    matrices = engine.calculate_for_dict(result.aligned_data, add_column=False)

    assert matrices["macd"].shape == (result.num_timesteps, result.num_assets)
    assert not matrices["macd"].isna().any().any()
    assert not matrices["signal"].isna().any().any()
    assert not matrices["hist"].isna().any().any()


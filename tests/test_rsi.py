"""
tests/test_rsi.py
Bộ kiểm thử chỉ báo RSI (Relative Strength Index Tests).

Ticket: FEAT-003 (P1 - Thành viên A)
Sprint: 2

Mục đích kiểm thử:
1. Kiểm tra nghiệm thu: Giá liên tục tăng thì RSI tiệm cận / bằng 100.0.
2. Kiểm tra nghiệm thu: Giá liên tục giảm thì RSI tiệm cận / bằng 0.0.
3. Kiểm tra ca biên: Giá đi ngang không đổi thì RSI = 50.0.
4. Kiểm tra tùy chọn scaled=True: toàn bộ giá trị RSI nằm trong đoạn [0.0, 1.0].
5. Kiểm tra RSIEngine với dictionary đa tài sản xuất ma trận [T, N].
6. Kiểm tra tính toán RSI trên dữ liệu thật của 25 cổ phiếu trong data/raw/.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.alignment import CalendarAligner
from src.features.rsi import RSIEngine, compute_rsi


def test_monotonically_increasing_prices_rsi_approaches_100():
    """Kiểm tra tiêu chuẩn nghiệm thu: Giá liên tục tăng thì RSI tiệm cận 100."""
    dates = pd.bdate_range("2024-01-01", periods=30)
    # Giá tăng liên tục từ 100 lên 129
    prices = pd.Series([100.0 + i for i in range(30)], index=dates)

    rsi = compute_rsi(prices, period=14, fill_na=True)
    # Sau chu kỳ 14 ngày (từ ngày 14 trở đi), RSI phải bằng hoặc tiệm cận 100.0
    assert rsi.iloc[-1] >= 99.9, f"Expected RSI >= 99.9 for pure bull run, got {rsi.iloc[-1]}"


def test_monotonically_decreasing_prices_rsi_approaches_0():
    """Kiểm tra tiêu chuẩn nghiệm thu: Giá liên tục giảm thì RSI tiệm cận 0."""
    dates = pd.bdate_range("2024-01-01", periods=30)
    # Giá giảm liên tục từ 200 xuống 171
    prices = pd.Series([200.0 - i for i in range(30)], index=dates)

    rsi = compute_rsi(prices, period=14, fill_na=True)
    # Sau chu kỳ 14 ngày, RSI phải tiệm cận hoặc bằng 0.0
    assert rsi.iloc[-1] <= 0.1, f"Expected RSI <= 0.1 for pure bear run, got {rsi.iloc[-1]}"


def test_flat_prices_rsi_is_neutral():
    """Kiểm tra ca biên: Giá hoàn toàn đi ngang thì RSI = 50.0."""
    dates = pd.bdate_range("2024-01-01", periods=20)
    prices = pd.Series([100.0] * 20, index=dates)

    rsi = compute_rsi(prices, period=14, fill_na=True)
    for val in rsi:
        assert abs(val - 50.0) < 1e-8


def test_scaled_rsi_bounds():
    """Kiểm tra tùy chọn scaled=True: giá trị RSI luôn nằm trong [0.0, 1.0]."""
    dates = pd.bdate_range("2024-01-01", periods=50)
    # Dữ liệu giá dao động ngẫu nhiên
    np.random.seed(42)
    random_prices = 100.0 + np.cumsum(np.random.randn(50))
    prices = pd.Series(random_prices, index=dates)

    scaled_rsi = compute_rsi(prices, period=14, scaled=True, fill_na=True)
    assert (scaled_rsi >= 0.0).all(), "Scaled RSI cannot be below 0.0"
    assert (scaled_rsi <= 1.0).all(), "Scaled RSI cannot be above 1.0"


def test_rsi_engine_multi_asset_dict():
    """Kiểm tra RSIEngine trên dictionary đa tài sản."""
    dates = pd.bdate_range("2024-01-01", periods=25)
    data_dict = {
        "AAPL": pd.DataFrame({"close": [100.0 + i * 0.5 for i in range(25)]}, index=dates),
        "MSFT": pd.DataFrame({"close": [200.0 - i * 0.5 for i in range(25)]}, index=dates),
    }

    engine = RSIEngine(period=14, scaled=False, price_col="close")

    # Chế độ 1: Trả về ma trận RSI [T, N]
    rsi_matrix = engine.calculate_for_dict(data_dict, add_column=False)
    assert isinstance(rsi_matrix, pd.DataFrame)
    assert rsi_matrix.shape == (25, 2)
    assert list(rsi_matrix.columns) == ["AAPL", "MSFT"]
    assert not rsi_matrix.isna().any().any()

    # Chế độ 2: Gắn thêm cột vào DataFrame
    updated_dict = engine.calculate_for_dict(data_dict, add_column=True, rsi_col_name="rsi_14")
    assert "rsi_14" in updated_dict["AAPL"].columns
    assert "rsi_14" in updated_dict["MSFT"].columns


def test_rsi_on_real_market_data():
    """Kiểm tra tính toán RSI trên toàn bộ dữ liệu 25 mã cổ phiếu thật."""
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        pytest.skip("data/raw directory not found, skipping real market data test.")

    aligner = CalendarAligner(method="intersection")
    result = aligner.align_from_directory(raw_dir)

    engine = RSIEngine(period=14, scaled=True, price_col="close")
    rsi_matrix = engine.calculate_for_dict(result.aligned_data, add_column=False)

    # 25 mã, 2,765 ngày
    assert rsi_matrix.shape == (result.num_timesteps, result.num_assets)
    # Không được có NaN và giá trị trong khoảng [0, 1]
    assert not rsi_matrix.isna().any().any()
    assert (rsi_matrix >= 0.0).all().all()
    assert (rsi_matrix <= 1.0).all().all()

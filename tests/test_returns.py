"""
tests/test_returns.py
Bộ kiểm thử tính toán chuỗi lợi suất hàng ngày (Daily Returns Tests).

Ticket: FEAT-001 (P0 - Thành viên A)
Sprint: 2

Mục đích kiểm thử:
1. Kiểm tra tính chính xác của Simple Return: R_t = (P_t - P_{t-1}) / P_{t-1} với sai số < 10^-8.
2. Kiểm tra tính chính xác của Log Return: r_t = ln(P_t / P_{t-1}) với sai số < 10^-8.
3. Kiểm tra xử lý giá trị t=0 (fill_zero=True điền 0.0).
4. Kiểm tra ném lỗi khi giá <= 0 trong phép tính Log Return.
5. Kiểm tra phương thức calculate_for_dict() trả về DataFrame [T, N] ma trận lợi suất
   chuẩn bị chuyển giao cho Thành viên B tính tương quan đồ thị.
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.features.returns import (
    ReturnsEngine,
    compute_log_returns,
    compute_simple_returns,
)


@pytest.fixture
def sample_price_series() -> pd.Series:
    """Chuỗi giá mẫu 5 ngày với biến động xác định."""
    dates = pd.bdate_range("2024-01-01", periods=5)
    # 100 -> 110 (+10%), 110 -> 99 (-10%), 99 -> 99 (0%), 99 -> 118.8 (+20%)
    return pd.Series([100.0, 110.0, 99.0, 99.0, 118.8], index=dates, name="Close")


def test_simple_returns_precision(sample_price_series: pd.Series):
    """Kiểm tra độ chính xác của Simple Return với sai số < 10^-8."""
    returns = compute_simple_returns(sample_price_series, fill_zero=True)

    expected = [0.0, 0.10, -0.10, 0.0, 0.20]
    for actual_val, expected_val in zip(returns, expected):
        assert abs(actual_val - expected_val) < 1e-8, f"Expected {expected_val}, got {actual_val}"


def test_log_returns_precision():
    """Kiểm tra độ chính xác của Log Return với các bội số của e."""
    dates = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.Series([100.0, 100.0 * math.e, 100.0], index=dates)

    log_rets = compute_log_returns(prices, fill_zero=True)

    assert abs(log_rets.iloc[0] - 0.0) < 1e-8
    assert abs(log_rets.iloc[1] - 1.0) < 1e-8  # ln(e) = 1.0
    assert abs(log_rets.iloc[2] - (-1.0)) < 1e-8  # ln(1/e) = -1.0


def test_log_returns_invalid_price_raises():
    """Kiểm tra ném ngoại lệ khi có giá <= 0 khi tính log returns."""
    dates = pd.bdate_range("2024-01-01", periods=3)
    prices_zero = pd.Series([100.0, 0.0, 105.0], index=dates)
    with pytest.raises(ValueError, match="strictly positive"):
        compute_log_returns(prices_zero)

    prices_neg = pd.Series([100.0, -5.0, 105.0], index=dates)
    with pytest.raises(ValueError, match="strictly positive"):
        compute_log_returns(prices_neg)


def test_returns_engine_single_dataframe():
    """Kiểm tra ReturnsEngine xử lý bảng OHLCV của 1 mã cổ phiếu."""
    dates = pd.bdate_range("2024-01-01", periods=4)
    df = pd.DataFrame(
        {
            "Open": [99.0, 101.0, 102.0, 103.0],
            "Close": [100.0, 105.0, 102.9, 102.9],
        },
        index=dates,
    )
    engine = ReturnsEngine(return_type="simple", price_col="close", fill_zero=True)
    res = engine.calculate(df)

    assert isinstance(res, pd.Series)
    assert abs(res.iloc[1] - 0.05) < 1e-8  # 100 -> 105 (+5%)


def test_returns_engine_multi_asset_dict():
    """Kiểm tra ReturnsEngine xử lý dictionary đa tài sản và xuất ma trận [T, N]."""
    dates = pd.bdate_range("2024-01-01", periods=3)
    data_dict = {
        "AAPL": pd.DataFrame({"close": [100.0, 110.0, 121.0]}, index=dates),
        "MSFT": pd.DataFrame({"close": [200.0, 200.0, 220.0]}, index=dates),
    }

    engine = ReturnsEngine(return_type="simple", price_col="close")

    # Chế độ 1: Trả về ma trận lợi suất [T, N] cho Thành viên B
    ret_matrix = engine.calculate_for_dict(data_dict, add_column=False)
    assert isinstance(ret_matrix, pd.DataFrame)
    assert ret_matrix.shape == (3, 2)
    assert list(ret_matrix.columns) == ["AAPL", "MSFT"]
    assert abs(ret_matrix.loc[dates[1], "AAPL"] - 0.10) < 1e-8
    assert abs(ret_matrix.loc[dates[1], "MSFT"] - 0.00) < 1e-8

    # Chế độ 2: Gắn thêm cột return vào từng DataFrame
    updated_dict = engine.calculate_for_dict(data_dict, add_column=True, return_col_name="return")
    assert "return" in updated_dict["AAPL"].columns
    assert "return" in updated_dict["MSFT"].columns

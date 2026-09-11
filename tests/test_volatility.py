"""
tests/test_volatility.py
Bộ kiểm thử tính toán độ biến động trượt (Rolling Volatility Tests).

Ticket: FEAT-002 (P0 - Thành viên A)
Sprint: 2

Mục đích kiểm thử:
1. Kiểm tra tính chính xác của độ lệch chuẩn mẫu trong cửa sổ trượt W ngày.
2. Kiểm tra chuỗi lợi suất không đổi (hằng số) có độ biến động bằng 0.0.
3. Kiểm tra công thức niên độ hóa: sigma_ann = sigma_daily * sqrt(252).
4. Kiểm tra tham số min_periods=1 loại bỏ triệt để các giá trị NaN.
5. Kiểm tra VolatilityEngine xử lý dictionary đa tài sản xuất ma trận [T, N].
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.features.volatility import VolatilityEngine, compute_rolling_volatility


def test_constant_returns_zero_volatility():
    """Kiểm tra chuỗi lợi suất bằng phẳng có độ lệch chuẩn bằng 0."""
    dates = pd.bdate_range("2024-01-01", periods=10)
    returns = pd.Series([0.01] * 10, index=dates)

    vol = compute_rolling_volatility(returns, window=5, min_periods=2)
    # Từ điểm thứ 2 trở đi, độ lệch chuẩn của [0.01, 0.01] bằng 0.0
    for val in vol.iloc[1:]:
        assert abs(val - 0.0) < 1e-8


def test_annualized_volatility_calculation():
    """Kiểm tra công thức nhân sqrt(252) cho độ biến động niên độ hóa."""
    dates = pd.bdate_range("2024-01-01", periods=5)
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02], index=dates)

    daily_vol = compute_rolling_volatility(returns, window=3, annualized=False)
    ann_vol = compute_rolling_volatility(returns, window=3, annualized=True)

    for d, a in zip(daily_vol, ann_vol):
        assert abs(a - d * math.sqrt(252)) < 1e-8


def test_min_periods_no_nans():
    """Kiểm tra min_periods=1 và fill_zero=True đảm bảo không có giá trị NaN."""
    dates = pd.bdate_range("2024-01-01", periods=20)
    returns = pd.Series(np.random.randn(20) * 0.02, index=dates)

    vol = compute_rolling_volatility(returns, window=10, min_periods=1, fill_zero=True)
    assert not vol.isna().any(), "Volatility series must not contain NaN values"


def test_invalid_window_raises():
    """Kiểm tra ném ngoại lệ khi window <= 1."""
    returns = pd.Series([0.01, 0.02])
    with pytest.raises(ValueError, match="Window size must be greater than 1"):
        compute_rolling_volatility(returns, window=1)


def test_volatility_engine_multi_asset_dict():
    """Kiểm tra VolatilityEngine trên dictionary đa tài sản."""
    dates = pd.bdate_range("2024-01-01", periods=10)
    data_dict = {
        "AAPL": pd.DataFrame({"return": [0.01, -0.01, 0.02, 0.0, 0.01, -0.02, 0.01, 0.0, 0.01, -0.01]}, index=dates),
        "MSFT": pd.DataFrame({"return": [0.02, 0.01, -0.01, 0.0, 0.01, 0.02, -0.01, 0.01, 0.0, -0.02]}, index=dates),
    }

    engine = VolatilityEngine(window=5, annualized=False, return_col="return")

    # Chế độ 1: Trả về ma trận biến động [T, N]
    vol_matrix = engine.calculate_for_dict(data_dict, add_column=False)
    assert isinstance(vol_matrix, pd.DataFrame)
    assert vol_matrix.shape == (10, 2)
    assert list(vol_matrix.columns) == ["AAPL", "MSFT"]
    assert not vol_matrix.isna().any().any()

    # Chế độ 2: Gắn thêm cột vào DataFrame
    updated_dict = engine.calculate_for_dict(data_dict, add_column=True, vol_col_name="vol_5d")
    assert "vol_5d" in updated_dict["AAPL"].columns
    assert "vol_5d" in updated_dict["MSFT"].columns

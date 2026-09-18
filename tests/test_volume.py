"""
tests/test_volume.py
Bộ kiểm thử chuẩn hóa khối lượng giao dịch (Volume Normalization Tests).

Ticket: FEAT-005 (P1 - Thành viên A)
Sprint: 3

Mục đích kiểm thử:
1. Kiểm tra biến đổi log1p: volume = 0 => log_vol = 0.
2. Bắt lỗi khi khối lượng âm (< 0).
3. Kiểm tra tính toán Rolling Z-score:
   - Khối lượng hằng số => Z-score = 0.
   - Đột biến khối lượng (Volume Spike) => Z-score dương mạnh (> 2.0).
4. Kiểm tra ném lỗi khi window <= 1.
5. Kiểm tra VolumeEngine trên dictionary đa tài sản.
6. Kiểm tra chuẩn hóa khối lượng trên toàn bộ 25 mã cổ phiếu thật.
"""

import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.alignment import CalendarAligner
from src.features.volume import (
    VolumeEngine,
    compute_log_volume,
    compute_normalized_volume,
)


def test_log_volume_transformation():
    """Kiểm tra log1p: volume = 0 => 0, volume = e - 1 => 1.0."""
    dates = pd.bdate_range("2024-01-01", periods=3)
    vol = pd.Series([0.0, math.e - 1.0, 1000.0], index=dates)

    log_vol = compute_log_volume(vol)
    assert abs(log_vol.iloc[0] - 0.0) < 1e-8
    assert abs(log_vol.iloc[1] - 1.0) < 1e-8


def test_negative_volume_raises_error():
    """Kiểm tra ném lỗi khi khối lượng giao dịch âm."""
    vol = pd.Series([100.0, -10.0, 50.0])
    with pytest.raises(ValueError, match="không thể âm"):
        compute_log_volume(vol)


def test_constant_volume_zero_zscore():
    """Kiểm tra khi khối lượng không đổi thì Z-score bằng 0."""
    dates = pd.bdate_range("2024-01-01", periods=15)
    vol = pd.Series([1_000_000.0] * 15, index=dates)

    norm_vol = compute_normalized_volume(vol, window=5, fill_zero=True)
    for val in norm_vol:
        assert abs(val - 0.0) < 1e-6


def test_volume_spike_detection():
    """Kiểm tra khi có phiên đột biến khối lượng, Z-score tăng vọt (> 2.0)."""
    dates = pd.bdate_range("2024-01-01", periods=25)
    # 24 phiên đều đặn ~100k, phiên cuối tăng vọt lên 5 triệu
    vol_values = [100_000.0 + (i % 5) * 1000.0 for i in range(24)] + [5_000_000.0]
    vol = pd.Series(vol_values, index=dates)

    norm_vol = compute_normalized_volume(vol, window=20)
    # Phiên cuối cùng đột biến
    assert norm_vol.iloc[-1] > 2.0, f"Kỳ vọng Z-score > 2.0 khi có volume spike, nhận được {norm_vol.iloc[-1]}"


def test_invalid_window_raises_error():
    """Kiểm tra ném lỗi khi window <= 1."""
    vol = pd.Series([1000.0, 2000.0])
    with pytest.raises(ValueError, match="lớn hơn 1"):
        compute_normalized_volume(vol, window=1)


def test_volume_engine_multi_asset_dict():
    """Kiểm tra VolumeEngine trên dictionary đa tài sản."""
    dates = pd.bdate_range("2024-01-01", periods=25)
    data_dict = {
        "AAPL": pd.DataFrame({"volume": [1_000_000.0 + i * 50_000.0 for i in range(25)]}, index=dates),
        "MSFT": pd.DataFrame({"volume": [2_000_000.0 - i * 30_000.0 for i in range(25)]}, index=dates),
    }

    engine = VolumeEngine(window=10)

    # Chế độ 1: Trả về ma trận khối lượng chuẩn hóa [T, N]
    norm_matrix = engine.calculate_for_dict(data_dict, add_column=False)
    assert isinstance(norm_matrix, pd.DataFrame)
    assert norm_matrix.shape == (25, 2)
    assert list(norm_matrix.columns) == ["AAPL", "MSFT"]
    assert not norm_matrix.isna().any().any()

    # Chế độ 2: Gắn thêm cột vào từng DataFrame
    updated_dict = engine.calculate_for_dict(data_dict, add_column=True)
    assert "volume_norm" in updated_dict["AAPL"].columns
    assert "volume_norm" in updated_dict["MSFT"].columns


def test_volume_normalization_on_real_data():
    """Kiểm tra chuẩn hóa khối lượng trên toàn bộ 25 mã cổ phiếu thật."""
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        pytest.skip("data/raw directory not found, skipping real market test.")

    aligner = CalendarAligner(method="intersection")
    result = aligner.align_from_directory(raw_dir)

    engine = VolumeEngine(window=20)
    norm_matrix = engine.calculate_for_dict(result.aligned_data, add_column=False)

    assert norm_matrix.shape == (result.num_timesteps, result.num_assets)
    assert not norm_matrix.isna().any().any()
    assert not np.isinf(norm_matrix.to_numpy()).any()


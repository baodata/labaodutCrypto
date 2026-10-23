# kiểm thử cho tầng tính toán và chuẩn hóa đặc trưng (Feature Layer)

import numpy as np
import pytest

from src.data.loader import load_market_data
from src.features.pipeline import build_feature_pipeline
from src.features.scaler import FeatureScaler
from src.features.technical import FEATURE_NAMES, compute_technical_features


def test_compute_technical_features():
    # tạo dữ liệu giá và volume giả lập với T=50 ngày, N=3 cổ phiếu
    np.random.seed(42)
    t, n = 50, 3
    base_price = np.array([100.0, 50.0, 20.0])
    returns = np.random.normal(0.001, 0.02, size=(t, n))
    price = base_price * np.exp(np.cumsum(returns, axis=0))
    volume = np.random.uniform(1e5, 1e6, size=(t, n))

    features, names = compute_technical_features(price, volume)

    assert features.shape == (t, n, len(FEATURE_NAMES))
    assert names == FEATURE_NAMES

    # kiểm tra RSI-14 luôn nằm trong khoảng [0, 1] (bỏ qua dòng 0 do diff khởi tạo)
    rsi_idx = names.index("rsi_14")
    rsi_vals = features[1:, :, rsi_idx]
    assert np.all(rsi_vals >= 0.0)
    assert np.all(rsi_vals <= 1.0)

    # kiểm tra độ biến động 20 ngày luôn >= 0 (sau khoảng thời gian khởi động 20 ngày)
    vol_idx = names.index("volatility_20d")
    vol_vals = features[20:, :, vol_idx]
    assert np.all(vol_vals >= 0.0)


def test_feature_scaler_no_leakage():
    # kiểm tra scaler chỉ học từ tập Train, không bị rò rỉ dữ liệu từ tập Test
    np.random.seed(42)
    train_data = np.random.normal(loc=10.0, scale=2.0, size=(100, 5, 4)).astype(np.float32)
    test_data = np.random.normal(loc=50.0, scale=10.0, size=(50, 5, 4)).astype(np.float32)

    scaler = FeatureScaler(clip_value=5.0)
    scaler.fit(train_data)

    initial_mean = scaler.mean.copy()
    initial_std = scaler.std.copy()

    # biến đổi tập Test không được phép làm thay đổi mean và std của scaler
    test_scaled = scaler.transform(test_data)
    assert np.array_equal(scaler.mean, initial_mean)
    assert np.array_equal(scaler.std, initial_std)

    # kiểm tra cắt ngưỡng ngoại lai clip_value
    assert np.all(test_scaled <= 5.0)
    assert np.all(test_scaled >= -5.0)


def test_build_feature_pipeline_integration():
    # kiểm thử tích hợp trên toàn bộ dữ liệu 24 cổ phiếu thực tế
    full_data, splits = load_market_data("configs/assets.yaml")
    scaled_splits, scaler = build_feature_pipeline(full_data, splits, warmup_days=30)

    assert "train" in scaled_splits
    assert "val" in scaled_splits
    assert "test" in scaled_splits

    train_tensor = scaled_splits["train"]
    val_tensor = scaled_splits["val"]
    test_tensor = scaled_splits["test"]

    # kiểm tra kích thước
    # train ban đầu là 1763 ngày, bỏ 30 ngày warmup còn 1733 ngày
    assert train_tensor.T == 1733
    assert val_tensor.T == 501
    assert test_tensor.T == 501

    assert train_tensor.N == 24
    assert train_tensor.F == 7

    # kiểm tra tuyệt đối không có NaN hoặc Inf
    for name, tensor in scaled_splits.items():
        assert not np.isnan(tensor.values).any(), f"NaN xuất hiện trong {name}"
        assert not np.isinf(tensor.values).any(), f"Inf xuất hiện trong {name}"
        # giá trị đã scale phải nằm trong đoạn [-5, 5]
        assert np.all(tensor.values >= -5.0)
        assert np.all(tensor.values <= 5.0)

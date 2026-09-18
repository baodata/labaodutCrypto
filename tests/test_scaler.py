"""
tests/test_scaler.py
Bộ kiểm thử chuẩn hóa đặc trưng độc quyền trên tập Train (Train-Only Scaler Tests).

Ticket: SPLIT-002 (P0 - Thành viên A)
Sprint: 3

Mục đích kiểm thử:
1. Kiểm tra nguyên tắc Train-Only: Fit chỉ diễn ra trên tập Train.
2. Kiểm tra dữ liệu tập Train sau chuẩn hóa đạt xấp xỉ Mean ~ 0 và Std ~ 1.
3. Kiểm tra biến đổi tập Validation và Test hoàn toàn sử dụng mu_train và sigma_train đóng băng.
4. Kiểm tra lưu (save) và nạp (load) thông số JSON tại data/processed/scaler_params.json.
5. Tương thích hoàn hảo với khuôn đúc MarketDataTensor [T, N, F].
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.split import TemporalSplitter
from src.features.pipeline import FeaturePipeline
from src.features.scaler import MarketFeatureScaler
from src.utils.data_types import MarketDataTensor


@pytest.fixture
def synthetic_train_test_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tạo tập Train và Test nhân tạo với phân phối khác nhau."""
    np.random.seed(42)
    # Train: phân phối N(10, 2)
    train_vals = np.random.normal(loc=10.0, scale=2.0, size=(100, 2))
    # Test: phân phối N(15, 3)
    test_vals = np.random.normal(loc=15.0, scale=3.0, size=(50, 2))

    train_df = pd.DataFrame(train_vals, columns=["f1", "f2"])
    test_df = pd.DataFrame(test_vals, columns=["f1", "f2"])
    return train_df, test_df


def test_scaler_fit_transform_train_only(synthetic_train_test_data: tuple[pd.DataFrame, pd.DataFrame]):
    """Kiểm tra scaler biến đổi tập Train thành chuẩn N(0, 1) và giữ nguyên tham số khi transform Test."""
    train_df, test_df = synthetic_train_test_data

    scaler = MarketFeatureScaler(clip_range=None)
    train_scaled = scaler.fit_transform(train_df)

    # Train sau scale phải có mean ~ 0 và std ~ 1
    assert abs(train_scaled["f1"].mean()) < 0.1
    assert abs(train_scaled["f1"].std() - 1.0) < 0.1

    # Transform test
    test_scaled = scaler.transform(test_df)
    # Vì Test có phân phối gốc (mean=15) cao hơn Train (mean=10), nên mean sau scale phải dương
    assert test_scaled["f1"].mean() > 1.5, "Test scale phải dùng mu_train, nên giá trị trung bình phải lệch dương"


def test_scaler_save_load_json(tmp_path: Path, synthetic_train_test_data: tuple[pd.DataFrame, pd.DataFrame]):
    """Kiểm tra lưu và tải tham số scaler từ JSON."""
    train_df, test_df = synthetic_train_test_data
    scaler = MarketFeatureScaler()
    scaler.fit(train_df)

    json_path = tmp_path / "scaler_params.json"
    saved_path = scaler.save(json_path)
    assert saved_path.exists()

    loaded_scaler = MarketFeatureScaler.load(json_path)
    assert loaded_scaler.is_fitted is True
    assert loaded_scaler.feature_names == scaler.feature_names
    assert loaded_scaler.means == scaler.means
    assert loaded_scaler.stds == scaler.stds

    # Kết quả biến đổi phải trùng khớp 100%
    orig_trans = scaler.transform(test_df)
    load_trans = loaded_scaler.transform(test_df)
    pd.testing.assert_frame_equal(orig_trans, load_trans)


def test_scaler_on_market_data_tensor():
    """Kiểm tra scaler hoạt động mượt mà trên MarketDataTensor [T, N, F]."""
    tensor_data = np.random.normal(loc=5.0, scale=2.0, size=(100, 3, 4)).astype(np.float32)
    dates = [f"2024-01-{i+1:02d}" for i in range(100)]
    tickers = ["AAPL", "MSFT", "NVDA"]
    feature_names = ["ret", "vol", "rsi", "norm_vol"]

    market_tensor = MarketDataTensor(
        tensor=tensor_data,
        tickers=tickers,
        feature_names=feature_names,
        dates=dates,
    )

    scaler = MarketFeatureScaler(clip_range=(-5.0, 5.0))
    scaler.fit(market_tensor)
    scaled_tensor = scaler.transform(market_tensor)

    assert isinstance(scaled_tensor, MarketDataTensor)
    assert scaled_tensor.tensor.shape == (100, 3, 4)
    # Kiểm tra không có NaN và Inf
    scaled_tensor.validate()


def test_end_to_end_train_val_test_pipeline():
    """Kiểm tra quy trình hoàn chỉnh: Raw Data -> Pipeline -> Split -> Train-Only Scaler."""
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        pytest.skip("data/raw directory not found, skipping end-to-end test.")

    # 1. Pipeline tính 6 đặc trưng
    pipeline = FeaturePipeline()
    _, full_tensor = pipeline.run_from_raw(raw_dir=raw_dir, export_parquet=True)

    # 2. Split thời gian
    splitter = TemporalSplitter()
    split_res = splitter.split_market_data_tensor(full_tensor)

    # 3. Fit scaler CHỈ trên tập Train
    scaler = MarketFeatureScaler()
    scaler.fit(split_res.train)

    # 4. Transform toàn bộ 3 tập
    train_scaled = scaler.transform(split_res.train)
    val_scaled = scaler.transform(split_res.val)
    test_scaled = scaler.transform(split_res.test)

    # 5. Lưu tham số chính thức
    out_scaler_path = Path("data/processed/scaler_params.json")
    scaler.save(out_scaler_path)

    assert out_scaler_path.exists()
    assert train_scaled.tensor.shape[0] == len(split_res.train.dates)
    assert val_scaled.tensor.shape[0] == len(split_res.val.dates)
    assert test_scaled.tensor.shape[0] == len(split_res.test.dates)


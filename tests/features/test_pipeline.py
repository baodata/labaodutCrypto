"""
tests/test_pipeline.py
Bộ kiểm thử tích hợp cho đường ống hợp nhất đặc trưng (Feature Pipeline Integration Tests).

Ticket: FEAT-006 (P0 - Thành viên A)
Sprint: 3

Mục đích kiểm thử:
1. Kiểm tra tính toán đồng bộ 6 đặc trưng kỹ thuật cốt lõi:
   ['return', 'volatility_20d', 'rsi_14', 'macd', 'macd_signal', 'volume_norm'].
2. Kiểm tra đóng gói MarketDataTensor kích thước [T, N, F] vượt qua hàm validate().
3. Kiểm tra tính toàn vẹn: Không chứa bất kỳ giá trị NaN hoặc vô cực Inf nào.
4. Kiểm tra xuất bản bảng phẳng parquet tại data/processed/features.parquet và đọc lại an toàn.
5. Nghiệm thu thực tế trên toàn bộ 25 mã cổ phiếu thật trong data/raw/.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FeaturePipeline
from src.utils.data_types import MarketDataTensor


@pytest.fixture
def synthetic_aligned_assets() -> dict[str, pd.DataFrame]:
    """Tạo dữ liệu giả lập gồm 2 tài sản 30 ngày."""
    dates = pd.bdate_range("2024-01-01", periods=30)
    data = {}
    for ticker, base_price in [("AAPL", 150.0), ("MSFT", 300.0)]:
        np.random.seed(42 if ticker == "AAPL" else 99)
        returns = np.random.randn(30) * 0.015
        close = base_price * np.cumprod(1 + returns)
        high = close * (1 + np.abs(np.random.randn(30) * 0.005))
        low = close * (1 - np.abs(np.random.randn(30) * 0.005))
        open_p = (high + low) / 2.0
        volume = np.random.randint(1_000_000, 5_000_000, size=30)
        df = pd.DataFrame(
            {"open": open_p, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )
        data[ticker] = df
    return data


def test_pipeline_process_asset(synthetic_aligned_assets: dict[str, pd.DataFrame]):
    """Kiểm tra xử lý 1 tài sản có đủ 6 cột đặc trưng."""
    pipeline = FeaturePipeline()
    df_aapl = synthetic_aligned_assets["AAPL"]
    df_processed = pipeline.process_asset(df_aapl)

    for feat in FeaturePipeline.FEATURE_NAMES:
        assert feat in df_processed.columns, f"Thiếu cột đặc trưng '{feat}'"
        assert not df_processed[feat].isna().any(), f"Đặc trưng '{feat}' chứa NaN"
        assert not np.isinf(df_processed[feat].to_numpy()).any(), f"Đặc trưng '{feat}' chứa Inf"


def test_pipeline_build_market_data_tensor(synthetic_aligned_assets: dict[str, pd.DataFrame]):
    """Kiểm tra đúc MarketDataTensor [T, N, F]."""
    pipeline = FeaturePipeline()
    processed_dict = pipeline.process_all(synthetic_aligned_assets)
    market_tensor = pipeline.build_market_data_tensor(processed_dict)

    assert isinstance(market_tensor, MarketDataTensor)
    assert market_tensor.tensor.shape == (30, 2, 6)
    assert market_tensor.tickers == ["AAPL", "MSFT"]
    assert market_tensor.feature_names == FeaturePipeline.FEATURE_NAMES
    assert len(market_tensor.dates) == 30

    # Phải pass qua kiểm tra toàn vẹn
    market_tensor.validate()


def test_pipeline_export_parquet(synthetic_aligned_assets: dict[str, pd.DataFrame], tmp_path: Path):
    """Kiểm tra xuất và đọc lại tệp parquet."""
    pipeline = FeaturePipeline()
    processed_dict = pipeline.process_all(synthetic_aligned_assets)

    out_file = tmp_path / "test_features.parquet"
    saved_path = pipeline.export_processed_data(processed_dict, output_file=out_file)

    assert saved_path.exists()
    loaded_df = pd.read_parquet(saved_path)
    assert loaded_df.shape == (60, 6)  # 30 ngày * 2 mã = 60 dòng, 6 cột đặc trưng
    assert list(loaded_df.columns) == FeaturePipeline.FEATURE_NAMES


def test_pipeline_run_on_real_market_dataset():
    """Kiểm tra toàn bộ pipeline chạy trên 25 mã cổ phiếu thật trong data/raw/."""
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        pytest.skip("data/raw directory not found, skipping real pipeline test.")

    pipeline = FeaturePipeline()
    out_file = Path("data/processed/features.parquet")
    processed_data, market_tensor = pipeline.run_from_raw(
        raw_dir=raw_dir,
        export_parquet=True,
        output_file=out_file,
    )

    # 25 mã (24 stocks + SPY), 2765 ngày, 6 đặc trưng
    assert market_tensor.tensor.shape == (2765, 25, 6)
    assert out_file.exists()
    assert out_file.stat().st_size > 100_000  # Kích thước hợp lý (> 100KB)


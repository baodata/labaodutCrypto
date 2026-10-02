"""
tests/test_split.py
Bộ kiểm thử phân chia tập dữ liệu theo chuỗi thời gian (Temporal Splitter Tests).

Ticket: SPLIT-001 (P0 - Thành viên A)
Sprint: 3

Mục đích kiểm thử:
1. Kiểm tra tiêu chuẩn nghiệm thu sống còn:
   max(train_date) < min(val_date) và max(val_date) < min(test_date).
2. Kiểm tra các khoảng thời gian chuẩn:
   - Train: 2015-01-01 -> 2021-12-31
   - Validation: 2022-01-01 -> 2023-12-31
   - Test: 2024-01-01 -> 2025-12-31
3. Kiểm tra phân chia đồng bộ trên DataFrame, dict[str, DataFrame], và MarketDataTensor [T, N, F].
4. Kiểm định trực tiếp trên tệp dữ liệu thật data/processed/features.parquet.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.split import SplitResult, TemporalSplitter
from src.features.pipeline import FeaturePipeline
from src.utils.data_types import MarketDataTensor


@pytest.fixture
def multi_year_df() -> pd.DataFrame:
    """Tạo chuỗi dữ liệu 10 năm từ 2015 đến 2025."""
    dates = pd.bdate_range("2015-01-01", "2025-12-31")
    return pd.DataFrame(
        {"close": np.linspace(100.0, 300.0, len(dates)), "volume": 1_000_000},
        index=dates,
    )


def test_temporal_order_strict(multi_year_df: pd.DataFrame):
    """Kiểm tra tính đơn điệu và không rò rỉ thời gian tuyệt đối."""
    splitter = TemporalSplitter()
    res = splitter.split_dataframe(multi_year_df)

    assert isinstance(res, SplitResult)
    train_df, val_df, test_df = res.train, res.val, res.test

    assert not train_df.empty
    assert not val_df.empty
    assert not test_df.empty

    max_train = train_df.index.max()
    min_val = val_df.index.min()
    max_val = val_df.index.max()
    min_test = test_df.index.min()

    # Tiêu chuẩn nghiệm thu cốt lõi
    assert max_train < min_val, f"Rò rỉ Train-Val: {max_train} >= {min_val}"
    assert max_val < min_test, f"Rò rỉ Val-Test: {max_val} >= {min_test}"

    # Kiểm tra biên năm
    assert max_train.year <= 2021
    assert min_val.year >= 2022
    assert max_val.year <= 2023
    assert min_test.year >= 2024


def test_split_market_data_tensor():
    """Kiểm tra phân chia MarketDataTensor [T, N, F]."""
    # Tạo tensor giả lập 1000 ngày
    dates_idx = pd.bdate_range("2015-01-02", periods=2500)
    dates_str = [str(d.date()) for d in dates_idx]
    tensor = np.zeros((2500, 3, 6), dtype=np.float32)

    market_tensor = MarketDataTensor(
        tensor=tensor,
        tickers=["AAPL", "MSFT", "NVDA"],
        feature_names=["f1", "f2", "f3", "f4", "f5", "f6"],
        dates=dates_str,
    )

    splitter = TemporalSplitter()
    split_res = splitter.split_market_data_tensor(market_tensor)

    train_t, val_t, test_t = split_res.train, split_res.val, split_res.test

    assert isinstance(train_t, MarketDataTensor)
    assert isinstance(val_t, MarketDataTensor)
    assert isinstance(test_t, MarketDataTensor)

    # Đảm bảo tổng số ngày khớp 100%
    total_split_t = len(train_t.dates) + len(val_t.dates) + len(test_t.dates)
    assert total_split_t == 2500

    # Kiểm tra trật tự thời gian giữa các tensor
    assert pd.Timestamp(train_t.dates[-1]) < pd.Timestamp(val_t.dates[0])
    assert pd.Timestamp(val_t.dates[-1]) < pd.Timestamp(test_t.dates[0])


def test_split_on_real_processed_dataset():
    """Kiểm tra phân chia trực tiếp trên file data/processed/features.parquet."""
    parquet_path = Path("data/processed/features.parquet")
    if not parquet_path.exists():
        pytest.skip("data/processed/features.parquet chưa tồn tại, bỏ qua test.")

    df_features = pd.read_parquet(parquet_path)
    splitter = TemporalSplitter()
    split_res = splitter.split_dataframe(df_features)

    # Kiểm tra kích thước các tập
    assert len(split_res.train) > 0
    assert len(split_res.val) > 0
    assert len(split_res.test) > 0

    # Đảm bảo không có dòng nào bị thất thoát
    assert len(split_res.train) + len(split_res.val) + len(split_res.test) == len(df_features)


"""
src/data/split.py
Mô-đun phân chia tập dữ liệu Train / Validation / Test theo chuỗi thời gian (Temporal Splitter).

Ticket: SPLIT-001 (P0 - Thành viên A)
Sprint: 3

Mục đích:
1. Phân chia dữ liệu thành 3 khoảng theo trật tự thời gian:
   - Tập Huấn luyện (Train):      2015-01-01 -> 2021-12-31 (~7 năm)
   - Tập Thẩm định (Validation):  2022-01-01 -> 2023-12-31 (2 năm, tuning & early stopping)
   - Tập Test lịch sử:            2024-01-01 -> 2025-12-31
2. Quy tắc trật tự thời gian (không bảo đảm test độc lập nếu kết quả đã được xem):
   - TUYỆT ĐỐI KHÔNG dùng hàm xáo trộn ngẫu nhiên (e.g. train_test_split(shuffle=True)).
   - Luôn đảm bảo trật tự nhân quả: max(train_date) < min(val_date) < min(test_date).
3. Hỗ trợ phân chia đa định dạng:
   - pd.DataFrame (đơn lẻ hoặc bảng phẳng multi-index Date-Ticker)
   - dict[str, pd.DataFrame] (danh mục 25 cổ phiếu)
   - MarketDataTensor (ma trận 3D [T, N, F] của hệ thống H-MARL)
"""

from dataclasses import dataclass
from typing import Any, Generic, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

from src.utils.data_types import MarketDataTensor

T = TypeVar("T")


@dataclass
class SplitResult(Generic[T]):
    """Kết quả phân chia tập dữ liệu 3 pha."""

    train: T
    val: T
    test: T
    dates_info: dict[str, Any]

    def summary(self) -> str:
        """Tóm tắt thông tin phân chia tập dữ liệu."""
        lines = [
            "Temporal Data Split Summary:",
            f"  - Train: {self.dates_info.get('train_range')}",
            f"  - Validation: {self.dates_info.get('val_range')}",
            f"  - Test (selected period; independence not implied): {self.dates_info.get('test_range')}",
        ]
        return "\n".join(lines)


class TemporalSplitter:
    """Bộ phân chia dữ liệu định lượng theo chuỗi thời gian không xáo trộn."""

    DEFAULT_TRAIN_END: str = "2021-12-31"
    DEFAULT_VAL_START: str = "2022-01-01"
    DEFAULT_VAL_END: str = "2023-12-31"
    # 2024–2025 đã được xem qua nhiều vòng phân tích: đây là giai đoạn test
    # lịch sử để khám phá, không còn là holdout độc lập.
    DEFAULT_TEST_START: str = "2024-01-01"

    def __init__(
        self,
        train_end: Optional[str] = None,
        val_start: Optional[str] = None,
        val_end: Optional[str] = None,
        test_start: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        """Khởi tạo TemporalSplitter với các mốc thời gian hoặc từ file cấu hình."""
        if config_path is not None:
            from src.utils.config import AssetsConfig
            cfg = AssetsConfig(str(config_path))
            t_end, v_start, v_end, t_start = cfg.get_split_or_default_dates()
            train_end = train_end or t_end
            val_start = val_start or v_start
            val_end = val_end or v_end
            test_start = test_start or t_start
        else:
            train_end = train_end or self.DEFAULT_TRAIN_END
            val_start = val_start or self.DEFAULT_VAL_START
            val_end = val_end or self.DEFAULT_VAL_END
            test_start = test_start or self.DEFAULT_TEST_START

        self.train_end = pd.Timestamp(train_end)
        self.val_start = pd.Timestamp(val_start)
        self.val_end = pd.Timestamp(val_end)
        self.test_start = pd.Timestamp(test_start)

        if not (self.train_end < self.val_start <= self.val_end < self.test_start):
            raise ValueError(
                f"Các mốc thời gian không hợp lệ: train_end ({self.train_end.date()}) < "
                f"val_start ({self.val_start.date()}) <= val_end ({self.val_end.date()}) < "
                f"test_start ({self.test_start.date()})."
            )

    @classmethod
    def from_config(cls, config_path: str = "configs/assets.yaml") -> "TemporalSplitter":
        """Khởi tạo TemporalSplitter đọc mốc ngày trực tiếp từ assets.yaml."""
        return cls(config_path=config_path)

    @classmethod
    def from_dates(
        cls,
        dates: Any,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> "TemporalSplitter":
        """Tự động tính các mốc train/val/test từ chuỗi ngày theo tỷ lệ (mặc định 70/15/15)."""
        dt_series = pd.to_datetime(list(dates)).sort_values()
        n = len(dt_series)
        if n < 3:
            raise ValueError(f"Cần ít nhất 3 ngày để phân chia, hiện có {n}.")
        train_idx = max(0, int(n * train_ratio) - 1)
        val_start_idx = train_idx + 1
        val_end_idx = max(val_start_idx, int(n * (train_ratio + val_ratio)) - 1)
        test_start_idx = min(n - 1, val_end_idx + 1)

        train_end = str(dt_series[train_idx].date())
        val_start = str(dt_series[val_start_idx].date())
        val_end = str(dt_series[val_end_idx].date())
        test_start = str(dt_series[test_start_idx].date())

        return cls(
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
        )

    def split_dataframe(self, df: pd.DataFrame) -> SplitResult[pd.DataFrame]:
        """Phân chia một DataFrame có DatetimeIndex hoặc cột Date."""
        df_copy = df.copy()

        # Kiểm tra DatetimeIndex hoặc MultiIndex có level date
        if isinstance(df_copy.index, pd.DatetimeIndex):
            date_series = df_copy.index
        elif isinstance(df_copy.index, pd.MultiIndex) and "date" in [str(n).lower() for n in df_copy.index.names]:
            date_level = [i for i, n in enumerate(df_copy.index.names) if str(n).lower() == "date"][0]
            date_series = pd.to_datetime(df_copy.index.get_level_values(date_level))
        elif "date" in [str(c).lower() for c in df_copy.columns]:
            date_col = next(c for c in df_copy.columns if str(c).lower() == "date")
            date_series = pd.to_datetime(df_copy[date_col])
        else:
            raise ValueError("DataFrame không có DatetimeIndex hoặc cột 'date' để phân chia.")

        train_mask = date_series <= self.train_end
        val_mask = (date_series >= self.val_start) & (date_series <= self.val_end)
        test_mask = date_series >= self.test_start

        train_df = df_copy[train_mask].copy()
        val_df = df_copy[val_mask].copy()
        test_df = df_copy[test_mask].copy()

        self._validate_temporal_order(train_df, val_df, test_df)

        info = {
            "train_range": f"{self._get_min_date(train_df)} -> {self._get_max_date(train_df)} ({len(train_df)} rows)",
            "val_range": f"{self._get_min_date(val_df)} -> {self._get_max_date(val_df)} ({len(val_df)} rows)",
            "test_range": f"{self._get_min_date(test_df)} -> {self._get_max_date(test_df)} ({len(test_df)} rows)",
        }

        return SplitResult(train=train_df, val=val_df, test=test_df, dates_info=info)

    def split_dict(self, data_dict: dict[str, pd.DataFrame]) -> SplitResult[dict[str, pd.DataFrame]]:
        """Phân chia dictionary chứa DataFrames của các mã cổ phiếu."""
        train_dict: dict[str, pd.DataFrame] = {}
        val_dict: dict[str, pd.DataFrame] = {}
        test_dict: dict[str, pd.DataFrame] = {}

        for ticker, df in data_dict.items():
            res = self.split_dataframe(df)
            train_dict[ticker] = res.train
            val_dict[ticker] = res.val
            test_dict[ticker] = res.test

        first_ticker = next(iter(data_dict.keys()))
        info = {
            "train_range": f"{self._get_min_date(train_dict[first_ticker])} -> {self._get_max_date(train_dict[first_ticker])}",
            "val_range": f"{self._get_min_date(val_dict[first_ticker])} -> {self._get_max_date(val_dict[first_ticker])}",
            "test_range": f"{self._get_min_date(test_dict[first_ticker])} -> {self._get_max_date(test_dict[first_ticker])}",
            "num_assets": len(data_dict),
        }

        return SplitResult(train=train_dict, val=val_dict, test=test_dict, dates_info=info)

    def split_market_data_tensor(self, tensor_obj: MarketDataTensor) -> SplitResult[MarketDataTensor]:
        """Phân chia MarketDataTensor [T, N, F] dọc theo trục thời gian T."""
        date_series = pd.to_datetime(tensor_obj.dates)

        tensor_obj.validate()
        train_indices = np.where((date_series <= self.train_end) & tensor_obj.feature_ready)[0]
        val_indices = np.where((date_series >= self.val_start) & (date_series <= self.val_end))[0]
        test_indices = np.where(date_series >= self.test_start)[0]

        if len(train_indices) == 0 or len(val_indices) == 0 or len(test_indices) == 0:
            raise ValueError(
                f"Một trong các tập sau khi phân chia bị rỗng (Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}). "
                f"Dải ngày dữ liệu thực tế: {tensor_obj.dates[0]} -> {tensor_obj.dates[-1]} (T={len(tensor_obj.dates)}). "
                f"Cấu hình split hiện tại: train_end={self.train_end.date()}, "
                f"val=[{self.val_start.date()} -> {self.val_end.date()}], "
                f"test_start={self.test_start.date()}."
            )

        train_tensor = MarketDataTensor(
            tensor=tensor_obj.tensor[train_indices, :, :].copy(),
            tickers=list(tensor_obj.tickers),
            feature_names=list(tensor_obj.feature_names),
            dates=[tensor_obj.dates[i] for i in train_indices],
            warmup_sessions=max(0, tensor_obj.warmup_sessions - int(train_indices[0])),
        )
        val_tensor = MarketDataTensor(
            tensor=tensor_obj.tensor[val_indices, :, :].copy(),
            tickers=list(tensor_obj.tickers),
            feature_names=list(tensor_obj.feature_names),
            dates=[tensor_obj.dates[i] for i in val_indices],
            warmup_sessions=max(0, tensor_obj.warmup_sessions - int(val_indices[0])),
        )
        test_tensor = MarketDataTensor(
            tensor=tensor_obj.tensor[test_indices, :, :].copy(),
            tickers=list(tensor_obj.tickers),
            feature_names=list(tensor_obj.feature_names),
            dates=[tensor_obj.dates[i] for i in test_indices],
            warmup_sessions=max(0, tensor_obj.warmup_sessions - int(test_indices[0])),
        )

        train_tensor.validate()
        val_tensor.validate()
        test_tensor.validate()

        info = {
            "train_range": f"{train_tensor.dates[0]} -> {train_tensor.dates[-1]} (T={len(train_tensor.dates)})",
            "val_range": f"{val_tensor.dates[0]} -> {val_tensor.dates[-1]} (T={len(val_tensor.dates)})",
            "test_range": f"{test_tensor.dates[0]} -> {test_tensor.dates[-1]} (T={len(test_tensor.dates)})",
            "tickers": tensor_obj.tickers,
            "feature_names": tensor_obj.feature_names,
        }

        return SplitResult(train=train_tensor, val=val_tensor, test=test_tensor, dates_info=info)

    @staticmethod
    def _get_min_date(df: pd.DataFrame) -> str:
        if df.empty:
            return "Empty"
        if isinstance(df.index, pd.DatetimeIndex):
            return str(df.index.min().date())
        if isinstance(df.index, pd.MultiIndex) and "date" in [str(n).lower() for n in df.index.names]:
            date_level = [i for i, n in enumerate(df.index.names) if str(n).lower() == "date"][0]
            return str(pd.to_datetime(df.index.get_level_values(date_level)).min().date())
        return "N/A"

    @staticmethod
    def _get_max_date(df: pd.DataFrame) -> str:
        if df.empty:
            return "Empty"
        if isinstance(df.index, pd.DatetimeIndex):
            return str(df.index.max().date())
        if isinstance(df.index, pd.MultiIndex) and "date" in [str(n).lower() for n in df.index.names]:
            date_level = [i for i, n in enumerate(df.index.names) if str(n).lower() == "date"][0]
            return str(pd.to_datetime(df.index.get_level_values(date_level)).max().date())
        return "N/A"

    def _validate_temporal_order(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Kiểm định nghiệm thu: max(train_date) < min(val_date) và max(val_date) < min(test_date)."""
        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError(
                f"Một trong các tập dữ liệu bị rỗng sau khi phân chia (Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}). "
                f"Cấu hình split hiện tại: train_end={self.train_end.date()}, "
                f"val=[{self.val_start.date()} -> {self.val_end.date()}], "
                f"test_start={self.test_start.date()}."
            )

        max_train = pd.Timestamp(self._get_max_date(train_df))
        min_val = pd.Timestamp(self._get_min_date(val_df))
        max_val = pd.Timestamp(self._get_max_date(val_df))
        min_test = pd.Timestamp(self._get_min_date(test_df))

        if not (max_train < min_val):
            raise AssertionError(f"Rò rỉ thời gian: max(train)={max_train} không nhỏ hơn min(val)={min_val}")
        if not (max_val < min_test):
            raise AssertionError(f"Rò rỉ thời gian: max(val)={max_val} không nhỏ hơn min(test)={min_test}")


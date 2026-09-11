"""
src/data/alignment.py
Bộ đồng bộ lịch giao dịch thị trường (Trading Calendar Alignment Engine).

Ticket: DATA-004 (P0 - Thành viên A)
Sprint: 2

Mục đích:
1. Đồng bộ hóa trục thời gian (DatetimeIndex) giữa tất cả các tài sản trong danh mục.
2. Đảm bảo cấu trúc dữ liệu thỏa mãn điều kiện nghiêm ngặt của Tensor [T, N, F]:
   - Toàn bộ N tài sản có cùng chính xác số lượng bước thời gian T: len(df_i) == T.
   - Trục thời gian hoàn toàn trùng khớp: df_i.index == df_j.index.
3. Hỗ trợ 2 phương pháp căn chỉnh chuẩn mực trong tài chính định lượng:
   - 'intersection' (mặc định): Cắt gọt theo tập giao các ngày giao dịch chung (Common Dates).
     Đảm bảo 100% dữ liệu là giao dịch thật, không có nội suy hay giả định giá.
   - 'union_ffill': Lấy tập hợp tất cả các ngày, điền giá hôm trước (Forward-fill: P_t = P_{t-1})
     và khối lượng bằng 0 (Volume = 0). Tuyệt đối KHÔNG dùng backward-fill để chống Look-ahead Bias.
4. Cung cấp tiện ích xuất bảng giá chung `to_price_panel()` kích thước [T, N] làm đầu vào
   cho ma trận tương quan của Thành viên B và các chỉ báo kỹ thuật tiếp theo.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd


@dataclass
class AlignmentResult:
    """Represents the outcome of a multi-asset calendar alignment."""

    aligned_data: dict[str, pd.DataFrame]
    common_dates: pd.DatetimeIndex
    num_assets: int
    num_timesteps: int
    dropped_dates: dict[str, list[pd.Timestamp]] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary of the alignment."""
        start_str = str(self.common_dates.min().date()) if len(self.common_dates) > 0 else "N/A"
        end_str = str(self.common_dates.max().date()) if len(self.common_dates) > 0 else "N/A"
        lines = [
            "Trading Calendar Alignment Summary:",
            f"  - Total Assets (N): {self.num_assets}",
            f"  - Total Timesteps (T): {self.num_timesteps}",
            f"  - Common Date Range: {start_str} -> {end_str}",
        ]
        if self.dropped_dates:
            total_dropped = sum(len(d) for d in self.dropped_dates.values())
            lines.append(f"  - Total Dropped Date Instances: {total_dropped}")
        return "\n".join(lines)


class CalendarAligner:
    """Aligns multiple asset OHLCV time series to a unified trading calendar."""

    def __init__(
        self,
        method: Literal["intersection", "union_ffill"] = "intersection",
    ) -> None:
        """Initialize CalendarAligner.

        Args:
            method: 'intersection' (default) keeps only common trading days.
                    'union_ffill' takes union of all dates, forward-filling missing prices.
        """
        if method not in ["intersection", "union_ffill"]:
            raise ValueError(f"Invalid method '{method}'. Choose 'intersection' or 'union_ffill'.")
        self.method = method

    def _ensure_datetime_index(self, df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
        """Ensure the DataFrame has a clean DatetimeIndex named 'Date'."""
        df_copy = df.copy()

        if not isinstance(df_copy.index, pd.DatetimeIndex):
            date_col = next((c for c in df_copy.columns if str(c).lower() == "date"), None)
            if date_col is not None:
                df_copy[date_col] = pd.to_datetime(df_copy[date_col])
                df_copy = df_copy.set_index(date_col)
            else:
                raise ValueError(
                    f"Asset '{ticker}' has no DatetimeIndex and no 'date' column found in columns: {list(df_copy.columns)}"
                )

        # Strip timezone if present to ensure perfect comparability
        if df_copy.index.tz is not None:
            df_copy.index = df_copy.index.tz_localize(None)

        df_copy.index.name = "Date"
        df_copy = df_copy.sort_index()
        # Drop duplicate timestamps if any
        if not df_copy.index.is_unique:
            df_copy = df_copy[~df_copy.index.duplicated(keep="last")]

        return df_copy

    def find_common_calendar(self, data_dict: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """Extract the unified trading calendar across all assets.

        Args:
            data_dict: Dictionary mapping ticker symbols to OHLCV DataFrames.

        Returns:
            pd.DatetimeIndex of aligned dates.
        """
        if not data_dict:
            raise ValueError("Cannot align an empty dictionary of assets.")

        # Standardize indexes first
        cleaned_indexes: list[pd.DatetimeIndex] = []
        for ticker, df in data_dict.items():
            clean_df = self._ensure_datetime_index(df, ticker=ticker)
            cleaned_indexes.append(clean_df.index)

        if self.method == "intersection":
            common = cleaned_indexes[0]
            for idx in cleaned_indexes[1:]:
                common = common.intersection(idx)
            return common.sort_values()

        elif self.method == "union_ffill":
            union = cleaned_indexes[0]
            for idx in cleaned_indexes[1:]:
                union = union.union(idx)
            return union.sort_values()

        raise ValueError(f"Unknown alignment method: {self.method}")

    def align(self, data_dict: dict[str, pd.DataFrame]) -> AlignmentResult:
        """Align all DataFrames in data_dict to the common calendar.

        Args:
            data_dict: Dictionary mapping ticker symbols to OHLCV DataFrames.

        Returns:
            AlignmentResult with aligned DataFrames and calendar metadata.
        """
        if not data_dict:
            raise ValueError("Input data_dict cannot be empty.")

        # Clean all input DataFrames
        cleaned_dict: dict[str, pd.DataFrame] = {
            ticker: self._ensure_datetime_index(df, ticker=ticker)
            for ticker, df in data_dict.items()
        }

        common_calendar = self.find_common_calendar(cleaned_dict)
        if len(common_calendar) == 0:
            raise ValueError("Intersection of trading dates resulted in 0 common trading days.")

        aligned_data: dict[str, pd.DataFrame] = {}
        dropped_dates: dict[str, list[pd.Timestamp]] = {}

        for ticker, df in cleaned_dict.items():
            if self.method == "intersection":
                # Dropped dates: dates in df not in common_calendar
                dropped = df.index.difference(common_calendar).tolist()
                if dropped:
                    dropped_dates[ticker] = dropped

                # Select only common calendar dates
                aligned_df = df.loc[common_calendar].copy()

            elif self.method == "union_ffill":
                # Reindex to full union
                reindexed_df = df.reindex(common_calendar)

                # Identify price columns vs volume columns
                price_cols = [
                    c for c in reindexed_df.columns
                    if any(p in str(c).lower() for p in ["open", "high", "low", "close", "adj"])
                ]
                vol_cols = [c for c in reindexed_df.columns if "vol" in str(c).lower()]

                # Forward-fill prices (P_t = P_{t-1}), never backward-fill!
                reindexed_df[price_cols] = reindexed_df[price_cols].ffill()
                # Volume on non-trading days is 0
                reindexed_df[vol_cols] = reindexed_df[vol_cols].fillna(0)

                # If an asset started trading after the common calendar start date, drop leading NaNs
                # to maintain strictly valid numbers
                aligned_df = reindexed_df.dropna().copy()

            aligned_data[ticker] = aligned_df

        # In intersection mode, verify that all lengths match exactly
        if self.method == "intersection":
            expected_len = len(common_calendar)
            for ticker, df in aligned_data.items():
                if len(df) != expected_len:
                    raise RuntimeError(
                        f"Alignment failed for '{ticker}': expected {expected_len} rows, got {len(df)}."
                    )

        result = AlignmentResult(
            aligned_data=aligned_data,
            common_dates=common_calendar,
            num_assets=len(aligned_data),
            num_timesteps=len(common_calendar),
            dropped_dates=dropped_dates,
            stats={
                "method": self.method,
                "start_date": str(common_calendar.min().date()),
                "end_date": str(common_calendar.max().date()),
            },
        )
        return result

    def align_from_directory(
        self,
        raw_dir: Path | str,
        file_pattern: str = "*.parquet",
    ) -> AlignmentResult:
        """Load, validate, and align all asset files from a raw directory.

        Args:
            raw_dir: Path to directory containing raw asset files.
            file_pattern: Glob pattern to match files (e.g. '*.parquet' or '*.csv').

        Returns:
            AlignmentResult.
        """
        dir_path = Path(raw_dir)
        if not dir_path.exists():
            raise FileNotFoundError(f"Raw data directory does not exist: {dir_path}")

        files = sorted(dir_path.glob(file_pattern))
        if not files:
            raise FileNotFoundError(f"No files matching '{file_pattern}' found in {dir_path}")

        data_dict: dict[str, pd.DataFrame] = {}
        for file_path in files:
            ticker = file_path.stem.upper()
            if file_path.suffix == ".parquet":
                df = pd.read_parquet(file_path)
            elif file_path.suffix == ".csv":
                df = pd.read_csv(file_path)
            else:
                continue
            data_dict[ticker] = df

        return self.align(data_dict)

    @staticmethod
    def to_price_panel(
        aligned_dict: dict[str, pd.DataFrame],
        price_col: str = "close",
    ) -> pd.DataFrame:
        """Extract a 2D Price Panel [T, N] from aligned assets dictionary.

        Args:
            aligned_dict: Dictionary of aligned OHLCV DataFrames.
            price_col: Column to extract (case-insensitive, e.g. 'close' or 'Close').

        Returns:
            pd.DataFrame of shape [T, N] with DatetimeIndex and ticker columns.
        """
        series_dict: dict[str, pd.Series] = {}

        for ticker, df in aligned_dict.items():
            matching_col = next(
                (c for c in df.columns if str(c).strip().lower() == price_col.lower()),
                None,
            )
            if matching_col is None:
                raise KeyError(
                    f"Price column '{price_col}' not found in asset '{ticker}' columns: {list(df.columns)}"
                )
            series_dict[ticker] = df[matching_col]

        panel = pd.DataFrame(series_dict)
        panel.index.name = "Date"
        return panel.sort_index()


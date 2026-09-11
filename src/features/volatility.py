"""
src/features/volatility.py
Mô-đun tính toán độ biến động trượt (Rolling Volatility Engine).

Ticket: FEAT-002 (P0 - Thành viên A)
Sprint: 2

Mục đích:
1. Đo lường rủi ro biến động giá cổ phiếu thông qua độ lệch chuẩn mẫu (Sample Standard Deviation)
   của chuỗi lợi suất trong một cửa sổ thời gian trượt W ngày (mặc định W = 20 ngày giao dịch).
2. Cung cấp 2 chế độ tính toán:
   - Daily Volatility: sigma_daily = std(R_{t-W+1 ... t})
   - Annualized Volatility: sigma_ann = sigma_daily * sqrt(252)
     (Chuẩn hóa theo 252 ngày giao dịch hàng năm tại thị trường Mỹ).
3. Hỗ trợ tham số min_periods:
   - Mặc định min_periods = 1 giúp hạn chế tối đa các giá trị NaN trong giai đoạn khởi động (warm-up),
     giúp mô hình RL có thể quan sát đặc trưng ngay từ những ngày đầu.
4. Tương thích hoàn hảo với:
   - pd.Series: Chuỗi lợi suất đơn lẻ.
   - pd.DataFrame: Ma trận lợi suất đa tài sản [T, N] hoặc bảng OHLCV.
   - dict[str, pd.DataFrame]: Từ điển dữ liệu toàn bộ vũ trụ 25 mã cổ phiếu.
"""

import math
from typing import Union

import numpy as np
import pandas as pd


def compute_rolling_volatility(
    returns: Union[pd.Series, pd.DataFrame],
    window: int = 20,
    annualized: bool = False,
    min_periods: int = 1,
    fill_zero: bool = True,
) -> Union[pd.Series, pd.DataFrame]:
    """Tính toán độ lệch chuẩn mẫu trong cửa sổ trượt W ngày.

    Args:
        returns: Series hoặc DataFrame chứa chuỗi lợi suất (returns).
        window: Độ dài cửa sổ trượt tính toán (mặc định 20 ngày ~ 1 tháng giao dịch).
        annualized: Nếu True, nhân với căn bậc hai của 252 ngày (sqrt(252)).
        min_periods: Số lượng quan sát tối thiểu trong cửa sổ (mặc định 1 để giảm NaN).
        fill_zero: Nếu True, thay thế các giá trị NaN còn sót lại bằng 0.0.

    Returns:
        Series hoặc DataFrame chứa độ biến động trượt.
    """
    if window <= 1:
        raise ValueError(f"Window size must be greater than 1, got {window}.")

    vol = returns.rolling(window=window, min_periods=min_periods).std(ddof=1)

    if annualized:
        vol = vol * math.sqrt(252)

    if fill_zero:
        vol = vol.fillna(0.0)

    return vol


class VolatilityEngine:
    """Động cơ tính toán và quản lý độ biến động trượt cho hệ thống định lượng."""

    def __init__(
        self,
        window: int = 20,
        annualized: bool = False,
        min_periods: int = 1,
        fill_zero: bool = True,
        return_col: str = "return",
    ) -> None:
        """Khởi tạo VolatilityEngine.

        Args:
            window: Độ dài cửa sổ trượt (mặc định 20 ngày).
            annualized: Có niên độ hóa (nhân sqrt(252)) hay không.
            min_periods: Số phiên tối thiểu trong cửa sổ để bắt đầu tính.
            fill_zero: Có điền 0.0 cho các giá trị NaN hay không.
            return_col: Tên cột lợi suất trong DataFrame (nếu truyền bảng OHLCV).
        """
        self.window = window
        self.annualized = annualized
        self.min_periods = min_periods
        self.fill_zero = fill_zero
        self.return_col = return_col.lower()

    def calculate(
        self,
        data: Union[pd.Series, pd.DataFrame],
    ) -> Union[pd.Series, pd.DataFrame]:
        """Tính toán độ biến động trượt cho Series hoặc DataFrame.

        Args:
            data: Series lợi suất, hoặc DataFrame bảng lợi suất [T, N], hoặc DataFrame có cột 'return'.

        Returns:
            Series hoặc DataFrame độ biến động tương ứng.
        """
        if isinstance(data, pd.Series):
            returns_series = data
        elif isinstance(data, pd.DataFrame):
            cols_lower = [str(c).strip().lower() for c in data.columns]
            if self.return_col in cols_lower:
                matching_col = data.columns[cols_lower.index(self.return_col)]
                returns_series = data[matching_col]
            else:
                # Toàn bộ DataFrame là ma trận lợi suất đa tài sản [T, N]
                return compute_rolling_volatility(
                    data,
                    window=self.window,
                    annualized=self.annualized,
                    min_periods=self.min_periods,
                    fill_zero=self.fill_zero,
                )
        else:
            raise TypeError(f"Unsupported data type '{type(data).__name__}'. Expected pd.Series or pd.DataFrame.")

        return compute_rolling_volatility(
            returns_series,
            window=self.window,
            annualized=self.annualized,
            min_periods=self.min_periods,
            fill_zero=self.fill_zero,
        )

    def calculate_for_dict(
        self,
        aligned_dict: dict[str, pd.DataFrame],
        add_column: bool = True,
        vol_col_name: str = "volatility_20d",
    ) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
        """Tính độ biến động trượt cho toàn bộ danh mục tài sản trong dictionary.

        Args:
            aligned_dict: Dictionary {ticker: DataFrame}.
            add_column: Nếu True, gắn trực tiếp cột độ biến động vào từng DataFrame.
                        Nếu False, trả về DataFrame ma trận độ biến động [T, N].
            vol_col_name: Tên cột mới khi add_column=True.

        Returns:
            dict[str, pd.DataFrame] nếu add_column=True,
            hoặc pd.DataFrame [T, N] nếu add_column=False.
        """
        if not aligned_dict:
            raise ValueError("Input aligned_dict cannot be empty.")

        if add_column:
            updated_dict: dict[str, pd.DataFrame] = {}
            for ticker, df in aligned_dict.items():
                df_copy = df.copy()
                vol_series = self.calculate(df_copy)
                df_copy[vol_col_name] = vol_series
                updated_dict[ticker] = df_copy
            return updated_dict
        else:
            vol_matrix: dict[str, pd.Series] = {}
            for ticker, df in aligned_dict.items():
                vol_matrix[ticker] = self.calculate(df)
            result_df = pd.DataFrame(vol_matrix)
            result_df.index.name = "Date"
            return result_df

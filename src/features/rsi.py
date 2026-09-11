"""
src/features/rsi.py
Mô-đun tính toán chỉ báo RSI (Relative Strength Index Engine).

Ticket: FEAT-003 (P1 - Thành viên A)
Sprint: 2

Mục đích:
1. Cài đặt chuẩn xác công thức Relative Strength Index theo phương pháp làm mượt
   Wilder (Wilder's Smoothing, J. Welles Wilder Jr., 1978) với chu kỳ mặc định N = 14 ngày.
2. Công thức tính toán:
   - Delta = P_t - P_{t-1}
   - Gain = max(Delta, 0), Loss = max(-Delta, 0)
   - AvgGain_t = (AvgGain_{t-1} * (N - 1) + Gain_t) / N
   - AvgLoss_t = (AvgLoss_{t-1} * (N - 1) + Loss_t) / N
   - RS = AvgGain / AvgLoss
   - RSI = 100 - (100 / (1 + RS))
3. Xử lý các ca biên tài chính nghiêm ngặt:
   - Giá liên tục tăng (Loss = 0): RSI tiệm cận / bằng 100.0.
   - Giá liên tục giảm (Gain = 0): RSI tiệm cận / bằng 0.0.
   - Giá đi ngang không đổi (Gain = 0 và Loss = 0): RSI = 50.0.
4. Hỗ trợ tùy chọn chuẩn hóa (scaled=True) đưa RSI về đoạn [0.0, 1.0] phù hợp với
   không gian quan sát chuẩn hóa của mô hình Reinforcement Learning (PPO).
5. Hỗ trợ đầy đủ pd.Series, pd.DataFrame [T, N], và dict[str, pd.DataFrame].
"""

from typing import Union

import numpy as np
import pandas as pd


def compute_rsi(
    prices: Union[pd.Series, pd.DataFrame],
    period: int = 14,
    scaled: bool = False,
    fill_na: bool = True,
) -> Union[pd.Series, pd.DataFrame]:
    """Tính toán Relative Strength Index (RSI) theo công thức làm mượt Wilder.

    Args:
        prices: Series hoặc DataFrame chứa giá đóng cửa (Close prices > 0).
        period: Chu kỳ tính toán (mặc định 14 phiên).
        scaled: Nếu True, chia cho 100.0 để đưa về khoảng [0.0, 1.0] cho RL.
        fill_na: Nếu True, điền 50.0 (mức trung tính) cho các ngày khởi động đầu tiên.

    Returns:
        Series hoặc DataFrame chứa chỉ báo RSI.
    """
    if period <= 1:
        raise ValueError(f"Period must be greater than 1, got {period}.")

    # Tính biến động giá Delta = P_t - P_{t-1}
    delta = prices.diff()

    # Tách thành Gain và Loss
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's Smoothing tương đương với exponential moving average với alpha = 1 / period (com = period - 1)
    # Phương pháp: adjust=False mô phỏng chính xác công thức đệ quy của Wilder:
    # Avg_t = (Avg_{t-1} * (N - 1) + Val_t) / N = Avg_{t-1} * (1 - 1/N) + Val_t * (1/N)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # Tính Relative Strength (RS) và RSI
    # Tránh chia cho 0:
    # Nếu avg_loss == 0 và avg_gain > 0 -> RSI = 100.0
    # Nếu avg_gain == 0 và avg_loss == 0 -> RSI = 50.0
    # Nếu avg_gain == 0 và avg_loss > 0 -> RSI = 0.0
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Xử lý các trường hợp avg_loss == 0
    zero_loss_mask = (avg_loss == 0) & (avg_gain > 0)
    zero_movement_mask = (avg_loss == 0) & (avg_gain == 0)

    if isinstance(prices, pd.DataFrame):
        rsi = rsi.mask(zero_loss_mask, 100.0)
        rsi = rsi.mask(zero_movement_mask, 50.0)
    else:
        rsi = rsi.where(~zero_loss_mask, 100.0)
        rsi = rsi.where(~zero_movement_mask, 50.0)

    if fill_na:
        # Trong giai đoạn khởi động chưa đủ period ngày, gán mức 50.0 (trung tính)
        rsi = rsi.fillna(50.0)

    if scaled:
        rsi = rsi / 100.0

    return rsi


class RSIEngine:
    """Động cơ tính toán và quản lý chỉ báo RSI cho hệ thống định lượng."""

    def __init__(
        self,
        period: int = 14,
        scaled: bool = False,
        fill_na: bool = True,
        price_col: str = "close",
    ) -> None:
        """Khởi tạo RSIEngine.

        Args:
            period: Chu kỳ tính RSI (mặc định 14 ngày).
            scaled: Đưa về khoảng [0, 1] hay giữ thang điểm [0, 100].
            fill_na: Điền mức trung tính 50.0 cho giai đoạn khởi động hay không.
            price_col: Tên cột giá dùng để tính (mặc định 'close').
        """
        self.period = period
        self.scaled = scaled
        self.fill_na = fill_na
        self.price_col = price_col.lower()

    def calculate(
        self,
        data: Union[pd.Series, pd.DataFrame],
    ) -> Union[pd.Series, pd.DataFrame]:
        """Tính toán RSI cho một chuỗi Series hoặc DataFrame.

        Args:
            data: Series giá, DataFrame bảng OHLCV, hoặc ma trận giá [T, N].

        Returns:
            Series hoặc DataFrame chứa chỉ báo RSI.
        """
        if isinstance(data, pd.Series):
            price_series = data
        elif isinstance(data, pd.DataFrame):
            cols_lower = [str(c).strip().lower() for c in data.columns]
            if self.price_col in cols_lower:
                matching_col = data.columns[cols_lower.index(self.price_col)]
                price_series = data[matching_col]
            else:
                # Toàn bộ DataFrame là ma trận giá đa tài sản [T, N]
                return compute_rsi(
                    data,
                    period=self.period,
                    scaled=self.scaled,
                    fill_na=self.fill_na,
                )
        else:
            raise TypeError(f"Unsupported data type '{type(data).__name__}'. Expected pd.Series or pd.DataFrame.")

        return compute_rsi(
            price_series,
            period=self.period,
            scaled=self.scaled,
            fill_na=self.fill_na,
        )

    def calculate_for_dict(
        self,
        aligned_dict: dict[str, pd.DataFrame],
        add_column: bool = True,
        rsi_col_name: str = "rsi_14",
    ) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
        """Tính RSI cho toàn bộ danh mục tài sản trong dictionary.

        Args:
            aligned_dict: Dictionary {ticker: DataFrame OHLCV}.
            add_column: Nếu True, gắn trực tiếp cột RSI vào từng DataFrame.
                        Nếu False, trả về DataFrame ma trận RSI [T, N].
            rsi_col_name: Tên cột mới khi add_column=True.

        Returns:
            dict[str, pd.DataFrame] nếu add_column=True,
            hoặc pd.DataFrame [T, N] ma trận RSI nếu add_column=False.
        """
        if not aligned_dict:
            raise ValueError("Input aligned_dict cannot be empty.")

        if add_column:
            updated_dict: dict[str, pd.DataFrame] = {}
            for ticker, df in aligned_dict.items():
                df_copy = df.copy()
                rsi_series = self.calculate(df_copy)
                df_copy[rsi_col_name] = rsi_series
                updated_dict[ticker] = df_copy
            return updated_dict
        else:
            rsi_matrix: dict[str, pd.Series] = {}
            for ticker, df in aligned_dict.items():
                rsi_matrix[ticker] = self.calculate(df)
            result_df = pd.DataFrame(rsi_matrix)
            result_df.index.name = "Date"
            return result_df

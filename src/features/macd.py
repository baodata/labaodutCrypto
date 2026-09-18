"""
src/features/macd.py
Mô-đun tính toán chỉ báo MACD (Moving Average Convergence Divergence Engine).

Ticket: FEAT-004 (P1 - Thành viên A)
Sprint: 3

Mục đích:
1. Tính toán chỉ báo MACD theo chuẩn phân tích kỹ thuật của Gerald Appel:
   - Fast EMA (chu kỳ mặc định 12 phiên): EMA_12(P_t)
   - Slow EMA (chu kỳ mặc định 26 phiên): EMA_26(P_t)
   - MACD Line = EMA_12 - EMA_26
   - Signal Line = EMA_9(MACD Line)
   - MACD Histogram = MACD Line - Signal Line
2. Cung cấp các đại lượng phản ánh cả động lượng (momentum) và xu hướng (trend),
   giúp tác tử RL nhận diện các điểm đảo chiều và phân kỳ giá.
3. Hỗ trợ xử lý đa định dạng:
   - pd.Series: Chuỗi giá một cổ phiếu.
   - pd.DataFrame: Ma trận giá [T, N] hoặc bảng OHLCV.
   - dict[str, pd.DataFrame]: Danh mục toàn bộ 25 mã cổ phiếu.
4. Kiểm soát giai đoạn khởi động (warm-up) với tùy chọn fill_zero=True.
"""

from typing import Tuple, Union

import pandas as pd


def compute_macd(
    prices: Union[pd.Series, pd.DataFrame],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    fill_zero: bool = True,
) -> Tuple[Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame]]:
    """Tính toán MACD Line, Signal Line và Histogram từ chuỗi giá.

    Args:
        prices: Series giá hoặc DataFrame ma trận giá [T, N] (yêu cầu giá > 0).
        fast_period: Chu kỳ đường trung bình động nhanh (mặc định 12).
        slow_period: Chu kỳ đường trung bình động chậm (mặc định 26).
        signal_period: Chu kỳ đường tín hiệu (mặc định 9).
        fill_zero: Nếu True, điền 0.0 cho các giá trị NaN khởi động.

    Returns:
        Tuple 3 phần tử gồm: (macd_line, signal_line, histogram)
    """
    if fast_period >= slow_period:
        raise ValueError(f"fast_period ({fast_period}) phải nhỏ hơn slow_period ({slow_period}).")
    if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
        raise ValueError("Tất cả chu kỳ MACD phải là số nguyên dương lớn hơn 0.")

    # Tính toán EMA theo chuẩn pandas: ewm với span=N và adjust=False (đệ quy chuẩn Gerald Appel)
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    if fill_zero:
        macd_line = macd_line.fillna(0.0)
        signal_line = signal_line.fillna(0.0)
        histogram = histogram.fillna(0.0)

    return macd_line, signal_line, histogram


class MACDEngine:
    """Động cơ tính toán và quản lý chỉ báo MACD cho hệ thống định lượng H-MARL."""

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        fill_zero: bool = True,
        price_col: str = "close",
    ) -> None:
        """Khởi tạo MACDEngine.

        Args:
            fast_period: Chu kỳ EMA nhanh (mặc định 12).
            slow_period: Chu kỳ EMA chậm (mặc định 26).
            signal_period: Chu kỳ Signal line (mặc định 9).
            fill_zero: Có điền 0.0 cho NaN khởi động hay không.
            price_col: Tên cột giá dùng để tính toán (mặc định 'close').
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.fill_zero = fill_zero
        self.price_col = price_col.lower()

    def calculate(
        self,
        data: Union[pd.Series, pd.DataFrame],
    ) -> Tuple[Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame]]:
        """Tính toán MACD cho một chuỗi Series hoặc DataFrame.

        Args:
            data: Series giá, DataFrame bảng OHLCV, hoặc ma trận giá [T, N].

        Returns:
            Tuple: (macd_line, signal_line, histogram)
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
                return compute_macd(
                    data,
                    fast_period=self.fast_period,
                    slow_period=self.slow_period,
                    signal_period=self.signal_period,
                    fill_zero=self.fill_zero,
                )
        else:
            raise TypeError(f"Unsupported data type '{type(data).__name__}'. Expected pd.Series or pd.DataFrame.")

        return compute_macd(
            price_series,
            fast_period=self.fast_period,
            slow_period=self.slow_period,
            signal_period=self.signal_period,
            fill_zero=self.fill_zero,
        )

    def calculate_for_dict(
        self,
        aligned_dict: dict[str, pd.DataFrame],
        add_column: bool = True,
        macd_col_name: str = "macd",
        signal_col_name: str = "macd_signal",
        hist_col_name: str = "macd_hist",
    ) -> Union[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """Tính MACD cho toàn bộ danh mục tài sản trong dictionary.

        Args:
            aligned_dict: Dictionary {ticker: DataFrame OHLCV}.
            add_column: Nếu True, gắn trực tiếp các cột MACD vào từng DataFrame.
                        Nếu False, trả về dictionary chứa 3 ma trận [T, N] tương ứng:
                        {'macd': df, 'signal': df, 'hist': df}.

        Returns:
            dict[str, pd.DataFrame] cập nhật cột, hoặc dict chứa 3 ma trận [T, N].
        """
        if not aligned_dict:
            raise ValueError("Input aligned_dict cannot be empty.")

        if add_column:
            updated_dict: dict[str, pd.DataFrame] = {}
            for ticker, df in aligned_dict.items():
                df_copy = df.copy()
                macd, signal, hist = self.calculate(df_copy)
                df_copy[macd_col_name] = macd
                df_copy[signal_col_name] = signal
                df_copy[hist_col_name] = hist
                updated_dict[ticker] = df_copy
            return updated_dict
        else:
            macd_dict: dict[str, pd.Series] = {}
            signal_dict: dict[str, pd.Series] = {}
            hist_dict: dict[str, pd.Series] = {}

            for ticker, df in aligned_dict.items():
                macd, signal, hist = self.calculate(df)
                macd_dict[ticker] = macd
                signal_dict[ticker] = signal
                hist_dict[ticker] = hist

            return {
                "macd": pd.DataFrame(macd_dict),
                "signal": pd.DataFrame(signal_dict),
                "hist": pd.DataFrame(hist_dict),
            }


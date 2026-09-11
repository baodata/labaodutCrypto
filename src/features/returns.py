"""
src/features/returns.py
Mô-đun tính toán chuỗi lợi suất hàng ngày (Daily Returns Engine).

Ticket: FEAT-001 (P0 - Thành viên A)
Sprint: 2

Mục đích:
1. Chuyển đổi chuỗi giá không dừng (non-stationary price series P_t) thành chuỗi lợi suất
   dừng (stationary returns series R_t), làm đầu vào chuẩn tắc cho Reinforcement Learning.
2. Cung cấp 2 công thức tính toán cốt lõi:
   - Simple / Arithmetic Return: R_t = (P_t - P_{t-1}) / P_{t-1}
     (Chuẩn mực để tính toán giá trị danh mục và PnL thực tế).
   - Logarithmic Return: r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
     (Chuẩn mực trong phân tích định lượng và lý thuyết đồ thị tương quan).
3. Hỗ trợ xử lý linh hoạt đa định dạng:
   - pd.Series: Một chuỗi giá đơn lẻ.
   - pd.DataFrame: Bảng giá đa tài sản [T, N] hoặc bảng OHLCV của 1 mã.
   - dict[str, pd.DataFrame]: Từ điển dữ liệu toàn bộ vũ trụ 25 mã cổ phiếu.
4. Kiểm soát chặt chẽ giá trị khởi tạo tại t=0 (fill_zero=True điền 0.0, hoặc giữ np.nan).
5. Đảm bảo độ chính xác số học cao (sai số < 10^-8).
"""

from typing import Literal, Union, overload

import numpy as np
import pandas as pd


def compute_simple_returns(
    prices: Union[pd.Series, pd.DataFrame],
    fill_zero: bool = True,
) -> Union[pd.Series, pd.DataFrame]:
    """Tính toán chuỗi lợi suất đơn (Simple / Arithmetic Return): R_t = (P_t - P_{t-1}) / P_{t-1}.

    Args:
        prices: Series giá hoặc DataFrame bảng giá [T, N] (yêu cầu giá > 0).
        fill_zero: Nếu True, thay thế giá trị NaN ở dòng đầu tiên bằng 0.0.

    Returns:
        Series hoặc DataFrame chứa lợi suất đơn có cùng cấu trúc Index/Columns.
    """
    returns = prices.pct_change()
    if fill_zero:
        returns = returns.fillna(0.0)
    return returns


def compute_log_returns(
    prices: Union[pd.Series, pd.DataFrame],
    fill_zero: bool = True,
) -> Union[pd.Series, pd.DataFrame]:
    """Tính toán chuỗi lợi suất logarit (Logarithmic Return): r_t = ln(P_t / P_{t-1}).

    Args:
        prices: Series giá hoặc DataFrame bảng giá [T, N] (yêu cầu giá > 0).
        fill_zero: Nếu True, thay thế giá trị NaN ở dòng đầu tiên bằng 0.0.

    Returns:
        Series hoặc DataFrame chứa lợi suất log có cùng cấu trúc Index/Columns.
    """
    if (prices <= 0).any().any() if isinstance(prices, pd.DataFrame) else (prices <= 0).any():
        raise ValueError("Prices must be strictly positive (> 0) to compute logarithmic returns.")

    log_returns = np.log(prices / prices.shift(1))
    if fill_zero:
        log_returns = log_returns.fillna(0.0)
    return log_returns


class ReturnsEngine:
    """Động cơ tính toán và quản lý lợi suất cho hệ thống định lượng H-MARL."""

    def __init__(
        self,
        return_type: Literal["simple", "log"] = "simple",
        price_col: str = "close",
        fill_zero: bool = True,
    ) -> None:
        """Khởi tạo ReturnsEngine.

        Args:
            return_type: 'simple' cho R_t = (P_t - P_{t-1}) / P_{t-1},
                         'log' cho r_t = ln(P_t / P_{t-1}).
            price_col: Tên cột giá sử dụng để tính (mặc định 'close').
            fill_zero: Nếu True, điền 0.0 cho ngày đầu tiên.
        """
        if return_type not in ["simple", "log"]:
            raise ValueError(f"Invalid return_type '{return_type}'. Choose 'simple' or 'log'.")
        self.return_type = return_type
        self.price_col = price_col.lower()
        self.fill_zero = fill_zero

    def calculate(
        self,
        data: Union[pd.Series, pd.DataFrame],
    ) -> Union[pd.Series, pd.DataFrame]:
        """Tính toán lợi suất cho một chuỗi Series hoặc DataFrame.

        Nếu truyền vào DataFrame chứa nhiều cột OHLCV, hàm sẽ tự động tìm cột giá
        phù hợp (self.price_col). Nếu DataFrame là ma trận giá [T, N], hàm sẽ tính
        lợi suất cho toàn bộ N tài sản.

        Args:
            data: Series hoặc DataFrame.

        Returns:
            Series hoặc DataFrame lợi suất tương ứng.
        """
        if isinstance(data, pd.Series):
            target_series = data
        elif isinstance(data, pd.DataFrame):
            # Kiểm tra xem DataFrame là bảng OHLCV của 1 mã hay ma trận giá đa tài sản [T, N]
            cols_lower = [str(c).strip().lower() for c in data.columns]
            if self.price_col in cols_lower:
                # Bảng OHLCV 1 mã: chỉ tính trên cột giá chỉ định
                matching_col = data.columns[cols_lower.index(self.price_col)]
                target_series = data[matching_col]
            else:
                # Ma trận giá đa tài sản [T, N]
                if self.return_type == "simple":
                    return compute_simple_returns(data, fill_zero=self.fill_zero)
                return compute_log_returns(data, fill_zero=self.fill_zero)
        else:
            raise TypeError(f"Unsupported data type '{type(data).__name__}'. Expected pd.Series or pd.DataFrame.")

        if self.return_type == "simple":
            return compute_simple_returns(target_series, fill_zero=self.fill_zero)
        return compute_log_returns(target_series, fill_zero=self.fill_zero)

    def calculate_for_dict(
        self,
        aligned_dict: dict[str, pd.DataFrame],
        add_column: bool = True,
        return_col_name: str = "return",
    ) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
        """Tính toán lợi suất cho toàn bộ danh mục tài sản trong dictionary.

        Args:
            aligned_dict: Dictionary {ticker: DataFrame OHLCV}.
            add_column: Nếu True, gắn trực tiếp cột lợi suất mới vào từng DataFrame.
                        Nếu False, trả về một DataFrame ma trận lợi suất [T, N].
            return_col_name: Tên cột mới khi add_column=True.

        Returns:
            dict[str, pd.DataFrame] nếu add_column=True,
            hoặc pd.DataFrame [T, N] ma trận lợi suất nếu add_column=False.
        """
        if not aligned_dict:
            raise ValueError("Input aligned_dict cannot be empty.")

        if add_column:
            updated_dict: dict[str, pd.DataFrame] = {}
            for ticker, df in aligned_dict.items():
                df_copy = df.copy()
                ret_series = self.calculate(df_copy)
                df_copy[return_col_name] = ret_series
                updated_dict[ticker] = df_copy
            return updated_dict
        else:
            # Tạo ma trận lợi suất [T, N]
            returns_matrix: dict[str, pd.Series] = {}
            for ticker, df in aligned_dict.items():
                returns_matrix[ticker] = self.calculate(df)
            result_df = pd.DataFrame(returns_matrix)
            result_df.index.name = "Date"
            return result_df

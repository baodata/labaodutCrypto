"""
src/features/volume.py
Mô-đun chuẩn hóa khối lượng giao dịch (Volume Normalization Engine).

Ticket: FEAT-005 (P1 - Thành viên A)
Sprint: 3

Mục đích:
1. Chuẩn hóa dữ liệu khối lượng giao dịch (Volume) vốn có đặc thù lệch phải nghiêm trọng
   (heavy-tailed, right-skewed distribution) và dao động ở biên độ cực lớn
   (từ hàng trăm nghìn đến hàng trăm triệu cổ phiếu).
2. Quy trình biến đổi 2 tầng chuẩn định lượng:
   - Tầng 1: Log transform: V_log = ln(1 + Volume) = np.log1p(Volume).
     Đảm bảo khi Volume = 0 thì V_log = 0, nén phân phối về dạng gần chuẩn (Gaussian-like).
   - Tầng 2: Rolling Z-score: Z_t = (V_log,t - mean_W) / (std_W + eps)
     với cửa sổ trượt W ngày (mặc định W = 20 ngày giao dịch).
3. Nguyên tắc sinh tử: Cửa sổ trượt W chỉ sử dụng thông tin trong quá khứ [t-W+1 ... t],
   TUYỆT ĐỐI KHÔNG dùng trung bình toàn bộ chuỗi thời gian (Full-sample mean) để chống Look-ahead Bias.
4. Hỗ trợ xử lý đa định dạng:
   - pd.Series: Chuỗi khối lượng của 1 cổ phiếu.
   - pd.DataFrame: Ma trận khối lượng [T, N] hoặc bảng OHLCV.
   - dict[str, pd.DataFrame]: Danh mục 25 mã cổ phiếu.
"""

from typing import Union, TypeVar, cast

import numpy as np
import pandas as pd

T = TypeVar("T", pd.Series, pd.DataFrame)


def compute_log_volume(
    volume: T,
) -> T:
    """Tính toán logarit tự nhiên của khối lượng: ln(1 + Volume).

    Args:
        volume: Series hoặc DataFrame chứa khối lượng (yêu cầu volume >= 0).

    Returns:
        Series hoặc DataFrame đã áp dụng log1p.
    """
    if (volume.to_numpy() < 0).any():
        raise ValueError("Khối lượng giao dịch không thể âm (< 0).")
    return np.log1p(volume)  # type: ignore


def compute_normalized_volume(
    volume: T,
    window: int = 20,
    min_periods: int = 1,
    eps: float = 1e-8,
    fill_zero: bool = True,
) -> T:
    """Chuẩn hóa khối lượng qua 2 tầng: Log1p và Rolling Z-score W ngày.

    Args:
        volume: Series hoặc DataFrame chứa khối lượng giao dịch.
        window: Độ dài cửa sổ trượt (mặc định 20 phiên).
        min_periods: Số phiên quan sát tối thiểu trong cửa sổ (mặc định 1).
        eps: Hằng số chống chia cho 0 (mặc định 1e-8).
        fill_zero: Nếu True, điền 0.0 cho các giá trị NaN khởi động.

    Returns:
        Series hoặc DataFrame chứa khối lượng đã chuẩn hóa.
    """
    if window <= 1:
        raise ValueError(f"Window size phải lớn hơn 1, nhận được {window}.")

    log_vol = compute_log_volume(volume)

    # Tính trung bình và độ lệch chuẩn trượt chỉ dùng dữ liệu quá khứ
    rolling_mean = log_vol.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = log_vol.rolling(window=window, min_periods=min_periods).std(ddof=1).fillna(0.0)

    # Rolling Z-score
    norm_vol = (log_vol - rolling_mean) / (rolling_std + eps)

    if fill_zero:
        norm_vol = norm_vol.fillna(0.0)

    return norm_vol


class VolumeEngine:
    """Động cơ chuẩn hóa và quản lý đặc trưng khối lượng cho hệ thống định lượng."""

    def __init__(
        self,
        window: int = 20,
        min_periods: int = 1,
        eps: float = 1e-8,
        fill_zero: bool = True,
        volume_col: str = "volume",
    ) -> None:
        """Khởi tạo VolumeEngine.

        Args:
            window: Cửa sổ trượt Z-score (mặc định 20).
            min_periods: Số phiên tối thiểu (mặc định 1).
            eps: Epsilon chống chia cho 0 (1e-8).
            fill_zero: Điền 0.0 cho NaN hay không.
            volume_col: Tên cột volume (mặc định 'volume').
        """
        self.window = window
        self.min_periods = min_periods
        self.eps = eps
        self.fill_zero = fill_zero
        self.volume_col = volume_col.lower()

    def calculate(
        self,
        data: Union[pd.Series, pd.DataFrame],
    ) -> Union[pd.Series, pd.DataFrame]:
        """Tính toán khối lượng chuẩn hóa cho Series hoặc DataFrame.

        Args:
            data: Series khối lượng, DataFrame bảng OHLCV, hoặc ma trận khối lượng [T, N].

        Returns:
            Series hoặc DataFrame khối lượng đã chuẩn hóa.
        """
        if isinstance(data, pd.Series):
            vol_series = data
        elif isinstance(data, pd.DataFrame):
            cols_lower = [str(c).strip().lower() for c in data.columns]
            if self.volume_col in cols_lower:
                matching_col = data.columns[cols_lower.index(self.volume_col)]
                vol_series = data[matching_col]
            else:
                # Toàn bộ DataFrame là ma trận khối lượng đa tài sản [T, N]
                return compute_normalized_volume(
                    data,
                    window=self.window,
                    min_periods=self.min_periods,
                    eps=self.eps,
                    fill_zero=self.fill_zero,
                )
        else:
            raise TypeError(f"Unsupported data type '{type(data).__name__}'. Expected pd.Series or pd.DataFrame.")

        return compute_normalized_volume(
            vol_series,
            window=self.window,
            min_periods=self.min_periods,
            eps=self.eps,
            fill_zero=self.fill_zero,
        )

    def calculate_for_dict(
        self,
        aligned_dict: dict[str, pd.DataFrame],
        add_column: bool = True,
        vol_col_name: str = "volume_norm",
    ) -> Union[dict[str, pd.DataFrame], pd.DataFrame]:
        """Tính chuẩn hóa khối lượng cho toàn bộ danh mục tài sản trong dictionary.

        Args:
            aligned_dict: Dictionary {ticker: DataFrame OHLCV}.
            add_column: Nếu True, gắn trực tiếp cột vào từng DataFrame.
                        Nếu False, trả về DataFrame ma trận khối lượng chuẩn hóa [T, N].

        Returns:
            dict[str, pd.DataFrame] cập nhật, hoặc pd.DataFrame ma trận [T, N].
        """
        if not aligned_dict:
            raise ValueError("Input aligned_dict cannot be empty.")

        if add_column:
            updated_dict: dict[str, pd.DataFrame] = {}
            for ticker, df in aligned_dict.items():
                df_copy = df.copy()
                norm_vol = self.calculate(df_copy)
                df_copy[vol_col_name] = norm_vol
                updated_dict[ticker] = df_copy
            return updated_dict
        else:
            norm_matrix: dict[str, pd.Series] = {}
            for ticker, df in aligned_dict.items():
                norm_matrix[ticker] = cast(pd.Series, self.calculate(df))
            result_df = pd.DataFrame(norm_matrix)
            result_df.index.name = "Date"
            return result_df


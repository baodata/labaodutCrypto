"""
src/graph/correlation.py
Module tính toán ma trận tương quan trượt (Rolling Correlation).

- Ticket: GRAPH-001 (Sprint 2 - Member B)

FILE NÀY ĐỂ LÀM GÌ?
- Input: Dữ liệu tỷ suất lợi nhuận (Return) của các tài sản theo thời gian.
- Áp dụng một tính toán sự tương quan trong cỡ 20 ngày (Pearson Correlation) 
  giữa biến động giá của từng cặp cổ phiếu.
- Output: Trả về các ma trận tương quan tương ứng với mỗi ngày, cho biết cặp cổ phiếu nào 
  đang diễn biến giống nhau.
"""

import pandas as pd
import numpy as np

def compute_rolling_correlation(returns_df: pd.DataFrame, window_size: int = 20) -> pd.DataFrame:
    """
    Tính ma trận tương quan trượt cho các tài sản.
    
    Args:
        returns_df: DataFrame với Index là Datetime, Columns là danh sách Ticker (AAPL, MSFT...), 
                    Values là Daily Return.
        window_size: Kích thước cửa sổ tính toán (số phiên giao dịch, mặc định 20).
        
    Returns:
        DataFrame chứa ma trận tương quan trượt.
        Index sẽ là MultiIndex: (Date, Ticker), Columns là danh sách Ticker.
    """
    # rolling.corr(): tính ma trận tương quan.
    # Kết quả sẽ tự động tạo MultiIndex (Date, Ticker1) -> Ticker2
    rolling_corr = returns_df.rolling(window=window_size).corr()
    
    # Ở những ngày đầu (chưa đủ window_size) dữ liệu sẽ là NaN, ta giữ nguyên.
    return rolling_corr

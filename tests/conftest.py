"""
cái này tạo data giả lập để test
"""

import pytest
import pandas as pd
import numpy as np

@pytest.fixture(scope="session")
def synthetic_returns_df():
    """Tạo DataFrame Returns giả lập cho 3 cổ phiếu trong 50 ngày (Phục vụ GRAPH-001)"""
    np.random.seed(42)
    # 50 ngày làm việc (Business days)
    dates = pd.date_range(start="2024-01-01", periods=50, freq="B")
    tickers = ["AAPL", "MSFT", "NVDA"]
    
    # Tạo lợi nhuận giả ngẫu nhiên có phân phối chuẩn
    returns_data = np.random.normal(loc=0.001, scale=0.02, size=(50, 3))
    
    # Ép MSFT và AAPL có tương quan cao (bằng cách cho MSFT = AAPL + nhiễu nhỏ)
    returns_data[:, 1] = returns_data[:, 0] + np.random.normal(0, 0.005, size=50)
    
    df = pd.DataFrame(returns_data, index=dates, columns=tickers)
    return df

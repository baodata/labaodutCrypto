"""
tests/test_data_leakage.py
Ticket: LEAK-001 (Sprint 3 - Member B)
Mục đích: Kiểm toán chống rò rỉ dữ liệu tương lai (Look-ahead bias) vào các phép tính của ngày hiện tại.
"""
import pandas as pd
import numpy as np
import pytest
from src.features.returns import compute_simple_returns
from src.features.volatility import compute_rolling_volatility
from src.features.rsi import compute_rsi

class TestDataLeakage:
    def test_no_lookahead_bias_in_returns(self):
        """Đảm bảo việc tính Tỷ suất lợi nhuận ngày t không dùng giá ngày t+1"""
        # Bảng giá 5 ngày
        prices = pd.DataFrame({
            'AAPL': [100.0, 102.0, 101.0, 105.0, 110.0],
            'MSFT': [200.0, 205.0, 202.0, 210.0, 215.0]
        }, index=pd.date_range("2024-01-01", periods=5))
        
        # Kịch bản 1: Chỉ đưa cho hàm 3 ngày đầu tiên (Tương lai bị giấu đi hoàn toàn)
        returns_3_days = compute_simple_returns(prices.iloc[:3], fill_zero=True)
        
        # Kịch bản 2: Đưa cho hàm cả 5 ngày (Có tương lai)
        returns_5_days = compute_simple_returns(prices, fill_zero=True)
        
        # Nếu không có Leakage, kết quả của 3 ngày đầu trong 2 kịch bản phải KHỚP NHAU 100%
        pd.testing.assert_frame_equal(returns_3_days.iloc[:3], returns_5_days.iloc[:3])

    def test_no_lookahead_bias_in_volatility(self):
        """Đảm bảo Volatility không lấy dữ liệu tương lai để tính độ lệch chuẩn"""
        returns = pd.DataFrame({
            'AAPL': np.random.randn(30)
        })
        
        vol_15_days = compute_rolling_volatility(returns.iloc[:15], window=10)
        vol_30_days = compute_rolling_volatility(returns, window=10)
        
        pd.testing.assert_frame_equal(vol_15_days.iloc[:15], vol_30_days.iloc[:15])

    def test_no_lookahead_bias_in_rsi(self):
        """Đảm bảo RSI của ngày t chỉ dùng giá từ t trở về quá khứ"""
        prices = pd.DataFrame({
            'AAPL': np.linspace(100, 150, 30)
        })
        
        rsi_20_days = compute_rsi(prices.iloc[:20], period=14)
        rsi_30_days = compute_rsi(prices, period=14)
        
        pd.testing.assert_frame_equal(rsi_20_days.iloc[:20], rsi_30_days.iloc[:20])

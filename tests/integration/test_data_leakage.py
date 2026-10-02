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

    def test_no_lookahead_bias_in_macd(self):
        """Đảm bảo tính toán MACD không rò rỉ dữ liệu tương lai (LEAK-002)"""
        from src.features.macd import MACDEngine
        prices = pd.DataFrame({
            'AAPL': np.linspace(100, 200, 50)
        })
        engine = MACDEngine(fast_period=12, slow_period=26, signal_period=9)
        
        # Scenario 1: Calculate with 30 days
        macd_30, sig_30, hist_30 = engine.calculate(prices.iloc[:30])
        
        # Scenario 2: Calculate with 50 days
        macd_50, sig_50, hist_50 = engine.calculate(prices)
        
        # Should be exactly equal up to day 30
        pd.testing.assert_frame_equal(macd_30.iloc[:30], macd_50.iloc[:30])
        pd.testing.assert_frame_equal(sig_30.iloc[:30], sig_50.iloc[:30])
        pd.testing.assert_frame_equal(hist_30.iloc[:30], hist_50.iloc[:30])

    def test_no_lookahead_bias_in_volume_norm(self):
        """Đảm bảo chuẩn hóa Volume (Z-score trượt) không nhìn vào tương lai (LEAK-002)"""
        from src.features.volume import VolumeEngine
        volumes = pd.DataFrame({
            'AAPL': np.random.randint(1000, 5000, 50)
        })
        engine = VolumeEngine(window=20)
        
        # Scenario 1: Calculate with 30 days
        norm_30 = engine.calculate(volumes.iloc[:30])
        
        # Scenario 2: Calculate with 50 days
        norm_50 = engine.calculate(volumes)
        
        # Should be exactly equal up to day 30
        pd.testing.assert_frame_equal(norm_30.iloc[:30], norm_50.iloc[:30])

    def test_scaler_no_future_leakage(self):
        """Đảm bảo Scaler chỉ được học trên tập Train, không chạm vào Val/Test (LEAK-002)"""
        from src.features.scaler import MarketFeatureScaler
        
        # Giả lập data
        train_df = pd.DataFrame({'feature1': np.random.randn(100) * 2 + 5})
        val_df = pd.DataFrame({'feature1': np.random.randn(50) * 10 + 50}) # Data phân phối khác
        
        scaler = MarketFeatureScaler()
        scaler.fit(train_df, feature_names=['feature1'])
        
        # Mean/std phải là của tập train, tuyệt đối không bị ảnh hưởng bởi val_df lớn
        assert np.isclose(scaler.means['feature1'], train_df['feature1'].mean())
        assert np.isclose(scaler.stds['feature1'], train_df['feature1'].std() + scaler.eps)
        
        # Việc biến đổi Val phải dùng tham số của Train
        val_transformed = scaler.transform(val_df)
        expected_val = (val_df - scaler.means['feature1']) / scaler.stds['feature1']
        expected_val = np.clip(expected_val, scaler.clip_range[0], scaler.clip_range[1])
        
        pd.testing.assert_frame_equal(val_transformed, expected_val)

    def test_split_temporal_strictness(self):
        """Đảm bảo việc chia Train/Val/Test tuyệt đối theo thời gian, không rò rỉ (LEAK-002)"""
        from src.data.split import TemporalSplitter
        
        dates = pd.date_range("2020-01-01", "2024-12-31")
        df = pd.DataFrame({'price': np.random.randn(len(dates))}, index=dates)
        
        splitter = TemporalSplitter(
            train_end="2021-12-31",
            val_start="2022-01-01",
            val_end="2023-12-31",
            test_start="2024-01-01"
        )
        
        result = splitter.split_dataframe(df)
        
        max_train = result.train.index.max()
        min_val = result.val.index.min()
        max_val = result.val.index.max()
        min_test = result.test.index.min()
        
        assert max_train < min_val, f"Rò rỉ Train -> Val: {max_train} >= {min_val}"
        assert max_val < min_test, f"Rò rỉ Val -> Test: {max_val} >= {min_test}"

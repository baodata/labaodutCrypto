import pytest
import numpy as np
from src.env.risk import RiskCalculator

class TestRiskCalculator:
    def test_single_asset_volatility(self):
        """Test rủi ro khi dồn 100% vào 1 tài sản."""
        calc = RiskCalculator(window=60)
        
        # Lịch sử 1 tài sản: Phương sai của nó
        returns_history = np.random.randn(100, 1) * 0.02
        weights = np.array([1.0, 0.0]) # 100% Asset, 0% Cash
        
        expected_std = np.std(returns_history[-60:], ddof=1)
        portfolio_vol = calc.compute_portfolio_volatility(weights, returns_history)
        
        assert np.isclose(portfolio_vol, expected_std)

    def test_all_cash_zero_risk(self):
        """Test khi cầm 100% tiền mặt thì rủi ro = 0."""
        calc = RiskCalculator()
        
        returns_history = np.random.randn(100, 5) # 5 tài sản cực kỳ biến động
        weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) # 100% Cash
        
        portfolio_vol = calc.compute_portfolio_volatility(weights, returns_history)
        
        assert np.isclose(portfolio_vol, 0.0)

    def test_diversification_benefit(self):
        """Đảm bảo đa dạng hóa (diversification) làm giảm rủi ro."""
        calc = RiskCalculator()
        
        # Tạo 2 tài sản hoàn toàn độc lập (correlation gần bằng 0)
        np.random.seed(42)
        asset1 = np.random.randn(100, 1) * 0.02
        asset2 = np.random.randn(100, 1) * 0.02
        returns_history = np.hstack([asset1, asset2])
        
        # Rủi ro nếu dồn 100% vào tài sản 1
        vol_all_asset1 = calc.compute_portfolio_volatility(np.array([1.0, 0.0, 0.0]), returns_history)
        
        # Rủi ro nếu chia đều 50-50
        vol_diversified = calc.compute_portfolio_volatility(np.array([0.5, 0.5, 0.0]), returns_history)
        
        # Đa dạng hóa phải giảm thiểu rủi ro
        assert vol_diversified < vol_all_asset1

    def test_not_enough_data(self):
        """Test trường hợp lịch sử chưa đủ 2 ngày."""
        calc = RiskCalculator()
        returns_history = np.array([[0.01, 0.02]]) # T=1
        weights = np.array([0.5, 0.5, 0.0])
        
        # Ma trận Covariance sẽ = 0, nên rủi ro = 0
        portfolio_vol = calc.compute_portfolio_volatility(weights, returns_history)
        assert portfolio_vol == 0.0

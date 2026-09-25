import pytest
import numpy as np
from src.env.returns import PortfolioReturnEngine

class TestPortfolioReturnEngine:
    def test_gross_return_all_assets(self):
        """Kiểm thử khi dồn 100% vốn vào cổ phiếu, không giữ tiền mặt."""
        engine = PortfolioReturnEngine(risk_free_rate_annual=0.0)
        
        # N=3, weights = [N+1]
        weights = np.array([0.5, 0.3, 0.2, 0.0]) # Tiền mặt = 0
        asset_returns = np.array([0.01, -0.02, 0.05])
        
        # Kỳ vọng: 0.5*0.01 + 0.3*-0.02 + 0.2*0.05 + 0 = 0.005 - 0.006 + 0.010 = 0.009
        gross_ret = engine.compute_gross_return(weights, asset_returns)
        assert np.isclose(gross_ret, 0.009)

    def test_gross_return_all_cash(self):
        """Kiểm thử khi giữ 100% tiền mặt."""
        # 25.2% lãi năm => 0.1% lãi ngày (để dễ tính)
        engine = PortfolioReturnEngine(risk_free_rate_annual=0.252, trading_days_per_year=252)
        
        weights = np.array([0.0, 0.0, 0.0, 1.0]) # 100% Tiền mặt
        asset_returns = np.array([-0.5, -0.9, -0.99]) # Cổ phiếu sập sàn
        
        # Gross return phải bằng đúng lãi suất tiền mặt ngày (0.252/252 = 0.001)
        gross_ret = engine.compute_gross_return(weights, asset_returns)
        assert np.isclose(gross_ret, 0.001)

    def test_mixed_portfolio(self):
        """Kiểm thử danh mục hỗn hợp có cả lãi cổ phiếu và lãi tiền mặt."""
        engine = PortfolioReturnEngine(risk_free_rate_annual=0.0252, trading_days_per_year=252) # Lãi ngày = 0.0001
        
        weights = np.array([0.4, 0.1, 0.5]) # Cash = 0.5
        asset_returns = np.array([0.1, -0.1])
        
        # Cổ phiếu: 0.4*0.1 + 0.1*-0.1 = 0.04 - 0.01 = 0.03
        # Tiền mặt: 0.5 * 0.0001 = 0.00005
        # Tổng: 0.03005
        gross_ret = engine.compute_gross_return(weights, asset_returns)
        assert np.isclose(gross_ret, 0.03005)
        
    def test_shape_mismatch_raises_error(self):
        """Kiểm tra báo lỗi khi truyền sai kích thước."""
        engine = PortfolioReturnEngine()
        weights = np.array([0.5, 0.5])
        asset_returns = np.array([0.1, 0.2]) # Thiếu cash node trong weights
        
        with pytest.raises(ValueError):
            engine.compute_gross_return(weights, asset_returns)

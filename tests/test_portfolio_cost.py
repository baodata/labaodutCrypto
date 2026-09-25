import pytest
import numpy as np
from src.env.cost import TransactionCostEngine

class TestTransactionCostEngine:
    def test_drifted_weights_sum_to_one(self):
        """Đảm bảo tỷ trọng sau khi trôi vẫn giữ nguyên tổng = 1.0"""
        engine = TransactionCostEngine()
        
        old_weights = np.array([0.4, 0.4, 0.2]) # N=2, Cash=0.2
        asset_returns = np.array([0.10, -0.05]) # Tài sản 1 tăng, tài sản 2 giảm
        cash_return = 0.001
        
        # R_p = 0.4*0.1 + 0.4*-0.05 + 0.2*0.001 = 0.04 - 0.02 + 0.0002 = 0.0202
        portfolio_gross_return = 0.0202
        
        drifted = engine.compute_drifted_weights(old_weights, asset_returns, cash_return, portfolio_gross_return)
        
        # Tài sản 1 phải nở ra vì tăng 10%
        assert drifted[0] > 0.4
        # Tài sản 2 phải teo lại vì giảm 5%
        assert drifted[1] < 0.4
        
        # Tổng tỷ trọng trôi vẫn phải = 1
        assert np.isclose(np.sum(drifted), 1.0)

    def test_zero_turnover_zero_cost(self):
        """Nếu AI chấp nhận tỷ trọng trôi tự nhiên (không giao dịch), phí = 0."""
        engine = TransactionCostEngine(cost_rate=0.001)
        
        drifted_weights = np.array([0.45, 0.35, 0.20])
        # AI nói "Cứ để thế"
        target_weights = np.array([0.45, 0.35, 0.20])
        
        cost = engine.compute_cost_rate(target_weights, drifted_weights)
        assert np.isclose(cost, 0.0)

    def test_full_reshuffle_cost(self):
        """Nếu xáo trộn toàn bộ danh mục, phí bị trừ phải cao."""
        engine = TransactionCostEngine(cost_rate=0.001)
        
        # Từ 100% tài sản 1 -> 100% tài sản 2
        drifted_weights = np.array([1.0, 0.0, 0.0])
        target_weights = np.array([0.0, 1.0, 0.0])
        
        # Turnover = |0-1| + |1-0| + |0-0| = 2.0
        # Cost = 0.001 * 2.0 = 0.002 (0.2%)
        cost = engine.compute_cost_rate(target_weights, drifted_weights)
        assert np.isclose(cost, 0.002)

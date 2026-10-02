import pytest
import numpy as np
from src.env.rebalance import RebalanceEngine

class TestRebalanceEngine:
    def test_project_valid_weights(self):
        """Nếu weights đã chuẩn rồi, hàm không được làm méo mó giá trị."""
        weights = np.array([0.2, 0.3, 0.5])
        projected = RebalanceEngine.project_weights(weights)
        
        np.testing.assert_array_almost_equal(projected, weights)
        assert np.isclose(np.sum(projected), 1.0)

    def test_project_negative_weights(self):
        """Kiểm tra chức năng Long-only (Triệt tiêu số âm)."""
        weights = np.array([1.5, -0.5, 0.5])
        projected = RebalanceEngine.project_weights(weights)
        
        # -0.5 sẽ bị biến thành 0
        # Còn lại [1.5, 0, 0.5] có tổng = 2.0
        # Chuẩn hóa về 1.0 sẽ là [0.75, 0, 0.25]
        expected = np.array([0.75, 0.0, 0.25])
        
        np.testing.assert_array_almost_equal(projected, expected)
        assert np.isclose(np.sum(projected), 1.0)

    def test_project_all_zero_or_negative_fallback(self):
        """Trường hợp xấu nhất: AI bị điên xuất toàn số âm. Hệ thống phải ép ra Tiền mặt."""
        weights = np.array([-1.0, -2.0, -3.0, -0.1])
        projected = RebalanceEngine.project_weights(weights)
        
        # Mong đợi: [0, 0, 0, 1.0] (Node cuối là Cash)
        expected = np.array([0.0, 0.0, 0.0, 1.0])
        
        np.testing.assert_array_almost_equal(projected, expected)
        assert np.isclose(np.sum(projected), 1.0)

    def test_project_exceed_one(self):
        """Nếu tổng weights > 1, phải tự ép lại thành 1."""
        weights = np.array([2.0, 2.0])
        projected = RebalanceEngine.project_weights(weights)
        
        expected = np.array([0.5, 0.5])
        np.testing.assert_array_almost_equal(projected, expected)
        assert np.isclose(np.sum(projected), 1.0)

import pytest
import numpy as np
from src.env.drawdown import DrawdownTracker

class TestDrawdownTracker:
    def test_drawdown_reset(self):
        """Đảm bảo reset khởi tạo giá trị chính xác."""
        tracker = DrawdownTracker()
        tracker.reset(100.0)
        assert tracker.peak_value == 100.0
        assert tracker.max_drawdown == 0.0
        
        with pytest.raises(ValueError):
            tracker.reset(-10)

    def test_drawdown_uptrend(self):
        """Khi danh mục liên tục lập đỉnh mới, Drawdown phải luôn = 0."""
        tracker = DrawdownTracker()
        tracker.reset(100.0)
        
        dd1 = tracker.update(110.0) # Tăng 10%
        assert np.isclose(dd1, 0.0)
        assert tracker.peak_value == 110.0
        assert tracker.max_drawdown == 0.0
        
        dd2 = tracker.update(120.0) # Tiếp tục tăng
        assert np.isclose(dd2, 0.0)
        assert tracker.peak_value == 120.0

    def test_drawdown_downtrend_and_recovery(self):
        """Kiểm tra chức năng tính Drawdown khi vốn sụt giảm và phục hồi."""
        tracker = DrawdownTracker()
        tracker.reset(100.0)
        
        # Sụt giảm lần 1 (100 -> 90) => Giảm 10%
        dd1 = tracker.update(90.0)
        assert np.isclose(dd1, -0.10)
        assert np.isclose(tracker.max_drawdown, -0.10)
        assert tracker.peak_value == 100.0
        
        # Sụt giảm sâu hơn (100 -> 80) => Giảm 20%
        dd2 = tracker.update(80.0)
        assert np.isclose(dd2, -0.20)
        assert np.isclose(tracker.max_drawdown, -0.20)
        
        # Phục hồi một nửa (100 -> 90) => DD hiện tại là -10%, nhưng MDD vẫn phải giữ -20%
        dd3 = tracker.update(90.0)
        assert np.isclose(dd3, -0.10)
        assert np.isclose(tracker.max_drawdown, -0.20)
        assert tracker.peak_value == 100.0
        
        # Lập đỉnh mới (100 -> 150) => DD hiện tại = 0, MDD = -20%, Đỉnh = 150
        dd4 = tracker.update(150.0)
        assert np.isclose(dd4, 0.0)
        assert np.isclose(tracker.max_drawdown, -0.20)
        assert tracker.peak_value == 150.0
        
        # Sụt từ đỉnh mới (150 -> 75) => Giảm 50%, MDD phá kỷ lục cũ (-20%)
        dd5 = tracker.update(75.0)
        assert np.isclose(dd5, -0.50)
        assert np.isclose(tracker.max_drawdown, -0.50)
        assert tracker.peak_value == 150.0

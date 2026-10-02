import pytest
import numpy as np
from src.env.reward import RewardEngine

class TestRewardEngine:
    def test_standard_reward(self):
        """Kiểm tra Reward cơ bản (beta=1, scaling=1)."""
        engine = RewardEngine(beta=1.0, scaling_factor=1.0)
        
        gross = 0.05
        cost = 0.01
        
        # Reward = 0.05 - 1.0*0.01 = 0.04
        reward = engine.compute_reward(gross, cost)
        net = engine.compute_net_return(gross, cost)
        
        assert np.isclose(reward, 0.04)
        assert np.isclose(net, 0.04)

    def test_reward_shaping_beta(self):
        """Kiểm tra Reward Shaping (phạt phí nặng) với beta > 1."""
        # beta = 3.0: Phạt phí gấp 3 lần để răn đe AI
        engine = RewardEngine(beta=3.0, scaling_factor=1.0)
        
        gross = 0.05
        cost = 0.01
        
        # Reward = 0.05 - 3.0*0.01 = 0.02
        reward = engine.compute_reward(gross, cost)
        
        # Nhưng tài khoản thật chỉ mất 0.01 => Net return vẫn là 0.04
        net = engine.compute_net_return(gross, cost)
        
        assert np.isclose(reward, 0.02)
        assert np.isclose(net, 0.04)

    def test_reward_scaling(self):
        """Kiểm tra hệ số phóng đại Reward (tránh Vanishing Gradient)."""
        engine = RewardEngine(beta=1.0, scaling_factor=100.0)
        
        gross = 0.012
        cost = 0.002
        
        # Raw reward = 0.010
        # Scaled = 0.010 * 100 = 1.0
        reward = engine.compute_reward(gross, cost)
        
        # Net return (thực tế) vẫn là 0.010
        net = engine.compute_net_return(gross, cost)
        
        assert np.isclose(reward, 1.0)
        assert np.isclose(net, 0.010)
        
    def test_negative_reward(self):
        """Khi lỗ và mất phí, reward phải âm sâu."""
        engine = RewardEngine(beta=2.0, scaling_factor=10.0)
        
        gross = -0.03 # Lỗ 3%
        cost = 0.01   # Mất thêm 1% phí do trade bậy
        
        # Raw = -0.03 - 2.0*0.01 = -0.05
        # Scaled = -0.05 * 10 = -0.5
        reward = engine.compute_reward(gross, cost)
        net = engine.compute_net_return(gross, cost)
        
        assert np.isclose(reward, -0.5)
        assert np.isclose(net, -0.04) # Thực tế lỗ 4%

"""
src/env/portfolio_env.py
Môi trường Giao dịch Chứng khoán (Trading Environment) chuẩn Gymnasium.

Ticket: ENV-001 -> ENV-007 (Sprint 4 - Thành viên A)

Mục đích:
1. Mô phỏng Sàn giao dịch thực tế (Có phí giao dịch, trượt giá).
2. Quản lý Trạng thái tài khoản (Số dư, Tỷ trọng, Max Drawdown).
3. Đóng vai trò làm Trọng tài phân xử Lãi/Lỗ (Reward) cho mạng GNN của B.
"""
import numpy as np
import yaml
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any

class PortfolioEnv(gym.Env):
    """Môi trường tối ưu hóa danh mục đầu tư đa tài sản."""
    
    def __init__(self, market_tensor, config_path: str = "configs/env.yaml"):
        super(PortfolioEnv, self).__init__()
        
        # 1. Đọc cấu hình luật chơi từ env.yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['environment']
            
        self.initial_balance = self.config['initial_balance']
        self.fee = self.config['transaction_fee_pct'] + self.config['slippage_pct']
        
        # Dữ liệu từ Pipeline của A
        # market_tensor.tensor có shape [T, N, F]
        self.features = market_tensor.tensor 
        self.T, self.N, self.F = self.features.shape
        
        # Trích xuất cột Returns (Giả định nằm ở Feature 0) để tính lãi/lỗ
        self.returns = self.features[:, :, 0]
        
        # 2. Định nghĩa Không gian Hành động & Quan sát
        # Action: Tỷ trọng danh mục [N], giá trị từ 0 đến 1
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.N,), dtype=np.float32)
        # Observation: Mảng đặc trưng [N, F]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.N, self.F), dtype=np.float32)
        
        self.reset()
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Thiết lập lại sàn giao dịch về ngày đầu tiên (ENV-001)"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.portfolio_value = self.initial_balance
        self.peak_value = self.initial_balance
        self.weights = np.zeros(self.N)  # Khởi đầu cầm 100% tiền mặt, 0% cổ phiếu
        
        # Theo dõi rủi ro
        self.max_drawdown = 0.0
        
        obs = self._get_obs()
        info = self._get_info()
        return obs, info
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Khớp lệnh và dịch chuyển thời gian (ENV-002 -> ENV-007)
        Args:
            action: Tỷ trọng danh mục do GNN/Actor đề xuất (Đã qua Softmax)
        """
        # 1. Tính toán chi phí giao dịch (ENV-004)
        # Phí bị trừ khi tỷ trọng mua/bán thay đổi so với ngày hôm qua
        transaction_cost_pct = np.sum(np.abs(action - self.weights)) * self.fee
        
        # Cập nhật danh mục mới
        self.weights = action
        
        # 2. Tính lợi nhuận sinh ra trong ngày hôm nay (ENV-002)
        # Lợi nhuận danh mục = Tổng (Tỷ trọng * Lợi nhuận từng cổ)
        day_return = self.returns[self.current_step]
        portfolio_return = np.sum(self.weights * day_return)
        
        # Lợi nhuận thực tế sau khi trừ phí
        net_return = portfolio_return - transaction_cost_pct
        
        # 3. Cập nhật Tài khoản & Tính Max Drawdown (ENV-005)
        self.portfolio_value *= (1 + net_return)
        
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
            
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # 4. Tính Phần thưởng (Reward V1 - ENV-007)
        # Thưởng dựa trên lợi nhuận, trừ đi hình phạt nếu Drawdown quá lớn
        reward = (net_return * self.config['reward_scaling']) - (drawdown * 0.1)
        
        # 5. Dịch chuyển thời gian
        self.current_step += 1
        terminated = bool(self.current_step >= self.T - 1)
        truncated = False
        
        if self.portfolio_value <= 0:  # Cháy tài khoản
            terminated = True
            reward = -100.0
            
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, float(reward), terminated, truncated, info
        
    def _get_obs(self) -> np.ndarray:
        return self.features[self.current_step]
        
    def _get_info(self) -> Dict[str, Any]:
        return {
            'portfolio_value': self.portfolio_value,
            'max_drawdown': self.max_drawdown
        }

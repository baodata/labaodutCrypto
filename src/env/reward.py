"""
src/env/reward.py
Ticket: ENV-007 (Sprint 4 - Thành viên A)

Bộ máy thiết kế Hàm phần thưởng (Reward Function).
Đóng vai trò làm giáo viên chấm điểm cho PPO Agent.
"""

class RewardEngine:
    """Bộ xử lý Phần thưởng cho AI."""
    
    def __init__(self, beta: float = 1.0, scaling_factor: float = 1.0):
        """
        Khởi tạo RewardEngine.
        
        Args:
            beta: Hệ số phạt phí giao dịch (Mặc định 1.0 là phạt đúng phí thật.
                  Nếu > 1.0 thì đóng vai trò Reward Shaping để ép AI bớt trade).
            scaling_factor: Hệ số nhân phóng to Reward (ví dụ 100.0) 
                            giúp Gradient của PPO không bị vanishing.
        """
        self.beta = float(beta)
        self.scaling_factor = float(scaling_factor)
        
    def compute_reward(self, gross_return: float, transaction_cost: float) -> float:
        """
        Tính phần thưởng cho PPO Agent.
        Công thức: Reward_t = (R_p,t - beta * TC_t) * scaling_factor
        
        Args:
            gross_return (R_p,t): Lợi nhuận gộp.
            transaction_cost (TC_t): Phần trăm phí giao dịch bị mất do xáo trộn.
            
        Returns:
            Điểm phần thưởng (Float) đã scale.
        """
        raw_reward = gross_return - (self.beta * transaction_cost)
        return float(raw_reward * self.scaling_factor)
        
    def compute_net_return(self, gross_return: float, transaction_cost: float) -> float:
        """
        Tính Lợi nhuận ròng thực tế (Net Return).
        Chỉ dùng để mô phỏng sự tăng giảm Đô la thật sự trong tài khoản.
        (Tuyệt đối KHÔNG bị bóp méo bởi beta hay scaling_factor).
        
        Args:
            gross_return: Lợi nhuận gộp.
            transaction_cost: Phí giao dịch thực.
            
        Returns:
            Lợi nhuận ròng (Net Return) = R_p,t - TC_t
        """
        return float(gross_return - transaction_cost)

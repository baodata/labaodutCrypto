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
        
    def compute_reward(
        self,
        gross_return: float,
        transaction_cost: float,
        benchmark_return: float = 0.0,
    ) -> float:
        """
        Tính phần thưởng cho PPO Agent.
        Công thức: Reward_t = (R_p,t - baseline_t - beta * TC_t) * scaling_factor
        
        Args:
            gross_return (R_p,t): Lợi nhuận gộp.
            transaction_cost (TC_t): Phần trăm phí giao dịch bị mất do xáo trộn.
            benchmark_return: Lợi nhuận tham chiếu bị trừ khỏi reward, không ảnh hưởng
                đến lợi nhuận tài khoản thực tế.
            
        Returns:
            Điểm phần thưởng (Float) đã scale.
        """
        raw_reward = gross_return - benchmark_return - (self.beta * transaction_cost)
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


class DifferentialSharpeReward:
    """QNT-014: DSR computed with PRE-update EMA moments of net excess.

    Reference is explicit at the environment: risk-free or equal-weight active.
    State is [A, B, count], with readiness deterministically derived from count.
    Degenerate variance emits zero; optional clipping is applied after DSR.
    """
    def __init__(self, eta=.01, epsilon=1e-8, warmup=20, clip=10., scaling=1.):
        import numpy as np
        if not 0 < eta <= 1 or epsilon <= 0 or warmup < 2 or clip <= 0 or not np.isfinite([eta,epsilon,clip,scaling]).all():
            raise ValueError("Invalid DSR configuration.")
        self.eta, self.epsilon, self.warmup, self.clip, self.scaling = eta, epsilon, warmup, clip, scaling
        self.reset()

    def reset(self):
        self.a = self.b = 0.
        self.count = 0
        self.last_raw = 0.
        self.clipped = 0

    @property
    def ready(self):
        return self.count >= self.warmup

    def state_dict(self):
        return dict(a=self.a, b=self.b, count=self.count, last_raw=self.last_raw, clipped=self.clipped)

    def load_state_dict(self, state):
        import numpy as np
        if not np.isfinite([state['a'], state['b']]).all() or state['b'] < 0 or state['count'] < 0:
            raise ValueError("Invalid DSR state.")
        self.a, self.b, self.count = float(state['a']), float(state['b']), int(state['count'])
        self.last_raw, self.clipped = state.get('last_raw', 0.), state.get('clipped', 0)

    def update(self, net_excess_return):
        import numpy as np
        r = float(net_excess_return)
        if not np.isfinite(r):
            raise ValueError("DSR requires finite realized returns.")
        da, db = r-self.a, r*r-self.b
        variance = self.b-self.a*self.a
        raw = (self.b*da - .5*self.a*db) / max(variance, self.epsilon)**1.5 if self.ready and variance > self.epsilon else 0.
        self.a += self.eta*da
        self.b += self.eta*db
        self.count += 1
        self.last_raw = float(raw)
        self.clipped += int(abs(raw) > self.clip)
        return float(np.clip(raw, -self.clip, self.clip)*self.scaling)

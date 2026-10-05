import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from typing import Dict, Any

def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """
    Tính tổng tích lũy có chiết khấu (discounted cumulative sum) một cách hiệu quả.
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class PPOBuffer:
    """
    Bộ đệm lưu trữ quỹ đạo (trajectory) cho một epoch của huấn luyện PPO.
    """
    def __init__(self, max_size: int, obs_dim: int, act_dim: int, gamma: float = 0.99, lam: float = 0.95):
        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((max_size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(max_size, dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.ret_buf = np.zeros(max_size, dtype=np.float32)
        self.val_buf = np.zeros(max_size, dtype=np.float32)
        self.logp_buf = np.zeros(max_size, dtype=np.float32)
        
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, max_size

    def store(self, obs: np.ndarray, act: np.ndarray, rew: float, val: float, logp: float):
        """
        Lưu trữ một bước chuyển (transition).
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val: float = 0.0):
        """
        Tính toán lợi thế GAE-Lambda và phần thưởng tích lũy khi kết thúc một quỹ đạo.
        """
        if not np.isfinite(last_val):
            raise ValueError("last_val phải hữu hạn.")
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # GAE: delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # Returns-to-go: discounted cumulative rewards
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self) -> Dict[str, torch.Tensor]:
        """
        Trả về dữ liệu dưới dạng torch tensors, và chuẩn hóa lợi thế (advantages).
        """
        assert self.ptr == self.max_size  # buffer must be full
        self.ptr, self.path_start_idx = 0, 0
        
        # Advantage normalization
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

class PPOUpdater:
    """
    Thực hiện cập nhật PPO theo hàm mục tiêu bị cắt xén (clipped objective).
    """
    def __init__(self, clip_ratio: float = 0.2, target_kl: float = 0.01, 
                 value_loss_coef: float = 0.5, entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5):
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def update(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
               buffer_data: Dict[str, torch.Tensor], train_iters: int = 10) -> Dict[str, Any]:
        """
        Thực hiện nhiều epoch cập nhật PPO.
        """
        if train_iters <= 0:
            raise ValueError("train_iters phải lớn hơn 0.")

        obs = buffer_data['obs']
        act = buffer_data['act']
        ret = buffer_data['ret']
        adv = buffer_data['adv']
        old_logp = buffer_data['logp']

        stats = {'policy_loss': [], 'value_loss': [], 'entropy': [], 'kl': []}

        for i in range(train_iters):
            # Model cần cung cấp hàm `evaluate` trả về log_prob, entropy và value
            log_prob, entropy, value = model.evaluate(obs, act)

            # PPO objective
            ratio = torch.exp(log_prob - old_logp)
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

            # Value loss
            loss_v = ((value.reshape(-1) - ret.reshape(-1)) ** 2).mean()

            # Entropy
            ent = entropy.mean()

            # Total loss
            loss = loss_pi + self.value_loss_coef * loss_v - self.entropy_coef * ent

            # KL divergence for early stopping
            with torch.no_grad():
                approx_kl = (old_logp - log_prob).mean().item()
            
            stats['policy_loss'].append(loss_pi.item())
            stats['value_loss'].append(loss_v.item())
            stats['entropy'].append(ent.item())
            stats['kl'].append(approx_kl)

            if approx_kl > 1.5 * self.target_kl:
                break

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            optimizer.step()

        return {
            'LossPi': np.mean(stats['policy_loss']),
            'LossV': np.mean(stats['value_loss']),
            'Entropy': np.mean(stats['entropy']),
            'KL': stats['kl'][-1],
            'StopIter': i + 1
        }

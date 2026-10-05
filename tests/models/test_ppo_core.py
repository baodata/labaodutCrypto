import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from src.training.ppo_core import discount_cumsum, PPOBuffer, PPOUpdater

def test_discount_cumsum():
    """Verify discount_cumsum computes discounted sum correctly."""
    x = np.array([1, 1, 1, 1], dtype=np.float32)
    y = discount_cumsum(x, 0.99)
    assert np.isclose(y[3], 1.0)
    assert np.isclose(y[2], 1.99)
    assert np.isclose(y[1], 2.9701)
    assert np.isclose(y[0], 3.940399)

def test_ppo_buffer_store_and_get():
    """Verify PPO buffer stores transitions, computes advantages and normalizes them."""
    buf = PPOBuffer(max_size=10, obs_dim=4, act_dim=2)
    for _ in range(10):
        obs = np.random.rand(4)
        act = np.random.rand(2)
        rew = np.random.rand()
        val = np.random.rand()
        logp = np.random.rand()
        buf.store(obs, act, rew, val, logp)
    
    buf.finish_path(last_val=0.5)
    data = buf.get()
    
    assert data['obs'].shape == (10, 4)
    assert data['act'].shape == (10, 2)
    assert data['ret'].shape == (10,)
    assert data['adv'].shape == (10,)
    assert data['logp'].shape == (10,)
    
    # Verify adv is normalized
    assert torch.isclose(data['adv'].mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(data['adv'].std(), torch.tensor(1.0), atol=1e-1)

class SimpleActorCritic(nn.Module):
    """Simple model to test PPO updater."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Linear(obs_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(obs_dim, 1)

    def evaluate(self, obs, act):
        mu = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        logp = dist.log_prob(act).sum(axis=-1)
        ent = dist.entropy().mean()
        val = self.critic(obs).squeeze()
        return logp, ent, val

def test_ppo_updater_reduces_loss():
    """Verify PPO updater executes optimization steps and reduces value loss."""
    obs_dim = 4
    act_dim = 2
    model = SimpleActorCritic(obs_dim, act_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    # Set target_kl high enough to prevent early stopping
    updater = PPOUpdater(target_kl=100.0)
    
    buf = PPOBuffer(max_size=100, obs_dim=obs_dim, act_dim=act_dim)
    
    # Dùng model thực tế để lấy logp chuẩn, tránh KL phân kỳ ngay lập tức
    obs_all = np.random.randn(100, obs_dim).astype(np.float32)
    act_all = np.random.randn(100, act_dim).astype(np.float32)
    with torch.no_grad():
        logp_all, _, val_all = model.evaluate(torch.as_tensor(obs_all), torch.as_tensor(act_all))
    
    for i in range(100):
        buf.store(obs_all[i], act_all[i], rew=1.0, val=val_all[i].item(), logp=logp_all[i].item())
    buf.finish_path(last_val=0.0)
    data = buf.get()
    
    _, _, val_old = model.evaluate(data['obs'], data['act'])
    old_loss_v = ((val_old.squeeze() - data['ret'])**2).mean().item()
    
    stats = updater.update(model, optimizer, data, train_iters=10)
    
    _, _, val_new = model.evaluate(data['obs'], data['act'])
    new_loss_v = ((val_new.squeeze() - data['ret'])**2).mean().item()
    
    assert new_loss_v < old_loss_v, f"Value loss did not decrease: {old_loss_v} -> {new_loss_v}"
    assert 'LossPi' in stats
    assert 'LossV' in stats

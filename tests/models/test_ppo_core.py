import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from src.models.actor_critic import SimplexActorCritic
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
    assert 'ExplainedVar' in stats
    assert 'FirstKL' in stats
    assert 'StopKL' in stats
    assert stats['NumMinibatchesAttempted'] == stats['NumMinibatches']


def test_dirichlet_act_and_evaluate_have_matching_log_probabilities():
    torch.manual_seed(7)
    model = SimplexActorCritic(observation_dim=8, action_dim=4, hidden_dim=16)
    obs = torch.randn(64, 8)

    with torch.no_grad():
        actions, old_logp, _ = model.act(obs)
        new_logp, _, _ = model.evaluate(obs, actions)
        log_ratio = new_logp - old_logp
        initial_kl = (((torch.exp(log_ratio) - 1.0) - log_ratio).mean()).item()

    assert torch.allclose(new_logp, old_logp, atol=1e-6, rtol=1e-6)
    assert abs(initial_kl) < 1e-6


def test_ppo_learns_clear_synthetic_positive_asset_signal():
    """A stationary asset with a clear positive reward should gain allocation."""
    torch.manual_seed(17)
    np.random.seed(17)
    model = SimplexActorCritic(observation_dim=1, action_dim=3, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    updater = PPOUpdater(target_kl=0.05, entropy_coef=0.0)
    obs_batch = torch.ones(128, 1)

    with torch.no_grad():
        initial_action = model.act(torch.ones(1), deterministic=True)[0]
    initial_weight = float(initial_action[0])

    for _ in range(25):
        buffer = PPOBuffer(
            max_size=len(obs_batch), obs_dim=1, act_dim=3, gamma=0.0, lam=0.0
        )
        with torch.no_grad():
            actions, log_probs, values = model.act(obs_batch)
        action_np = actions.numpy()
        # Asset 0 earns 1% per step; asset 1 and cash earn zero.
        rewards = action_np[:, 0]
        for index in range(len(obs_batch)):
            buffer.store(
                obs_batch[index].numpy(),
                action_np[index],
                float(rewards[index]),
                float(values[index]),
                float(log_probs[index]),
            )
        buffer.finish_path(last_val=0.0)
        updater.update(
            model,
            optimizer,
            buffer.get(),
            train_iters=4,
            batch_size=len(obs_batch),
        )

    with torch.no_grad():
        final_action = model.act(torch.ones(1), deterministic=True)[0]
    final_weight = float(final_action[0])

    assert final_weight > initial_weight + 0.25
    assert final_weight > 0.65

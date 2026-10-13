"""
tests/models/test_gnn_actor_critic.py
Unit tests for GNN-004: GNNActorCritic (GAT & GCN) integration with PPO.
"""
import numpy as np
import pytest
import torch

from src.models.actor_critic import build_actor_critic, GNNActorCritic
from src.training.ppo_core import PPOBuffer, PPOUpdater


@pytest.mark.parametrize("encoder", ["gcn", "gat"])
def test_gnn_actor_critic_act_and_evaluate_single_obs(encoder):
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_assets, n_features, lookback = 4, 3, 5
    action_dim = n_assets + 1
    tickers = ["AAPL", "MSFT", "NVDA", "AMD"]
    
    model = build_actor_critic(
        encoder=encoder,
        n_assets=n_assets,
        n_features=n_features,
        lookback=lookback,
        action_dim=action_dim,
        hidden_dim=16,
        gnn_heads=2,
        gnn_knn=2,
        tickers=tickers,
    )
    assert isinstance(model, GNNActorCritic)
    
    # 1D single observation
    hist_dim = lookback * n_assets * n_features
    port_dim = action_dim
    obs = torch.randn(hist_dim + port_dim)
    # Portfolio weights in simplex
    obs[hist_dim:] = torch.softmax(torch.randn(port_dim), dim=-1)
    
    action, log_prob, value = model.act(obs, deterministic=False)
    assert action.shape == (action_dim,)
    assert log_prob.shape == ()
    assert value.shape == ()
    assert torch.all(action >= 0.0)
    assert torch.isclose(action.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.isfinite(log_prob)
    assert torch.isfinite(value)
    
    # Deterministic action
    det_action, det_log_prob, det_val = model.act(obs, deterministic=True)
    assert det_action.shape == (action_dim,)
    assert torch.all(det_action >= 0.0)
    assert torch.isclose(det_action.sum(), torch.tensor(1.0), atol=1e-5)


@pytest.mark.parametrize("encoder", ["gcn", "gat"])
def test_gnn_actor_critic_batch_and_ppo_update(encoder):
    torch.manual_seed(42)
    np.random.seed(42)
    
    batch_size, n_assets, n_features, lookback = 8, 4, 3, 5
    action_dim = n_assets + 1
    tickers = ["AAPL", "MSFT", "NVDA", "AMD"]
    
    model = build_actor_critic(
        encoder=encoder,
        n_assets=n_assets,
        n_features=n_features,
        lookback=lookback,
        action_dim=action_dim,
        hidden_dim=16,
        gnn_heads=2,
        gnn_knn=2,
        tickers=tickers,
    )
    
    hist_dim = lookback * n_assets * n_features
    port_dim = action_dim
    obs_dim = hist_dim + port_dim
    
    batch_obs = torch.randn(batch_size, obs_dim)
    # Ensure portfolio state in valid simplex
    for b in range(batch_size):
        batch_obs[b, hist_dim:] = torch.softmax(torch.randn(port_dim), dim=-1)
        
    actions, log_probs, values = model.act(batch_obs)
    assert actions.shape == (batch_size, action_dim)
    assert log_probs.shape == (batch_size,)
    assert values.shape == (batch_size,)
    assert torch.all(actions >= 0.0)
    assert torch.allclose(actions.sum(dim=-1), torch.ones(batch_size), atol=1e-5)
    
    # Evaluate interface
    eval_log_probs, entropy, eval_values = model.evaluate(batch_obs, actions)
    assert eval_log_probs.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert eval_values.shape == (batch_size,)
    assert torch.allclose(eval_log_probs, log_probs, atol=1e-5, rtol=1e-5)
    
    # PPO Buffer and Update verification
    buffer = PPOBuffer(
        max_size=batch_size,
        obs_dim=obs_dim,
        act_dim=action_dim,
        gamma=0.99,
        lam=0.95,
    )
    for b in range(batch_size):
        buffer.store(
            obs=batch_obs[b].numpy(),
            act=actions[b].detach().numpy(),
            rew=float(np.random.randn()),
            val=float(values[b].item()),
            logp=float(log_probs[b].item()),
        )
    buffer.finish_path(last_val=0.0)
    
    buffer_data = buffer.get()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    updater = PPOUpdater(clip_ratio=0.2, target_kl=0.05)
    
    stats = updater.update(
        model=model,
        optimizer=optimizer,
        buffer_data=buffer_data,
        train_iters=2,
        batch_size=4,
    )
    
    assert "LossPi" in stats or "policy_loss" in stats
    assert "LossV" in stats or "value_loss" in stats
    assert "Entropy" in stats or "entropy" in stats


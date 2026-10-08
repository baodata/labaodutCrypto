import numpy as np
import pytest
import torch

from src.models.actor_critic import build_actor_critic
from src.training.ppo_core import PPOBuffer, PPOUpdater


@pytest.mark.parametrize("encoder", ["lstm", "gru"])
def test_recurrent_actor_critic_round_trips_and_updates_ppo(encoder):
    torch.manual_seed(23)
    np.random.seed(23)
    batch_size, lookback, n_assets, n_features = 8, 5, 3, 2
    action_dim = n_assets + 1
    model_config = {
        "encoder": encoder,
        "n_assets": n_assets,
        "n_features": n_features,
        "lookback": lookback,
        "action_dim": action_dim,
        "hidden_dim": 16,
        "rnn_layers": 1,
        "rnn_dropout": 0.0,
    }
    model = build_actor_critic(**model_config)

    # Match the packed observation layout used by PPOBuffer:
    # flattened chronological market history, then risky and cash weights.
    history = torch.randn(batch_size, lookback, n_assets, n_features)
    portfolio_state = torch.softmax(torch.randn(batch_size, action_dim), dim=-1)
    packed_obs = torch.cat((history.flatten(start_dim=1), portfolio_state), dim=-1)
    restored_history, restored_portfolio = model._unflatten_obs(packed_obs)
    assert torch.equal(restored_history, history)
    assert torch.equal(restored_portfolio, portfolio_state)

    actions, old_log_probs, values = model.act(packed_obs)
    new_log_probs, entropy, evaluated_values = model.evaluate(packed_obs, actions)
    assert actions.shape == (batch_size, action_dim)
    assert torch.allclose(actions.sum(dim=-1), torch.ones(batch_size), atol=1e-6)
    assert torch.allclose(new_log_probs, old_log_probs, atol=1e-6, rtol=1e-6)
    assert evaluated_values.shape == (batch_size,)
    assert entropy.shape == (batch_size,)

    buffer = PPOBuffer(
        max_size=batch_size,
        obs_dim=packed_obs.shape[-1],
        act_dim=action_dim,
        gamma=0.0,
        lam=0.0,
    )
    rewards = np.linspace(-0.2, 0.2, batch_size, dtype=np.float32)
    for i in range(batch_size):
        buffer.store(
            packed_obs[i].numpy(),
            actions[i].detach().numpy(),
            float(rewards[i]),
            float(values[i].detach()),
            float(old_log_probs[i].detach()),
        )
    buffer.finish_path()

    updater = PPOUpdater(target_kl=100.0, entropy_coef=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    stats = updater.update(
        model,
        optimizer,
        buffer.get(),
        train_iters=1,
        batch_size=batch_size,
    )
    assert stats["NumMinibatches"] == 1
    assert np.isfinite(stats["LossPi"])
    assert np.isfinite(stats["LossV"])
    assert model.rnn.weight_ih_l0.grad is not None
    assert torch.isfinite(model.rnn.weight_ih_l0.grad).all()
    assert model.rnn.weight_ih_l0.grad.abs().sum() > 0

    # The same metadata used for checkpoint reconstruction recreates the model.
    restored_model = build_actor_critic(**model_config)
    restored_model.load_state_dict(model.state_dict())
    with torch.no_grad():
        expected_actions = model.act(packed_obs, deterministic=True)[0]
        restored_actions = restored_model.act(packed_obs, deterministic=True)[0]
    assert torch.allclose(expected_actions, restored_actions, atol=1e-7, rtol=1e-7)


def test_ppo_training_defaults_to_gru_encoder():
    from src.training.train_ppo import build_parser

    args = build_parser().parse_args([])
    assert args.encoder == "gru"

"""MLP Actor-Critic baseline with actions constrained to the portfolio simplex."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet


class SimplexActorCritic(nn.Module):
    """PPO policy for raw market features and portfolio state.

    The actor uses a Dirichlet distribution, so every sampled action is
    non-negative and sums to one (including the cash allocation).
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        if observation_dim <= 0 or action_dim <= 1 or hidden_dim <= 0:
            raise ValueError("Dimensions must be positive and action_dim must include assets and cash.")

        self.trunk = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def _forward_distribution(self, obs: torch.Tensor) -> tuple[Dirichlet, torch.Tensor]:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        hidden = self.trunk(obs)
        # Keep concentrations >= 1 to avoid singular density at the simplex
        # boundary and zero-valued samples that make PPO log-probabilities infinite.
        concentration = (F.softplus(self.actor_head(hidden)) + 1.0).clamp(max=100.0)
        return Dirichlet(concentration), self.critic_head(hidden).squeeze(-1)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return action, log probability, and value for one observation or a batch."""
        single_observation = obs.ndim == 1
        distribution, value = self._forward_distribution(obs)
        action = distribution.mean if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action)
        if single_observation:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            value = value.squeeze(0)
        return action, log_prob, value

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPOUpdater interface: log probability, entropy, and state value."""
        distribution, value = self._forward_distribution(obs)
        return distribution.log_prob(actions), distribution.entropy(), value

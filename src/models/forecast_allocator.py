"""Forecast-only centralized PPO, sector actors and a conditional manager/worker policy.

All policies expose the same PPOUpdater interface. The hierarchical likelihood is
p(manager | state) * product p(worker_g | state, sampled manager budgets).
Workers see the same global information; this isolates action organization rather
than giving the hierarchical model extra market inputs. One daily team reward and
one centralized critic are used; this is cooperative joint PPO, not independent PPO.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from src.evaluation.cost_aware_allocator import asset_bounds


METHODS = ("convex", "ppo", "sector_marl", "hmarl")


def project_box(values, lower, upper, budget):
    """Euclidean projection onto a bounded simplex, for single vectors or batches."""
    values = torch.as_tensor(values, dtype=torch.float64)
    lower = torch.as_tensor(lower, dtype=values.dtype, device=values.device)
    upper = torch.as_tensor(upper, dtype=values.dtype, device=values.device)
    budget = torch.as_tensor(budget, dtype=values.dtype, device=values.device)
    if budget.ndim == values.ndim - 1:
        budget = budget.unsqueeze(-1)
    if (torch.any(lower > upper) or torch.any(budget < lower.sum(-1, keepdim=True) - 1e-9)
            or torch.any(budget > upper.sum(-1, keepdim=True) + 1e-9)):
        raise ValueError("Infeasible bounded simplex budget.")
    # Sum(clip(v-lambda,l,u)) is piecewise linear; its breakpoints are
    # v-u and v-l. Locate the enclosing segment in parallel, then interpolate.
    # This avoids 50 Python/Torch iterations per worker on every rollout step.
    points = torch.cat((values-upper, values-lower), -1).sort(-1).values
    levels = torch.clamp(values.unsqueeze(-2)-points.unsqueeze(-1), min=lower, max=upper).sum(-1)
    index = (levels > budget).sum(-1, keepdim=True).clamp(1, points.shape[-1]-1)
    lo, hi = points.gather(-1, index-1), points.gather(-1, index)
    f_lo, f_hi = levels.gather(-1, index-1), levels.gather(-1, index)
    fraction = ((f_lo-budget)/(f_lo-f_hi).clamp(min=1e-30)).clamp(0, 1)
    return torch.clamp(values-(lo+fraction*(hi-lo)), min=lower, max=upper)


def actor(input_dim, output_dim, hidden_dim):
    network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, output_dim))
    # Every deterministic initial policy reproduces the same matched 1/N core.
    nn.init.zeros_(network[-1].weight)
    nn.init.zeros_(network[-1].bias)
    return network


class ForecastAllocatorPolicy(nn.Module):
    def __init__(self, observation_dim, groups, *, method, cash_ratio, tilt_fraction,
                 max_asset_weight, hidden_dim=32, action_std=.1):
        super().__init__()
        if method not in METHODS[1:] or not groups or any(not g for g in groups):
            raise ValueError("Specify PPO/sector/H-MARL and nonempty asset groups.")
        self.groups = [list(g) for g in groups]
        self.n = sum(map(len, groups))
        if sorted(i for group in groups for i in group) != list(range(self.n)):
            raise ValueError("Groups must partition the ordered asset universe exactly once.")
        if not np.isfinite(action_std) or action_std <= 0 or hidden_dim <= 0:
            raise ValueError("Positive action noise and hidden width required.")
        self.method, self.cash, self.tilt = method, cash_ratio, tilt_fraction
        self.action_std, self.observation_dim = float(action_std), observation_dim
        core, lower, upper = asset_bounds(self.n, cash_ratio, tilt_fraction, max_asset_weight)
        for name, values in (("core", core), ("lower", lower), ("upper", upper)):
            self.register_buffer(name, torch.tensor(values, dtype=torch.float64))
        self.register_buffer("group_core", torch.tensor([core[g].sum() for g in groups], dtype=torch.float64))
        self.register_buffer("group_lower", torch.tensor([lower[g].sum() for g in groups], dtype=torch.float64))
        self.register_buffer("group_upper", torch.tensor([upper[g].sum() for g in groups], dtype=torch.float64))
        count = len(groups)
        self.action_dim = self.n + (count if method == "hmarl" else 0)
        self.critic = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.Tanh(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        if method == "ppo":
            self.single_actor = actor(observation_dim, self.n, hidden_dim)
        else:
            self.workers = nn.ModuleList([actor(observation_dim + count, len(g), hidden_dim) for g in groups])
            if method == "hmarl":
                self.manager = actor(observation_dim, count, hidden_dim)

    def budgets(self, manager_action=None):
        if manager_action is None:
            return self.group_core
        proposed = self.group_core * (1 + self.tilt * torch.tanh(manager_action.double()))
        return project_box(proposed, self.group_lower, self.group_upper, 1-self.cash)

    def _draw(self, state, deterministic=False, actions=None):
        if state.ndim == 1:
            state = state.unsqueeze(0)
        batch, count = len(state), len(self.groups)
        logp = torch.zeros(batch, device=state.device)
        entropy = torch.zeros_like(logp)
        parts = []

        def draw(mean, observed):
            distribution = Normal(mean, self.action_std)
            sample = observed if observed is not None else mean if deterministic else distribution.sample()
            return sample, distribution.log_prob(sample).sum(-1), distribution.entropy().sum(-1)

        if self.method == "ppo":
            sample, logp, entropy = draw(self.single_actor(state), actions)
            return sample, logp, entropy, self.critic(state).squeeze(-1)
        if self.method == "hmarl":
            manager, lp, ent = draw(self.manager(state), None if actions is None else actions[:, :count])
            budgets = self.budgets(manager)
            logp, entropy = logp+lp, entropy+ent
            parts.append(manager)
            local = None if actions is None else actions[:, count:]
        else:
            budgets = self.group_core.expand(batch, -1)
            local = actions
        worker_state = torch.cat((state, (budgets / (1-self.cash)).float()), dim=-1)
        scores = torch.zeros((batch, self.n), device=state.device)
        for indices, worker in zip(self.groups, self.workers):
            sample, lp, ent = draw(worker(worker_state), None if local is None else local[:, indices])
            scores[:, indices] = sample
            logp, entropy = logp+lp, entropy+ent
        parts.append(scores)
        return torch.cat(parts, -1), logp, entropy, self.critic(state).squeeze(-1)

    def act(self, obs, deterministic=False):
        single = obs.ndim == 1
        action, lp, _, value = self._draw(obs, deterministic=deterministic)
        return (action[0], lp[0], value[0]) if single else (action, lp, value)

    def evaluate(self, obs, actions):
        _, lp, entropy, value = self._draw(obs, actions=actions)
        return lp, entropy, value

    def target_tensors(self, action):
        """Differentiable batched action mapping, shared by imitation and execution."""
        action = torch.as_tensor(action, device=self.core.device, dtype=torch.float64)
        single = action.ndim == 1
        if single:
            action = action.unsqueeze(0)
        if action.ndim != 2 or action.shape[-1] != self.action_dim or not torch.isfinite(action).all():
            raise ValueError("Invalid policy action.")
        if self.method == "ppo":
            stocks = project_box(self.core * (1+self.tilt*torch.tanh(action)), self.lower, self.upper, 1-self.cash)
            budgets = torch.stack([stocks[:, g].sum(-1) for g in self.groups], -1)
        else:
            count = len(self.groups)
            budgets = self.budgets(action[:, :count]) if self.method == "hmarl" else self.group_core.expand(len(action), -1)
            scores = action[:, count:] if self.method == "hmarl" else action
            stocks = torch.zeros((len(action), self.n), dtype=self.core.dtype, device=self.core.device)
            for index, group in enumerate(self.groups):
                proposed = budgets[:, index:index+1] / len(group) * (1+self.tilt*torch.tanh(scores[:, group]))
                stocks[:, group] = project_box(proposed, self.lower[group], self.upper[group], budgets[:, index])
        cash = torch.full((len(action), 1), self.cash, dtype=stocks.dtype, device=stocks.device)
        targets = torch.cat((stocks, cash), -1)
        return (targets[0], budgets[0]) if single else (targets, budgets)

    @torch.no_grad()
    def target(self, action):
        """Map one stored latent action to execution targets; PPO scores latent actions."""
        if np.shape(action) != (self.action_dim,):
            raise ValueError("Execution needs one action vector.")
        targets, budgets = self.target_tensors(action)
        return targets.cpu().numpy(), budgets.cpu().numpy()

"""Reusable policy backtests, weight diagnostics and financial summaries. Owner: A."""

from __future__ import annotations

from torch import nn
import numpy as np
import torch

from src.env.trading_env import TradingEnv
from src.evaluation.backtester import BacktestResult
from src.evaluation.backtester import DeterministicBacktester
from src.evaluation.baselines import EqualWeightBaseline
from src.evaluation.metrics import FinancialMetrics
from src.models.market_cache import prepare_market_cache
from src.models.policy_adapter import graph_tensors_from_obs
from src.models.policy_adapter import history_mask_from_obs
from src.models.policy_adapter import observation_vector


def action_for_model(
    model: nn.Module,
    obs: dict[str, np.ndarray],
    deterministic: bool = False,
    market_embedding: torch.Tensor | None = None,
) -> np.ndarray:
    """Evaluate model action from observation dict."""
    device = next(model.parameters()).device
    with torch.no_grad():
        state = torch.as_tensor(observation_vector(obs), dtype=torch.float32, device=device)
        history_mask = history_mask_from_obs(obs, device)
        graph = graph_tensors_from_obs(obs, device)
        if market_embedding is not None:
            graph = {"market_embedding": market_embedding}
        if graph is None:
            action, _, _ = model.act(state, deterministic=deterministic, history_mask=history_mask)
        else:
            action, _, _ = model.act(state, deterministic=deterministic, graph=graph, history_mask=history_mask)
    return action.detach().cpu().numpy().astype(np.float32)


def evaluate_policy(
    model: nn.Module,
    env: TradingEnv,
) -> BacktestResult:
    result, _, _ = evaluate_policy_with_actions(model, env)
    return result


def evaluate_policy_with_actions(
    model: nn.Module,
    env: TradingEnv,
) -> tuple[BacktestResult, np.ndarray, np.ndarray]:
    """Backtest deterministically and collect policy and blended target weights."""
    model.eval()
    market_cache = prepare_market_cache(model, env)
    policy_actions: list[np.ndarray] = []
    blended_targets: list[np.ndarray] = []

    def act(obs: dict[str, np.ndarray]) -> np.ndarray:
        embedding = market_cache.get(model, env.history_start_index + env.current_step) if market_cache else None
        action = action_for_model(model, obs, deterministic=True, market_embedding=embedding)
        policy_target, execution_target = env.project_action(action)
        policy_actions.append(policy_target.copy())
        blended_targets.append(execution_target.copy())
        return action

    result = DeterministicBacktester(env).run_agent(act)
    if env.cash_funded_execution:
        blended_targets = result.daily_execution_weights
    return result, np.asarray(policy_actions), np.asarray(blended_targets)


def policy_weight_summary(
    actions: np.ndarray,
    n_assets: int,
    asset_names: list[str] | None = None,
) -> dict[str, float]:
    """Summarize per-step allocation tilt and temporal weight variability."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != n_assets + 1 or len(actions) == 0:
        raise ValueError("actions phải có shape [steps, n_assets + 1] và không rỗng.")
    if asset_names is not None and len(asset_names) != n_assets:
        raise ValueError("asset_names phải có một tên cho mỗi tài sản.")
    asset_weights = actions[:, :n_assets]
    stock_sum = asset_weights.sum(axis=1, keepdims=True)
    normalized_stock_weights = np.where(
        stock_sum > 1e-12, asset_weights / stock_sum, 1.0 / n_assets
    )
    per_step_deviation = normalized_stock_weights - (1.0 / n_assets)
    summary = {
        "mean_stock_allocation": float(actions[:, :n_assets].sum(axis=1).mean()),
        "mean_cash_weight": float(actions[:, n_assets].mean()),
        "mean_asset_weight": float(asset_weights.mean()),
        "cross_asset_weight_std": float(asset_weights.std(axis=1).mean()),
        "mean_abs_deviation_1n": float(np.abs(per_step_deviation).mean()),
        "max_abs_deviation_1n": float(np.abs(per_step_deviation).max()),
        "mean_temporal_weight_std": float(actions.std(axis=0).mean()),
    }
    temporal_std = actions.std(axis=0)
    labels = asset_names or [f"asset_{index}" for index in range(n_assets)]
    for asset_name, value in zip(labels, temporal_std[:n_assets]):
        summary[f"asset_{asset_name}_weight_std_time"] = float(value)
    summary["cash_weight_std_time"] = float(temporal_std[n_assets])
    return summary


def metric_summary(
    result: BacktestResult,
    risk_free_rate_annual: float,
) -> dict[str, float]:
    summary = FinancialMetrics(
        daily_net_returns=result.daily_net_returns,
        risk_free_rate_annual=risk_free_rate_annual,
        daily_turnover=result.daily_turnover,
        daily_costs=result.daily_costs,
    ).summary_dict()
    if result.extended_diagnostics:
        for field in ('daily_commission','daily_slippage','daily_impact'):
            summary['total_'+field.removeprefix('daily_')] = float(np.sum(getattr(result, field)))
        summary['hold_fraction'] = float(np.mean(result.daily_hold))
        summary['participation_max'] = float(np.max(result.daily_participation_max))
    summary["profit"] = float(result.portfolio_values[-1] - result.portfolio_values[0])
    summary["final_value"] = float(result.portfolio_values[-1])
    return summary


def evaluate_equal_weight(
    env: TradingEnv,
    open_prices: np.ndarray,
    num_assets: int,
    transaction_cost_rate: float,
    risk_free_rate_annual: float,
    cash_ratio: float = 0.0,
) -> tuple[dict[str, float], BacktestResult]:
    """Evaluate 1/N (hoặc 1/N + cash) reference qua cùng bộ máy TradingEnv."""
    result = DeterministicBacktester(env).run_strategy(
        EqualWeightBaseline(
            open_prices=open_prices,
            num_assets=num_assets,
            transaction_cost_rate=transaction_cost_rate,
            risk_free_rate_annual=risk_free_rate_annual,
            cash_ratio=cash_ratio,
        )
    )
    return metric_summary(result, risk_free_rate_annual), result

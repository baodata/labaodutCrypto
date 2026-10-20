"""Learn allocation only from causal OOF forecasts; select checkpoints on inner."""
from __future__ import annotations

import copy
from pathlib import Path
import random
import time

import numpy as np
import torch

from src.env.orders import PortfolioOrder
from src.evaluation.backtester import DeterministicBacktester
from src.evaluation.cost_aware_allocator import cost_aware_target
from src.evaluation.decision_data import ForecastEpisode
from src.evaluation.policy_evaluator import metric_summary
from src.models.forecast_allocator import ForecastAllocatorPolicy
from src.training.ppo_core import PPOBuffer, PPOUpdater
from src.utils.artifacts import atomic_csv, atomic_json, rng_state
from src.training.decision_probes import fixed_train_windows, evaluate_train_windows


def simulate(data, episode, method, policy=None):
    env, decisions = data.env(episode), []
    matched_env = data.env(episode)
    matched_env.reset()

    def strategy(obs, info):
        position = episode.start + env.current_step
        if position not in episode.forecasts:
            matched_env.step(PortfolioOrder.hold())
            return PortfolioOrder.hold()
        if method == "matched_1n":
            target, budgets = data.core_target, data.group_prior*(1-data.cash)
        elif method == "convex":
            target, diagnostic = cost_aware_target(episode.forecasts[position][0], data.risk.covariance[position],
                env.portfolio.weights, horizon=episode.horizon, action_alpha=data.alpha,
                cash_ratio=data.cash, tilt_fraction=data.allocation["tilt_fraction"], max_asset_weight=data.cap,
                risk_aversion=data.allocation["risk_aversion"], cost_rate=data.cost_rate,
                cost_multiplier=data.allocation["cost_multiplier"])
            budgets = np.array([target[g].sum() for g in data.groups])
        else:
            policy.eval()
            state = torch.from_numpy(data.state(episode, position, env.portfolio.weights, matched_env.portfolio.weights))
            with torch.no_grad():
                action, _, _ = policy.act(state, deterministic=True)
            target, budgets = policy.target(action)
        decisions.append(dict(decision_date=data.tensor.dates[position],
            **{f"target_{ticker}": float(w) for ticker, w in zip(data.tensor.tickers+["CASH"], target)},
            **{f"budget_{name}": float(w) for name, w in zip(data.group_names, budgets)}))
        matched_env.step(data.core_target)
        return target

    result = DeterministicBacktester(env).run_strategy(strategy)
    return result, decisions


def train_decision_policy(data, selection, method, seed, settings, output):
    """Joint PPO for sector workers/manager; no predictor parameters are loaded."""
    import pandas as pd

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    training, inner = data.episode(selection, "train_oof"), data.episode(selection, "inner")
    policy_config = dict(observation_dim=len(data.state(training, training.start, np.r_[np.zeros(len(data.core)), 1.])),
        groups=data.groups, method=method, cash_ratio=data.cash, tilt_fraction=data.allocation["tilt_fraction"],
        max_asset_weight=data.cap, hidden_dim=settings["hidden_dim"], action_std=settings["action_std"])
    policy = ForecastAllocatorPolicy(**policy_config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=settings["learning_rate"])
    updater = PPOUpdater(target_kl=settings["target_kl"], entropy_coef=0., max_grad_norm=.5)
    risk_free = float(data.environment["risk_free_rate"])
    reference, _ = simulate(data, inner, "matched_1n")
    matched = metric_summary(reference, risk_free)
    initial_result, _ = simulate(data, inner, method, policy)
    initial = metric_summary(initial_result, risk_free)
    if abs(initial["sharpe_ratio"]-matched["sharpe_ratio"]) > 1e-8:
        raise RuntimeError("Initial deterministic policy must match the core baseline.")
    best_score = initial["sharpe_ratio"]
    best_state = copy.deepcopy(policy.state_dict())
    best_epoch, best_metrics = 0, initial
    recent = [best_score]*settings["selection_window"]
    stale = 0
    patience_best_score, patience_best_epoch = best_score, 0
    best_trained_score, best_trained_epoch = -np.inf, 0
    min_delta = settings.get("patience_min_delta", 1e-5)
    windows = fixed_train_windows(training, settings.get("train_probe_windows", 3),
                                  settings.get("train_probe_days", 126))
    atomic_json(output / "train_probe_windows.json", dict(
        selection="evenly_spaced_train_oof_only_no_rng_no_inner_selection", reset="cash_for_each_window",
        aggregation="mean_per_window_metrics_overlapping_windows_are_not_independent_samples",
        windows=[dict(window_id=i, first_decision=data.tensor.dates[w.start],
            last_decision=data.tensor.dates[w.end-2], last_mark=data.tensor.dates[w.end-1],
            decisions=w.end-w.start-1) for i, w in enumerate(windows)]))
    probe_baselines = [simulate(data, window, "matched_1n")[0] for window in windows]
    probe, probe_rows = evaluate_train_windows(data, windows, policy, method, probe_baselines, simulate)
    probe_history = [dict(epoch=0, **row) for row in probe_rows]
    rows = [dict(epoch=0, checkpoint_status="untrained_reference", **initial, **probe,
        inner_sharpe_delta_vs_matched=initial["sharpe_ratio"]-matched["sharpe_ratio"],
        patience_best_score=patience_best_score, epochs_without_progress=0)]
    atomic_csv(pd.DataFrame(probe_history), output / "train_probe_metrics.csv", index=False)
    atomic_csv(pd.DataFrame(rows), output / "training_log.csv", index=False)

    def save(name, epoch, metrics):
        payload = dict(policy_config=policy_config, model_state_dict=policy.state_dict(),
            optimizer_state_dict=optimizer.state_dict(), rng_state=rng_state(),
            episode_rng_state=rng.bit_generator.state, epoch=epoch,
            checkpoint_status="untrained_reference" if epoch == 0 else "trained",
            checkpoint_role="diagnostic_only_not_promoted" if name == "best_trained.pth" else
                            "latest_training_state_not_promoted" if name == "last.pth" else "selected_for_evaluation",
            predictor_lock_digest=data.lock["lock_digest"], selection=selection,
            training_split="train_oof", selection_split="inner", validation_metrics=metrics,
            settings=settings, reward="reward_scale_times_daily_net_active_return_vs_synchronized_matched_core",
            training_state=dict(patience_best_score=patience_best_score, patience_best_epoch=patience_best_epoch,
                epochs_without_progress=stale, best_trained_score=best_trained_score,
                best_trained_epoch=best_trained_epoch, promoted_score=best_score,
                promoted_epoch=best_epoch, recent_scores=recent),
            team_credit="shared_team_reward_joint_conditional_likelihood_centralized_critic",
            manager_period_sessions=1, no_predictor_updates=True)
        # Run directory is unique; replace atomically for interrupted-run integrity.
        temporary = output / (name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(output / name)

    save("best.pth", 0, initial)
    for epoch in range(1, settings["epochs"]+1):
        started = time.perf_counter()
        policy.train()
        specs = []
        decisions_available = training.end-training.start-1
        window_max = min(settings["window_max"], decisions_available)
        if window_max < settings["window_min"]:
            raise ValueError("Insufficient causal OOF sessions for allocation training windows.")
        for _ in range(settings["episodes_per_update"]):
            steps = int(rng.integers(settings["window_min"], window_max+1))
            begin = int(rng.integers(training.start, training.end-steps))
            specs.append(ForecastEpisode(begin, begin+steps+1, training.horizon, training.forecasts))
        buffer = PPOBuffer(sum(e.end-e.start-1 for e in specs), policy.observation_dim,
                           policy.action_dim, gamma=settings["gamma"], lam=settings["gae_lambda"])
        episode_rewards = []
        for episode in specs:
            env, matched_env = data.env(episode), data.env(episode)
            env.reset()
            matched_env.reset()
            total_reward = 0.
            while env.current_step < episode.end-episode.start-1:
                position = episode.start+env.current_step
                state = data.state(episode, position, env.portfolio.weights, matched_env.portfolio.weights)
                with torch.no_grad():
                    action, logp, value = policy.act(torch.from_numpy(state))
                target, _ = policy.target(action)
                _, _, _, _, info = env.step(target)
                _, _, _, _, matched_info = matched_env.step(data.core_target)
                reward = settings["reward_scale"]*(info["net_return"]-matched_info["net_return"])
                buffer.store(state, action.numpy(), reward, value.item(), logp.item())
                total_reward += reward
            # Random windows are truncations, not market terminals. Only the
            # end of the available OOF phase is terminal.
            next_position = episode.end-1
            if next_position in training.forecasts:
                with torch.no_grad():
                    _, _, value = policy.act(torch.from_numpy(data.state(training, next_position,
                        env.portfolio.weights, matched_env.portfolio.weights)), deterministic=True)
                last_value = value.item()
            else:
                last_value = 0.
            buffer.finish_path(last_value)
            episode_rewards.append(total_reward)
        rollout_diagnostics = dict(rollout_reward_mean_step=float(buffer.rew_buf.mean()),
            rollout_reward_std_step=float(buffer.rew_buf.std()),
            rollout_return_std=float(buffer.ret_buf.std()), rollout_advantage_std=float(buffer.adv_buf.std()))
        update = updater.update(policy, optimizer, buffer.get(), train_iters=settings["ppo_iters"],
                                batch_size=settings["batch_size"])
        probe, probe_rows = evaluate_train_windows(data, windows, policy, method, probe_baselines, simulate)
        probe_history.extend(dict(epoch=epoch, **row) for row in probe_rows)
        result, _ = simulate(data, inner, method, policy)
        metrics = metric_summary(result, risk_free)
        recent = (recent+[metrics["sharpe_ratio"]])[-settings["selection_window"]:]
        smoothed = float(np.mean(recent))
        progress_improved = np.isfinite(smoothed) and smoothed > patience_best_score+min_delta
        if progress_improved:
            patience_best_score, patience_best_epoch, stale = smoothed, epoch, 0
        elif epoch >= settings["min_epochs"]:
            stale += 1
        diagnostic_improved = np.isfinite(metrics["sharpe_ratio"]) and metrics["sharpe_ratio"] > best_trained_score
        if diagnostic_improved:
            best_trained_score, best_trained_epoch = metrics["sharpe_ratio"], epoch
        # Select an actual checkpoint using its own metric. A rolling mean is a
        # separate stability gate, never reported as that checkpoint's Sharpe.
        eligible = (epoch >= settings["min_epochs"]
            and smoothed >= initial["sharpe_ratio"]+settings["selection_margin"]
            and metrics["sharpe_ratio"] > best_score)
        if eligible:
            best_score, best_epoch, best_metrics = metrics["sharpe_ratio"], epoch, metrics
            best_state = copy.deepcopy(policy.state_dict())
        rows.append(dict(epoch=epoch, checkpoint_status="trained", **metrics, **probe, **rollout_diagnostics,
            smoothed_inner_sharpe=smoothed, reward_mean=float(np.mean(episode_rewards)),
            inner_sharpe_delta_vs_matched=metrics["sharpe_ratio"]-matched["sharpe_ratio"],
            patience_improved=bool(progress_improved), patience_best_score=patience_best_score,
            epochs_without_progress=stale, best_trained_epoch=best_trained_epoch,
            policy_loss=update["LossPi"], value_loss=update["LossV"], kl=update["KL"],
            kl_first=update["FirstKL"], kl_max=update["MaxKL"], kl_stop=update["StopKL"],
            ppo_minibatches_completed=update["NumMinibatches"],
            ppo_minibatches_attempted=update["NumMinibatchesAttempted"],
            ppo_epochs_completed=update["StopIter"], ppo_stopped_early=update["StoppedEarly"],
            explained_variance=update["ExplainedVar"], elapsed_seconds=time.perf_counter()-started,
            selected=bool(eligible)))
        if eligible:
            save("best.pth", epoch, metrics)
        if diagnostic_improved:
            save("best_trained.pth", epoch, metrics)
        atomic_csv(pd.DataFrame(rows), output / "training_log.csv", index=False)
        atomic_csv(pd.DataFrame(probe_history), output / "train_probe_metrics.csv", index=False)
        save("last.pth", epoch, metrics)
        print(f"  {selection['fold_id']} {method} seed={seed} epoch={epoch}/{settings['epochs']} "
              f"inner Sharpe={metrics['sharpe_ratio']:.4f} Δmatched={metrics['sharpe_ratio']-matched['sharpe_ratio']:+.4f} "
              f"train probe ΔSharpe={probe['train_probe_mean_sharpe_delta_vs_matched']:+.4f} "
              f"patience={stale}/{settings['patience']} "
              f"KL={update['KL']:.4g} ({time.perf_counter()-started:.1f}s)", flush=True)
        if settings["patience"] and stale >= settings["patience"]:
            break
    policy.load_state_dict(best_state)
    policy.eval()
    return policy, dict(checkpoint_epoch=best_epoch,
        checkpoint_status="trained" if best_epoch else "untrained_reference",
        inner_sharpe=best_metrics["sharpe_ratio"], inner_sharpe_delta_vs_matched=best_score-matched["sharpe_ratio"],
        parameter_count=sum(p.numel() for p in policy.parameters()))

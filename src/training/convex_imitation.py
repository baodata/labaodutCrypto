"""Behavior cloning of causal convex decisions; predictor/critic stay frozen."""
from __future__ import annotations

from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import yaml

from src.env.orders import PortfolioOrder
from src.evaluation.backtester import DeterministicBacktester
from src.evaluation.cost_aware_allocator import cost_aware_target
from src.evaluation.decision_data import DecisionData, ROOT
from src.evaluation.policy_evaluator import metric_summary
from src.evaluation.predictor_lock import load_predictor_lock
from src.evaluation.signal_allocation import load_allocation_config
from src.models.forecast_allocator import ForecastAllocatorPolicy, project_box
from src.training.decision_probes import fixed_train_windows
from src.training.decision_trainer import simulate
from src.utils.artifacts import atomic_csv, atomic_json, file_hashes, new_run_directory, rng_state


def load_imitation_config(path):
    from src.utils.config import resolve_config_path
    settings = yaml.safe_load(resolve_config_path(path).read_text())
    integers = {"seed", "hidden_dim", "epochs", "batch_size", "evaluate_every", "cpu_threads",
                "train_probe_windows", "train_probe_days"}
    scalars = {"action_std", "learning_rate", "max_grad_norm", "inverse_clip_epsilon"}
    if (not isinstance(settings, dict) or set(settings) != integers | scalars | {"version", "fold", "method"}
            or settings["version"] != "convex-imitation-v1" or settings["method"] != "ppo"):
        raise ValueError("This capacity diagnostic supports the unconstrained-sector PPO actor only.")
    if any(type(settings[k]) is not int or settings[k] < (0 if k == "seed" else 1) for k in integers):
        raise ValueError("Invalid imitation count/seed setting.")
    if (not np.isfinite([settings[k] for k in scalars]).all() or any(settings[k] <= 0 for k in scalars)
            or settings["inverse_clip_epsilon"] >= .01):
        raise ValueError("Invalid imitation learning/noise/clip setting.")
    return settings


def teacher_target(data, episode, position, weights):
    return cost_aware_target(episode.forecasts[position][0], data.risk.covariance[position], weights,
        horizon=episode.horizon, action_alpha=data.alpha, cash_ratio=data.cash,
        tilt_fraction=data.allocation["tilt_fraction"], max_asset_weight=data.cap,
        risk_aversion=data.allocation["risk_aversion"], cost_rate=data.cost_rate,
        cost_multiplier=data.allocation["cost_multiplier"])[0]


def collect_rollout(data, episode, policy=None, *, persistence=False):
    """Teacher forcing when policy=None; otherwise teacher queried at learner holdings.

    Collect each pair before stepping prices. No realized label or next Open is
    passed to the solver or actor. Benchmark holdings advance after the decision.
    """
    env, matched = data.env(episode), data.env(episode)
    matched.reset()
    states, targets, positions, emitted = [], [], [], []

    def act(obs, info):
        position = episode.start+env.current_step
        if position not in episode.forecasts:
            matched.step(PortfolioOrder.hold())
            return PortfolioOrder.hold()
        state = data.state(episode, position, env.portfolio.weights, matched.portfolio.weights)
        target = teacher_target(data, episode, position, env.portfolio.weights.copy())
        if persistence:
            action = np.r_[project_box(env.portfolio.weights[:-1], data.lower, data.upper, 1-data.cash).numpy(), data.cash]
        elif policy is None:
            action = target
        else:
            with torch.no_grad():
                latent = policy.single_actor(torch.from_numpy(state))
                action = policy.target(latent)[0]
        states.append(state)
        targets.append(target)
        emitted.append(action)
        positions.append(position)
        matched.step(data.core_target)
        return action

    result = DeterministicBacktester(env).run_strategy(act)
    return result, dict(states=np.asarray(states, dtype=np.float32), targets=np.asarray(targets),
                        positions=np.asarray(positions), emitted=np.asarray(emitted))


def predict_targets(policy, states, batch_size):
    policy.eval()
    with torch.no_grad():
        return torch.cat([policy.target_tensors(policy.single_actor(torch.from_numpy(x)))[0]
                          for x in np.array_split(states, max(1, int(np.ceil(len(states)/batch_size))))]).cpu().numpy()


def fidelity(predicted, teacher, core, radius):
    error = predicted[:, :-1]-teacher[:, :-1]
    denominator = float(np.mean((teacher[:, :-1]-core)**2))
    mse = float(np.mean(error**2))
    return dict(mae_bps=float(np.mean(np.abs(error))*1e4), rmse_bps=float(np.sqrt(mse)*1e4),
        normalized_weight_mse=float(np.mean((error/radius)**2)),
        mse_improvement_vs_core=1-mse/denominator if denominator > 1e-24 else None,
        teacher_mean_abs_tilt_bps=float(np.mean(np.abs(teacher[:, :-1]-core))*1e4),
        predicted_mean_abs_tilt_bps=float(np.mean(np.abs(predicted[:, :-1]-core))*1e4))


def run_convex_imitation(lock_path, allocation_path, config_path, raw_dir, assets_path, env_path, output_dir=None):
    from src.utils.config import resolve_config_path
    config_path = resolve_config_path(config_path)
    allocation_path = resolve_config_path(allocation_path)
    assets_path = resolve_config_path(assets_path)
    env_path = resolve_config_path(env_path)
    settings, allocation = load_imitation_config(config_path), load_allocation_config(allocation_path)
    lock, bank = load_predictor_lock(lock_path)
    selection = next((s for s in lock["selections"] if s["fold_id"] == settings["fold"]), None)
    if selection is None or settings["seed"] not in bank.metadata["protocol"]["seeds"]:
        raise ValueError("Imitation fold/seed must belong to the frozen protocol.")
    if allocation["tilt_fraction"] <= 0:
        raise ValueError("Imitation needs a nonzero active allocation radius.")
    if selection["family"] not in allocation["models"] or selection["horizon_sessions"] not in allocation["horizons"]:
        raise ValueError("Allocator configuration does not cover the locked predictor.")
    output = Path(output_dir) if output_dir else new_run_directory(ROOT / "models/runs", "convex_imitation")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Imitation output already contains artifacts.")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(settings["cpu_threads"])
    torch.manual_seed(settings["seed"])
    np.random.seed(settings["seed"])
    random.seed(settings["seed"])
    rng = np.random.default_rng(settings["seed"])
    manifest = dict(status="running", settings=settings, predictor_lock_digest=lock["lock_digest"],
        selection=selection, fit_split="train_oof", evaluation_split="inner", outer_evaluated=False,
        test_evaluated=False, checkpoint_selection="fixed_final_epoch_no_inner_selection",
        loss="MSE_of_stock_targets_divided_by_core_times_tilt_squared_exact_execution_mapping",
        state_preprocessing="original_raw_decision_state_no_new_normalizer", critic_trained=False,
        teacher_state_distribution="continuous_train_teacher_rollout_plus_fixed_train_windows",
        source_hashes=file_hashes(list((ROOT / "src").rglob("*.py")) + [ROOT / "scripts/train_convex_imitation.py"], ROOT),
        config_hashes=file_hashes([Path(config_path), Path(allocation_path), Path(assets_path), Path(env_path)], ROOT),
        data_hashes=bank.metadata["data_hashes"])
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "predictor_lock.json", lock)
    print(f"Imitation artifacts: {output}", flush=True)
    try:
        data = DecisionData(bank, lock, allocation, raw_dir, assets_path, env_path, output)
        train, inner = data.episode(selection, "train_oof"), data.episode(selection, "inner")
        windows = [train]+fixed_train_windows(train, settings["train_probe_windows"], settings["train_probe_days"])
        atomic_json(output / "teacher_windows.json", dict(windows=[dict(window_id=i,
            first_decision=data.tensor.dates[w.start], last_mark=data.tensor.dates[w.end-1],
            kind="continuous_oof" if i == 0 else "fixed_train_window") for i, w in enumerate(windows)]))
        pairs, catalog = [], []
        teacher_train = None
        for index, window in enumerate(windows):
            result, samples = collect_rollout(data, window)
            if index == 0:
                teacher_train = result
            pairs.append(samples)
            catalog.extend(dict(split="train_oof", window_id=index, decision_date=data.tensor.dates[int(p)])
                           for p in samples["positions"])
        teacher_inner, heldout = collect_rollout(data, inner)
        catalog.extend(dict(split="inner", window_id=-1, decision_date=data.tensor.dates[int(p)]) for p in heldout["positions"])
        states = np.concatenate([p["states"] for p in pairs])
        targets = np.concatenate([p["targets"] for p in pairs])
        np.savez_compressed(output / "teacher_pairs.npz", train_states=states, train_targets=targets,
                            inner_states=heldout["states"], inner_targets=heldout["targets"])
        atomic_csv(pd.DataFrame(catalog), output / "teacher_samples.csv", index=False)
        policy_config = dict(observation_dim=states.shape[1], groups=data.groups, method="ppo", cash_ratio=data.cash,
            tilt_fraction=allocation["tilt_fraction"], max_asset_weight=data.cap,
            hidden_dim=settings["hidden_dim"], action_std=settings["action_std"])
        policy = ForecastAllocatorPolicy(**policy_config)
        # Critic is intentionally untouched. Loss follows the deterministic
        # actor mean through the very same tanh/projection as live execution.
        for parameter in policy.critic.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.Adam(policy.single_actor.parameters(), lr=settings["learning_rate"])
        radius = data.core*allocation["tilt_fraction"]
        x, y, scale = torch.from_numpy(states), torch.from_numpy(targets[:, :-1]), torch.from_numpy(radius)
        # Analytic inverse gives an approximate representability floor without
        # learning; finite tanh scores approximate vertices, never change bounds.
        z = (targets[:, :-1]-data.core)/radius
        inverse = np.arctanh(np.clip(z, -1+settings["inverse_clip_epsilon"], 1-settings["inverse_clip_epsilon"]))
        with torch.no_grad():
            reconstructed = policy.target_tensors(torch.from_numpy(inverse))[0].numpy()
        inverse_metrics = fidelity(reconstructed, targets, data.core, radius)
        atomic_json(output / "action_mapping_floor.json", inverse_metrics)
        history = []

        def evaluate(epoch, elapsed, gradient):
            for split, inputs, teacher in (("train_oof", states, targets), ("inner", heldout["states"], heldout["targets"])):
                prediction = predict_targets(policy, inputs, settings["batch_size"])
                n = policy.n
                holdings_offset = 2*n+n*n
                with torch.no_grad():
                    copied = project_box(torch.from_numpy(inputs[:, holdings_offset:holdings_offset+n]),
                                         policy.lower, policy.upper, 1-policy.cash).numpy()
                persistence_error = copied-teacher[:, :-1]
                actor_error = prediction[:, :-1]-teacher[:, :-1]
                ablated = inputs.copy()
                ablated[:, :2*n] = 0
                zero_forecast = predict_targets(policy, ablated, settings["batch_size"])
                history.append(dict(epoch=epoch, split=split, samples=len(inputs), elapsed_seconds=elapsed,
                    last_gradient_norm_before_clip=gradient,
                    persistence_mae_bps=float(np.abs(persistence_error).mean()*1e4),
                    persistence_rmse_bps=float(np.sqrt(np.mean(persistence_error**2))*1e4),
                    mse_improvement_vs_persistence=1-float(np.mean(actor_error**2))/max(float(np.mean(persistence_error**2)), 1e-24),
                    zero_forecast_target_change_bps=float(np.abs(prediction-zero_forecast).mean()*1e4),
                    **fidelity(prediction, teacher, data.core, radius)))
            atomic_csv(pd.DataFrame(history), output / "imitation_learning_curves.csv", index=False)
            last = history[-2:]
            print(f"epoch={epoch:03d}/{settings['epochs']} "
                  f"train MAE={last[0]['mae_bps']:.3f}bp inner MAE={last[1]['mae_bps']:.3f}bp "
                  f"train relative MSE={last[0]['normalized_weight_mse']:.4f}", flush=True)

        evaluate(0, 0., None)
        started = time.perf_counter()
        for epoch in range(1, settings["epochs"]+1):
            policy.train()
            for indices in np.array_split(rng.permutation(len(x)), int(np.ceil(len(x)/settings["batch_size"]))):
                optimizer.zero_grad(set_to_none=True)
                predicted = policy.target_tensors(policy.single_actor(x[indices]))[0][:, :-1]
                loss = ((predicted-y[indices])/scale).square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite imitation loss.")
                loss.backward()
                gradient = float(torch.nn.utils.clip_grad_norm_(policy.single_actor.parameters(), settings["max_grad_norm"]))
                if not np.isfinite(gradient):
                    raise RuntimeError("Nonfinite imitation gradient.")
                optimizer.step()
            if epoch == 1 or epoch % settings["evaluate_every"] == 0 or epoch == settings["epochs"]:
                evaluate(epoch, time.perf_counter()-started, gradient)
        policy.eval()
        torch.save(dict(model_state_dict=policy.state_dict(), policy_config=policy_config,
            optimizer_state_dict=optimizer.state_dict(), rng_state=rng_state(), epoch=settings["epochs"], settings=settings,
            checkpoint_role="convex_imitation_diagnostic_not_promoted", critic_trained=False,
            predictor_lock_digest=lock["lock_digest"], tickers=data.tensor.tickers,
            state_preprocessing="original_raw_decision_state_no_new_normalizer", selection=selection), output / "imitation.pth")
        finances, daily, closed_fidelity = [], [], []
        for split, episode, teacher in (("train_oof", train, teacher_train), ("inner", inner, teacher_inner)):
            learner, samples = collect_rollout(data, episode, policy)
            persistence, _ = collect_rollout(data, episode, persistence=True)
            baseline = simulate(data, episode, "matched_1n")[0]
            closed_fidelity.append(dict(split=split, state_distribution="learner_own_holdings",
                **fidelity(samples["emitted"], samples["targets"], data.core, radius)))
            reference = metric_summary(baseline, float(data.environment["risk_free_rate"]))
            for name, result in (("matched_1n", baseline), ("teacher_convex", teacher),
                                 ("persistence_projected", persistence), ("imitation", learner)):
                metrics = metric_summary(result, float(data.environment["risk_free_rate"]))
                finances.append(dict(split=split, method=name, **metrics,
                    sharpe_delta_vs_matched=metrics["sharpe_ratio"]-reference["sharpe_ratio"]))
                daily.append(pd.DataFrame(dict(split=split, method=name, date=data.tensor.dates[episode.start+1:episode.end],
                    net_return=result.daily_net_returns, turnover=result.daily_turnover, costs=result.daily_costs)))
        atomic_csv(pd.DataFrame(finances), output / "portfolio_metrics.csv", index=False)
        atomic_csv(pd.concat(daily, ignore_index=True), output / "daily_returns.csv", index=False)
        atomic_csv(pd.DataFrame(closed_fidelity), output / "closed_loop_fidelity.csv", index=False)
        atomic_json(output / "decision.json", dict(decision="IMITATION_DIAGNOSTIC_ONLY", independent_alpha_claim=False,
            promotion_gate_changed=False, predictor_updated=False, critic_trained=False, outer_evaluated=False,
            test_evaluated=False, inner_used_for_checkpoint_selection=False,
            caveat="One-step fidelity on teacher states differs from closed-loop portfolio performance; fixed-budget raw-state MLP fit."))
        manifest.update(status="completed", train_samples=len(states), inner_samples=len(heldout["states"]),
            unique_train_dates=len({r["decision_date"] for r in catalog if r["split"] == "train_oof"}),
            artifact_hashes=file_hashes([output / "teacher_pairs.npz", output / "imitation.pth"], output))
        print(pd.DataFrame(finances)[["split", "method", "sharpe_ratio", "sharpe_delta_vs_matched"]].to_string(index=False), flush=True)
        return output
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        atomic_json(output / "manifest.json", manifest)

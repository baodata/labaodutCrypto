"""Compare a locked predictor with convex, PPO, sector MARL and H-MARL decisions."""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.evaluation.decision_data import DecisionData, ROOT
from src.evaluation.metrics import block_bootstrap_excess_returns
from src.evaluation.policy_evaluator import metric_summary
from src.evaluation.predictor_lock import load_predictor_lock
from src.evaluation.signal_allocation import load_allocation_config, _holm_adjust
from src.models.forecast_allocator import METHODS
from src.training.decision_trainer import simulate, train_decision_policy
from src.utils.artifacts import atomic_json, atomic_csv, file_hashes, new_run_directory


def load_decision_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if (not isinstance(config, dict) or set(config) != {"version", "seeds", "methods", "training"}
            or config["version"] != "portfolio-decision-v1"):
        raise ValueError("Invalid portfolio decision configuration.")
    if config["methods"] != list(METHODS):
        raise ValueError("Comparison must include convex, PPO, sector MARL and H-MARL in that order.")
    if (not config["seeds"] or len(config["seeds"]) != len(set(config["seeds"]))
            or any(type(s) is not int or s < 0 for s in config["seeds"])):
        raise ValueError("Unique nonnegative integer seeds required.")
    settings = config["training"]
    # Older decision profiles retain their CLI/config compatibility.
    for name, value in dict(patience_min_delta=1e-5, train_probe_windows=3, train_probe_days=126).items():
        settings.setdefault(name, value)
    integers = {"epochs", "min_epochs", "patience", "selection_window", "hidden_dim", "episodes_per_update",
                "window_min", "window_max", "ppo_iters", "batch_size", "cpu_threads", "train_probe_windows", "train_probe_days"}
    scalars = {"selection_margin", "action_std", "learning_rate", "target_kl", "gamma", "gae_lambda", "reward_scale", "patience_min_delta"}
    if set(settings) != integers | scalars:
        raise ValueError("Training parameters must be explicit and use supported keys.")
    if any(type(settings[k]) is not int or settings[k] < (0 if k == "patience" else 1) for k in integers):
        raise ValueError("Invalid training count/window parameter.")
    if (not np.isfinite([settings[k] for k in scalars]).all()
            or any(settings[k] <= 0 for k in scalars - {"selection_margin", "patience_min_delta"})
            or min(settings["selection_margin"], settings["patience_min_delta"]) < 0 or settings["gamma"] > 1 or settings["gae_lambda"] > 1
            or settings["window_min"] > settings["window_max"] or settings["min_epochs"] > settings["epochs"]):
        raise ValueError("Invalid training settings.")
    return config


def summarize_decisions(metrics, daily, protocol, horizon):
    """Paired temporal bootstrap on seed-mean returns, never treat seeds as new days."""
    # Target projection/CSV round trips can differ by float64 ULPs. Do not count
    # a 1e-16 return difference as an economically positive fold or bootstrap it
    # into spurious evidence. This tolerance is in decimal return units.
    tolerance = 1e-12
    outer = daily[daily.split == "outer"]
    keys = ["fold_id", "date"]
    panels = {}
    for method, rows in outer.groupby("method", sort=False):
        if rows.duplicated(keys+["seed"]).any():
            raise ValueError("Duplicate daily seed/fold/session result.")
        panels[method] = rows.groupby(keys, sort=True).net_return.mean()
    for panel in panels.values():
        if not panel.index.equals(panels["matched_1n"].index):
            raise ValueError("Allocator comparison requires identical outer sessions.")
    contrasts = [(m, "matched_1n") for m in METHODS]
    contrasts += list(zip(METHODS[1:], METHODS[:-1])) + [("hmarl", "convex")]
    records = []
    for candidate, reference in contrasts:
        left, right = panels[candidate], panels[reference]
        candidate_returns, reference_returns = left.to_numpy().copy(), right.to_numpy()
        if np.max(np.abs(candidate_returns-reference_returns)) <= tolerance:
            candidate_returns = reference_returns.copy()
        stats = block_bootstrap_excess_returns(candidate_returns, reference_returns,
            block_size=max(horizon, protocol["bootstrap"]["block_size"]),
            num_samples=protocol["bootstrap"]["samples"], seed=protocol["bootstrap"]["seed"],
            groups=left.index.get_level_values("fold_id").to_numpy())
        # Convex/core have one deterministic run; learned methods retain seed
        # variability in metrics_by_fold.csv and are averaged only within fold.
        means = metrics[metrics.split == "outer"].groupby(["method", "fold_id"])[
            ["sharpe_ratio", "cumulative_return", "max_drawdown", "total_turnover", "total_costs"]].mean()
        a, b = means.loc[candidate], means.loc[reference]
        if not a.index.equals(b.index):
            raise ValueError("Allocator metrics have different folds.")
        sharpe_delta = float((a.sharpe_ratio-b.sharpe_ratio).mean())
        records.append(dict(candidate=candidate, reference=reference,
            mean_sharpe_delta=0. if abs(sharpe_delta) <= tolerance else sharpe_delta,
            positive_folds=int((a.cumulative_return-b.cumulative_return > tolerance).sum()),
            worst_drawdown_degradation=float((b.max_drawdown-a.max_drawdown).max()),
            mean_turnover_delta=float((a.total_turnover-b.total_turnover).mean()),
            mean_cost_delta=float((a.total_costs-b.total_costs).mean()),
            annual_net_excess=stats["mean_excess_annualized"], ci_lower_95=stats["ci_lower_95"],
            ci_upper_95=stats["ci_upper_95"], p_value=stats["p_value_positive_excess"]))
    summary = pd.DataFrame(records)
    summary["holm_p_value"] = _holm_adjust(summary.p_value.to_numpy())
    gate = protocol["gate"]
    summary["passes_portfolio_gate"] = ((summary.positive_folds >= gate["minimum_positive_folds"])
        & (summary.annual_net_excess >= gate["minimum_annualized_net_excess"])
        & (summary.ci_lower_95 > gate["minimum_ci_lower"]) & (summary.holm_p_value < .05)
        & (summary.mean_sharpe_delta > gate["minimum_mean_sharpe_delta"])
        & (summary.worst_drawdown_degradation <= gate["maximum_drawdown_degradation"]))
    return summary


def compare_portfolio_decisions(lock_path, allocation_path, decision_path, raw_dir, assets_path,
                                env_path, output_dir=None, bank_dir=None, *, development_only=False, fold_id=None):
    lock, bank = load_predictor_lock(lock_path, bank_dir)
    allocation, config = load_allocation_config(allocation_path), load_decision_config(decision_path)
    if fold_id is not None and not development_only:
        raise ValueError("A single-fold run is for development-only learning curves; full comparison requires all folds.")
    selections = [s for s in lock["selections"] if fold_id is None or s["fold_id"] == fold_id]
    if not selections:
        raise ValueError("Requested fold is absent from the predictor lock.")
    if not set(config["seeds"]).issubset(bank.metadata["protocol"]["seeds"]):
        raise ValueError("Allocator seeds must belong to the frozen research protocol.")
    if (lock["horizon_sessions"] not in allocation["horizons"]
            or not {s["family"] for s in lock["selections"]}.issubset(allocation["models"])):
        raise ValueError("Allocation config must cover locked families and horizon.")
    output = Path(output_dir) if output_dir else new_run_directory(ROOT / "models/runs", "portfolio_decision")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output already has artifacts; choose a new run.")
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(status="running", predictor_lock_digest=lock["lock_digest"], config=config,
        development_only=development_only, selected_folds=[s["fold_id"] for s in selections],
        outer_evaluated=not development_only,
        allocation_config=allocation, test_evaluated=False, outer_used_for_selection=False,
        predictor_selection="per_fold_inner_rank_IC_fixed_horizon",
        checkpoint_selection="same_inner_period_reused_for_all_allocators_nested_development_not_confirmation",
        training="causal_candidate_OOF_selected_offline_after_inner_selection_not_historical_live_simulation",
        actor_inputs="same_forecasts_full_causal_covariance_close_holdings_and_known_costs_no_OHLCV",
        team_credit="shared_daily_net_active_reward_joint_PPO_centralized_critic",
        period_status=bank.metadata["protocol"]["period_status"],
        caveat="Development outer results were explored previously; this is not a fresh holdout.",
        source_hashes=file_hashes(list((ROOT / "src").rglob("*.py"))+
            [ROOT / "scripts/compare_portfolio_decisions.py"], ROOT),
        config_hashes=file_hashes([Path(allocation_path), Path(decision_path), Path(assets_path), Path(env_path)], ROOT),
        data_hashes=bank.metadata["data_hashes"])
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "predictor_lock.json", lock)
    metrics, daily, decisions = [], [], []
    torch.set_num_threads(config["training"]["cpu_threads"])
    try:
        data = DecisionData(bank, lock, allocation, raw_dir, assets_path, env_path, output)

        def record(selection, phase, method, seed, result, actions, metadata):
            episode = data.episode(selection, phase)
            financial = metric_summary(result, float(data.environment["risk_free_rate"]))
            identity = dict(fold_id=selection["fold_id"], split=phase, method=method, seed=seed,
                predictor=selection["family"], horizon_sessions=selection["horizon_sessions"])
            metrics.append(dict(identity, **financial, **metadata,
                mean_realized_cash=float(result.weights_history[1:, -1].mean()),
                max_target_bound_violation=max([max(float(np.max(data.lower-np.array(
                    [row[f"target_{t}"] for t in data.tensor.tickers]))),
                    float(np.max(np.array([row[f"target_{t}"] for t in data.tensor.tickers])-data.upper)), 0.)
                    for row in actions] or [0.])))
            daily.append(pd.DataFrame(dict(identity, date=data.tensor.dates[episode.start+1:episode.end],
                net_return=result.daily_net_returns, turnover=result.daily_turnover, costs=result.daily_costs)))
            decisions.extend([dict(identity, **row) for row in actions])
            atomic_csv(pd.DataFrame(metrics), output / "metrics_by_fold.csv", index=False)
            atomic_csv(pd.concat(daily, ignore_index=True), output / "daily_returns.csv", index=False)
            atomic_csv(pd.DataFrame(decisions), output / "allocation_decisions.csv", index=False)

        for selection in selections:
            print(f"Locked {selection['fold_id']}: {selection['family']} h={selection['horizon_sessions']}", flush=True)
            # All fitting/selection in this fold finishes before any outer evaluation.
            trained = []
            for method, seed in itertools.product(METHODS[1:], config["seeds"]):
                policy, metadata = train_decision_policy(data, selection, method, seed, config["training"],
                    output / "policies" / selection["fold_id"] / f"{method}_seed{seed}")
                trained.append((method, seed, policy, metadata))
                result, actions = simulate(data, data.episode(selection, "inner"), method, policy)
                record(selection, "inner", method, seed, result, actions, metadata)
            for phase in (("inner",) if development_only else ("inner", "outer")):
                episode = data.episode(selection, phase)
                for method in ("matched_1n", "convex"):
                    result, actions = simulate(data, episode, method)
                    record(selection, phase, method, -1, result, actions,
                           dict(checkpoint_epoch=-1, checkpoint_status="deterministic", parameter_count=0))
                if phase == "outer":
                    for method, seed, policy, metadata in trained:
                        result, actions = simulate(data, episode, method, policy)
                        record(selection, phase, method, seed, result, actions, metadata)
        metrics, daily = pd.DataFrame(metrics), pd.concat(daily, ignore_index=True)
        base = metrics[metrics.method == "matched_1n"].set_index(["fold_id", "split"])
        metrics["sharpe_delta_vs_matched_baseline"] = [row.sharpe_ratio-base.loc[
            (row.fold_id, row.split), "sharpe_ratio"] for row in metrics.itertuples()]
        atomic_csv(metrics, output / "metrics_by_fold.csv", index=False)
        curves = []
        for path in sorted((output / "policies").glob("*/*/training_log.csv")):
            curve = pd.read_csv(path)
            curve["fold_id"] = path.parents[1].name
            curve["method"], seed_text = path.parent.name.rsplit("_seed", 1)
            curve["seed"] = int(seed_text)
            curves.append(curve)
        learning_curves = pd.concat(curves, ignore_index=True)
        atomic_csv(learning_curves, output / "learning_curves.csv", index=False)
        if development_only:
            atomic_json(output / "decision.json", dict(decision="DEVELOPMENT_LEARNING_CURVES_ONLY",
                outer_evaluated=False, test_evaluated=False, independent_alpha_claim=False,
                predictor_lock_digest=lock["lock_digest"], promotion_gate_unchanged=True,
                patience_rule="smoothed_inner_high_water_mark_plus_patience_min_delta_independent_of_promotion"))
            manifest.update(status="completed", comparisons=0)
            return learning_curves, output
        summary = summarize_decisions(metrics, daily, data.protocol, lock["horizon_sessions"])
        atomic_csv(summary, output / "paired_comparison.csv", index=False)
        columns = ["sharpe_ratio", "sharpe_delta_vs_matched_baseline", "cumulative_return", "total_turnover", "total_costs"]
        seed_summary = metrics[metrics.seed >= 0].groupby(["split", "method", "fold_id"])[
            ["sharpe_ratio", "sharpe_delta_vs_matched_baseline", "cumulative_return", "total_turnover", "total_costs"]].agg(["mean", "std"])
        atomic_csv(seed_summary, output / "seed_fold_summary.csv")
        macro_seed = metrics[metrics.seed >= 0].groupby(["split", "method", "seed"])[columns].mean()
        atomic_csv(macro_seed, output / "seed_macro_fold_metrics.csv")
        atomic_csv(macro_seed.groupby(level=["split", "method"]).agg(["mean", "std"]), output / "seeds_summary.csv")
        # No automatic escalation based on a winner in already explored periods.
        atomic_json(output / "decision.json", dict(decision="REVIEW_DEVELOPMENT_RESULTS",
            test_evaluated=False, independent_alpha_claim=False, predictor_lock_digest=lock["lock_digest"],
            portfolio_gate_passes=int(summary.passes_portfolio_gate.sum()),
            positive_inner_ic_folds=sum(s["positive_inner_ic"] for s in lock["selections"]),
            signal_uncertainty_gate_evaluated=False, multiplicity="Holm_over_8_paired_contrasts",
            numeric_return_tolerance=1e-12,
            bootstrap="time_blocks_within_fold_on_seed_mean_returns_seeds_are_not_independent_market_days",
            confidence_intervals="pointwise_not_simultaneous_conditional_on_fitted_policies",
            baseline_initialization="100_percent_cash_alpha_ramp_equal_for_every_branch",
            topology_note="sector_MARL_fixed_group_budgets_HMARL_learned_daily_manager_no_multiscale_claim"))
        manifest.update(status="completed", comparisons=len(summary))
        print(summary.to_string(index=False), flush=True)
        return summary, output
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        atomic_json(output / "manifest.json", manifest)

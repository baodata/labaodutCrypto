"""Evaluate one frozen RSI/momentum tilt against scheduled 1/N baselines."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_equal_weight_tilt import allocation_diagnostics, make_tilt_policy
from src.env.cost import resolve_transaction_cost_components
from src.evaluation.backtester import DeterministicBacktester
from src.evaluation.metrics import FinancialMetrics
from src.features.pipeline import FeaturePipeline
from src.training.data_preparation import close_price_matrix, open_price_matrix
from src.training.env_factory import create_env
from src.evaluation.policy_evaluator import metric_summary
from src.utils.data_types import MarketDataTensor
from src.utils.artifacts import atomic_csv, atomic_json, file_hashes


def bootstrap_mean_ci(
    differences: list[np.ndarray],
    block_size: int,
    n_bootstrap: int,
    confidence: float,
    seed: int = 20261014,
) -> tuple[float, float, float]:
    """Paired circular moving-block bootstrap, resampling within each period."""
    if not differences or n_bootstrap <= 0 or block_size <= 0 or not 0 < confidence < 1:
        raise ValueError("Thông số bootstrap không hợp lệ.")
    if any(np.asarray(v).ndim != 1 or len(v) < block_size or not np.isfinite(v).all() for v in differences):
        raise ValueError("Chuỗi bootstrap phải hữu hạn và đủ dài cho block.")
    rng = np.random.default_rng(seed)
    n_total = sum(len(values) for values in differences)
    observed = sum(float(values.sum()) for values in differences) / n_total * 252.0
    boot_sums = np.zeros(n_bootstrap, dtype=np.float64)
    for values in differences:
        n = len(values)
        blocks_per_rep = int(np.ceil(n / block_size))
        starts = rng.integers(0, n, size=(n_bootstrap, blocks_per_rep))
        offsets = np.arange(block_size)
        indices = (starts[..., None] + offsets) % n
        samples = values[indices].reshape(n_bootstrap, -1)[:, :n]
        boot_sums += samples.sum(axis=1)
    estimates = boot_sums / n_total * 252.0
    tail = (1.0 - confidence) / 2.0
    return observed, float(np.quantile(estimates, tail)), float(np.quantile(estimates, 1.0 - tail))


def validate_gate(gate: dict) -> None:
    candidate = gate["candidate"]
    acceptance = gate["acceptance_gate"]
    if candidate["rsi_orientation"] not in {"momentum", "contrarian"}:
        raise ValueError("RSI orientation phải là momentum hoặc contrarian.")
    for key in ("cash_ratio", "action_alpha", "initial_action_alpha", "tilt_fraction_of_1n", "no_trade_l1"):
        if not np.isfinite(candidate[key]):
            raise ValueError(f"Candidate {key} phải hữu hạn.")
    if not 0 <= candidate["cash_ratio"] < 1 or not 0 < candidate["action_alpha"] <= 1 or not 0 < candidate["initial_action_alpha"] <= 1:
        raise ValueError("Cash/alpha không hợp lệ.")
    if not 0 <= candidate["tilt_fraction_of_1n"] <= 1 or candidate["no_trade_l1"] < 0:
        raise ValueError("Tilt/no-trade không hợp lệ.")
    weights = candidate["score_weights"]
    if set(weights) != {"mom_12_1_rank", "rsi_14_rank"} or not np.isfinite(list(weights.values())).all() or min(weights.values()) < 0 or not np.isclose(sum(weights.values()), 1):
        raise ValueError("Score weights không hợp lệ.")
    if not gate["periods"] or len({p["name"] for p in gate["periods"]}) != len(gate["periods"]):
        raise ValueError("Periods phải không rỗng, tên không trùng.")
    intervals = sorted((pd.Timestamp(p["start"]), pd.Timestamp(p["end"])) for p in gate["periods"])
    if any(a >= b for a, b in intervals) or any(b >= c for (_, b), (c, _) in zip(intervals, intervals[1:])):
        raise ValueError("Periods phải tăng dần và không chồng lấn.")
    if not gate["baselines"] or len({b["name"] for b in gate["baselines"]}) != len(gate["baselines"]):
        raise ValueError("Baselines phải không rỗng, tên không trùng.")
    positive_counts = [candidate["rebalance_every_sessions"]] + [b["rebalance_every_sessions"] for b in gate["baselines"]] + [acceptance["block_bootstrap_size_sessions"], acceptance["bootstrap_replicates"], acceptance["minimum_positive_periods"]]
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in positive_counts):
        raise ValueError("Lịch/độ dài block/số lần bootstrap phải là số nguyên dương.")
    if not 0 < acceptance["confidence_level"] < 1:
        raise ValueError("confidence_level phải thuộc (0,1).")
    if not 1 <= acceptance["minimum_positive_periods"] <= len(gate["periods"]):
        raise ValueError("minimum_positive_periods vượt số giai đoạn.")
    for key in ("minimum_annualized_net_excess_return", "minimum_bootstrap_lower_bound_annualized_excess", "maximum_mean_drawdown_degradation", "minimum_mean_sharpe_delta"):
        if not np.isfinite(acceptance[key]):
            raise ValueError(f"Ngưỡng {key} phải hữu hạn.")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-config", type=Path, default=PROJECT_ROOT / "configs/core/tilt_robustness_gate.yaml")
    parser.add_argument("--env-config", type=Path, default=PROJECT_ROOT / "configs/core/env.yaml")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments")
    args = parser.parse_args(argv)
    gate_path = args.gate_config
    with gate_path.open("r", encoding="utf-8") as stream:
        gate = yaml.safe_load(stream)
    validate_gate(gate)
    candidate = gate["candidate"]
    acceptance = gate["acceptance_gate"]
    env_path = args.env_config
    with env_path.open("r", encoding="utf-8") as stream:
        env_config = yaml.safe_load(stream)["environment"]

    processed, full_tensor = FeaturePipeline().run_from_raw(
        raw_dir=args.raw_dir,
        export_parquet=False,
    )
    tickers = list(full_tensor.tickers)
    dates = [pd.Timestamp(date) for date in full_tensor.dates]
    date_index = {date.strftime("%Y-%m-%d"): i for i, date in enumerate(dates)}
    all_open = open_price_matrix(processed, tickers)
    all_close = close_price_matrix(processed, tickers, full_tensor.dates)
    feature_names = list(full_tensor.feature_names)
    feature_indices = {
        name: feature_names.index(name) for name in ("rsi_14", "mom_12_1")
    }
    reward_beta = float(env_config["reward_beta"])
    reward_free = float(env_config["risk_free_rate"])
    transaction_cost_rate = resolve_transaction_cost_components(env_config)[2]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "resolved_gate": gate, "resolved_environment": env_config,
        "feature_parameters": FeaturePipeline().parameters,
        "source_hashes": file_hashes(list((PROJECT_ROOT / "src").rglob("*.py")) + list((PROJECT_ROOT / "scripts").glob("*.py")), PROJECT_ROOT),
        "data_hashes": file_hashes(args.raw_dir.glob("*.parquet"), PROJECT_ROOT),
        "config_hashes": file_hashes([gate_path, env_path, PROJECT_ROOT / "configs/core/assets.yaml"], PROJECT_ROOT),
        "status": "running",
    }
    atomic_json(output_dir / "tilt_robustness_manifest.json", manifest)

    fold_rows: list[dict[str, float | str | int]] = []
    period_results: dict[str, dict[str, tuple[object, dict[str, float]]]] = {}
    for period in gate["periods"]:
        start = date_index[period["start"]]
        end = date_index[period["end"]] + 1
        if start < full_tensor.warmup_sessions:
            raise ValueError(f"{period['name']} bắt đầu trước khi feature warm-up hoàn tất.")
        period_dates = full_tensor.dates[start:end]
        period_tensor = MarketDataTensor(
            tensor=full_tensor.tensor[start:end].copy(),
            tickers=tickers,
            feature_names=feature_names,
            dates=list(period_dates),
        )
        period_tensor.validate()
        period_open = all_open[start:end]
        period_close = all_close[start:end]
        period_features = full_tensor.tensor[start:end].astype(np.float64, copy=False)

        def make_env():
            return create_env(
                period_tensor,
                period_open,
                env_path,
                reward_beta,
                reward_scaling=float(env_config.get("reward_scaling", 1.0)),
                history_tensor=full_tensor,
                history_start_index=start,
                reward_baseline="equal_weight",
                lookback_window=20,
                action_alpha=float(candidate["action_alpha"]),
                close_prices=period_close,
            )

        policies = {
            "frozen_weekly_RSI_momentum_tilt": make_tilt_policy(
                period_features,
                feature_indices,
                len(tickers),
                float(candidate["cash_ratio"]),
                float(candidate["tilt_fraction_of_1n"]),
                int(candidate["rebalance_every_sessions"]),
                float(candidate["no_trade_l1"]),
                1 if candidate["rsi_orientation"] == "momentum" else -1,
                initial_action_alpha=float(candidate.get("initial_action_alpha", 1.0)),
                score_weights=(float(candidate["score_weights"]["mom_12_1_rank"]), float(candidate["score_weights"]["rsi_14_rank"])),
            )
        }
        for baseline in gate["baselines"]:
            name = str(baseline["name"])
            policies[name] = make_tilt_policy(
                np.zeros_like(period_features),
                feature_indices,
                len(tickers),
                float(candidate["cash_ratio"]),
                0.0,
                int(baseline["rebalance_every_sessions"]),
                0.0,
                1,
                initial_action_alpha=float(candidate.get("initial_action_alpha", 1.0)),
            )

        period_results[period["name"]] = {}
        for strategy_name, policy in policies.items():
            result = DeterministicBacktester(make_env()).run_strategy(policy)
            metrics = metric_summary(result, reward_free)
            period_results[period["name"]][strategy_name] = (result, metrics)
            interval = (
                int(candidate["rebalance_every_sessions"])
                if strategy_name == "frozen_weekly_RSI_momentum_tilt"
                else next(int(b["rebalance_every_sessions"]) for b in gate["baselines"] if b["name"] == strategy_name)
            )
            fold_rows.append({
                "period": period["name"],
                "strategy": strategy_name,
                "sharpe_convention": FinancialMetrics.SHARPE_CONVENTION,
                "start": period["start"],
                "end": period["end"],
                "steps": result.total_steps,
                "cumulative_return": metrics["cumulative_return"],
                "annualized_return": metrics["annualized_return"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "total_turnover": metrics["total_turnover"],
                "total_costs": metrics["total_costs"],
                "initial_action_alpha": float(candidate.get("initial_action_alpha", 1.0)),
                **allocation_diagnostics(result, interval),
            })

    comparison_rows: list[dict[str, float | str | int | bool]] = []
    candidate_name = "frozen_weekly_RSI_momentum_tilt"
    for baseline in gate["baselines"]:
        baseline_name = str(baseline["name"])
        paired_differences: list[np.ndarray] = []
        positive_periods = 0
        sharpe_deltas: list[float] = []
        drawdown_degradations: list[float] = []
        period_excesses: list[float] = []
        for period in gate["periods"]:
            candidate_result, candidate_metrics = period_results[period["name"]][candidate_name]
            baseline_result, baseline_metrics = period_results[period["name"]][baseline_name]
            if len(candidate_result.daily_net_returns) != len(baseline_result.daily_net_returns):
                raise RuntimeError(f"Daily returns do not align for {period['name']} vs {baseline_name}.")
            daily_difference = (
                np.asarray(candidate_result.daily_net_returns, dtype=np.float64)
                - np.asarray(baseline_result.daily_net_returns, dtype=np.float64)
            )
            paired_differences.append(daily_difference)
            period_excess = float(np.mean(daily_difference) * 252.0)
            period_excesses.append(period_excess)
            positive_periods += int(period_excess > 0.0)
            sharpe_deltas.append(candidate_metrics["sharpe_ratio"] - baseline_metrics["sharpe_ratio"])
            drawdown_degradations.append(baseline_metrics["max_drawdown"] - candidate_metrics["max_drawdown"])

        mean_excess, ci_low, ci_high = bootstrap_mean_ci(
            paired_differences,
            int(acceptance["block_bootstrap_size_sessions"]),
            int(acceptance["bootstrap_replicates"]),
            float(acceptance["confidence_level"]),
            seed=20261014 + len(comparison_rows),
        )
        mean_sharpe_delta = float(np.mean(sharpe_deltas))
        mean_drawdown_degradation = float(np.mean(drawdown_degradations))
        passed = (
            positive_periods >= int(acceptance["minimum_positive_periods"])
            and mean_excess >= float(acceptance["minimum_annualized_net_excess_return"])
            and ci_low > float(acceptance["minimum_bootstrap_lower_bound_annualized_excess"])
            and mean_sharpe_delta > float(acceptance["minimum_mean_sharpe_delta"])
            and mean_drawdown_degradation <= float(acceptance["maximum_mean_drawdown_degradation"])
        )
        comparison_rows.append({
            "candidate": candidate_name,
            "baseline": baseline_name,
            "positive_periods": positive_periods,
            "period_count": len(gate["periods"]),
            "period_annualized_net_excess": ";".join(f"{value:+.4%}" for value in period_excesses),
            "mean_annualized_net_excess": mean_excess,
            "block_bootstrap_ci95_low": ci_low,
            "block_bootstrap_ci95_high": ci_high,
            "mean_sharpe_delta": mean_sharpe_delta,
            "mean_drawdown_degradation": mean_drawdown_degradation,
            "gate_pass": bool(passed),
        })

    fold_frame = pd.DataFrame(fold_rows)
    comparison_frame = pd.DataFrame(comparison_rows)
    fold_path = output_dir / "tilt_robustness_periods.csv"
    comparison_path = output_dir / "tilt_robustness_gate.csv"
    atomic_csv(fold_frame, fold_path, index=False)
    atomic_csv(comparison_frame, comparison_path, index=False)
    manifest.update(status="completed", overall_gate_pass=bool(comparison_frame.gate_pass.all()))
    atomic_json(output_dir / "tilt_robustness_manifest.json", manifest)

    print("Frozen strategy robustness evaluation; no 2024-2025 results loaded or evaluated.")
    print(f"Costs: total transaction cost rate={transaction_cost_rate:.3%}; alpha={candidate['action_alpha']:.2f}")
    print("Full initial allocation with fees; hold dates keep shares; Sharpe uses arithmetic daily excess returns.")
    print("Predeclared acceptance gate:")
    print(yaml.safe_dump(acceptance, sort_keys=False).rstrip())
    print("\nBy period:")
    print(fold_frame.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print("\nGate summary:")
    print(comparison_frame.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"Overall gate (all baselines): {'PASS' if manifest['overall_gate_pass'] else 'FAIL'}")
    print(f"Saved period results: {fold_path}")
    print(f"Saved gate results: {comparison_path}")


if __name__ == "__main__":
    main()

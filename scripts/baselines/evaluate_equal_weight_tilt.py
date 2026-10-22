#!/usr/bin/env python3
"""
==============================================================================
TÊN SCRIPT: scripts/evaluate_equal_weight_tilt.py
MỤC ĐÍCH:
    Quét lưới tham số (exploratory validation sweep) cho chiến lược danh mục đều 1/N
    kết hợp độ lệch (tilt) theo chỉ số RSI và Động lượng 12-1 (Momentum 12-1).
    Khảo sát các chu kỳ tái cân bằng (rebalance periods) và tỷ lệ tilt để tìm
    bộ tham số tối ưu hóa Sharpe Ratio so với benchmark 1/N truyền thống.
    
    BƯỚC TIẾP THEO: Sau khi tìm được tham số tối ưu, chạy scripts/evaluate_tilt_robustness.py
    để kiểm định độ bền vững (robustness test) bằng phương pháp Bootstrap.

HƯỚNG DẪN CHẠY (CLI USAGE):
    # 1. Chạy quét tham số mặc định và lưu bảng kết quả:
    python scripts/evaluate_equal_weight_tilt.py --output experiments/tilt_sweep.csv

    # 2. Tùy chỉnh chi phí giao dịch hoặc tỷ lệ tiền mặt:
    python scripts/evaluate_equal_weight_tilt.py --cash-ratio 0.05 --action-alpha 0.25

ĐẦU VÀO CẦN THIẾT (PREREQUISITES):
    - Dữ liệu nến thô trong: data/raw/*.parquet
    - configs/core/env.yaml, configs/core/assets.yaml

ĐẦU RA SINH RA (OUTPUT ARTIFACTS):
    - File kết quả CSV (mặc định: experiments/tilt_sweep.csv)
    - File manifest JSON lưu hash dữ liệu và cấu hình
    - In ra bảng Top 12 bộ tham số tilt có Sharpe Delta cao nhất
==============================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import TemporalSplitter
from src.env.cost import resolve_transaction_cost_components
from src.env.orders import PortfolioOrder
from src.evaluation.backtester import DeterministicBacktester
from src.evaluation.metrics import FinancialMetrics
from src.features.pipeline import FeaturePipeline
from src.utils.artifacts import atomic_csv, atomic_json, file_hashes
from src.training.data_preparation import close_price_matrix, open_price_matrix
from src.training.env_factory import create_env
from src.evaluation.policy_evaluator import metric_summary


def _cross_sectional_rank(values: np.ndarray) -> np.ndarray:
    """Map finite cross-sectional values to [-1, 1], with average tie ranks."""
    if not np.isfinite(values).all():
        raise ValueError("Signal contains non-finite values.")
    if len(values) < 2:
        return np.zeros_like(values, dtype=np.float64)
    ranked = pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float64)
    return 2.0 * (ranked - 0.5)


def _bounded_simplex(raw: np.ndarray, total: float, lower: float, upper: float) -> np.ndarray:
    """Project onto equal per-asset bounds while preserving the stock budget."""
    if len(raw) * lower > total + 1e-12 or len(raw) * upper < total - 1e-12:
        raise ValueError("Tilt bounds cannot contain the requested stock budget.")
    lo = float(np.min(raw - upper)) - 1.0
    hi = float(np.max(raw - lower)) + 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        candidate = np.clip(raw - mid, lower, upper)
        if candidate.sum() > total:
            lo = mid
        else:
            hi = mid
    weights = np.clip(raw - (lo + hi) / 2.0, lower, upper)
    # Remove tiny bisection residue without violating the box constraints.
    residue = total - float(weights.sum())
    free = (weights > lower + 1e-12) & (weights < upper - 1e-12)
    if np.any(free):
        weights[free] += residue / int(free.sum())
    return weights


def _make_signal(
    features: np.ndarray,
    feature_indices: dict[str, int],
    rsi_orientation: int,
    score_weights: tuple[float, float] = (0.5, 0.5),
) -> np.ndarray:
    rsi_rank = _cross_sectional_rank(features[:, feature_indices["rsi_14"]])
    momentum_rank = _cross_sectional_rank(features[:, feature_indices["mom_12_1"]])
    # The robustness manifest supplies these weights and the RSI direction.
    return score_weights[0] * momentum_rank + score_weights[1] * rsi_orientation * rsi_rank


def make_tilt_policy(
    features: np.ndarray,
    feature_indices: dict[str, int],
    n_assets: int,
    cash_ratio: float,
    tilt_fraction: float,
    rebalance_every: int,
    no_trade_l1: float,
    rsi_orientation: int,
    initial_action_alpha: float = 1.0,
    score_weights: tuple[float, float] = (0.5, 0.5),
):
    """Enter at the first open; trade only on scheduled dates outside the band.

    The first order uses alpha=1 by default and incurs normal transaction costs.
    Later rebalances use the environment's alpha; hold orders keep share counts.
    """
    if not isinstance(rebalance_every, int) or rebalance_every <= 0:
        raise ValueError("rebalance_every phải là số nguyên dương.")
    if not 0 <= cash_ratio < 1 or not 0 <= tilt_fraction <= 1 or not np.isfinite(no_trade_l1) or no_trade_l1 < 0:
        raise ValueError("Cash/tilt/no-trade không hợp lệ.")
    if rsi_orientation not in (-1, 1) or len(score_weights) != 2 or not np.isfinite(score_weights).all() or min(score_weights) < 0 or not np.isclose(sum(score_weights), 1.0):
        raise ValueError("Score weights phải không âm, tổng 1; RSI orientation phải ±1.")
    base = (1.0 - cash_ratio) / n_assets
    targets = np.zeros((len(features), n_assets + 1), dtype=np.float64)
    for t, row in enumerate(features):
        signal = _make_signal(row, feature_indices, rsi_orientation, score_weights)
        raw_stock_weights = base * (1.0 + tilt_fraction * signal)
        stock_weights = _bounded_simplex(
            raw_stock_weights,
            total=1.0 - cash_ratio,
            lower=base * (1.0 - tilt_fraction),
            upper=base * (1.0 + tilt_fraction),
        )
        targets[t, :-1] = stock_weights
        targets[t, -1] = cash_ratio

    def policy(obs: dict, info: dict | None = None) -> np.ndarray | PortfolioOrder:
        step = int((info or {}).get("step", 0))
        desired = targets[min(step, len(targets) - 1)]
        if step == 0:
            return PortfolioOrder.rebalance(desired, action_alpha=initial_action_alpha)
        current = np.concatenate(
            (
                np.asarray(obs["portfolio_weights"], dtype=np.float64),
                np.asarray(obs["cash_ratio"], dtype=np.float64).reshape(-1),
            )
        )
        if step % rebalance_every != 0:
            return PortfolioOrder.hold()
        if float(np.abs(desired - current).sum()) < no_trade_l1:
            return PortfolioOrder.hold()
        return desired.astype(np.float32)

    return policy


def allocation_diagnostics(result, rebalance_every: int) -> dict[str, float | int]:
    """Expose actual cash and any turnover on days declared as hold."""
    hold_days = np.arange(result.total_steps) % rebalance_every != 0
    return {
        "mean_cash_weight": float(result.weights_history[1:, -1].mean()),
        "first_close_cash_weight": float(result.weights_history[1, -1]),
        "unscheduled_turnover": float(result.daily_turnover[hold_days].sum()),
        "unscheduled_costs": float(result.daily_costs[hold_days].sum()),
        "unscheduled_trading_days": int(np.count_nonzero(result.daily_turnover[hold_days] > 1e-8)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--env-config", type=Path, default=PROJECT_ROOT / "configs/core/env.yaml")
    parser.add_argument("--train-end", default=TemporalSplitter.DEFAULT_TRAIN_END)
    parser.add_argument("--val-start", default=TemporalSplitter.DEFAULT_VAL_START)
    parser.add_argument("--val-end", default=TemporalSplitter.DEFAULT_VAL_END)
    parser.add_argument("--test-start", default=TemporalSplitter.DEFAULT_TEST_START)
    parser.add_argument("--cash-ratio", type=float, default=0.04)
    parser.add_argument("--action-alpha", type=float, default=0.25)
    parser.add_argument("--reward-beta", type=float, default=None)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "experiments/equal_weight_tilt_validation.csv")
    args = parser.parse_args()

    if not 0.0 <= args.cash_ratio < 1.0:
        raise ValueError("--cash-ratio must be in [0, 1).")
    if not 0.0 < args.action_alpha <= 1.0:
        raise ValueError("--action-alpha must be in (0, 1].")

    with args.env_config.open("r", encoding="utf-8") as stream:
        env_config = yaml.safe_load(stream)["environment"]
    reward_beta = float(env_config["reward_beta"] if args.reward_beta is None else args.reward_beta)
    processed, full_tensor = FeaturePipeline().run_from_raw(
        raw_dir=args.raw_dir,
        export_parquet=False,
    )
    split = TemporalSplitter(
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
    ).split_market_data_tensor(full_tensor)

    dates = list(full_tensor.dates)
    date_index = {str(date): i for i, date in enumerate(dates)}
    val_start = date_index[str(split.val.dates[0])]
    val_end = date_index[str(split.val.dates[-1])] + 1
    tickers = list(full_tensor.tickers)
    all_open = open_price_matrix(processed, tickers)
    all_close = close_price_matrix(processed, tickers, dates)
    feature_names = list(full_tensor.feature_names)
    feature_indices = {name: feature_names.index(name) for name in ("rsi_14", "mom_12_1")}
    val_features = full_tensor.tensor[val_start:val_end].astype(np.float64, copy=False)
    val_open = all_open[val_start:val_end]
    val_close = all_close[val_start:val_end]

    transaction_cost_rate = resolve_transaction_cost_components(env_config)[2]
    risk_free = float(env_config["risk_free_rate"])

    def make_env():
        return create_env(
            split.val,
            val_open,
            args.env_config,
            reward_beta,
            reward_scaling=float(env_config.get("reward_scaling", 1.0)),
            history_tensor=full_tensor,
            history_start_index=val_start,
            reward_baseline="equal_weight",
            lookback_window=20,
            action_alpha=args.action_alpha,
            close_prices=val_close,
        )

    periods = {"weekly_5d": 5, "monthly_21d": 21}
    tilt_fractions = (0.25, 0.50, 1.00)
    no_trade_bands = (0.0, 0.01, 0.03)
    records: list[dict[str, float | str | int]] = []
    # Use the same full initial allocation as the weekly/monthly references.
    daily_baseline_result = DeterministicBacktester(make_env()).run_strategy(
        make_tilt_policy(
            np.zeros_like(val_features), feature_indices, len(tickers),
            args.cash_ratio, 0.0, 1, 0.0, 1,
        )
    )
    daily_baseline_metrics = metric_summary(daily_baseline_result, risk_free)
    for period_name, rebalance_every in periods.items():
        baseline_result = DeterministicBacktester(make_env()).run_strategy(
            make_tilt_policy(
                np.zeros_like(val_features), feature_indices, len(tickers),
                args.cash_ratio, 0.0, rebalance_every, 0.0, 1,
            )
        )
        baseline_metrics = metric_summary(baseline_result, risk_free)
        records.append({
            "strategy": "Scheduled 1/N + cash",
            "sharpe_convention": FinancialMetrics.SHARPE_CONVENTION,
            "rsi_orientation": "none",
            "rebalance": period_name,
            "tilt_fraction_of_1n": 0.0,
            "no_trade_l1": 0.0,
            "action_alpha": args.action_alpha,
            "initial_action_alpha": 1.0,
            "cash_ratio": args.cash_ratio,
            "cumulative_return": baseline_metrics["cumulative_return"],
            "annualized_return": baseline_metrics["annualized_return"],
            "sharpe_ratio": baseline_metrics["sharpe_ratio"],
            "max_drawdown": baseline_metrics["max_drawdown"],
            "total_turnover": baseline_metrics["total_turnover"],
            "total_costs": baseline_metrics["total_costs"],
            **allocation_diagnostics(baseline_result, rebalance_every),
            "sharpe_delta_vs_scheduled_1n": 0.0,
            "return_delta_vs_scheduled_1n": 0.0,
            "sharpe_delta_vs_daily_1n": baseline_metrics["sharpe_ratio"] - daily_baseline_metrics["sharpe_ratio"],
            "scheduled_1n_sharpe": baseline_metrics["sharpe_ratio"],
            "scheduled_1n_return": baseline_metrics["cumulative_return"],
            "scheduled_1n_turnover": baseline_metrics["total_turnover"],
            "daily_1n_sharpe": daily_baseline_metrics["sharpe_ratio"],
        })

        for orientation, orientation_name in ((1, "rsi_momentum"), (-1, "rsi_contrarian")):
            for tilt_fraction in tilt_fractions:
                for no_trade_l1 in no_trade_bands:
                    result = DeterministicBacktester(make_env()).run_strategy(
                        make_tilt_policy(
                            val_features, feature_indices, len(tickers), args.cash_ratio,
                            tilt_fraction, rebalance_every, no_trade_l1, orientation,
                        )
                    )
                    metrics = metric_summary(result, risk_free)
                    records.append({
                        "strategy": "RSI+mom12_1 tilt",
                        "sharpe_convention": FinancialMetrics.SHARPE_CONVENTION,
                        "rsi_orientation": orientation_name,
                        "rebalance": period_name,
                        "tilt_fraction_of_1n": tilt_fraction,
                        "no_trade_l1": no_trade_l1,
                        "action_alpha": args.action_alpha,
                        "initial_action_alpha": 1.0,
                        "cash_ratio": args.cash_ratio,
                        "cumulative_return": metrics["cumulative_return"],
                        "annualized_return": metrics["annualized_return"],
                        "sharpe_ratio": metrics["sharpe_ratio"],
                        "max_drawdown": metrics["max_drawdown"],
                        "total_turnover": metrics["total_turnover"],
                        "total_costs": metrics["total_costs"],
                        **allocation_diagnostics(result, rebalance_every),
                        "sharpe_delta_vs_scheduled_1n": metrics["sharpe_ratio"] - baseline_metrics["sharpe_ratio"],
                        "return_delta_vs_scheduled_1n": metrics["cumulative_return"] - baseline_metrics["cumulative_return"],
                        "sharpe_delta_vs_daily_1n": metrics["sharpe_ratio"] - daily_baseline_metrics["sharpe_ratio"],
                        "scheduled_1n_sharpe": baseline_metrics["sharpe_ratio"],
                        "scheduled_1n_return": baseline_metrics["cumulative_return"],
                        "scheduled_1n_turnover": baseline_metrics["total_turnover"],
                        "daily_1n_sharpe": daily_baseline_metrics["sharpe_ratio"],
                    })

    output = pd.DataFrame.from_records(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_csv(output, args.output, index=False)
    atomic_json(args.output.with_suffix(".manifest.json"), {
        "status": "completed", "study_type": "exploratory_validation_sweep",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "resolved_environment": env_config, "feature_parameters": FeaturePipeline().parameters,
        "validation_dates": [split.val.dates[0], split.val.dates[-1]],
        "sharpe_convention": FinancialMetrics.SHARPE_CONVENTION,
        "source_hashes": file_hashes(list((PROJECT_ROOT / "src").rglob("*.py")) + [Path(__file__)], PROJECT_ROOT),
        "data_hashes": file_hashes(args.raw_dir.glob("*.parquet"), PROJECT_ROOT),
        "config_hashes": file_hashes([args.env_config, PROJECT_ROOT / "configs/core/assets.yaml"], PROJECT_ROOT),
    })
    print("Exploratory validation sweep: RSI/momentum long-only tilt around 1/N")
    print(f"Validation: {split.val.dates[0]} to {split.val.dates[-1]}; assets={len(tickers)}")
    print(f"alpha={args.action_alpha:g}; cash={args.cash_ratio:.1%}; total cost rate={transaction_cost_rate:.3%}")
    print("First order: full target at next open with fees; later hold dates incur no turnover.")
    print("Sharpe: arithmetic daily excess return / sample standard deviation, annualized.")
    print("Parameters were not fit on validation; validation has already been used in prior research.")
    print(output.sort_values("sharpe_delta_vs_scheduled_1n", ascending=False).head(12).to_string(index=False))
    print(f"Saved full grid: {args.output}")


if __name__ == "__main__":
    main()

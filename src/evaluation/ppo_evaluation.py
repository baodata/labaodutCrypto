"""Historical PPO comparison, bootstrap reporting and metric persistence. Owner: A."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse

from torch import nn
import numpy as np
import pandas as pd

from src.evaluation.backtester import DeterministicBacktester
from src.evaluation.baselines import BuyAndHoldBaseline
from src.evaluation.baselines import CashBaseline
from src.evaluation.baselines import EqualWeightBaseline
from src.evaluation.metrics import FinancialMetrics
from src.evaluation.metrics import block_bootstrap_excess_returns
from src.evaluation.policy_evaluator import evaluate_policy_with_actions
from src.evaluation.policy_evaluator import policy_weight_summary
from src.training.config import resolve_training_config
from src.training.env_factory import create_env
from src.training.run_artifacts import finish_run_manifest
from src.utils.artifacts import atomic_csv


def save_or_append_metrics(
    metrics_path: Path,
    new_metrics: pd.DataFrame,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Lưu bảng metrics vào CSV:
    - Nếu file đã tồn tại và overwrite=False: Giữ nguyên các hàng baseline (1/N, Buy & Hold, Cash)
      ở trên cùng, giữ lại các mô hình PPO cũ đã chạy trước đó, và ghi nối tiếp (hoặc cập nhật)
      kết quả của mô hình PPO mới.
    - Đảm bảo 3 hàng baseline luôn đứng ở đầu file CSV.
    """
    baseline_names = ["Equal Weight (1/N)", "Buy & Hold", "Cash"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite or not metrics_path.exists():
        combined_df = new_metrics.copy()
    else:
        try:
            existing_df = pd.read_csv(metrics_path, index_col=0)
            # Lấy baseline từ new_metrics nếu có, hoặc từ existing_df
            base_df = new_metrics.loc[[b for b in baseline_names if b in new_metrics.index]]
            if base_df.empty:
                base_df = existing_df.loc[[b for b in baseline_names if b in existing_df.index]]

            # Lấy các mô hình PPO đã lưu trước đó (loại bỏ baseline, loại bỏ 'PPO' cũ và PPO đang chạy)
            curr_ppo_names = [idx for idx in new_metrics.index if idx not in baseline_names]
            old_ppos = existing_df.loc[
                [idx for idx in existing_df.index if idx not in baseline_names and idx not in curr_ppo_names and idx != "PPO"]
            ]

            # Lấy PPO của lượt chạy hiện tại
            curr_ppos = new_metrics.loc[curr_ppo_names]

            # Ghép lại bằng pd.concat
            combined_df = pd.concat([base_df, old_ppos, curr_ppos], axis=0)
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể đọc file metrics cũ ({e}), tạo file mới.")
            combined_df = new_metrics.copy()

    # Sắp xếp hàng: 3 baselines ở đầu tiên, theo sau là các PPO runs
    top_rows = [b for b in baseline_names if b in combined_df.index]
    other_rows = [r for r in combined_df.index if r not in baseline_names]
    combined_df = combined_df.loc[top_rows + other_rows]

    combined_df.to_csv(metrics_path)
    return combined_df


def evaluate_test(
    args: argparse.Namespace,
    model: nn.Module,
    context: dict[str, Any],
) -> pd.DataFrame:
    resolve_training_config(args)
    if args.evaluation_scope != "test":
        raise ValueError("Chế độ validation không cho phép backtest test.")
    try:
        return _evaluate_test(args, model, context)
    except KeyboardInterrupt:
        if hasattr(args, "_run_dir"):
            finish_run_manifest(args, "interrupted_during_evaluation")
        raise
    except Exception as exc:
        if hasattr(args, "_run_dir"):
            finish_run_manifest(args, "evaluation_failed", error=str(exc))
        raise


def _evaluate_test(
    args: argparse.Namespace,
    model: nn.Module,
    context: dict[str, Any],
) -> pd.DataFrame:
    from src.evaluation.periods import describe_test_period, warn_for_explored_test
    period = describe_test_period(context['test_tensor'], 'test')
    warn_for_explored_test(period)
    print("[4/6] Backtest trên giai đoạn test đã cấu hình (tính độc lập chưa được bảo đảm)...")
    test_env = create_env(
        context["test_tensor"],
        context["test_open"],
        args.env_config,
        context["reward_beta"],
        reward_scaling=context["reward_scaling"],
        history_tensor=context["scaled_full_tensor"],
        history_start_index=context["test_history_start"],
        reward_baseline=args.reward_baseline,
        lookback_window=context["lookback"],
        action_alpha=context["action_alpha"],
        graph_adjacency_sequence=context["graph_adjacency_sequence"],
        graph_edge_mask_sequence=context["graph_edge_mask_sequence"],
        graph_start_index=context["test_history_start"],
        close_prices=context["test_close"],
    )
    backtester = DeterministicBacktester(test_env)
    ppo_result, test_policy_actions, test_blended_targets = evaluate_policy_with_actions(
        model, test_env
    )
    test_policy_weights = policy_weight_summary(
        test_policy_actions, len(context["tickers"]), context["tickers"]
    )
    test_blended_weights = policy_weight_summary(
        test_blended_targets, len(context["tickers"]), context["tickers"]
    )
    # Baselines provide their own target weights; alpha=1 makes each target
    # take effect immediately and keeps PPO smoothing out of their metrics.
    baseline_env = create_env(
        context["test_tensor"],
        context["test_open"],
        args.env_config,
        context["reward_beta"],
        reward_scaling=context["reward_scaling"],
        history_tensor=context["scaled_full_tensor"],
        history_start_index=context["test_history_start"],
        lookback_window=context["lookback"],
        action_alpha=1.0,
        close_prices=context["test_close"],
    )
    baseline_backtester = DeterministicBacktester(baseline_env)
    n_assets = len(context["tickers"])
    matched_cash = float(context.get("matched_cash_ratio", 1.0 / (n_assets + 1)))
    matched_name = f"Matched 1/N + Cash {100 * matched_cash:g}% (same alpha)"
    env_config = context["env_config"]
    matched_baseline_env = create_env(
        context["test_tensor"],
        context["test_open"],
        args.env_config,
        context["reward_beta"],
        reward_scaling=context["reward_scaling"],
        history_tensor=context["scaled_full_tensor"],
        history_start_index=context["test_history_start"],
        lookback_window=context["lookback"],
        action_alpha=context["action_alpha"],
        close_prices=context["test_close"],
    )
    matched_baseline_result = DeterministicBacktester(
        matched_baseline_env
    ).run_strategy(
        EqualWeightBaseline(
            open_prices=context["test_open"],
            num_assets=n_assets,
            transaction_cost_rate=float(env_config["transaction_cost_rate"]),
            risk_free_rate_annual=float(env_config["risk_free_rate"]),
            cash_ratio=matched_cash,
        )
    )

    # Thứ tự: 1/N, Buy & Hold, Cash ở đầu tiên, theo sau là các baselines và PPO
    results = {
        "Equal Weight (1/N)": baseline_backtester.run_strategy(
            EqualWeightBaseline(
                open_prices=context["test_open"],
                num_assets=n_assets,
                transaction_cost_rate=float(env_config["transaction_cost_rate"]),
                risk_free_rate_annual=float(env_config["risk_free_rate"]),
                cash_ratio=0.0,
            )
        ),
        "Buy & Hold": baseline_backtester.run_strategy(
            BuyAndHoldBaseline(
                open_prices=context["test_open"],
                num_assets=n_assets,
                transaction_cost_rate=float(env_config["transaction_cost_rate"]),
                risk_free_rate_annual=float(env_config["risk_free_rate"]),
            )
        ),
        "Cash": baseline_backtester.run_strategy(
            CashBaseline(
                num_days=len(context["test_tensor"].dates) - 1,
                risk_free_rate_annual=float(env_config["risk_free_rate"]),
            )
        ),
        "1/N + Cash 4.4%": baseline_backtester.run_strategy(
            EqualWeightBaseline(
                open_prices=context["test_open"],
                num_assets=n_assets,
                transaction_cost_rate=float(env_config["transaction_cost_rate"]),
                risk_free_rate_annual=float(env_config["risk_free_rate"]),
                cash_ratio=0.044,
            )
        ),
        matched_name: matched_baseline_result,
        "PPO": ppo_result,
    }

    metrics = backtester.compare(
        results,
        risk_free_rate_annual=float(env_config["risk_free_rate"]),
    )
    cbb_stats = block_bootstrap_excess_returns(
        ppo_result.daily_net_returns,
        matched_baseline_result.daily_net_returns,
        block_size=args.bootstrap_block_size,
        num_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    cbb_1n_stats = block_bootstrap_excess_returns(
        ppo_result.daily_net_returns,
        results["Equal Weight (1/N)"].daily_net_returns,
        block_size=args.bootstrap_block_size,
        num_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    metrics["profit"] = pd.Series(
        {
            name: float(result.portfolio_values[-1] - result.portfolio_values[0])
            for name, result in results.items()
        }
    )
    metrics["mean_policy_cash_weight"] = np.nan
    metrics["mean_policy_abs_deviation_1n"] = np.nan
    metrics["mean_policy_cross_asset_weight_std"] = np.nan
    metrics["mean_policy_temporal_weight_std"] = np.nan
    metrics["mean_blended_cash_weight"] = np.nan
    metrics.loc["PPO", "mean_policy_cash_weight"] = test_policy_weights[
        "mean_cash_weight"
    ]
    metrics.loc["PPO", "mean_policy_abs_deviation_1n"] = test_policy_weights[
        "mean_abs_deviation_1n"
    ]
    metrics.loc["PPO", "mean_policy_cross_asset_weight_std"] = test_policy_weights[
        "cross_asset_weight_std"
    ]
    metrics.loc["PPO", "mean_policy_temporal_weight_std"] = test_policy_weights[
        "mean_temporal_weight_std"
    ]
    metrics.loc["PPO", "mean_blended_cash_weight"] = test_blended_weights[
        "mean_cash_weight"
    ]
    for index in range(n_assets):
        ticker = context["tickers"][index]
        policy_column = f"policy_asset_{ticker}_weight_std_time"
        blended_column = f"blended_asset_{ticker}_weight_std_time"
        metrics[policy_column] = np.nan
        metrics[blended_column] = np.nan
        metrics.loc["PPO", policy_column] = test_policy_weights[
            f"asset_{ticker}_weight_std_time"
        ]
        metrics.loc["PPO", blended_column] = test_blended_weights[
            f"asset_{ticker}_weight_std_time"
        ]
    metrics["policy_cash_weight_std_time"] = np.nan
    metrics["blended_cash_weight_std_time"] = np.nan
    metrics.loc["PPO", "policy_cash_weight_std_time"] = test_policy_weights[
        "cash_weight_std_time"
    ]
    metrics.loc["PPO", "blended_cash_weight_std_time"] = test_blended_weights[
        "cash_weight_std_time"
    ]
    metrics["checkpoint_epoch"] = np.nan
    metrics["checkpoint_status"] = "none"
    metrics["sharpe_convention"] = FinancialMetrics.SHARPE_CONVENTION
    metrics["test_period_status"] = period["status"]
    metrics["evaluation_start"] = period["start"]
    metrics["evaluation_end"] = period["end"]
    metrics.loc["PPO", "checkpoint_epoch"] = float(context.get("checkpoint_epoch", -1))
    metrics.loc["PPO", "checkpoint_status"] = str(context.get("checkpoint_status", "unknown"))
    metrics["cbb_excess_annual_vs_1n_cash"] = np.nan
    metrics["cbb_ci_lower_95"] = np.nan
    metrics["cbb_ci_upper_95"] = np.nan
    metrics["cbb_p_value"] = np.nan
    metrics["cbb_baseline"] = matched_name
    metrics["matched_baseline_cash_ratio"] = matched_cash
    metrics["cbb_p_value_method"] = "one_sided_centered_block_bootstrap_null_mean_zero"
    metrics["sharpe_delta_vs_matched_baseline"] = np.nan
    metrics.loc["PPO", "sharpe_delta_vs_matched_baseline"] = (
        metrics.loc["PPO", "sharpe_ratio"]
        - metrics.loc[matched_name, "sharpe_ratio"]
    )
    metrics.loc["PPO", "cbb_excess_annual_vs_1n_cash"] = cbb_stats["mean_excess_annualized"]
    metrics.loc["PPO", "cbb_ci_lower_95"] = cbb_stats["ci_lower_95"]
    metrics.loc["PPO", "cbb_ci_upper_95"] = cbb_stats["ci_upper_95"]
    metrics.loc["PPO", "cbb_p_value"] = cbb_stats["p_value_positive_excess"]

    # Ghi lại các thông số cấu hình môi trường và mô hình
    meta_info = {
        "Equal Weight (1/N)": {
            "encoder": "-",
            "reward_beta": "-",
            "lookback": "-",
            "action_alpha": "1.0",
            "cost_rate": str(env_config["transaction_cost_rate"]),
            "risk_free_rate": str(env_config["risk_free_rate"]),
            "seed": "-",
        },
        "Buy & Hold": {
            "encoder": "-",
            "reward_beta": "-",
            "lookback": "-",
            "action_alpha": "1.0",
            "cost_rate": str(env_config["transaction_cost_rate"]),
            "risk_free_rate": str(env_config["risk_free_rate"]),
            "seed": "-",
        },
        "Cash": {
            "encoder": "-",
            "reward_beta": "-",
            "lookback": "-",
            "action_alpha": "-",
            "cost_rate": "0.0",
            "risk_free_rate": str(env_config["risk_free_rate"]),
            "seed": "-",
        },
        "1/N + Cash 4.4%": {
            "encoder": "-",
            "reward_beta": "-",
            "lookback": "-",
            "action_alpha": "1.0",
            "cost_rate": str(env_config["transaction_cost_rate"]),
            "risk_free_rate": str(env_config["risk_free_rate"]),
            "seed": "-",
        },
        matched_name: {
            "encoder": "-",
            "reward_beta": "-",
            "lookback": "-",
            "action_alpha": str(context["action_alpha"]),
            "cost_rate": str(env_config["transaction_cost_rate"]),
            "risk_free_rate": str(env_config["risk_free_rate"]),
            "seed": "-",
        },
        "PPO": {
            "encoder": str(args.encoder),
            "reward_beta": str(context["reward_beta"]),
            "lookback": str(context["lookback"]),
            "action_alpha": str(context["action_alpha"]),
            "cost_rate": str(env_config["transaction_cost_rate"]),
            "risk_free_rate": str(env_config["risk_free_rate"]),
            "seed": str(args.seed),
        },
    }
    meta_cols = ["encoder", "reward_beta", "lookback", "action_alpha", "cost_rate", "risk_free_rate", "seed"]
    for col in meta_cols:
        metrics[col] = [meta_info.get(s, {}).get(col, "-") for s in metrics.index]

    # Đưa các cột cấu hình ra phía trước các cột tài chính
    perf_cols = [c for c in metrics.columns if c not in meta_cols]
    metrics = metrics[meta_cols + perf_cols]

    # Xác định tên định danh cho PPO run khi lưu tổng hợp (sprint5_metrics.csv)
    ppo_label = getattr(args, "run_name", None)
    if not ppo_label:
        parts = [f"PPO ({args.encoder}", f"beta={context['reward_beta']}"]
        if context["lookback"] != 20:
            parts.append(f"w={context['lookback']}")
        if args.seed != 7:
            parts.append(f"seed={args.seed}")
        ppo_label = ", ".join(parts) + ")"

    # Lưu và ghi tiếp (append) vào file metrics_output (sprint5_metrics.csv)
    metrics_to_append = metrics.rename(index={"PPO": ppo_label})
    save_or_append_metrics(
        args.metrics_output,
        metrics_to_append,
        overwrite=getattr(args, "overwrite_metrics", False),
    )

    if "run_dir" in context:
        atomic_csv(metrics, context["run_dir"] / "metrics.csv")
        daily = pd.DataFrame({name: result.daily_net_returns for name, result in results.items()}, index=context["test_tensor"].dates[1:])
        daily.index.name = "date"
        atomic_csv(daily, context["run_dir"] / "daily_returns.csv")
        finish_run_manifest(args, "completed", best_epoch=context["checkpoint_epoch"],
                            test_period=period, resolved_split_dates=context.get("resolved_split_dates", {}))

        print("[5/6] Metrics test (profit và costs theo đơn vị tiền tài khoản):")
    print(metrics.to_string(float_format=lambda value: f"{value:.6f}"))
    print(
        "  PPO mean weights | "
        f"target cash={test_policy_weights['mean_cash_weight']:.2%}, "
        f"target mean |asset weight - 1/N|="
        f"{test_policy_weights['mean_abs_deviation_1n']:.3%}, "
        f"blended cash={test_blended_weights['mean_cash_weight']:.2%}"
    )
    print(
        f"  Bootstrap vs {matched_name} | "
        f"excess annual return={cbb_stats['mean_excess_annualized']:.2%} "
        f"(95% CI: [{cbb_stats['ci_lower_95']:.2%}, {cbb_stats['ci_upper_95']:.2%}]), "
        f"p-value(H0 mean excess=0)={cbb_stats['p_value_positive_excess']:.4f}"
    )
    print(
        f"  Bootstrap vs 1/N | "
        f"excess annual return={cbb_1n_stats['mean_excess_annualized']:.2%} "
        f"(95% CI: [{cbb_1n_stats['ci_lower_95']:.2%}, {cbb_1n_stats['ci_upper_95']:.2%}]), "
        f"p-value(H0 mean excess=0)={cbb_1n_stats['p_value_positive_excess']:.4f}"
    )
    print(f"[6/6] Đã lưu metrics: {args.metrics_output}")
    return metrics

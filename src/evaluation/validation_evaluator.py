"""Development-only PPO reporting with the same execution engine and baselines."""

import numpy as np
import pandas as pd

from src.evaluation.backtester import DeterministicBacktester
from src.evaluation.baselines import BuyAndHoldBaseline, CashBaseline, EqualWeightBaseline
from src.evaluation.metrics import block_bootstrap_excess_returns
from src.evaluation.policy_evaluator import evaluate_policy, metric_summary
from src.training.env_factory import create_env
from src.training.run_artifacts import finish_run_manifest
from src.utils.artifacts import atomic_csv, atomic_json


def evaluate_validation(args, model, context):
    n = len(context["tickers"])
    opening, closing = context["val_open"], context["val_close"]
    config = context["env_config"]
    risk_free = float(config["risk_free_rate"])

    def env(alpha, graph=False):
        return create_env(
            context["val_tensor"], opening, args.env_config, context["reward_beta"],
            reward_scaling=context["reward_scaling"], close_prices=closing,
            history_tensor=context["scaled_full_tensor"],
            history_start_index=context["val_history_start"], lookback_window=context["lookback"],
            action_alpha=alpha, graph_start_index=context["val_history_start"],
            graph_adjacency_sequence=context["graph_adjacency_sequence"] if graph else None,
            graph_edge_mask_sequence=context["graph_edge_mask_sequence"] if graph else None,
        )

    def equal(cash):
        return EqualWeightBaseline(opening, n, float(config["transaction_cost_rate"]), risk_free, cash_ratio=cash)

    results = {
        "PPO": evaluate_policy(model, env(context["action_alpha"], graph=True)),
        "Matched 1/N + Cash": DeterministicBacktester(env(context["action_alpha"])).run_strategy(equal(context["matched_cash_ratio"])),
        "Equal Weight (1/N)": DeterministicBacktester(env(1.0)).run_strategy(equal(0.0)),
        "Buy & Hold": DeterministicBacktester(env(1.0)).run_strategy(
            BuyAndHoldBaseline(opening, n, float(config["transaction_cost_rate"]), risk_free)),
        "Cash": DeterministicBacktester(env(1.0)).run_strategy(
            CashBaseline(len(opening) - 1, risk_free)),
    }
    metrics = pd.DataFrame({name: metric_summary(result, risk_free) for name, result in results.items()}).T
    metrics.index.name = "Strategy"
    metrics["sharpe_delta_vs_matched_baseline"] = metrics.sharpe_ratio - metrics.loc["Matched 1/N + Cash", "sharpe_ratio"]
    if getattr(args, "quant_config", None):
        metrics["sharpe_delta_vs_matched_asset_baseline"] = metrics["sharpe_delta_vs_matched_baseline"]
        metrics["matched_baseline"] = "Matched 1/N + Cash"
    metrics["evaluation_scope"] = "validation"
    metrics["checkpoint_status"] = "none"
    metrics["checkpoint_epoch"] = np.nan
    metrics.loc["PPO", "checkpoint_status"] = context["checkpoint_status"]
    metrics.loc["PPO", "checkpoint_epoch"] = context["checkpoint_epoch"]
    bootstrap = block_bootstrap_excess_returns(
        results["PPO"].daily_net_returns, results["Matched 1/N + Cash"].daily_net_returns,
        block_size=args.bootstrap_block_size, num_samples=args.bootstrap_samples, seed=args.seed,
    )
    daily = pd.DataFrame({name: result.daily_net_returns for name, result in results.items()},
                         index=context["val_tensor"].dates[1:])
    daily.index.name = "date"
    atomic_csv(metrics, args.metrics_output)
    atomic_csv(metrics, context["run_dir"] / "validation_metrics.csv")
    atomic_csv(daily, context["run_dir"] / "validation_daily_returns.csv")
    atomic_json(context["run_dir"] / "validation_report.json", {
        "evaluation_scope": "validation", "test_evaluated": False,
        "learning_gate": context["learning_gate"], "bootstrap_vs_matched": bootstrap,
        "decision": "GO" if context["learning_gate"].get("passed") else "NO-GO",
        "selection_period": "inner" if "protocol_fold" in context else "validation",
        "evaluation_period": context.get("evaluation_period", "validation"),
        "protocol_fold": context.get("protocol_fold"),
        "untrained_selection_reference": context["reference_validation_metrics"],
        "matched_selection_reference": context["matched_validation_metrics"],
    })
    finish_run_manifest(args, "completed", evaluation_scope="validation", test_evaluated=False,
                        best_epoch=context["checkpoint_epoch"])
    period = context.get("evaluation_period", "validation")
    print(f"{period}: {context['val_tensor'].dates[0]} → {context['val_tensor'].dates[-1]} — không backtest test.")
    if "protocol_fold" in context:
        fold = context["protocol_fold"]
        print(f"Checkpoint/gate dùng inner {fold['inner_start']} → {fold['inner_end']}; bảng dưới là outer development.")
    print(metrics.to_string())
    print("Learning gate:", "GO" if context["learning_gate"].get("passed") else "NO-GO",
          f"({context['learning_gate']['status']})")
    return metrics

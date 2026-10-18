"""Validation-only evaluation of independently initialized policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse

import numpy as np
import pandas as pd
import torch
import yaml

from src.env.cost import resolve_transaction_cost_components
from src.evaluation.policy_evaluator import evaluate_equal_weight
from src.evaluation.policy_evaluator import evaluate_policy_with_actions
from src.evaluation.policy_evaluator import metric_summary
from src.evaluation.policy_evaluator import policy_weight_summary
from src.graph.dynamic_graph import build_dynamic_graph_sequence
from src.models.actor_critic import build_actor_critic
from src.models.policy_adapter import graph_tensors_from_obs
from src.models.policy_adapter import history_mask_from_obs
from src.models.policy_adapter import observation_vector
from src.training.data_preparation import close_price_matrix
from src.training.data_preparation import prepare_data
from src.training.env_factory import create_env
from src.training.run_artifacts import configure_run_artifacts
from src.training.run_artifacts import finish_run_manifest
from src.training.run_artifacts import record_effective_training_config
from src.utils.artifacts import atomic_csv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def evaluate_untrained_null(args: argparse.Namespace, seeds: list[int]) -> pd.DataFrame:
    configure_run_artifacts(args, "null")
    try:
        return _evaluate_untrained_null(args, seeds)
    except KeyboardInterrupt:
        finish_run_manifest(args, "interrupted")
        raise
    except Exception as exc:
        finish_run_manifest(args, "failed", error=str(exc))
        raise


def _evaluate_untrained_null(args: argparse.Namespace, seeds: list[int]) -> pd.DataFrame:
    """Evaluate independently initialized, deterministic policies on validation only."""
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("--untrained-null-seeds cần ít nhất 2 seed khác nhau.")
    if not 0.0 < args.action_alpha <= 1.0:
        raise ValueError("--action-alpha phải thuộc (0, 1].")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA được chọn nhưng không khả dụng.")

    with args.model_config.open("r", encoding="utf-8") as stream:
        configured_model_config = yaml.safe_load(stream)
    with args.env_config.open("r", encoding="utf-8") as stream:
        env_config = yaml.safe_load(stream)["environment"]
    reward_beta = (
        float(env_config["reward_beta"])
        if args.reward_beta is None
        else float(args.reward_beta)
    )
    reward_scaling = (
        float(env_config.get("reward_scaling", 1.0))
        if args.reward_scaling is None
        else float(args.reward_scaling)
    )
    if not np.isfinite(reward_beta) or reward_beta < 0:
        raise ValueError("reward_beta phải là số hữu hạn không âm.")
    if not np.isfinite(reward_scaling) or reward_scaling <= 0:
        raise ValueError("reward_scaling phải là số hữu hạn lớn hơn 0.")

    (
        full_tensor,
        processed_data,
        scaled_full_tensor,
        split,
        train_tensor,
        val_tensor,
        _unused_test_tensor,
        _unused_train_open,
        val_open,
        _unused_test_open,
        _unused_train_close,
        val_close,
        _unused_test_close,
        tickers,
    ) = prepare_data(args, reward_beta)
    full_date_index = {str(date): index for index, date in enumerate(full_tensor.dates)}
    val_history_start = full_date_index[str(split.val.dates[0])]
    lookback = int(env_config.get("lookback_window", 20) if args.lookback is None else args.lookback)
    if lookback <= 0:
        raise ValueError("--lookback phải lớn hơn 0.")
    n_assets, n_features = train_tensor.tensor.shape[1:]
    action_dim = n_assets + 1
    matched_cash = 1.0 / action_dim if args.matched_cash_ratio is None else float(args.matched_cash_ratio)
    if not np.isfinite(matched_cash) or not 0 <= matched_cash < 1:
        raise ValueError("matched-cash-ratio phải thuộc [0,1).")
    record_effective_training_config(args, matched_cash_ratio=matched_cash)
    encoder = args.encoder
    rnn_layers = int(args.rnn_layers)
    rnn_dropout = float(args.rnn_dropout)
    if rnn_layers <= 0 or not np.isfinite(rnn_dropout) or not 0 <= rnn_dropout < 1:
        raise ValueError("Cấu hình RNN không hợp lệ.")

    rl_config = configured_model_config.get("rl_agent", {})
    dirichlet_kappa = (
        float(args.dirichlet_kappa)
        if args.dirichlet_kappa is not None
        else (
            float(rl_config["dirichlet_kappa"])
            if rl_config.get("dirichlet_kappa") is not None
            else None
        )
    )
    if dirichlet_kappa is not None and dirichlet_kappa <= action_dim:
        raise ValueError(
            f"dirichlet_kappa ({dirichlet_kappa}) phải lớn hơn action_dim ({action_dim})."
        )
    model_kwargs: dict[str, Any] = {
        "encoder": encoder,
        "n_assets": n_assets,
        "n_features": n_features,
        "lookback": lookback,
        "action_dim": action_dim,
        "hidden_dim": args.hidden_dim,
        "rnn_layers": rnn_layers,
        "rnn_dropout": rnn_dropout,
        "dirichlet_kappa": dirichlet_kappa,
    }

    graph_adjacency_sequence = None
    graph_edge_mask_sequence = None
    if encoder == "gnn":
        gnn_config = configured_model_config.get("gnn", {})
        architecture = (
            args.gnn_architecture or str(gnn_config.get("architecture", "GAT"))
        ).lower()
        if architecture not in {"gcn", "gat"}:
            raise ValueError("GNN architecture phải là GCN hoặc GAT.")
        close_prices = close_price_matrix(processed_data, tickers, full_tensor.dates)
        close_returns = pd.DataFrame(
            close_prices,
            index=pd.DatetimeIndex(pd.to_datetime(full_tensor.dates)),
            columns=tickers,
        ).pct_change(fill_method=None)
        graph_adjacency_sequence, graph_edge_mask_sequence = build_dynamic_graph_sequence(
            close_returns,
            full_tensor.dates,
            tickers,
            window_size=int(gnn_config.get("correlation_window", 60)),
            threshold=float(gnn_config.get("threshold_corr", 0.5)),
            graph_mode=str(gnn_config.get("graph_mode", "knn")).lower(),
            top_k=int(gnn_config.get("top_k", 5)),
            sector_config_path=str(PROJECT_ROOT / "configs/assets.yaml"),
        )
        model_kwargs.update(
            {
                "gnn_architecture": architecture,
                "gnn_hidden_channels": int(gnn_config.get("hidden_channels", 64) if args.gnn_hidden_channels is None else args.gnn_hidden_channels),
                "gnn_out_channels": int(gnn_config.get("out_channels", 32) if args.gnn_out_channels is None else args.gnn_out_channels),
                "gnn_heads": int(gnn_config.get("heads", 4) if args.gnn_heads is None else args.gnn_heads),
                "temporal_encoder": "gru",
            }
        )

    device = torch.device(args.device)
    risk_free_rate = float(env_config["risk_free_rate"])
    transaction_cost_rate = resolve_transaction_cost_components(env_config)[2]
    baseline_env = create_env(
        val_tensor,
        val_open,
        args.env_config,
        reward_beta,
        reward_scaling=reward_scaling,
        history_tensor=scaled_full_tensor,
        history_start_index=val_history_start,
        lookback_window=lookback,
        action_alpha=1.0,
        graph_adjacency_sequence=graph_adjacency_sequence,
        graph_edge_mask_sequence=graph_edge_mask_sequence,
        graph_start_index=val_history_start,
        close_prices=val_close,
    )
    baseline_metrics, _ = evaluate_equal_weight(
        baseline_env,
        val_open,
        n_assets,
        transaction_cost_rate,
        risk_free_rate,
        cash_ratio=0.044,
    )
    matched_baseline_env = create_env(
        val_tensor,
        val_open,
        args.env_config,
        reward_beta,
        reward_scaling=reward_scaling,
        history_tensor=scaled_full_tensor,
        history_start_index=val_history_start,
        lookback_window=lookback,
        action_alpha=args.action_alpha,
        graph_adjacency_sequence=graph_adjacency_sequence,
        graph_edge_mask_sequence=graph_edge_mask_sequence,
        graph_start_index=val_history_start,
        close_prices=val_close,
    )
    matched_baseline_metrics, _ = evaluate_equal_weight(
        matched_baseline_env,
        val_open,
        n_assets,
        transaction_cost_rate,
        risk_free_rate,
        cash_ratio=matched_cash,
    )

    rows: list[dict[str, float | int | str]] = []
    print(
        f"Validation null: encoder={encoder}, seeds={len(seeds)}, "
        f"lookback={lookback}, action_alpha={args.action_alpha:g}; "
        "không train, không đánh giá test."
    )
    print(
        f"1/N + cash 4.4% | Sharpe={baseline_metrics['sharpe_ratio']:.4f}, "
        f"return={baseline_metrics['cumulative_return']:.2%}"
    )
    print(
        f"Matched 1/N + cash {matched_cash:.2%}, action_alpha={args.action_alpha:g} | "
        f"Sharpe={matched_baseline_metrics['sharpe_ratio']:.4f}, "
        f"return={matched_baseline_metrics['cumulative_return']:.2%}"
    )
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = build_actor_critic(**model_kwargs).to(device)
        model.cache_batch_size = args.cache_batch_size
        model.cache_temporal_batch_size = args.cache_temporal_batch_size
        val_env = create_env(
            val_tensor,
            val_open,
            args.env_config,
            reward_beta,
            reward_scaling=reward_scaling,
            history_tensor=scaled_full_tensor,
            history_start_index=val_history_start,
            reward_baseline=args.reward_baseline,
            lookback_window=lookback,
            action_alpha=args.action_alpha,
            graph_adjacency_sequence=graph_adjacency_sequence,
            graph_edge_mask_sequence=graph_edge_mask_sequence,
            graph_start_index=val_history_start,
            close_prices=val_close,
        )
        initial_obs, _ = val_env.reset()
        initial_state = torch.as_tensor(
            observation_vector(initial_obs), dtype=torch.float32, device=device
        )
        initial_graph = graph_tensors_from_obs(initial_obs, device)
        initial_diagnostics = model.diagnostics(initial_state, initial_graph, history_mask_from_obs(initial_obs, device))
        result, policy_actions, _ = evaluate_policy_with_actions(model, val_env)
        metrics = metric_summary(result, risk_free_rate)
        weight_summary = policy_weight_summary(policy_actions, n_assets)
        row: dict[str, float | int | str] = {
            "seed": seed,
            "encoder": encoder,
            "checkpoint_status": "untrained",
            "sharpe_ratio": metrics["sharpe_ratio"],
            "cumulative_return": metrics["cumulative_return"],
            "annualized_return": metrics["annualized_return"],
            "max_drawdown": metrics["max_drawdown"],
            "total_turnover": metrics["total_turnover"],
            "total_costs": metrics["total_costs"],
            "baseline_1n_cash_sharpe": baseline_metrics["sharpe_ratio"],
            "baseline_1n_cash_cumulative_return": baseline_metrics[
                "cumulative_return"
            ],
            "matched_baseline_sharpe": matched_baseline_metrics["sharpe_ratio"],
            "matched_baseline_cumulative_return": matched_baseline_metrics[
                "cumulative_return"
            ],
            "matched_baseline_cash_ratio": matched_cash,
            "matched_baseline_action_alpha": args.action_alpha,
            "reference_cash_ratio": 0.044,
            "reference_action_alpha": 1.0,
            "sharpe_delta_vs_1n_cash": (
                metrics["sharpe_ratio"] - baseline_metrics["sharpe_ratio"]
            ),
            "return_delta_vs_1n_cash": (
                metrics["cumulative_return"] - baseline_metrics["cumulative_return"]
            ),
            "sharpe_delta_vs_matched_baseline": (
                metrics["sharpe_ratio"] - matched_baseline_metrics["sharpe_ratio"]
            ),
            "return_delta_vs_matched_baseline": (
                metrics["cumulative_return"]
                - matched_baseline_metrics["cumulative_return"]
            ),
            "initial_concentration_mean": initial_diagnostics["concentration_mean"],
            "initial_kappa_mean": initial_diagnostics["kappa_mean"],
            "initial_entropy": initial_diagnostics["entropy"],
            **weight_summary,
        }
        rows.append(row)
        print(
            f"  seed={seed:>4} | Sharpe={metrics['sharpe_ratio']:.4f} "
            f"(Δmatched {row['sharpe_delta_vs_matched_baseline']:+.4f}) | "
            f"return={metrics['cumulative_return']:.2%} "
            f"(Δmatched {row['return_delta_vs_matched_baseline']:+.2%}) | "
            f"tilt={weight_summary['mean_abs_deviation_1n']:.3%}"
        )

    output = pd.DataFrame(rows)
    output_path = args.null_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_csv(output, output_path, index=False)
    atomic_csv(output, args._run_dir / "untrained_null_validation.csv", index=False)
    finish_run_manifest(args, "completed", seeds=seeds, model_config=model_kwargs,
                        resolved_model_config=configured_model_config,
                        resolved_environment=env_config,
                        feature_parameters=args._feature_parameters,
                        matched_cash_ratio=matched_cash)
    print("\nUntrained null summary (validation):")
    print(
        output[[
            "sharpe_ratio",
            "cumulative_return",
            "sharpe_delta_vs_1n_cash",
            "return_delta_vs_1n_cash",
            "sharpe_delta_vs_matched_baseline",
            "return_delta_vs_matched_baseline",
            "mean_abs_deviation_1n",
        ]]
        .agg(["mean", "std", "min", "max"])
        .to_string(float_format=lambda value: f"{value:.6f}")
    )
    print(f"Đã lưu: {output_path}")
    return output

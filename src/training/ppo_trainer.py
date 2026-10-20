"""Single-agent PPO rollout, updates and checkpoint selection. Owner: B."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import random
import shutil
import time
import warnings

from torch import nn
import numpy as np
import pandas as pd
import torch
import yaml

from src.env.cost import resolve_transaction_cost_components
from src.evaluation.metrics import FinancialMetrics
from src.evaluation.research_protocol import checkpoint_learning_gate
from src.evaluation.policy_evaluator import evaluate_equal_weight
from src.evaluation.policy_evaluator import evaluate_policy
from src.evaluation.policy_evaluator import evaluate_policy_with_actions
from src.evaluation.policy_evaluator import metric_summary
from src.evaluation.policy_evaluator import policy_weight_summary
from src.graph.dynamic_graph import build_dynamic_graph_sequence
from src.graph.multi_relation_graph import GRAPH_RELATIONS
from src.models.actor_critic import build_actor_critic
from src.models.market_cache import prepare_market_cache
from src.models.policy_adapter import graph_tensors_from_obs
from src.models.policy_adapter import history_mask_from_obs
from src.models.policy_adapter import observation_vector
from src.training.data_preparation import close_price_matrix
from src.training.data_preparation import prepare_data
from src.training.env_factory import create_env
from src.training.logger import TrainerLogger
from src.training.ppo_core import PPOBuffer
from src.training.ppo_core import PPOUpdater
from src.training.run_artifacts import configure_run_artifacts
from src.training.run_artifacts import finish_run_manifest
from src.training.run_artifacts import record_effective_training_config
from src.utils.artifacts import atomic_csv
from src.utils.artifacts import atomic_json
from src.utils.artifacts import restore_rng_state
from src.utils.artifacts import rng_state
from src.utils.data_types import MarketDataTensor
from src.utils.progress import PhaseProgress

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def train(args: argparse.Namespace) -> tuple[nn.Module, dict[str, Any]]:
    configure_run_artifacts(args)
    try:
        result = _train(args)
    except KeyboardInterrupt:
        finish_run_manifest(args, "interrupted")
        raise
    except Exception as exc:
        finish_run_manifest(args, "failed", error=str(exc))
        raise
    finish_run_manifest(args, "trained", best_epoch=result[1]["checkpoint_epoch"])
    return result


def _train(args: argparse.Namespace) -> tuple[nn.Module, dict[str, Any]]:
    if args.epochs <= 0 or args.ppo_iters <= 0 or args.episodes_per_update <= 0:
        raise ValueError("--epochs, --ppo-iters và --episodes-per-update phải lớn hơn 0.")
    if args.hidden_dim <= 0:
        raise ValueError("--hidden-dim phải lớn hơn 0.")
    if args.train_eval_every <= 0 or args.bootstrap_block_size <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("Lịch eval và thông số bootstrap phải dương.")
    if args.selection_window <= 0 or args.min_epochs < 0 or args.patience < 0:
        raise ValueError("selection-window phải dương; min-epochs và patience không âm.")
    if args.min_epochs > args.epochs:
        raise ValueError("--min-epochs không được lớn hơn --epochs.")
    if not 0.0 < args.action_alpha <= 1.0:
        raise ValueError("--action-alpha phải thuộc (0, 1].")
    if args.target_kl is not None and (
        not np.isfinite(args.target_kl) or args.target_kl <= 0
    ):
        raise ValueError("--target-kl phải là số hữu hạn lớn hơn 0.")
    if args.learning_rate is not None and (
        not np.isfinite(args.learning_rate) or args.learning_rate <= 0
    ):
        raise ValueError("--learning-rate phải là số hữu hạn lớn hơn 0.")
    if args.gamma is not None and (
        not np.isfinite(args.gamma) or not 0.0 < args.gamma <= 1.0
    ):
        raise ValueError("--gamma phải thuộc (0, 1].")
    if args.reward_scaling is not None and (
        not np.isfinite(args.reward_scaling) or args.reward_scaling <= 0
    ):
        raise ValueError("--reward-scaling phải là số hữu hạn lớn hơn 0.")
    if args.entropy_coef is not None and (
        not np.isfinite(args.entropy_coef) or args.entropy_coef < 0
    ):
        raise ValueError("--entropy-coef phải là số hữu hạn không âm.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA được chọn nhưng không khả dụng.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    with args.model_config.open("r", encoding="utf-8") as stream:
        model_config = yaml.safe_load(stream)
    configured_model_config = model_config
    with args.env_config.open("r", encoding="utf-8") as stream:
        env_config = yaml.safe_load(stream)["environment"]
    transaction_fee_pct, slippage_pct, transaction_cost_rate = (
        resolve_transaction_cost_components(env_config)
    )
    env_config.update(
        {
            "transaction_fee_pct": transaction_fee_pct,
            "slippage_pct": slippage_pct,
            "transaction_cost_rate": transaction_cost_rate,
        }
    )

    reward_beta = (
        float(env_config["reward_beta"])
        if args.reward_beta is None
        else float(args.reward_beta)
    )
    if not np.isfinite(reward_beta) or reward_beta < 0:
        raise ValueError("reward_beta phải là số hữu hạn không âm.")
    reward_scaling = (
        float(env_config["reward_scaling"])
        if args.reward_scaling is None
        else float(args.reward_scaling)
    )
    if not np.isfinite(reward_scaling) or reward_scaling <= 0:
        raise ValueError("reward_scaling phải là số hữu hạn lớn hơn 0.")

    (
        full_tensor,
        processed_data,
        scaled_full_tensor,
        split,
        train_tensor,
        val_tensor,
        _test_tensor,
        train_open,
        val_open,
        _test_open,
        train_close,
        val_close,
        _test_close,
        tickers,
    ) = prepare_data(args, reward_beta)
    full_date_index = {str(date): index for index, date in enumerate(full_tensor.dates)}
    train_history_start = full_date_index[str(split.train.dates[0])]
    val_history_start = full_date_index[str(split.val.dates[0])]
    test_history_start = full_date_index[str(split.test.dates[0])] if split.test is not None else None
    graph_metadata = None

    rl_config = model_config["rl_agent"]
    window_min = args.window_min_days
    window_max = min(args.window_max_days, len(train_tensor.dates))
    if window_min < 3 or window_max < window_min:
        raise ValueError(
            "Cửa sổ train phải có ít nhất 3 phiên và dữ liệu train phải đủ dài; "
            "kiểm tra --window-min-days/--window-max-days."
        )
    lookback = (
        int(args.lookback)
        if args.lookback is not None
        else int(env_config.get("lookback_window", 20))
    )
    if lookback <= 0:
        raise ValueError("--lookback phải lớn hơn 0.")
    batch_size = args.batch_size
    if batch_size <= 0:
        raise ValueError("--batch-size phải lớn hơn 0.")

    n_assets, n_features = train_tensor.tensor.shape[1:]
    matched_cash = 1.0 / (n_assets + 1) if args.matched_cash_ratio is None else float(args.matched_cash_ratio)
    if not np.isfinite(matched_cash) or not 0 <= matched_cash < 1:
        raise ValueError("matched-cash-ratio phải thuộc [0,1).")
    from src.training.quant_context import observation_context_dim
    reward_state_dim = observation_context_dim(scaled_full_tensor)
    obs_dim = lookback * n_assets * n_features + n_assets + 1 + reward_state_dim
    action_dim = n_assets + 1
    encoder = getattr(args, "encoder", "gru")
    if encoder not in {"mlp", "lstm", "gru", "gnn", "gcn", "gat"}:
        raise ValueError("--encoder phải là 'mlp', 'lstm', 'gru', 'gnn', 'gcn' hoặc 'gat'.")
    rnn_layers = int(getattr(args, "rnn_layers", 1))
    rnn_dropout = float(getattr(args, "rnn_dropout", 0.0))
    gnn_heads = int(getattr(args, "gnn_heads", 4))
    gnn_dropout = float(getattr(args, "gnn_dropout", 0.1))
    gnn_knn = int(getattr(args, "gnn_knn", 5))
    if rnn_layers <= 0:
        raise ValueError("--rnn-layers phải lớn hơn 0.")
    if not np.isfinite(rnn_dropout) or not 0.0 <= rnn_dropout < 1.0:
        raise ValueError("--rnn-dropout phải hữu hạn và thuộc [0, 1).")
    dirichlet_kappa = (
        float(args.dirichlet_kappa)
        if args.dirichlet_kappa is not None
        else (
            float(rl_config.get("dirichlet_kappa"))
            if rl_config.get("dirichlet_kappa") is not None
            else (
                float(configured_model_config.get("dirichlet_kappa"))
                if configured_model_config.get("dirichlet_kappa") is not None
                else None
            )
        )
    )
    if dirichlet_kappa is not None and dirichlet_kappa <= action_dim:
        raise ValueError(f"--dirichlet-kappa ({dirichlet_kappa}) phải lớn hơn action_dim ({action_dim}).")
    model_config = {
        "encoder": encoder,
        "portfolio_context_dim": reward_state_dim,
        "n_assets": n_assets,
        "n_features": n_features,
        "lookback": lookback,
        "action_dim": action_dim,
        "hidden_dim": args.hidden_dim,
        "rnn_layers": rnn_layers,
        "rnn_dropout": rnn_dropout,
        "dirichlet_kappa": dirichlet_kappa,
        "gnn_heads": gnn_heads,
        "gnn_dropout": gnn_dropout,
        "gnn_knn": gnn_knn,
        "tickers": tickers,
    }
    graph_adjacency_sequence = None
    graph_edge_mask_sequence = None
    if encoder == "gnn":
        gnn_config = configured_model_config.get("gnn", {})
        gnn_architecture = (
            args.gnn_architecture
            or str(gnn_config.get("architecture", "GAT")).lower()
        ).lower()
        if gnn_architecture not in {"gcn", "gat"}:
            raise ValueError("GNN architecture trong config phải là GCN hoặc GAT.")
        gnn_hidden_channels = int(gnn_config.get("hidden_channels", 64) if args.gnn_hidden_channels is None else args.gnn_hidden_channels)
        gnn_out_channels = int(gnn_config.get("out_channels", 32) if args.gnn_out_channels is None else args.gnn_out_channels)
        gnn_heads = int(gnn_config.get("heads", 4) if args.gnn_heads is None else args.gnn_heads)
        corr_window = int(gnn_config.get("correlation_window", 60))
        corr_threshold = float(gnn_config.get("threshold_corr", 0.5))
        graph_mode = str(gnn_config.get("graph_mode", "knn")).lower()
        graph_top_k = int(gnn_config.get("top_k", 5))
        if min(gnn_hidden_channels, gnn_out_channels, gnn_heads) <= 0:
            raise ValueError("Các kích thước GNN trong configs/model.yaml phải dương.")
        if graph_mode not in {"threshold", "knn", "dynamic_threshold"}:
            raise ValueError("gnn.graph_mode phải là threshold, knn hoặc dynamic_threshold.")
        if graph_top_k <= 0:
            raise ValueError("gnn.top_k phải là số nguyên dương.")
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
            window_size=corr_window,
            threshold=corr_threshold,
            graph_mode=graph_mode,
            top_k=graph_top_k,
            sector_config_path=str(args.assets_config),
            covariance_bank=(scaled_full_tensor.quant_context or {}).get("graph_risk"),
        )
        from src.training.quant_context import record_graph_diagnostics
        record_graph_diagnostics(args, graph_adjacency_sequence, graph_edge_mask_sequence)
        graph_metadata = {
            "source": "close_to_close_rolling_correlation_plus_sector_edges",
            "covariance_source": getattr(args, '_quant_config', {}).get('graph', {}).get('covariance_source', 'legacy'),
            "covariance_metadata": ((scaled_full_tensor.quant_context or {}).get('graph_risk').metadata
                if (scaled_full_tensor.quant_context or {}).get('graph_risk') is not None else None),
            "relation_channels": list(GRAPH_RELATIONS),
            "correlation_window": corr_window,
            "correlation_threshold": corr_threshold,
            "graph_mode": graph_mode,
            "top_k": graph_top_k if graph_mode == "knn" else None,
            "gnn_dropout": 0.0,
            "temporal_encoder": "gru",
            "asset_pooling": "learned_attention",
            "time_alignment": "features_date_t_graph_date_t_orders_from_open_t_plus_1",
        }
        model_config.update(
            {
                "gnn_architecture": gnn_architecture,
                "gnn_hidden_channels": gnn_hidden_channels,
                "gnn_out_channels": gnn_out_channels,
                "gnn_heads": gnn_heads,
                "temporal_encoder": "gru",
            }
        )
    model = build_actor_critic(**model_config).to(device)
    model.cache_batch_size = args.cache_batch_size
    model.cache_temporal_batch_size = args.cache_temporal_batch_size
    microbatch_size = args.microbatch_size
    if microbatch_size is None:
        microbatch_size = (16 if device.type == "cpu" else 32) if encoder == "gnn" else batch_size
    microbatch_size = min(microbatch_size, batch_size)
    resolved_training_config = record_effective_training_config(
        args,
        batch_size=batch_size,
        microbatch_size=microbatch_size,
        window_max_days=window_max,
        matched_cash_ratio=matched_cash,
    )

    learning_rate = (
        float(rl_config["learning_rate"])
        if args.learning_rate is None
        else float(args.learning_rate)
    )
    target_kl = (
        float(rl_config.get("target_kl", 0.03))
        if args.target_kl is None
        else float(args.target_kl)
    )
    gamma = (
        float(rl_config.get("gamma", 0.95))
        if args.gamma is None
        else float(args.gamma)
    )
    entropy_coef = (
        float(rl_config.get("entropy_coef", 0.01))
        if args.entropy_coef is None
        else float(args.entropy_coef)
    )
    if not np.isfinite(target_kl) or target_kl <= 0:
        raise ValueError("target_kl phải là số hữu hạn lớn hơn 0.")
    if not np.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("gamma phải thuộc (0, 1].")
    if not np.isfinite(entropy_coef) or entropy_coef < 0:
        raise ValueError("entropy_coef phải là số hữu hạn không âm.")
    value_clip_range = (
        rl_config.get("value_clip_range")
        if args.value_clip_range is None
        else args.value_clip_range
    )
    if value_clip_range is not None:
        value_clip_range = float(value_clip_range)
        if not np.isfinite(value_clip_range) or value_clip_range <= 0:
            raise ValueError("--value-clip-range phải là None hoặc số hữu hạn lớn hơn 0.")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    updater = PPOUpdater(
        clip_ratio=float(rl_config.get("clip_ratio", 0.2)),
        target_kl=target_kl,
        value_loss_coef=float(rl_config.get("value_loss_coef", 0.5)),
        entropy_coef=entropy_coef,
        max_grad_norm=float(rl_config.get("max_grad_norm", 0.5)),
        value_clip_range=value_clip_range,
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.training_log.parent.mkdir(parents=True, exist_ok=True)
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainerLogger(
        log_dir=str(args.tensorboard_dir),
        checkpoint_dir=str(args._run_dir / "checkpoints"),
        best_checkpoint_name="best.pth",
    )
    training_rows: list[dict[str, float]] = []
    best_score = -np.inf
    best_validation_metrics: dict[str, float] = {}
    recent_validation_scores: list[float] = []
    epochs_without_improvement = 0
    last_episode_reward_state = None

    def checkpoint_metadata(epoch: int, val_metrics: dict[str, float]) -> dict[str, Any]:
        from src.evaluation.periods import describe_test_period
        period = describe_test_period(_test_tensor, args.evaluation_scope)
        return {
            "epoch": epoch,
            "checkpoint_status": "untrained_fallback" if epoch == 0 else "trained",
            "quant_config": getattr(args, '_quant_config', {}),
            "market_context_features": (scaled_full_tensor.quant_context or {}).get("market_context_features", []),
            "last_episode_reward_state": last_episode_reward_state,
            "reward_state_convention": "reset_each_episode_state_in_observation" if getattr(args, "_quant_config", {}).get("reward", {}).get("mode") == "differential_sharpe" else None,
            "reward_beta": reward_beta,
            "reward_scaling": reward_scaling,
            "reward_baseline": args.reward_baseline,
            "return_convention": "adjusted_close_scaled_open_close_total_return",
            "sharpe_convention": FinancialMetrics.SHARPE_CONVENTION,
            "universe_methodology": "fixed_current_asset_list_without_point_in_time_membership",
            "test_period_status": period['status'],
            "test_period": period,
            "resolved_split_dates": args._resolved_split_dates,
            "accounting_precision": "float64",
            "transaction_fee_pct": transaction_fee_pct,
            "slippage_pct": slippage_pct,
            "transaction_cost_rate": transaction_cost_rate,
            "max_asset_weight": float(env_config.get("max_asset_weight", 1.0)),
            "target_kl": target_kl,
            "entropy_coef": entropy_coef,
            "gamma": gamma,
            "episodes_per_update": args.episodes_per_update,
            "learning_rate": learning_rate,
            "value_clip_range": value_clip_range,
            "validation_metrics": {
                name: float(value) for name, value in val_metrics.items()
            },
            "observation_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_dim": args.hidden_dim,
            "lookback": lookback,
            "history_mask_convention": "explicit_bool_mask_true_real_timestep_v1",
            "encoder": encoder,
            "actor_critic_topology": "separate_actor_critic_trunks_shared_encoder",
            "model_config": model_config.copy(),
            "n_assets": n_assets,
            "n_features": n_features,
            "rnn_layers": rnn_layers if encoder in {"lstm", "gru", "gnn"} else None,
            "rnn_dropout": rnn_dropout if encoder in {"lstm", "gru", "gnn"} else None,
            "graph_sequence": "rolling_correlation_plus_sector" if encoder == "gnn" else None,
            "graph_date_indexing": "observation_date_t" if encoder == "gnn" else None,
            "graph_config": graph_metadata,
            "graph_relation_channels": list(GRAPH_RELATIONS) if encoder == "gnn" else None,
            "gnn_heads": gnn_heads if encoder in {"gat", "gnn"} else None,
            "gnn_dropout": gnn_dropout if encoder in {"gat", "gcn"} else None,
            "gnn_knn": gnn_knn if encoder in {"gat", "gcn"} else None,
            "action_alpha": args.action_alpha,
            "tickers": tickers,
            "feature_names": full_tensor.feature_names,
            "scaler_path": str(args.scaler_output),
            "seed": args.seed,
            "scaler_params": args._scaler_params,
            "feature_parameters": args._feature_parameters,
            "resolved_model_config": configured_model_config,
            "resolved_training_config": resolved_training_config,
            "resolved_environment_config": env_config,
            "split_dates": split.dates_info,
            "run_manifest": args._manifest,
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": rng_state(),
            "matched_cash_ratio": matched_cash,
            "training_state": {
                "best_score": float(best_score), "best_validation_metrics": best_validation_metrics,
                "recent_validation_scores": recent_validation_scores,
                "epochs_without_improvement": epochs_without_improvement,
                "training_rows": training_rows,
            },
            "reference_validation_metrics": baseline_metrics,
            "reference_score": float(baseline_score),
            "learning_gate": checkpoint_learning_gate(
                epoch, val_metrics, baseline_metrics, matched_val_metrics,
                float(np.mean(recent_validation_scores)) - baseline_score,
                getattr(args, "_research_protocol", None),
            ),
            "matched_validation_metrics": matched_val_metrics,
            "training_parameters": {
                "normalization_warmup": getattr(args, "normalization_warmup", 0),
                "epochs": args.epochs, "ppo_iters": args.ppo_iters, "batch_size": batch_size,
                "window_min": window_min, "window_max": window_max,
                "gamma": gamma, "gae_lambda": float(rl_config.get("gae_lambda", 0.95)),
                "seed": args.seed, "episodes_per_update": args.episodes_per_update,
                "action_alpha": args.action_alpha, "reward_baseline": args.reward_baseline,
                "reward_scaling": reward_scaling, "reward_beta": reward_beta,
                "selection_metric": args.selection_metric, "selection_window": args.selection_window,
                "selection_margin": args.selection_margin,
                "learning_rate": learning_rate, "target_kl": target_kl,
                "entropy_coef": entropy_coef, "value_clip_range": value_clip_range,
                "matched_cash_ratio": matched_cash, "patience": args.patience,
                "min_epochs": args.min_epochs,
                "microbatch_size": microbatch_size, "cpu_threads": args.cpu_threads,
                "cache_batch_size": args.cache_batch_size,
                "cache_temporal_batch_size": args.cache_temporal_batch_size,
                "device": str(device), "train_eval_every": args.train_eval_every,
            },
        }

    print(
        f"[2/6] PPO {encoder.upper()} encoder: "
        + (
            f"obs={obs_dim} (flatten)"
            if encoder == "mlp"
            else (
                f"Dynamic Graph GNN (k-NN={gnn_knn}, D={args.hidden_dim})"
                if encoder in {"gat", "gcn"}
                else f"[{lookback}, {n_assets}, {n_features}]→h_{args.hidden_dim}"
            )
        )
        + f", actions={action_dim}, device={device}, minibatch={batch_size}."
    )
    if encoder == "gnn":
        print(
            f"  Dynamic graph: {model_config['gnn_architecture'].upper()}, "
            f"rolling window={graph_metadata['correlation_window']}, "
            f"graph={graph_metadata['graph_mode']}"
            + (
                f"(k={graph_metadata['top_k']})"
                if graph_metadata["graph_mode"] == "knn"
                else f"(threshold={graph_metadata['correlation_threshold']:g})"
            )
            + ", temporal=GRU, asset pooling=attention."
        )
    print(
        f"[3/6] Tối đa {args.epochs} updates, {args.episodes_per_update} cửa sổ/update "
        f"({window_min}-{window_max} phiên); action_alpha={args.action_alpha:g}, "
        f"beta={reward_beta:g}, reward_scale={reward_scaling:g}, "
        f"fee={transaction_fee_pct:.4%}, slippage={slippage_pct:.4%}, "
        f"total_cost_rate={transaction_cost_rate:.4%}, "
        f"max_asset_weight={float(env_config.get('max_asset_weight', 1.0)):.2%}, "
        f"reward_baseline={args.reward_baseline}, gamma={gamma:g}, "
        f"target_kl={target_kl:g}, entropy={entropy_coef:g}, lr={learning_rate:g}, "
        f"value_clip={'off' if value_clip_range is None else value_clip_range}."
    )
    print(f"  Runtime: CPU threads={torch.get_num_threads()}, effective minibatch={batch_size}, microbatch={microbatch_size}.", flush=True)
    print(f"  Run artifacts: {args._run_dir}", flush=True)
    try:
        train_eval_env = create_env(
            train_tensor,
            train_open,
            args.env_config,
            reward_beta,
            reward_scaling=reward_scaling,
            history_tensor=scaled_full_tensor,
            history_start_index=train_history_start,
            reward_baseline=args.reward_baseline,
            lookback_window=lookback,
            action_alpha=args.action_alpha,
            graph_adjacency_sequence=graph_adjacency_sequence,
            graph_edge_mask_sequence=graph_edge_mask_sequence,
            graph_start_index=train_history_start,
            close_prices=train_close,
        )
        train_baseline_env = create_env(
            train_tensor,
            train_open,
            args.env_config,
            reward_beta,
            reward_scaling=reward_scaling,
            history_tensor=scaled_full_tensor,
            history_start_index=train_history_start,
            lookback_window=lookback,
            action_alpha=1.0,
            close_prices=train_close,
        )
        # Keep an untrained validation reference as a checkpoint fallback.
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
        val_baseline_env = create_env(
            val_tensor,
            val_open,
            args.env_config,
            reward_beta,
            reward_scaling=reward_scaling,
            history_tensor=scaled_full_tensor,
            history_start_index=val_history_start,
            lookback_window=lookback,
            action_alpha=1.0,
            close_prices=val_close,
        )
        equal_weight_train_metrics, equal_weight_train_result = evaluate_equal_weight(
            train_baseline_env,
            train_open,
            n_assets,
            float(env_config["transaction_cost_rate"]),
            float(env_config["risk_free_rate"]),
            cash_ratio=0.0,
        )
        equal_weight_val_metrics, equal_weight_val_result = evaluate_equal_weight(
            val_baseline_env,
            val_open,
            n_assets,
            float(env_config["transaction_cost_rate"]),
            float(env_config["risk_free_rate"]),
            cash_ratio=0.0,
        )
        train_baseline_env.action_alpha = args.action_alpha
        val_baseline_env.action_alpha = args.action_alpha
        matched_train_metrics, matched_train_result = evaluate_equal_weight(
            train_baseline_env,
            train_open,
            n_assets,
            float(env_config["transaction_cost_rate"]),
            float(env_config["risk_free_rate"]),
            cash_ratio=matched_cash,
        )
        matched_val_metrics, matched_val_result = evaluate_equal_weight(
            val_baseline_env,
            val_open,
            n_assets,
            float(env_config["transaction_cost_rate"]),
            float(env_config["risk_free_rate"]),
            cash_ratio=matched_cash,
        )
        logger.log_metrics(
            0,
            {
                "Train/EqualWeightCumulativeReturn": equal_weight_train_metrics[
                    "cumulative_return"
                ],
                "Train/EqualWeightSharpe": equal_weight_train_metrics["sharpe_ratio"],
                "Train/EqualWeightCashCumulativeReturn": matched_train_metrics[
                    "cumulative_return"
                ],
                "Train/EqualWeightCashSharpe": matched_train_metrics["sharpe_ratio"],
                "Validation/EqualWeightCumulativeReturn": equal_weight_val_metrics[
                    "cumulative_return"
                ],
                "Validation/EqualWeightSharpe": equal_weight_val_metrics[
                    "sharpe_ratio"
                ],
                "Validation/EqualWeightCashCumulativeReturn": matched_val_metrics[
                    "cumulative_return"
                ],
                "Validation/EqualWeightCashSharpe": matched_val_metrics[
                    "sharpe_ratio"
                ],
            },
        )
        print(
            "  1/N reference | "
            f"train return={equal_weight_train_metrics['cumulative_return']:.2%}, "
            f"Sharpe={equal_weight_train_metrics['sharpe_ratio']:.3f} | "
            f"val return={equal_weight_val_metrics['cumulative_return']:.2%}, "
            f"Sharpe={equal_weight_val_metrics['sharpe_ratio']:.3f}"
        )
        print(
            f"  Matched 1/N + cash {matched_cash:.2%}, alpha={args.action_alpha:g} | "
            f"train return={matched_train_metrics['cumulative_return']:.2%}, "
            f"Sharpe={matched_train_metrics['sharpe_ratio']:.3f} | "
            f"val return={matched_val_metrics['cumulative_return']:.2%}, "
            f"Sharpe={matched_val_metrics['sharpe_ratio']:.3f}"
        )
        resume_payload = None
        if args.resume is not None:
            resume_payload = torch.load(args.resume, map_location=device, weights_only=False)
            if resume_payload.get("model_config") != model_config or "training_state" not in resume_payload:
                raise ValueError("Checkpoint không tương thích để resume model/training state.")
            for key in ("data_hashes", "source_hashes", "config_hashes"):
                if resume_payload["run_manifest"][key] != args._manifest[key]:
                    raise ValueError(f"Resume yêu cầu cùng {key}.")
            if resume_payload.get("scaler_params") != args._scaler_params:
                raise ValueError("Scaler của resume không khớp dữ liệu train.")
            model.load_state_dict(resume_payload["model_state_dict"])
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
            state = resume_payload["training_state"]
            best_score = float(state["best_score"])
            best_validation_metrics = state["best_validation_metrics"]
            recent_validation_scores = state["recent_validation_scores"]
            epochs_without_improvement = int(state["epochs_without_improvement"])
            training_rows = state["training_rows"]
            logger.best_reward = best_score
            baseline_metrics = resume_payload["reference_validation_metrics"]
            baseline_score = resume_payload["reference_score"]
            current_parameters = checkpoint_metadata(int(resume_payload["epoch"]), baseline_metrics)["training_parameters"]
            if current_parameters != resume_payload["training_parameters"]:
                raise ValueError("Resume yêu cầu cùng training_parameters (bao gồm epochs/scheduler horizon).")
            source_best = args.resume.parent / "best.pth"
            destination_best = logger.checkpoint_dir / "best.pth"
            if source_best.resolve() != destination_best.resolve():
                shutil.copy2(source_best, destination_best)
            restore_rng_state(resume_payload["rng_state"])
        else:
            baseline_result = evaluate_policy(model, val_env)
            baseline_metrics = metric_summary(
                baseline_result,
                risk_free_rate_annual=float(env_config["risk_free_rate"]),
            )
        if args.selection_metric == "sharpe":
            baseline_score = baseline_metrics["sharpe_ratio"]
        elif args.selection_metric == "cumulative_return":
            baseline_score = baseline_metrics["cumulative_return"]
        elif args.selection_metric == "excess_sharpe":
            baseline_score = (
                baseline_metrics["sharpe_ratio"] - matched_val_metrics["sharpe_ratio"]
            )
        elif args.selection_metric == "excess_return":
            baseline_score = (
                baseline_metrics["cumulative_return"] - matched_val_metrics["cumulative_return"]
            )
        else:
            baseline_score = baseline_metrics["sharpe_ratio"]
        logger.log_metrics(
            0,
            {
                "Reference/CumulativeReturn": baseline_metrics["cumulative_return"],
                "Reference/Profit": baseline_metrics["profit"],
                "Reference/Sharpe": baseline_metrics["sharpe_ratio"],
                "Reference/Turnover": baseline_metrics["total_turnover"],
            },
        )
        atomic_json(args._run_dir / "baseline_reference.json", {
            "selection_period": "inner" if getattr(args, "_research_protocol", None) else "validation",
            "untrained_validation": baseline_metrics,
            "equal_weight_train": equal_weight_train_metrics, "equal_weight_validation": equal_weight_val_metrics,
            "matched_train": matched_train_metrics, "matched_validation": matched_val_metrics,
            "matched_cash_ratio": matched_cash, "action_alpha": args.action_alpha,
            "baseline_action_alpha": 1.0, "initial_allocation": "blend_from_cash",
            "transaction_fee_pct": transaction_fee_pct, "slippage_pct": slippage_pct,
            "max_asset_weight": float(env_config.get("max_asset_weight", 1.0)),
            "timeline": "close_t_decision_open_t_plus_1_execution_close_t_plus_1_mark",
            "history_mask_convention": "explicit_bool_mask_true_real_timestep_v1",
        })
        if np.isfinite(baseline_score) and resume_payload is None:
            best_score = float(baseline_score)
            best_validation_metrics = baseline_metrics
            # Compare like with like: epoch 0's reference uses the same
            # selection-window mean as later checkpoints (repeated deterministic
            # evaluations of the unchanged initial policy).
            recent_validation_scores = [float(baseline_score)] * args.selection_window
            logger.save_checkpoint(
                0,
                model,
                optimizer,
                current_reward=best_score,
                selection_metric=f"smoothed_validation_{args.selection_metric}",
                metadata=checkpoint_metadata(0, baseline_metrics),
            )
            logger.save_latest(0, model, optimizer, checkpoint_metadata(0, baseline_metrics))
        print(
            f"  untrained validation reference | return={baseline_metrics['cumulative_return']:.2%} "
            f"| profit={baseline_metrics['profit']:+,.2f} "
            f"| total=${baseline_metrics['final_value']:,.2f} "
            f"| Sharpe={baseline_metrics['sharpe_ratio']:.3f} "
            f"| turnover={baseline_metrics['total_turnover']:.2f}"
        )

        start_epoch = int(resume_payload["epoch"]) + 1 if resume_payload else 1
        if resume_payload:
            print(f"  Resume tại epoch {start_epoch}; run_dir={args._run_dir}")
        for epoch in range(start_epoch, args.epochs + 1):
            t0_roll = time.perf_counter()
            model.train()
            cache_started = time.perf_counter()
            cache_progress = PhaseProgress(f"epoch {epoch:03d} market cache", args.progress_every) if encoder == "gnn" else None
            market_cache = prepare_market_cache(
                model, train_eval_env,
                progress=(lambda stage, completed, total: cache_progress.update(completed, total, stage)) if cache_progress else None,
            )
            cache_time = time.perf_counter() - cache_started
            episode_specs: list[
                tuple[int, int, MarketDataTensor, np.ndarray, np.ndarray]
            ] = []
            for _ in range(args.episodes_per_update):
                length = int(np.random.randint(window_min, window_max + 1))
                start = int(np.random.randint(0, len(train_tensor.dates) - length + 1))
                end = start + length
                episode_tensor = MarketDataTensor(
                    tensor=train_tensor.tensor[start:end].copy(),
                    tickers=list(train_tensor.tickers),
                    feature_names=list(train_tensor.feature_names),
                    dates=list(train_tensor.dates[start:end]),
                )
                episode_tensor.validate()
                episode_specs.append(
                    (
                        start,
                        length,
                        episode_tensor,
                        train_open[start:end].copy(),
                        train_close[start:end].copy(),
                    )
                )

            rollout_size = sum(length - 1 for _, length, _, _, _ in episode_specs)
            buffer = PPOBuffer(
                max_size=rollout_size,
                obs_dim=obs_dim,
                act_dim=action_dim,
                gamma=gamma,
                lam=float(rl_config.get("gae_lambda", 0.95)),
                graph_dim=n_assets if encoder == "gnn" else None,
                graph_lookback=lookback if encoder == "gnn" else 1,
                graph_relations=(len(GRAPH_RELATIONS) if encoder == "gnn" else 1),
                graph_adjacency_sequence=graph_adjacency_sequence,
                graph_edge_mask_sequence=graph_edge_mask_sequence,
                history_lookback=lookback,
            )
            episode_rewards: list[float] = []
            rollout_progress = PhaseProgress(f"epoch {epoch:03d} rollout", args.progress_every)
            for episode_index, (
                start,
                length,
                episode_tensor,
                episode_open,
                episode_close,
            ) in enumerate(
                episode_specs
            ):
                episode_global_start = train_history_start + start
                episode_env = create_env(
                    episode_tensor,
                    episode_open,
                    args.env_config,
                    reward_beta,
                    reward_scaling=reward_scaling,
                    history_tensor=scaled_full_tensor,
                    history_start_index=episode_global_start,
                    reward_baseline=args.reward_baseline,
                    lookback_window=lookback,
                    action_alpha=args.action_alpha,
                    graph_adjacency_sequence=graph_adjacency_sequence,
                    graph_edge_mask_sequence=graph_edge_mask_sequence,
                    graph_start_index=episode_global_start,
                    close_prices=episode_close,
                )
                obs, _ = episode_env.reset(
                    seed=args.seed + epoch * args.episodes_per_update + episode_index
                )
                episode_reward = 0.0
                done = False
                while not done:
                    obs_vector = observation_vector(obs)
                    state = torch.as_tensor(obs_vector, dtype=torch.float32, device=device)
                    history_mask = history_mask_from_obs(obs, device)
                    if market_cache is not None:
                        graph = {"market_embedding": market_cache.get(model, episode_global_start + episode_env.current_step)}
                    else:
                        graph = graph_tensors_from_obs(obs, device)
                    with torch.no_grad():
                        if graph is None:
                            policy_action, log_prob, value = model.act(state, history_mask=history_mask)
                        else:
                            policy_action, log_prob, value = model.act(state, graph=graph, history_mask=history_mask)
                    policy_action_np = policy_action.cpu().numpy().astype(np.float32)
                    next_obs, reward, terminated, truncated, _ = episode_env.step(
                        policy_action_np
                    )
                    # Store the sampled action and its log-probability; blending is a
                    # deterministic part of the action mapping into the environment.
                    buffer.store(
                        obs_vector,
                        policy_action_np,
                        float(reward),
                        float(value.item()),
                        float(log_prob.item()),
                        graph_day_index=(episode_global_start + episode_env.current_step - 1 if graph is not None else None),
                        history_mask=obs["history_mask"],
                    )
                    episode_reward += float(reward)
                    obs = next_obs
                    done = terminated or truncated
                    if buffer.ptr % 32 == 0 or done:
                        rollout_progress.update(buffer.ptr, rollout_size, f"episode {episode_index + 1}/{args.episodes_per_update}")

                # Random-window boundaries bootstrap the critic. The final train
                # boundary is terminal because no later training observations exist.
                if start + length >= len(train_tensor.dates):
                    last_value = 0.0
                else:
                    last_vector = observation_vector(obs)
                    last_state = torch.as_tensor(last_vector, dtype=torch.float32, device=device)
                    last_history_mask = history_mask_from_obs(obs, device)
                    if market_cache is not None:
                        last_graph = {"market_embedding": market_cache.get(model, episode_global_start + episode_env.current_step)}
                    else:
                        last_graph = graph_tensors_from_obs(obs, device)
                    with torch.no_grad():
                        if last_graph is None:
                            _, _, last_value_tensor = model.act(
                                last_state, deterministic=True, history_mask=last_history_mask
                            )
                        else:
                            _, _, last_value_tensor = model.act(
                                last_state, deterministic=True, graph=last_graph, history_mask=last_history_mask
                            )
                    last_value = float(last_value_tensor.item())
                buffer.finish_path(last_val=last_value)

                last_episode_reward_state = episode_env.reward_state_dict()
                episode_rewards.append(episode_reward)

            step_rewards = buffer.rew_buf[:buffer.ptr]
            step_reward_mean = float(np.mean(step_rewards))
            step_reward_std = float(np.std(step_rewards))
            t_roll = time.perf_counter() - t0_roll

            t0_upd = time.perf_counter()
            update_progress = PhaseProgress(f"epoch {epoch:03d} PPO update", args.progress_every)
            batch = {
                name: tensor.to(device)
                for name, tensor in buffer.get().items()
            }
            update_stats = updater.update(
                model,
                optimizer,
                batch,
                train_iters=args.ppo_iters,
                batch_size=batch_size,
                microbatch_size=microbatch_size,
                progress=lambda completed, total, piece, pieces: update_progress.update(completed, total, f"microbatch {piece}/{pieces}"),
            )
            update_progress.update(
                update_stats["NumMinibatches"], args.ppo_iters * int(np.ceil(rollout_size / batch_size)),
                f"KL stop={bool(update_stats['StoppedEarly'])}", force=True,
            )
            scheduler.step()
            t_upd = time.perf_counter() - t0_upd

            t0_eval = time.perf_counter()
            eval_progress = PhaseProgress(f"epoch {epoch:03d} evaluation", args.progress_every)
            train_evaluated = epoch == 1 or epoch == args.epochs or epoch % args.train_eval_every == 0
            if train_evaluated:
                train_eval_result, train_policy_actions, train_blended_targets = evaluate_policy_with_actions(model, train_eval_env)
                train_policy_weights = policy_weight_summary(train_policy_actions, n_assets, tickers)
                train_blended_weights = policy_weight_summary(train_blended_targets, n_assets, tickers)
                train_eval_metrics = metric_summary(train_eval_result, float(env_config["risk_free_rate"]))
                eval_progress.update(1, 2, "train hoàn tất", force=True)
            else:
                train_eval_metrics = {key: float("nan") for key in equal_weight_train_metrics}
                template = policy_weight_summary(np.full((1, action_dim), 1.0 / action_dim), n_assets, tickers)
                train_policy_weights = {key: float("nan") for key in template}
                train_blended_weights = train_policy_weights.copy()
            val_result, val_policy_actions, val_blended_targets = (
                evaluate_policy_with_actions(model, val_env)
            )
            val_policy_weights = policy_weight_summary(
                val_policy_actions, n_assets, tickers
            )
            val_blended_weights = policy_weight_summary(
                val_blended_targets, n_assets, tickers
            )
            val_metrics = metric_summary(
                val_result,
                risk_free_rate_annual=float(env_config["risk_free_rate"]),
            )
            t_eval = time.perf_counter() - t0_eval
            eval_progress.update(2 if train_evaluated else 1, 2 if train_evaluated else 1, "validation hoàn tất")
            train_sharpe_delta_1n = (
                train_eval_metrics["sharpe_ratio"]
                - equal_weight_train_metrics["sharpe_ratio"]
            )
            train_return_delta_1n = (
                train_eval_metrics["cumulative_return"]
                - equal_weight_train_metrics["cumulative_return"]
            )
            train_sharpe_delta_1n_cash = (
                train_eval_metrics["sharpe_ratio"]
                - matched_train_metrics["sharpe_ratio"]
            )
            train_return_delta_1n_cash = (
                train_eval_metrics["cumulative_return"]
                - matched_train_metrics["cumulative_return"]
            )
            val_sharpe_delta_1n = (
                val_metrics["sharpe_ratio"] - equal_weight_val_metrics["sharpe_ratio"]
            )
            val_return_delta_1n = (
                val_metrics["cumulative_return"]
                - equal_weight_val_metrics["cumulative_return"]
            )
            val_sharpe_delta_1n_cash = (
                val_metrics["sharpe_ratio"] - matched_val_metrics["sharpe_ratio"]
            )
            val_return_delta_1n_cash = (
                val_metrics["cumulative_return"]
                - matched_val_metrics["cumulative_return"]
            )
            if args.selection_metric == "sharpe":
                score = val_metrics["sharpe_ratio"]
            elif args.selection_metric == "cumulative_return":
                score = val_metrics["cumulative_return"]
            elif args.selection_metric == "excess_sharpe":
                score = val_sharpe_delta_1n_cash
            elif args.selection_metric == "excess_return":
                score = val_return_delta_1n_cash
            else:
                score = val_metrics["sharpe_ratio"]
            recent_validation_scores.append(float(score))
            recent_validation_scores = recent_validation_scores[-args.selection_window :]
            smoothed_score = float(np.mean(recent_validation_scores))
            learning_gate = checkpoint_learning_gate(
                epoch, val_metrics, baseline_metrics, matched_val_metrics,
                smoothed_score - baseline_score, getattr(args, "_research_protocol", None),
            )
            training_row = {
                "epoch": float(epoch),
                "train_evaluated": int(train_evaluated),
                "time_rollout": t_roll,
                "time_market_cache": cache_time,
                "graph_storage_mb": (graph_adjacency_sequence.nbytes + graph_edge_mask_sequence.nbytes) / 1e6 if encoder == "gnn" else 0.0,
                "time_update": t_upd,
                "time_eval": t_eval,
                "gamma": gamma,
                "target_kl": target_kl,
                "reward_scaling": reward_scaling,
                "episodes_per_update": float(args.episodes_per_update),
                "train_reward_mean": float(np.mean(episode_rewards)),
                "train_reward_std": float(np.std(episode_rewards)),
                "policy_loss": float(update_stats["LossPi"]),
                "value_loss": float(update_stats["LossV"]),
                "explained_variance": float(update_stats["ExplainedVar"]),
                "entropy": float(update_stats["Entropy"]),
                "kl": float(update_stats["KL"]),
                "kl_first": float(update_stats["FirstKL"]),
                "kl_max": float(update_stats["MaxKL"]),
                "kl_stop": float(update_stats["StopKL"]),
                "clip_fraction": float(update_stats["ClipFrac"]),
                "ppo_epochs_completed": float(update_stats["StopIter"]),
                "ppo_minibatches_completed": float(update_stats["NumMinibatches"]),
                "ppo_minibatches_attempted": float(update_stats["NumMinibatchesAttempted"]),
                "ppo_microbatches_completed": float(update_stats["NumMicrobatches"]),
                "ppo_stopped_early": float(update_stats["StoppedEarly"]),
                "train_cumulative_return": train_eval_metrics["cumulative_return"],
                "train_profit": train_eval_metrics["profit"],
                "train_sharpe": train_eval_metrics["sharpe_ratio"],
                "train_1n_cumulative_return": equal_weight_train_metrics[
                    "cumulative_return"
                ],
                "train_1n_sharpe": equal_weight_train_metrics["sharpe_ratio"],
                "train_return_delta_1n": train_return_delta_1n,
                "train_sharpe_delta_1n": train_sharpe_delta_1n,
                "step_reward_mean": step_reward_mean,
                "step_reward_std": step_reward_std,
                "train_1n_cash_cumulative_return": matched_train_metrics["cumulative_return"],
                "train_1n_cash_sharpe": matched_train_metrics["sharpe_ratio"],
                "train_return_delta_1n_cash": train_return_delta_1n_cash,
                "train_sharpe_delta_1n_cash": train_sharpe_delta_1n_cash,
                "validation_1n_cash_cumulative_return": matched_val_metrics["cumulative_return"],
                "validation_1n_cash_sharpe": matched_val_metrics["sharpe_ratio"],
                "validation_return_delta_1n_cash": val_return_delta_1n_cash,
                "validation_sharpe_delta_1n_cash": val_sharpe_delta_1n_cash,
                "validation_score": float(score),
                "validation_smoothed_score": smoothed_score,
                "validation_cumulative_return": val_metrics["cumulative_return"],
                "validation_profit": val_metrics["profit"],
                "validation_sharpe": val_metrics["sharpe_ratio"],
                "validation_1n_cumulative_return": equal_weight_val_metrics[
                    "cumulative_return"
                ],
                "validation_1n_sharpe": equal_weight_val_metrics["sharpe_ratio"],
                "validation_return_delta_1n": val_return_delta_1n,
                "validation_sharpe_delta_1n": val_sharpe_delta_1n,
                "validation_max_drawdown": val_metrics["max_drawdown"],
                "validation_turnover": val_metrics["total_turnover"],
                "validation_portfolio_value": float(val_result.portfolio_values[-1]),
            }
            training_row.update(
                {
                    f"train_policy_{name}": value
                    for name, value in train_policy_weights.items()
                }
            )
            training_row.update(
                {
                    f"train_blended_{name}": value
                    for name, value in train_blended_weights.items()
                }
            )
            training_row.update(
                {
                    f"validation_policy_{name}": value
                    for name, value in val_policy_weights.items()
                }
            )
            training_row.update(
                {
                    f"validation_blended_{name}": value
                    for name, value in val_blended_weights.items()
                }
            )
            for prefix, values in (('train',train_eval_metrics),('validation',val_metrics)):
                for name in ('total_commission','total_slippage','total_impact','hold_fraction','participation_max'):
                    if name in values:
                        training_row[f'{prefix}_{name}'] = values[name]
            if last_episode_reward_state is not None:
                training_row['last_episode_dsr_raw'] = last_episode_reward_state['last_raw']
                training_row['last_episode_dsr_clipped_fraction'] = last_episode_reward_state['clipped']/max(last_episode_reward_state['count'],1)
            training_rows.append(training_row)
            training_row.update({
                "validation_sharpe_delta_vs_untrained": learning_gate["sharpe_delta_vs_untrained"],
                "validation_selection_delta_vs_untrained": learning_gate["smoothed_selection_delta_vs_untrained"],
                "learning_gate_passed": int(learning_gate["passed"]),
            })
            epoch_metrics = {
                    "Validation/LearningGatePassed": int(learning_gate["passed"]),
                    "Validation/SharpeDeltaVsUntrained": learning_gate["sharpe_delta_vs_untrained"],
                    "Validation/SmoothedSelectionDeltaVsUntrained": learning_gate["smoothed_selection_delta_vs_untrained"],
                    "Time/Rollout": t_roll,
                    "Time/Update": t_upd,
                    "Time/Eval": t_eval,
                    "Train/RewardMean": float(np.mean(episode_rewards)),
                    "Train/RewardStd": float(np.std(episode_rewards)),
                    "Train/StepRewardMean": step_reward_mean,
                    "Train/StepRewardStd": step_reward_std,
                    "Train/PolicyLoss": float(update_stats["LossPi"]),
                    "Train/ValueLoss": float(update_stats["LossV"]),
                    "Train/ExplainedVariance": float(update_stats["ExplainedVar"]),
                    "Train/Entropy": float(update_stats["Entropy"]),
                    "Train/KL": float(update_stats["KL"]),
                    "Train/KLFirst": float(update_stats["FirstKL"]),
                    "Train/KLMax": float(update_stats["MaxKL"]),
                    "Train/KLStop": float(update_stats["StopKL"]),
                    "Train/ClipFraction": float(update_stats["ClipFrac"]),
                    "Train/PPOEpochsCompleted": float(update_stats["StopIter"]),
                    "Train/PPOMinibatchesCompleted": float(update_stats["NumMinibatches"]),
                    "Train/PPOMinibatchesAttempted": float(update_stats["NumMinibatchesAttempted"]),
                    "Train/StoppedEarly": float(update_stats["StoppedEarly"]),
                    "Train/CumulativeReturn": train_eval_metrics["cumulative_return"],
                    "Train/Profit": train_eval_metrics["profit"],
                    "Train/Sharpe": train_eval_metrics["sharpe_ratio"],
                    "Train/EqualWeightSharpe": equal_weight_train_metrics["sharpe_ratio"],
                    "Train/EqualWeightCumulativeReturn": equal_weight_train_metrics[
                        "cumulative_return"
                    ],
                    "Train/EqualWeightCashSharpe": matched_train_metrics["sharpe_ratio"],
                    "Train/EqualWeightCashCumulativeReturn": matched_train_metrics[
                        "cumulative_return"
                    ],
                    "Train/SharpeDeltaVs1N": train_sharpe_delta_1n,
                    "Train/ReturnDeltaVs1N": train_return_delta_1n,
                    "Train/SharpeDeltaVs1NCash": train_sharpe_delta_1n_cash,
                    "Train/ReturnDeltaVs1NCash": train_return_delta_1n_cash,
                    "Validation/CumulativeReturn": val_metrics["cumulative_return"],
                    "Validation/Profit": val_metrics["profit"],
                    "Validation/Sharpe": val_metrics["sharpe_ratio"],
                    "Validation/MaxDrawdown": val_metrics["max_drawdown"],
                    "Validation/Turnover": val_metrics["total_turnover"],
                    "Validation/PortfolioValue": val_result.portfolio_values[-1],
                    "Validation/EqualWeightSharpe": equal_weight_val_metrics[
                        "sharpe_ratio"
                    ],
                    "Validation/EqualWeightCumulativeReturn": equal_weight_val_metrics[
                        "cumulative_return"
                    ],
                    "Validation/EqualWeightCashSharpe": matched_val_metrics[
                        "sharpe_ratio"
                    ],
                    "Validation/EqualWeightCashCumulativeReturn": matched_val_metrics[
                        "cumulative_return"
                    ],
                    "Validation/SharpeDeltaVs1N": val_sharpe_delta_1n,
                    "Validation/ReturnDeltaVs1N": val_return_delta_1n,
                    "Validation/SharpeDeltaVs1NCash": val_sharpe_delta_1n_cash,
                    "Validation/ReturnDeltaVs1NCash": val_return_delta_1n_cash,
                    "Validation/SmoothedScore": smoothed_score,
                }
            epoch_metrics.update(
                {
                    f"Train/PolicyWeights/{name}": value
                    for name, value in train_policy_weights.items()
                }
            )
            epoch_metrics.update(
                {
                    f"Train/BlendedWeights/{name}": value
                    for name, value in train_blended_weights.items()
                }
            )
            epoch_metrics.update(
                {
                    f"Validation/PolicyWeights/{name}": value
                    for name, value in val_policy_weights.items()
                }
            )
            epoch_metrics.update(
                {
                    f"Validation/BlendedWeights/{name}": value
                    for name, value in val_blended_weights.items()
                }
            )
            logger.log_metrics(epoch, epoch_metrics)

            improved = (
                np.isfinite(smoothed_score)
                and smoothed_score > (best_score + args.selection_margin)
                and (not getattr(args, "_research_protocol", None) or learning_gate["passed"])
            )
            if improved:
                best_score = smoothed_score
                best_validation_metrics = val_metrics
                epochs_without_improvement = 0
                logger.save_checkpoint(
                    epoch,
                    model,
                    optimizer,
                    current_reward=best_score,
                    selection_metric=f"smoothed_validation_{args.selection_metric}",
                    metadata=checkpoint_metadata(epoch, val_metrics),
                )
            elif epoch >= args.min_epochs:
                epochs_without_improvement += 1

            logger.save_latest(epoch, model, optimizer, checkpoint_metadata(epoch, val_metrics))
            atomic_csv(pd.DataFrame(training_rows), args.training_log, index=False)
            atomic_csv(pd.DataFrame(training_rows), args._run_dir / "training_log.csv", index=False)

            print(
                f"  epoch {epoch:03d}/{args.epochs} "
                f"| time: roll={t_roll:.1f}s upd={t_upd:.1f}s eval={t_eval:.1f}s "
                f"| reward={np.mean(episode_rewards):.3f}±{np.std(episode_rewards):.3f} (step std={step_reward_std:.3f}) "
                f"| policy loss={update_stats['LossPi']:.4f} "
                f"| value loss={update_stats['LossV']:.4f} "
                f"| EV={update_stats['ExplainedVar']:.3f} "
                f"\n| train return={train_eval_metrics['cumulative_return']:.2%} "
                f"| train profit={train_eval_metrics['profit']:+,.2f} "
                f"| train total={train_eval_metrics['final_value']:,.2f} "
                f"| train Sharpe={train_eval_metrics['sharpe_ratio']:.3f} "
                f"(Δ1/N {train_sharpe_delta_1n:+.3f}, Δ1/N+cash {train_sharpe_delta_1n_cash:+.3f}) "
                f"| train return Δ1/N={train_return_delta_1n:+.2%} "
                f"\n| val return={val_metrics['cumulative_return']:.2%} "
                f"| val profit={val_metrics['profit']:+,.2f} "
                f"| val total={val_metrics['final_value']:,.2f} "
                f"| val Sharpe={val_metrics['sharpe_ratio']:.3f} "
                f"(Δ1/N {val_sharpe_delta_1n:+.3f}, Δ1/N+cash {val_sharpe_delta_1n_cash:+.3f}) "
                f"| val return Δ1/N={val_return_delta_1n:+.2%} (Δ1/N+cash {val_return_delta_1n_cash:+.2%}) "
                f"| val policy cash={val_policy_weights['mean_cash_weight']:.2%} "
                f"| val mean |w−1/N|={val_policy_weights['mean_abs_deviation_1n']:.3%} "
                f"| val temporal weight std={val_policy_weights['mean_temporal_weight_std']:.3%} "
                f"| val blended cash={val_blended_weights['mean_cash_weight']:.2%} "
                f"| val turnover={val_metrics['total_turnover']:.2f} "
                f"| entropy={update_stats['Entropy']:.4f} "
                f"| KL={update_stats['KL']:.5f} "
                f"| KLfirst={update_stats['FirstKL']:.2g} "
                f"| KLmax={update_stats['MaxKL']:.5f} "
                f"| KLstop={update_stats['StopKL']:.5f} "
                f"| clipfrac={update_stats['ClipFrac']:.3f} "
                f"| PPO={update_stats['NumMinibatches']}/"
                f"{update_stats['NumMinibatchesAttempted']} minibatches/"
                f"{update_stats['StopIter']} epochs "
                f"| lr={scheduler.get_last_lr()[0]:.2e} "
                f"| val MDD={val_metrics['max_drawdown']:.2%}"
            )
            if args.patience > 0 and epoch >= args.min_epochs:
                if epochs_without_improvement >= args.patience:
                    print(
                        f"  Early stopping: validation không cải thiện trong "
                        f"{args.patience} epochs."
                    )
                    break
    finally:
        atomic_csv(pd.DataFrame(training_rows), args.training_log, index=False)
        atomic_csv(pd.DataFrame(training_rows), args._run_dir / "training_log.csv", index=False)
        logger.close()

    if not best_validation_metrics:
        raise RuntimeError("Không tạo được checkpoint: validation score không hữu hạn.")
    best_path = logger.checkpoint_dir / "best.pth"
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    if best_path.resolve() != args.checkpoint.resolve():
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, args.checkpoint)
    saved_model_config = checkpoint.get("model_config")
    if saved_model_config is not None and saved_model_config != model_config:
        raise ValueError(
            "Checkpoint architecture không khớp với model vừa tạo: "
            f"saved={saved_model_config}, current={model_config}"
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    checkpoint_status = checkpoint.get("checkpoint_status", "legacy_or_unknown")
    if checkpoint_status == "untrained_fallback" or int(checkpoint.get("epoch", -1)) == 0:
        warnings.warn(
            f"Checkpoint tốt nhất là epoch 0 (policy chưa train). Kết quả {args.evaluation_scope} sẽ đo policy khởi tạo, "
            "không phải kết quả của một policy PPO đã học.",
            RuntimeWarning,
            stacklevel=2,
        )
    print(
        f"Best checkpoint: epoch {checkpoint['epoch']} | "
        f"status={checkpoint_status} | "
        f"validation {args.selection_metric}={checkpoint['selection_score']:.4f} | "
        f"val profit={best_validation_metrics['profit']:+,.2f}"
    )
    return model, {
        "val_tensor": val_tensor, "val_open": val_open, "val_close": val_close,
        "val_history_start": val_history_start,
        **getattr(args, "_outer_validation_context", {}),
        "reference_validation_metrics": baseline_metrics,
        "matched_validation_metrics": matched_val_metrics,
        "learning_gate": checkpoint.get("learning_gate", {}),
        "full_tensor": full_tensor,
        "scaled_full_tensor": scaled_full_tensor,
        "split": split,
        "test_tensor": _test_tensor,
        "test_open": _test_open,
        "test_close": _test_close,
        "tickers": tickers,
        "reward_beta": reward_beta,
        "reward_scaling": reward_scaling,
        "lookback": lookback,
        "test_history_start": test_history_start,
        "graph_adjacency_sequence": graph_adjacency_sequence,
        "graph_edge_mask_sequence": graph_edge_mask_sequence,
        "action_alpha": args.action_alpha,
        "env_config": env_config,
        "resolved_split_dates": args._resolved_split_dates,
        "best_validation_metrics": best_validation_metrics,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_status": checkpoint_status,
        "matched_cash_ratio": matched_cash,
        "run_dir": args._run_dir,
    }

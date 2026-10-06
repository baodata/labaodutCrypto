"""End-to-end single-agent PPO training and out-of-sample evaluation.

Run from the repository root with::

    python -m src.training.train_ppo --reward-beta 2.0

The policy receives engineered, train-normalized features. Raw OHLCV prices are
used to build those features and to simulate order execution in TradingEnv.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import TemporalSplitter
from src.env.trading_env import TradingEnv
from src.evaluation.backtester import BacktestResult, DeterministicBacktester
from src.evaluation.baselines import BuyAndHoldBaseline, CashBaseline, EqualWeightBaseline
from src.evaluation.metrics import FinancialMetrics
from src.features.pipeline import FeaturePipeline
from src.features.scaler import MarketFeatureScaler
from src.models.actor_critic import SimplexActorCritic
from src.training.logger import TrainerLogger
from src.training.ppo_core import PPOBuffer, PPOUpdater
from src.utils.data_types import MarketDataTensor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument(
        "--processed-output", type=Path, default=PROJECT_ROOT / "data/processed/features.parquet"
    )
    parser.add_argument("--model-config", type=Path, default=PROJECT_ROOT / "configs/model.yaml")
    parser.add_argument("--env-config", type=Path, default=PROJECT_ROOT / "configs/env.yaml")
    parser.add_argument("--train-end", default=TemporalSplitter.DEFAULT_TRAIN_END)
    parser.add_argument("--val-start", default=TemporalSplitter.DEFAULT_VAL_START)
    parser.add_argument("--val-end", default=TemporalSplitter.DEFAULT_VAL_END)
    parser.add_argument("--test-start", default=TemporalSplitter.DEFAULT_TEST_START)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--ppo-iters", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--window-min-days", type=int, default=126)
    parser.add_argument("--window-max-days", type=int, default=252)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument(
        "--action-alpha",
        type=float,
        default=0.25,
        help="Tỷ lệ tiến về danh mục policy mỗi bước; nhỏ hơn giúp giảm turnover.",
    )
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--selection-window", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Tùy chọn chạy nhiều seed, ví dụ: --seeds 7,17,27; xuất thêm mean/std test.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--reward-beta",
        type=float,
        default=None,
        help="Penalty multiplier for transaction costs; defaults to configs/env.yaml.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("sharpe", "cumulative_return"),
        default="sharpe",
        help="Validation metric used to select the best checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models/checkpoints/sprint5_best.pth",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=PROJECT_ROOT / "experiments/sprint5_metrics.csv",
    )
    parser.add_argument(
        "--training-log",
        type=Path,
        default=PROJECT_ROOT / "experiments/sprint5_training_log.csv",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=PROJECT_ROOT / "logs/ppo_training",
    )
    parser.add_argument(
        "--scaler-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/scaler_params.json",
    )
    return parser


def observation_vector(
    obs: dict[str, np.ndarray], env: TradingEnv, lookback: int
) -> np.ndarray:
    """Flatten the last lookback days of engineered features and portfolio state."""
    history_source = getattr(env, "history_tensor", env.market_tensor)
    current_step = int(getattr(env, "history_start_index", 0)) + int(env.current_step)
    start = max(0, current_step - lookback + 1)
    history = history_source.tensor[start : current_step + 1]
    if len(history) < lookback:
        padding = np.zeros(
            (lookback - len(history), *history.shape[1:]), dtype=np.float32
        )
        history = np.concatenate((padding, history), axis=0)
    return np.concatenate(
        (
            np.asarray(history, dtype=np.float32).reshape(-1),
            np.asarray(obs["portfolio_weights"], dtype=np.float32).reshape(-1),
            np.asarray(obs["cash_ratio"], dtype=np.float32).reshape(-1),
        )
    )


def blend_action(
    policy_action: np.ndarray,
    obs: dict[str, np.ndarray],
    action_alpha: float,
) -> np.ndarray:
    """Move gradually from current holdings toward the policy target."""
    current_weights = np.concatenate(
        (obs["portfolio_weights"], obs["cash_ratio"])
    ).astype(np.float32)
    blended = (1.0 - action_alpha) * current_weights + action_alpha * policy_action
    return blended.astype(np.float32)


def action_for_model(
    model: SimplexActorCritic,
    obs: dict[str, np.ndarray],
    env: TradingEnv,
    lookback: int,
    action_alpha: float,
    deterministic: bool = False,
) -> np.ndarray:
    device = next(model.parameters()).device
    state = torch.as_tensor(
        observation_vector(obs, env, lookback), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        action, _, _ = model.act(state, deterministic=deterministic)
    policy_action = action.detach().cpu().numpy().astype(np.float32)
    return blend_action(policy_action, obs, action_alpha)


def open_price_matrix(processed_data: dict[str, pd.DataFrame], tickers: list[str]) -> np.ndarray:
    columns: list[np.ndarray] = []
    expected_index = processed_data[tickers[0]].index
    for ticker in tickers:
        frame = processed_data[ticker]
        if not frame.index.equals(expected_index):
            raise ValueError(f"Lịch giao dịch của {ticker} không khớp với các mã còn lại.")
        open_col = next((column for column in frame.columns if str(column).lower() == "open"), None)
        if open_col is None:
            raise ValueError(f"Thiếu cột Open cần cho mô phỏng giao dịch: {ticker}")
        columns.append(frame[open_col].to_numpy(dtype=np.float64))
    prices = np.column_stack(columns)
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("Giá Open phải hữu hạn và lớn hơn 0.")
    return prices


def create_env(
    tensor,
    open_prices: np.ndarray,
    env_config: Path,
    reward_beta: float,
    history_tensor: MarketDataTensor | None = None,
    history_start_index: int = 0,
) -> TradingEnv:
    env = TradingEnv(
        market_tensor=tensor,
        open_prices=open_prices,
        config_path=str(env_config),
        reward_beta=reward_beta,
    )
    env.history_tensor = history_tensor if history_tensor is not None else tensor
    env.history_start_index = history_start_index
    return env


def evaluate_policy(
    model: SimplexActorCritic,
    env: TradingEnv,
    lookback: int,
    action_alpha: float,
) -> BacktestResult:
    model.eval()
    return DeterministicBacktester(env).run_agent(
        lambda obs: action_for_model(
            model,
            obs,
            env,
            lookback=lookback,
            action_alpha=action_alpha,
            deterministic=True,
        )
    )


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
    summary["profit"] = float(result.portfolio_values[-1] - result.portfolio_values[0])
    return summary


def prepare_data(args: argparse.Namespace, reward_beta: float):
    if not args.raw_dir.exists() or not any(args.raw_dir.glob("*.parquet")):
        raise FileNotFoundError(
            f"Không tìm thấy OHLCV parquet trong {args.raw_dir}. "
            "Hãy đặt dữ liệu raw vào thư mục này trước khi train."
        )

    print("[1/6] Tính feature, căn lịch và ghi processed parquet...")
    pipeline = FeaturePipeline()
    processed_data, full_tensor = pipeline.run_from_raw(
        raw_dir=args.raw_dir,
        export_parquet=True,
        output_file=args.processed_output,
    )
    split = TemporalSplitter(
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
    ).split_market_data_tensor(full_tensor)

    scaler = MarketFeatureScaler()
    scaler.fit(split.train)
    train_tensor = scaler.transform(split.train)
    val_tensor = scaler.transform(split.val)
    test_tensor = scaler.transform(split.test)
    scaled_full_tensor = scaler.transform(full_tensor)
    args.scaler_output.parent.mkdir(parents=True, exist_ok=True)
    scaler.save(args.scaler_output)

    tickers = list(full_tensor.tickers)
    all_open_prices = open_price_matrix(processed_data, tickers)
    date_to_index = {str(date): index for index, date in enumerate(full_tensor.dates)}

    def prices_for(partition) -> np.ndarray:
        indices = [date_to_index[str(date)] for date in partition.dates]
        return all_open_prices[indices].copy()

    print(split.summary())
    print(f"Features: {', '.join(full_tensor.feature_names)}")
    print(f"Transaction-cost reward beta: {reward_beta:g}")
    return (
        full_tensor,
        scaled_full_tensor,
        split,
        train_tensor,
        val_tensor,
        test_tensor,
        prices_for(split.train),
        prices_for(split.val),
        prices_for(split.test),
        tickers,
    )


def train(args: argparse.Namespace) -> tuple[SimplexActorCritic, dict[str, Any]]:
    if args.epochs <= 0 or args.ppo_iters <= 0 or args.episodes_per_update <= 0:
        raise ValueError("--epochs, --ppo-iters và --episodes-per-update phải lớn hơn 0.")
    if args.hidden_dim <= 0:
        raise ValueError("--hidden-dim phải lớn hơn 0.")
    if args.selection_window <= 0 or args.min_epochs < 0 or args.patience < 0:
        raise ValueError("selection-window phải dương; min-epochs và patience không âm.")
    if args.min_epochs > args.epochs:
        raise ValueError("--min-epochs không được lớn hơn --epochs.")
    if not 0.0 < args.action_alpha <= 1.0:
        raise ValueError("--action-alpha phải thuộc (0, 1].")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA được chọn nhưng không khả dụng.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    with args.model_config.open("r", encoding="utf-8") as stream:
        model_config = yaml.safe_load(stream)
    with args.env_config.open("r", encoding="utf-8") as stream:
        env_config = yaml.safe_load(stream)["environment"]

    reward_beta = (
        float(env_config["reward_beta"])
        if args.reward_beta is None
        else float(args.reward_beta)
    )
    if not np.isfinite(reward_beta) or reward_beta < 0:
        raise ValueError("reward_beta phải là số hữu hạn không âm.")

    (
        full_tensor,
        scaled_full_tensor,
        split,
        train_tensor,
        val_tensor,
        _test_tensor,
        train_open,
        val_open,
        _test_open,
        tickers,
    ) = prepare_data(args, reward_beta)
    full_date_index = {str(date): index for index, date in enumerate(full_tensor.dates)}
    val_history_start = full_date_index[str(split.val.dates[0])]
    test_history_start = full_date_index[str(split.test.dates[0])]

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
    if batch_size is None:
        batch_size = int(rl_config.get("batch_size", 256))
    if batch_size <= 0:
        raise ValueError("--batch-size phải lớn hơn 0.")

    n_assets, n_features = train_tensor.tensor.shape[1:]
    obs_dim = lookback * n_assets * n_features + n_assets + 1
    action_dim = n_assets + 1
    model = SimplexActorCritic(obs_dim, action_dim, hidden_dim=args.hidden_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(rl_config["learning_rate"]))
    updater = PPOUpdater(
        clip_ratio=float(rl_config.get("clip_ratio", 0.2)),
        target_kl=float(rl_config.get("target_kl", 0.01)),
        value_loss_coef=float(rl_config.get("value_loss_coef", 0.5)),
        entropy_coef=float(rl_config.get("entropy_coef", 0.01)),
        max_grad_norm=float(rl_config.get("max_grad_norm", 0.5)),
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.training_log.parent.mkdir(parents=True, exist_ok=True)
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainerLogger(
        log_dir=str(args.tensorboard_dir),
        checkpoint_dir=str(args.checkpoint.parent),
        best_checkpoint_name=args.checkpoint.name,
    )
    training_rows: list[dict[str, float]] = []
    best_score = -np.inf
    best_validation_metrics: dict[str, float] = {}
    recent_validation_scores: list[float] = []
    epochs_without_improvement = 0

    def checkpoint_metadata(epoch: int, val_metrics: dict[str, float]) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "reward_beta": reward_beta,
            "validation_metrics": {
                name: float(value) for name, value in val_metrics.items()
            },
            "observation_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_dim": args.hidden_dim,
            "lookback": lookback,
            "action_alpha": args.action_alpha,
            "tickers": tickers,
            "feature_names": full_tensor.feature_names,
            "scaler_path": str(args.scaler_output),
            "seed": args.seed,
        }

    print(
        f"[2/6] PPO MLP: obs={obs_dim} (lookback={lookback}), actions={action_dim}, "
        f"device={device}, minibatch={batch_size}."
    )
    print(
        f"[3/6] Tối đa {args.epochs} updates, {args.episodes_per_update} cửa sổ/update "
        f"({window_min}-{window_max} phiên); action_alpha={args.action_alpha:g}, "
        f"beta={reward_beta:g}."
    )
    try:
        # Keep an untrained validation reference as a checkpoint fallback.
        val_env = create_env(
            val_tensor,
            val_open,
            args.env_config,
            reward_beta,
            history_tensor=scaled_full_tensor,
            history_start_index=val_history_start,
        )
        baseline_result = evaluate_policy(
            model, val_env, lookback=lookback, action_alpha=args.action_alpha
        )
        baseline_metrics = metric_summary(
            baseline_result,
            risk_free_rate_annual=float(env_config["risk_free_rate"]),
        )
        baseline_score = (
            baseline_metrics["sharpe_ratio"]
            if args.selection_metric == "sharpe"
            else baseline_metrics["cumulative_return"]
        )
        logger.log_metrics(
            0,
            {
                "Reference/CumulativeReturn": baseline_metrics["cumulative_return"],
                "Reference/Profit": baseline_metrics["profit"],
                "Reference/Sharpe": baseline_metrics["sharpe_ratio"],
                "Reference/Turnover": baseline_metrics["total_turnover"],
            },
        )
        if np.isfinite(baseline_score):
            best_score = float(baseline_score)
            best_validation_metrics = baseline_metrics
            logger.save_checkpoint(
                0,
                model,
                optimizer,
                current_reward=best_score,
                selection_metric=f"validation_{args.selection_metric}",
                metadata=checkpoint_metadata(0, baseline_metrics),
            )
        print(
            f"  untrained validation reference | return={baseline_metrics['cumulative_return']:.2%} "
            f"| profit={baseline_metrics['profit']:+,.2f} "
            f"| Sharpe={baseline_metrics['sharpe_ratio']:.3f} "
            f"| turnover={baseline_metrics['total_turnover']:.2f}"
        )

        for epoch in range(1, args.epochs + 1):
            model.train()
            episode_specs: list[tuple[int, int, MarketDataTensor, np.ndarray]] = []
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
                    (start, length, episode_tensor, train_open[start:end].copy())
                )

            rollout_size = sum(length - 2 for _, length, _, _ in episode_specs)
            buffer = PPOBuffer(
                max_size=rollout_size,
                obs_dim=obs_dim,
                act_dim=action_dim,
                gamma=float(rl_config.get("gamma", 0.99)),
                lam=float(rl_config.get("gae_lambda", 0.95)),
            )
            episode_rewards: list[float] = []
            for episode_index, (start, length, episode_tensor, episode_open) in enumerate(
                episode_specs
            ):
                episode_env = create_env(
                    episode_tensor,
                    episode_open,
                    args.env_config,
                    reward_beta,
                    history_tensor=train_tensor,
                    history_start_index=start,
                )
                obs, _ = episode_env.reset(
                    seed=args.seed + epoch * args.episodes_per_update + episode_index
                )
                episode_reward = 0.0
                done = False
                while not done:
                    obs_vector = observation_vector(obs, episode_env, lookback)
                    state = torch.as_tensor(obs_vector, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        policy_action, log_prob, value = model.act(state)
                    policy_action_np = policy_action.cpu().numpy().astype(np.float32)
                    executed_action = blend_action(
                        policy_action_np, obs, args.action_alpha
                    )
                    next_obs, reward, terminated, truncated, _ = episode_env.step(
                        executed_action
                    )
                    # Store the sampled action and its log-probability; blending is a
                    # deterministic part of the action mapping into the environment.
                    buffer.store(
                        obs_vector,
                        policy_action_np,
                        float(reward),
                        float(value.item()),
                        float(log_prob.item()),
                    )
                    episode_reward += float(reward)
                    obs = next_obs
                    done = terminated or truncated

                # Random-window boundaries bootstrap the critic. The final train
                # boundary is terminal because no later training observations exist.
                if start + length >= len(train_tensor.dates):
                    last_value = 0.0
                else:
                    last_vector = observation_vector(obs, episode_env, lookback)
                    last_state = torch.as_tensor(
                        last_vector, dtype=torch.float32, device=device
                    )
                    with torch.no_grad():
                        _, _, last_value_tensor = model.act(
                            last_state, deterministic=True
                        )
                    last_value = float(last_value_tensor.item())
                buffer.finish_path(last_val=last_value)
                episode_rewards.append(episode_reward)

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
            )

            val_result = evaluate_policy(
                model, val_env, lookback=lookback, action_alpha=args.action_alpha
            )
            val_metrics = metric_summary(
                val_result,
                risk_free_rate_annual=float(env_config["risk_free_rate"]),
            )
            score = (
                val_metrics["sharpe_ratio"]
                if args.selection_metric == "sharpe"
                else val_metrics["cumulative_return"]
            )
            recent_validation_scores.append(float(score))
            recent_validation_scores = recent_validation_scores[-args.selection_window :]
            smoothed_score = float(np.mean(recent_validation_scores))
            training_row = {
                "epoch": float(epoch),
                "train_reward_mean": float(np.mean(episode_rewards)),
                "train_reward_std": float(np.std(episode_rewards)),
                "policy_loss": float(update_stats["LossPi"]),
                "value_loss": float(update_stats["LossV"]),
                "entropy": float(update_stats["Entropy"]),
                "kl": float(update_stats["KL"]),
                "clip_fraction": float(update_stats["ClipFrac"]),
                "ppo_epochs_completed": float(update_stats["StopIter"]),
                "ppo_minibatches_completed": float(update_stats["NumMinibatches"]),
                "ppo_stopped_early": float(update_stats["StoppedEarly"]),
                "validation_score": float(score),
                "validation_smoothed_score": smoothed_score,
                "validation_cumulative_return": val_metrics["cumulative_return"],
                "validation_profit": val_metrics["profit"],
                "validation_sharpe": val_metrics["sharpe_ratio"],
                "validation_max_drawdown": val_metrics["max_drawdown"],
                "validation_turnover": val_metrics["total_turnover"],
                "validation_portfolio_value": float(val_result.portfolio_values[-1]),
            }
            training_rows.append(training_row)
            logger.log_metrics(
                epoch,
                {
                    "Train/RewardMean": float(np.mean(episode_rewards)),
                    "Train/RewardStd": float(np.std(episode_rewards)),
                    "Train/PolicyLoss": float(update_stats["LossPi"]),
                    "Train/ValueLoss": float(update_stats["LossV"]),
                    "Train/Entropy": float(update_stats["Entropy"]),
                    "Train/KL": float(update_stats["KL"]),
                    "Train/ClipFraction": float(update_stats["ClipFrac"]),
                    "Train/PPOEpochsCompleted": float(update_stats["StopIter"]),
                    "Train/PPOMinibatchesCompleted": float(update_stats["NumMinibatches"]),
                    "Train/StoppedEarly": float(update_stats["StoppedEarly"]),
                    "Validation/CumulativeReturn": val_metrics["cumulative_return"],
                    "Validation/Profit": val_metrics["profit"],
                    "Validation/Sharpe": val_metrics["sharpe_ratio"],
                    "Validation/MaxDrawdown": val_metrics["max_drawdown"],
                    "Validation/Turnover": val_metrics["total_turnover"],
                    "Validation/PortfolioValue": val_result.portfolio_values[-1],
                    "Validation/SmoothedScore": smoothed_score,
                },
            )

            improved = (
                epoch >= args.min_epochs
                and np.isfinite(smoothed_score)
                and smoothed_score > best_score
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

            print(
                f"  epoch {epoch:03d}/{args.epochs} "
                f"| reward={np.mean(episode_rewards):.3f}±{np.std(episode_rewards):.3f} "
                f"| policy loss={update_stats['LossPi']:.4f} "
                f"| val return={val_metrics['cumulative_return']:.2%} "
                f"| val profit={val_metrics['profit']:+,.2f} "
                f"| val Sharpe={val_metrics['sharpe_ratio']:.3f} "
                f"| val turnover={val_metrics['total_turnover']:.2f} "
                f"| entropy={update_stats['Entropy']:.4f} "
                f"| KL={update_stats['KL']:.5f} "
                f"| clipfrac={update_stats['ClipFrac']:.3f} "
                f"| PPO={update_stats['NumMinibatches']} minibatches/"
                f"{update_stats['StopIter']} epochs "
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
        logger.close()

    if not best_validation_metrics:
        raise RuntimeError("Không tạo được checkpoint: validation score không hữu hạn.")
    pd.DataFrame(training_rows).to_csv(args.training_log, index=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(
        f"Best checkpoint: epoch {checkpoint['epoch']} | "
        f"validation {args.selection_metric}={checkpoint['selection_score']:.4f} | "
        f"val profit={best_validation_metrics['profit']:+,.2f}"
    )
    return model, {
        "full_tensor": full_tensor,
        "scaled_full_tensor": scaled_full_tensor,
        "split": split,
        "test_tensor": _test_tensor,
        "test_open": _test_open,
        "tickers": tickers,
        "reward_beta": reward_beta,
        "lookback": lookback,
        "test_history_start": test_history_start,
        "action_alpha": args.action_alpha,
        "env_config": env_config,
        "best_validation_metrics": best_validation_metrics,
    }


def evaluate_test(
    args: argparse.Namespace,
    model: SimplexActorCritic,
    context: dict[str, Any],
) -> pd.DataFrame:
    print("[4/6] Chạy checkpoint tốt nhất trên test out-of-sample...")
    test_env = create_env(
        context["test_tensor"],
        context["test_open"],
        args.env_config,
        context["reward_beta"],
        history_tensor=context["scaled_full_tensor"],
        history_start_index=context["test_history_start"],
    )
    backtester = DeterministicBacktester(test_env)
    n_assets = len(context["tickers"])
    env_config = context["env_config"]
    results = {
        "PPO": backtester.run_agent(
            lambda obs: action_for_model(
                model,
                obs,
                test_env,
                lookback=context["lookback"],
                action_alpha=context["action_alpha"],
                deterministic=True,
            )
        ),
        "Equal Weight (1/N)": backtester.run_strategy(
            EqualWeightBaseline(
                open_prices=context["test_open"],
                num_assets=n_assets,
                transaction_cost_rate=float(env_config["transaction_cost_rate"]),
                risk_free_rate_annual=float(env_config["risk_free_rate"]),
            )
        ),
        "Buy & Hold": backtester.run_strategy(
            BuyAndHoldBaseline(
                open_prices=context["test_open"],
                num_assets=n_assets,
                transaction_cost_rate=float(env_config["transaction_cost_rate"]),
                risk_free_rate_annual=float(env_config["risk_free_rate"]),
            )
        ),
        "Cash": backtester.run_strategy(
            CashBaseline(
                num_days=len(context["test_tensor"].dates) - 2,
                risk_free_rate_annual=float(env_config["risk_free_rate"]),
            )
        ),
    }
    metrics = backtester.compare(
        results,
        risk_free_rate_annual=float(env_config["risk_free_rate"]),
    )
    metrics["profit"] = pd.Series(
        {
            name: float(result.portfolio_values[-1] - result.portfolio_values[0])
            for name, result in results.items()
        }
    )
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics_output)
    print("[5/6] Metrics test (profit và costs theo đơn vị tiền tài khoản):")
    print(metrics.to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"[6/6] Đã lưu metrics: {args.metrics_output}")
    return metrics


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.seeds:
        model, context = train(args)
        evaluate_test(args, model, context)
        return

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) < 2:
        raise ValueError("--seeds cần ít nhất 2 seed, ví dụ: --seeds 7,17,27.")

    per_seed_metrics: dict[int, pd.DataFrame] = {}
    for seed in seeds:
        run_args = copy.copy(args)
        run_args.seed = seed
        run_args.checkpoint = (
            args.checkpoint.parent / f"seed_{seed}" / args.checkpoint.name
        )
        run_args.metrics_output = args.metrics_output.with_name(
            f"{args.metrics_output.stem}_seed{seed}{args.metrics_output.suffix}"
        )
        run_args.training_log = args.training_log.with_name(
            f"{args.training_log.stem}_seed{seed}{args.training_log.suffix}"
        )
        run_args.tensorboard_dir = args.tensorboard_dir / f"seed_{seed}"
        print(f"\n===== PPO run seed={seed} =====")
        model, context = train(run_args)
        per_seed_metrics[seed] = evaluate_test(run_args, model, context)

    stacked = pd.concat(per_seed_metrics, names=["seed", "strategy"])
    summary = stacked.groupby(level="strategy").agg(["mean", "std"])
    summary_path = args.metrics_output.with_name(
        f"{args.metrics_output.stem}_seeds_summary{args.metrics_output.suffix}"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path)
    print("\nTest mean ± std across seeds:")
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"Đã lưu tổng hợp nhiều seed: {summary_path}")


if __name__ == "__main__":
    main()

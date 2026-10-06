"""Train PPO on the temporal training split and compare on held-out test data.

Run from the repository root with:
    python scripts/run_sprint5_backtest.py --raw-dir data/raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import TemporalSplitter
from src.env.trading_env import TradingEnv
from src.evaluation.backtester import DeterministicBacktester
from src.evaluation.baselines import BuyAndHoldBaseline, CashBaseline, EqualWeightBaseline
from src.features.pipeline import FeaturePipeline
from src.features.scaler import MarketFeatureScaler
from src.models.actor_critic import SimplexActorCritic
from src.training.ppo_core import PPOBuffer, PPOUpdater


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument(
        "--processed-output", type=Path, default=PROJECT_ROOT / "data/processed/features.parquet"
    )
    parser.add_argument("--env-config", type=Path, default=PROJECT_ROOT / "configs/env.yaml")
    parser.add_argument("--train-end", default=TemporalSplitter.DEFAULT_TRAIN_END)
    parser.add_argument("--val-start", default=TemporalSplitter.DEFAULT_VAL_START)
    parser.add_argument("--val-end", default=TemporalSplitter.DEFAULT_VAL_END)
    parser.add_argument("--test-start", default=TemporalSplitter.DEFAULT_TEST_START)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--ppo-iters", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--checkpoint", type=Path, default=PROJECT_ROOT / "models/checkpoints/sprint5_best.pth"
    )
    parser.add_argument(
        "--metrics-output", type=Path, default=PROJECT_ROOT / "experiments/sprint5_metrics.csv"
    )
    return parser.parse_args()


def observation_vector(obs: dict) -> np.ndarray:
    """Flatten engineered/scaled [N,F] features and portfolio state for the MLP."""
    return np.concatenate(
        (
            np.asarray(obs["market_state"], dtype=np.float32).reshape(-1),
            np.asarray(obs["portfolio_weights"], dtype=np.float32).reshape(-1),
            np.asarray(obs["cash_ratio"], dtype=np.float32).reshape(-1),
        )
    )


def make_env(market_tensor, open_prices: np.ndarray, config_path: Path) -> TradingEnv:
    return TradingEnv(market_tensor, open_prices, config_path=str(config_path))


def action_for_model(model: SimplexActorCritic, obs: dict, deterministic: bool = False) -> np.ndarray:
    state = torch.as_tensor(
        observation_vector(obs), dtype=torch.float32, device=next(model.parameters()).device
    )
    with torch.no_grad():
        action, _, _ = model.act(state, deterministic=deterministic)
    return action.cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.ppo_iters <= 0:
        raise ValueError("--epochs và --ppo-iters phải lớn hơn 0.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA được chọn nhưng không khả dụng.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    if not args.raw_dir.exists() or not any(args.raw_dir.glob("*.parquet")):
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu OHLCV dạng parquet trong {args.raw_dir}. "
            "Hãy tải hoặc đặt dữ liệu vào thư mục này trước khi chạy Sprint 5."
        )

    print("[1/6] Tạo processed features và căn chỉnh giá Open từ dữ liệu OHLCV...")
    pipeline = FeaturePipeline()
    processed_data, full_tensor = pipeline.run_from_raw(
        raw_dir=args.raw_dir,
        export_parquet=True,
        output_file=args.processed_output,
    )

    tickers = list(full_tensor.tickers)
    open_columns = []
    for ticker in tickers:
        frame = processed_data[ticker]
        open_col = next((c for c in frame.columns if str(c).lower() == "open"), None)
        if open_col is None:
            raise ValueError(f"Thiếu cột Open cho mã {ticker}.")
        open_columns.append(frame[open_col].to_numpy(dtype=np.float64))
    full_open_prices = np.column_stack(open_columns)

    splitter = TemporalSplitter(
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
    )
    split = splitter.split_market_data_tensor(full_tensor)
    date_to_index = {str(date): i for i, date in enumerate(full_tensor.dates)}

    def opens_for(partition) -> np.ndarray:
        indices = [date_to_index[str(date)] for date in partition.dates]
        return full_open_prices[indices].copy()

    train_open = opens_for(split.train)
    val_open = opens_for(split.val)
    test_open = opens_for(split.test)

    scaler = MarketFeatureScaler()
    scaler.fit(split.train)
    train_tensor = scaler.transform(split.train)
    val_tensor = scaler.transform(split.val)
    test_tensor = scaler.transform(split.test)
    scaler.save(PROJECT_ROOT / "data/processed/scaler_params.json")
    print(split.summary())

    print("[2/6] Khởi tạo PPO MLP trên feature đã tính và chuẩn hóa (không dùng GNN)...")
    train_env = make_env(train_tensor, train_open, args.env_config)
    val_env = make_env(val_tensor, val_open, args.env_config)
    test_env = make_env(test_tensor, test_open, args.env_config)
    obs_dim = train_tensor.tensor.shape[1] * train_tensor.tensor.shape[2] + train_tensor.tensor.shape[1] + 1
    action_dim = train_tensor.tensor.shape[1] + 1
    model = SimplexActorCritic(obs_dim, action_dim, hidden_dim=args.hidden_dim).to(device)

    with open(PROJECT_ROOT / "configs/model.yaml", "r", encoding="utf-8") as stream:
        model_config = yaml.safe_load(stream)
    rl_config = model_config["rl_agent"]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(rl_config["learning_rate"]))
    updater = PPOUpdater(
        clip_ratio=float(rl_config.get("clip_ratio", 0.2)),
        entropy_coef=float(rl_config.get("entropy_coef", 0.01)),
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    best_validation_value = -np.inf
    train_log: list[dict[str, float]] = []
    rollout_size = len(train_tensor.dates) - 2
    if rollout_size <= 0:
        raise ValueError("Tập train cần tối thiểu 3 ngày để tạo rollout PPO.")

    print(f"[3/6] Huấn luyện PPO trong {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        obs, _ = train_env.reset(seed=args.seed + epoch)
        buffer = PPOBuffer(
            max_size=rollout_size,
            obs_dim=obs_dim,
            act_dim=action_dim,
            gamma=float(rl_config.get("gamma", 0.99)),
        )
        episode_reward = 0.0
        done = False

        while not done:
            state_np = observation_vector(obs)
            state = torch.as_tensor(state_np, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, log_prob, value = model.act(state)
            action_np = action.cpu().numpy().astype(np.float32)
            next_obs, reward, terminated, truncated, _ = train_env.step(action_np)
            buffer.store(
                state_np,
                action_np,
                float(reward),
                float(value.item()),
                float(log_prob.item()),
            )
            episode_reward += float(reward)
            obs = next_obs
            done = terminated or truncated

        buffer.finish_path(last_val=0.0)
        batch = {key: value.to(device) for key, value in buffer.get().items()}
        update_stats = updater.update(model, optimizer, batch, train_iters=args.ppo_iters)

        model.eval()
        val_backtester = DeterministicBacktester(val_env)
        val_result = val_backtester.run_agent(
            lambda val_obs: action_for_model(model, val_obs, deterministic=True)
        )
        validation_metrics = val_backtester.compare({"PPO": val_result}).iloc[0]
        validation_value = float(val_result.portfolio_values[-1])
        train_log.append(
            {
                "epoch": epoch,
                "train_reward": episode_reward,
                "policy_loss": float(update_stats["LossPi"]),
                "value_loss": float(update_stats["LossV"]),
                "validation_value": validation_value,
                "validation_sharpe": float(validation_metrics["sharpe_ratio"]),
            }
        )
        print(
            f"  epoch {epoch:03d}/{args.epochs} | reward={episode_reward:.3f} "
            f"| val=${validation_value:,.2f} | val Sharpe={validation_metrics['sharpe_ratio']:.3f}"
        )

        if validation_value > best_validation_value:
            best_validation_value = validation_value
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_value": validation_value,
                    "observation_dim": obs_dim,
                    "action_dim": action_dim,
                    "hidden_dim": args.hidden_dim,
                    "tickers": tickers,
                    "feature_names": full_tensor.feature_names,
                    "seed": args.seed,
                },
                args.checkpoint,
            )
        model.train()

    pd.DataFrame(train_log).to_csv(args.metrics_output.with_name("sprint5_training_log.csv"), index=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("[4/6] Nạp checkpoint tốt nhất và chạy test out-of-sample...")
    env_cfg = yaml.safe_load(args.env_config.read_text(encoding="utf-8"))["environment"]
    n_assets = len(tickers)
    cash_strategy = CashBaseline(
        num_days=len(test_tensor.dates) - 2,
        risk_free_rate_annual=float(env_cfg["risk_free_rate"]),
    )
    equal_weight_strategy = EqualWeightBaseline(
        open_prices=test_open,
        num_assets=n_assets,
        transaction_cost_rate=float(env_cfg["transaction_cost_rate"]),
        risk_free_rate_annual=float(env_cfg["risk_free_rate"]),
    )
    buy_and_hold_strategy = BuyAndHoldBaseline(
        open_prices=test_open,
        num_assets=n_assets,
        transaction_cost_rate=float(env_cfg["transaction_cost_rate"]),
        risk_free_rate_annual=float(env_cfg["risk_free_rate"]),
    )
    backtester = DeterministicBacktester(test_env)
    results = {
        "PPO": backtester.run_agent(lambda test_obs: action_for_model(model, test_obs, deterministic=True)),
        "Equal Weight (1/N)": backtester.run_strategy(equal_weight_strategy),
        "Buy & Hold": backtester.run_strategy(buy_and_hold_strategy),
        "Cash": backtester.run_strategy(cash_strategy),
    }

    print("[5/6] Tính metrics trên cùng TradingEnv và test prices...")
    metrics = backtester.compare(results)
    metrics.to_csv(args.metrics_output)
    print(metrics.to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"[6/6] Đã lưu metrics: {args.metrics_output}")
    print(f"Best validation checkpoint: {args.checkpoint} (epoch {checkpoint['epoch']})")


if __name__ == "__main__":
    main()

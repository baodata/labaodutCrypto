"""End-to-end single-agent PPO training and historical-period evaluation.

Run from the repository root with::

    python -m src.training.train_ppo --reward-beta 2.0

The policy receives engineered, train-normalized features. Raw OHLCV prices are
used to build those features and to simulate order execution in TradingEnv.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
import warnings
import shutil
import random
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import TemporalSplitter
from src.env.cost import resolve_transaction_cost_components
from src.env.trading_env import TradingEnv
from src.evaluation.backtester import BacktestResult, DeterministicBacktester
from src.evaluation.baselines import BuyAndHoldBaseline, CashBaseline, EqualWeightBaseline
from src.evaluation.metrics import FinancialMetrics, block_bootstrap_excess_returns
from src.evaluation.seed_summary import summarize_seed_metrics
from src.features.pipeline import FeaturePipeline
from src.features.scaler import MarketFeatureScaler
from src.graph.dynamic_graph import build_dynamic_graph_sequence
from src.graph.multi_relation_graph import GRAPH_RELATIONS
from src.models.actor_critic import build_actor_critic
from src.models.policy_adapter import observation_vector, graph_tensors_from_obs, history_mask_from_obs
from src.models.market_cache import prepare_market_cache
from src.training.logger import TrainerLogger
from src.training.ppo_core import PPOBuffer, PPOUpdater
from src.utils.data_types import MarketDataTensor
from src.utils.artifacts import atomic_csv, atomic_json, file_hashes, new_run_directory, rng_state, restore_rng_state
from src.utils.progress import PhaseProgress


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
    parser.add_argument("--microbatch-size", type=int, default=None, help="Số mẫu forward/backward mỗi lần; GNN mặc định 16 CPU / 32 CUDA, tích lũy gradient đủ minibatch.")
    parser.add_argument("--cpu-threads", type=int, default=4, help="Số thread PyTorch trên CPU; tránh dùng quá nhiều thread cho graph nhỏ.")
    parser.add_argument("--cache-batch-size", type=int, default=64, help="Số ngày mã hóa GNN mỗi lượt khi dựng cache.")
    parser.add_argument("--cache-temporal-batch-size", type=int, default=32, help="Số cửa sổ GRU mỗi lượt khi dựng cache.")
    parser.add_argument("--progress-every", type=float, default=10.0, help="Khoảng giây giữa các dòng tiến độ trong một giai đoạn train.")
    parser.add_argument("--episodes-per-update", type=int, default=32)
    parser.add_argument("--run-dir", type=Path, default=None, help="Thư mục artifact riêng cho run; mặc định tạo thư mục mới.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume từ last.pth với cùng cấu hình train.")
    parser.add_argument("--train-eval-every", type=int, default=5)
    parser.add_argument("--bootstrap-block-size", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--matched-cash-ratio", type=float, default=None, help="Cash cố định cho matched baseline; mặc định 1/(N+1).")
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Ghi đè gamma trong configs/model.yaml.",
    )
    parser.add_argument(
        "--reward-scaling",
        type=float,
        default=None,
        help="Ghi đè reward_scaling trong configs/env.yaml.",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="Ghi đè target_kl trong configs/model.yaml.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Ghi đè learning_rate trong configs/model.yaml.",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=None,
        help="Ghi đè entropy_coef trong configs/model.yaml.",
    )
    parser.add_argument(
        "--value-clip-range",
        type=float,
        default=None,
        help="Ngưỡng riêng cho value clipping; mặc định tắt nếu YAML để null.",
    )
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
    parser.add_argument(
        "--encoder",
        choices=("mlp", "lstm", "gru", "gnn"),
        default="gru",
        help="Loại encoder: gru mặc định; cũng hỗ trợ lstm, mlp và gnn động.",
    )
    parser.add_argument(
        "--gnn-architecture",
        choices=("gcn", "gat"),
        default=None,
        help="Kiến trúc GNN khi dùng --encoder gnn; mặc định đọc configs/model.yaml.",
    )
    parser.add_argument("--gnn-hidden-channels", type=int, default=None)
    parser.add_argument("--gnn-out-channels", type=int, default=None)
    parser.add_argument("--gnn-heads", type=int, default=None)
    parser.add_argument(
        "--rnn-layers",
        type=int,
        default=1,
        help="Số lớp RNN cho LSTM/GRU và GRU thời gian trong GNN.",
    )
    parser.add_argument(
        "--rnn-dropout",
        type=float,
        default=0.0,
        help="Dropout giữa các lớp RNN; GNN PPO yêu cầu 0 để giữ log-prob nhất quán.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Tùy chọn chạy nhiều seed, ví dụ: --seeds 7,17,27; xuất thêm mean/std test.",
    )
    parser.add_argument(
        "--untrained-null-seeds",
        default=None,
        help=(
            "Chỉ đánh giá policy chưa train trên validation với các seed phân tách bằng dấu phẩy; "
            "ví dụ: --untrained-null-seeds 7,17,27. Không chạy PPO hoặc đánh giá test."
        ),
    )
    parser.add_argument(
        "--null-output",
        type=Path,
        default=PROJECT_ROOT / "experiments/untrained_null_validation.csv",
        help="CSV kết quả policy khởi tạo ngẫu nhiên và baseline validation.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--reward-beta",
        type=float,
        default=None,
        help="Penalty multiplier for transaction costs; defaults to configs/env.yaml.",
    )
    parser.add_argument(
        "--reward-baseline",
        choices=("none", "equal_weight"),
        default="equal_weight",
        help="Equal-weight excess-return reward baseline (mặc định bật).",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("sharpe", "cumulative_return", "excess_sharpe", "excess_return"),
        default="sharpe",
        help="Validation metric used to select the best checkpoint.",
    )
    parser.add_argument(
        "--selection-margin",
        type=float,
        default=0.0,
        help="Biên tối thiểu yêu cầu để checkpoint sau vượt checkpoint trước.",
    )
    parser.add_argument(
        "--dirichlet-kappa",
        type=float,
        default=None,
        help="Tổng concentration kappa để tách mean khỏi variance của Dirichlet (ví dụ 200.0).",
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


def action_for_model(
    model: nn.Module,
    obs: dict[str, np.ndarray],
    deterministic: bool = False,
    market_embedding: torch.Tensor | None = None,
) -> np.ndarray:
    """Evaluate model action from observation dict."""
    device = next(model.parameters()).device
    with torch.no_grad():
        state = torch.as_tensor(observation_vector(obs), dtype=torch.float32, device=device)
        history_mask = history_mask_from_obs(obs, device)
        graph = graph_tensors_from_obs(obs, device)
        if market_embedding is not None:
            graph = {"market_embedding": market_embedding}
        if graph is None:
            action, _, _ = model.act(state, deterministic=deterministic, history_mask=history_mask)
        else:
            action, _, _ = model.act(state, deterministic=deterministic, graph=graph, history_mask=history_mask)
    return action.detach().cpu().numpy().astype(np.float32)



def _total_return_adjustment_factor(
    frame: pd.DataFrame,
    dates: pd.DatetimeIndex,
    ticker: str,
) -> np.ndarray:
    """Return Yahoo's Adj Close / Close factor aligned to the requested dates.

    Applying the same daily factor to raw Open and Close creates a consistent
    total-return price index. Ratios across dates then include cash distributions
    (and split adjustments) while within-day returns remain price returns.
    """
    close_col = next(
        (column for column in frame.columns if str(column).lower() == "close"),
        None,
    )
    adj_close_col = next(
        (
            column
            for column in frame.columns
            if str(column).lower().replace(" ", "_") in {"adj_close", "adjusted_close"}
        ),
        None,
    )
    if close_col is None:
        raise ValueError(f"Thiếu cột Close để tính hệ số total return: {ticker}")
    if adj_close_col is None:
        # Retain compatibility with price-only data sources, but make the loss
        # of dividend information explicit instead of silently implying total return.
        warnings.warn(
            f"{ticker} không có Adj Close; lợi nhuận sẽ chỉ dựa trên biến động giá, "
            "không bao gồm cổ tức.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.ones(len(dates), dtype=np.float64)

    aligned = frame.reindex(dates)
    close = aligned[close_col].to_numpy(dtype=np.float64)
    adj_close = aligned[adj_close_col].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(close).all()
        or not np.isfinite(adj_close).all()
        or np.any(close <= 0)
        or np.any(adj_close <= 0)
    ):
        raise ValueError(f"Close/Adj Close của {ticker} không hợp lệ hoặc thiếu ngày.")
    factor = adj_close / close
    if not np.isfinite(factor).all() or np.any(factor <= 0):
        raise ValueError(f"Hệ số Adj Close / Close của {ticker} không hợp lệ.")
    return factor


def open_price_matrix(processed_data: dict[str, pd.DataFrame], tickers: list[str]) -> np.ndarray:
    """Build Open marks on a total-return scale for portfolio return simulation."""
    columns: list[np.ndarray] = []
    expected_index = processed_data[tickers[0]].index
    for ticker in tickers:
        frame = processed_data[ticker]
        if not frame.index.equals(expected_index):
            raise ValueError(f"Lịch giao dịch của {ticker} không khớp với các mã còn lại.")
        open_col = next(
            (column for column in frame.columns if str(column).lower() == "open"),
            None,
        )
        if open_col is None:
            raise ValueError(f"Thiếu cột Open cần cho mô phỏng giao dịch: {ticker}")
        raw_open = frame[open_col].to_numpy(dtype=np.float64)
        dates = pd.DatetimeIndex(expected_index)
        factor = _total_return_adjustment_factor(frame, dates, ticker)
        columns.append(raw_open * factor)
    prices = np.column_stack(columns)
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("Giá Open phải hữu hạn và lớn hơn 0.")
    return prices


def close_price_matrix(
    processed_data: dict[str, pd.DataFrame],
    tickers: list[str],
    dates: list,
) -> np.ndarray:
    """Build Close marks on the same total-return scale as Open marks."""
    expected_index = pd.DatetimeIndex(pd.to_datetime(dates))
    columns: list[np.ndarray] = []
    for ticker in tickers:
        frame = processed_data[ticker]
        close_col = next(
            (column for column in frame.columns if str(column).lower() == "close"),
            None,
        )
        if close_col is None:
            raise ValueError(f"Thiếu cột Close để dựng correlation graph: {ticker}")
        aligned = frame.reindex(expected_index)[close_col].to_numpy(dtype=np.float64)
        if not np.isfinite(aligned).all() or np.any(aligned <= 0):
            raise ValueError(f"Giá Close của {ticker} không căn đủ theo lịch feature.")
        factor = _total_return_adjustment_factor(frame, expected_index, ticker)
        columns.append(aligned * factor)
    return np.column_stack(columns)


def create_env(
    tensor,
    open_prices: np.ndarray,
    env_config: Path,
    reward_beta: float,
    reward_scaling: float | None = None,
    history_tensor: MarketDataTensor | None = None,
    history_start_index: int = 0,
    reward_baseline: str = "none",
    lookback_window: int = 1,
    action_alpha: float = 1.0,
    graph_adjacency_sequence: np.ndarray | None = None,
    graph_edge_mask_sequence: np.ndarray | None = None,
    graph_start_index: int = 0,
    close_prices: np.ndarray | None = None,
) -> TradingEnv:
    return TradingEnv(
        market_tensor=tensor,
        open_prices=open_prices,
        close_prices=close_prices,
        config_path=str(env_config),
        reward_beta=reward_beta,
        reward_scaling=reward_scaling,
        reward_baseline=reward_baseline,
        history_tensor=history_tensor,
        history_start_index=history_start_index,
        lookback_window=lookback_window,
        action_alpha=action_alpha,
        graph_adjacency_sequence=graph_adjacency_sequence,
        graph_edge_mask_sequence=graph_edge_mask_sequence,
        graph_start_index=graph_start_index,
    )


def evaluate_policy(
    model: nn.Module,
    env: TradingEnv,
) -> BacktestResult:
    result, _, _ = evaluate_policy_with_actions(model, env)
    return result


def evaluate_policy_with_actions(
    model: nn.Module,
    env: TradingEnv,
) -> tuple[BacktestResult, np.ndarray, np.ndarray]:
    """Backtest deterministically and collect policy and blended target weights."""
    model.eval()
    market_cache = prepare_market_cache(model, env)
    policy_actions: list[np.ndarray] = []
    blended_targets: list[np.ndarray] = []

    def act(obs: dict[str, np.ndarray]) -> np.ndarray:
        embedding = market_cache.get(model, env.history_start_index + env.current_step) if market_cache else None
        action = action_for_model(model, obs, deterministic=True, market_embedding=embedding)
        policy_target, execution_target = env.project_action(action)
        policy_actions.append(policy_target.copy())
        blended_targets.append(execution_target.copy())
        return action

    result = DeterministicBacktester(env).run_agent(act)
    return result, np.asarray(policy_actions), np.asarray(blended_targets)


def policy_weight_summary(
    actions: np.ndarray,
    n_assets: int,
    asset_names: list[str] | None = None,
) -> dict[str, float]:
    """Summarize per-step allocation tilt and temporal weight variability."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != n_assets + 1 or len(actions) == 0:
        raise ValueError("actions phải có shape [steps, n_assets + 1] và không rỗng.")
    if asset_names is not None and len(asset_names) != n_assets:
        raise ValueError("asset_names phải có một tên cho mỗi tài sản.")
    asset_weights = actions[:, :n_assets]
    stock_sum = asset_weights.sum(axis=1, keepdims=True)
    normalized_stock_weights = np.where(
        stock_sum > 1e-12, asset_weights / stock_sum, 1.0 / n_assets
    )
    per_step_deviation = normalized_stock_weights - (1.0 / n_assets)
    summary = {
        "mean_stock_allocation": float(actions[:, :n_assets].sum(axis=1).mean()),
        "mean_cash_weight": float(actions[:, n_assets].mean()),
        "mean_asset_weight": float(asset_weights.mean()),
        "cross_asset_weight_std": float(asset_weights.std(axis=1).mean()),
        "mean_abs_deviation_1n": float(np.abs(per_step_deviation).mean()),
        "max_abs_deviation_1n": float(np.abs(per_step_deviation).max()),
        "mean_temporal_weight_std": float(actions.std(axis=0).mean()),
    }
    temporal_std = actions.std(axis=0)
    labels = asset_names or [f"asset_{index}" for index in range(n_assets)]
    for asset_name, value in zip(labels, temporal_std[:n_assets]):
        summary[f"asset_{asset_name}_weight_std_time"] = float(value)
    summary["cash_weight_std_time"] = float(temporal_std[n_assets])
    return summary


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
    summary["final_value"] = float(result.portfolio_values[-1])
    return summary


def evaluate_equal_weight(
    env: TradingEnv,
    open_prices: np.ndarray,
    num_assets: int,
    transaction_cost_rate: float,
    risk_free_rate_annual: float,
    cash_ratio: float = 0.0,
) -> tuple[dict[str, float], BacktestResult]:
    """Evaluate 1/N (hoặc 1/N + cash) reference qua cùng bộ máy TradingEnv."""
    result = DeterministicBacktester(env).run_strategy(
        EqualWeightBaseline(
            open_prices=open_prices,
            num_assets=num_assets,
            transaction_cost_rate=transaction_cost_rate,
            risk_free_rate_annual=risk_free_rate_annual,
            cash_ratio=cash_ratio,
        )
    )
    return metric_summary(result, risk_free_rate_annual), result


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
        export_parquet=args.resume is None,
        output_file=args.processed_output,
    )


    split = TemporalSplitter(
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
    ).split_market_data_tensor(full_tensor)
    with args.env_config.open("r", encoding="utf-8") as stream:
        configured_lookback = yaml.safe_load(stream)["environment"].get("lookback_window", 20)
    lookback = int(configured_lookback if args.lookback is None else args.lookback)
    if lookback <= 0:
        raise ValueError("--lookback phải lớn hơn 0.")
    if full_tensor.warmup_sessions:
        offset = lookback - 1
        if len(split.train.dates) <= offset + 2:
            raise ValueError("Train không đủ dữ liệu sau feature warm-up và lookback.")
        split.train = MarketDataTensor(
            tensor=split.train.tensor[offset:].copy(), tickers=list(split.train.tickers),
            feature_names=list(split.train.feature_names), dates=list(split.train.dates[offset:]),
        )
        split.dates_info["train_range"] = f"{split.train.dates[0]} -> {split.train.dates[-1]} (T={len(split.train.dates)}; warm-up excluded)"

    scaler = MarketFeatureScaler()
    scaler.fit(split.train)
    if args.resume is not None:
        previous = args._resume_payload
        if previous.get("scaler_params") != scaler.to_dict():
            raise ValueError("Scaler của resume không khớp dữ liệu/split train.")
        if previous.get("split_dates") != split.dates_info:
            raise ValueError("Resume yêu cầu cùng split_dates.")
    train_tensor = scaler.transform(split.train)
    val_tensor = scaler.transform(split.val)
    test_tensor = scaler.transform(split.test)
    scaled_full_tensor = scaler.transform(full_tensor)
    args.scaler_output.parent.mkdir(parents=True, exist_ok=True)
    scaler.save(args.scaler_output)
    args._scaler_params = scaler.to_dict()
    args._feature_parameters = pipeline.parameters
    if hasattr(args, "_run_dir"):
        scaler.save(args._run_dir / "scaler.json")

    tickers = list(full_tensor.tickers)
    all_open_prices = open_price_matrix(processed_data, tickers)
    all_close_prices = close_price_matrix(processed_data, tickers, full_tensor.dates)
    date_to_index = {str(date): index for index, date in enumerate(full_tensor.dates)}

    def prices_for(partition, price_matrix: np.ndarray) -> np.ndarray:
        indices = [date_to_index[str(date)] for date in partition.dates]
        return price_matrix[indices].copy()

    print(split.summary())
    print(f"Features: {', '.join(full_tensor.feature_names)}")
    print(f"Transaction-cost reward beta: {reward_beta:g}")
    print(
        "Research caveat: the configured asset list is fixed retrospectively; "
        "no point-in-time membership or delisted-asset history is supplied."
    )
    print("Portfolio return basis: Adj Close adjusted Open/Close (total return, dividends reinvested).")
    return (
        full_tensor,
        processed_data,
        scaled_full_tensor,
        split,
        train_tensor,
        val_tensor,
        test_tensor,
        prices_for(split.train, all_open_prices),
        prices_for(split.val, all_open_prices),
        prices_for(split.test, all_open_prices),
        prices_for(split.train, all_close_prices),
        prices_for(split.val, all_close_prices),
        prices_for(split.test, all_close_prices),
        tickers,
    )


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
def runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "torch", "torch-geometric", "gymnasium"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def configure_run_artifacts(args, kind="ppo"):
    if hasattr(args, "_run_dir"):
        return
    if min(args.cpu_threads, args.cache_batch_size, args.cache_temporal_batch_size) <= 0 or (args.microbatch_size is not None and args.microbatch_size <= 0):
        raise ValueError("Thread count và batch sizes phải lớn hơn 0.")
    if not np.isfinite(args.progress_every) or args.progress_every <= 0:
        raise ValueError("--progress-every phải hữu hạn và lớn hơn 0.")
    if torch.device(args.device).type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    if args.resume is not None and args.run_dir is None:
        run_dir = args.resume.resolve().parent.parent
    elif args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        if run_dir.exists() and any(run_dir.iterdir()) and args.resume is None:
            raise ValueError("run-dir đã có artifact; chọn thư mục mới hoặc dùng --resume.")
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = new_run_directory(PROJECT_ROOT / "models/runs", f"{kind}_{args.encoder}_seed{args.seed}")
    hashes = {
        "source_hashes": file_hashes(list((PROJECT_ROOT / "src").rglob("*.py")) + list((PROJECT_ROOT / "scripts").glob("*.py")), PROJECT_ROOT),
        "data_hashes": file_hashes(args.raw_dir.glob("*.parquet"), PROJECT_ROOT),
        "config_hashes": file_hashes([args.model_config, args.env_config, PROJECT_ROOT / "configs/assets.yaml"], PROJECT_ROOT),
    }
    if args.resume is not None:
        previous = torch.load(args.resume, map_location="cpu", weights_only=False)
        for key, current in hashes.items():
            if previous.get("run_manifest", {}).get(key) != current:
                raise ValueError(f"Resume yêu cầu cùng {key}; artifact cũ được giữ nguyên.")
        if previous.get("run_manifest", {}).get("runtime") != runtime_versions():
            raise ValueError("Resume yêu cầu cùng phiên bản Python/thư viện để giữ quỹ đạo train.")
        args._resume_payload = previous
    args._run_dir = run_dir
    defaults = build_parser().parse_args([])
    paths = {
        "checkpoint": run_dir / "checkpoints/best.pth",
        "training_log": run_dir / "training_log.csv", "metrics_output": run_dir / "metrics.csv",
        "null_output": run_dir / "untrained_null_validation.csv",
        "scaler_output": run_dir / "scaler.json", "processed_output": run_dir / "features.parquet",
        "tensorboard_dir": run_dir / "tensorboard",
    }
    for name, path in paths.items():
        if getattr(args, name) == getattr(defaults, name):
            setattr(args, name, path)
    args._manifest = {
        "status": "running", "kind": kind, "run_dir": str(run_dir),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith("_")},
        "runtime": runtime_versions(),
        **hashes,
    }
    atomic_json(run_dir / "manifest.json", args._manifest)


def finish_run_manifest(args, status, **fields):
    args._manifest.update(status=status, **fields)
    atomic_json(args._run_dir / "manifest.json", args._manifest)


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
    test_history_start = full_date_index[str(split.test.dates[0])]
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
    if batch_size is None:
        batch_size = int(rl_config.get("batch_size", 256))
    if batch_size <= 0:
        raise ValueError("--batch-size phải lớn hơn 0.")

    n_assets, n_features = train_tensor.tensor.shape[1:]
    matched_cash = 1.0 / (n_assets + 1) if args.matched_cash_ratio is None else float(args.matched_cash_ratio)
    if not np.isfinite(matched_cash) or not 0 <= matched_cash < 1:
        raise ValueError("matched-cash-ratio phải thuộc [0,1).")
    obs_dim = lookback * n_assets * n_features + n_assets + 1
    action_dim = n_assets + 1
    encoder = getattr(args, "encoder", "gru")
    if encoder not in {"mlp", "lstm", "gru", "gnn"}:
        raise ValueError("--encoder phải là 'mlp', 'lstm', 'gru' hoặc 'gnn'.")
    rnn_layers = int(getattr(args, "rnn_layers", 1))
    rnn_dropout = float(getattr(args, "rnn_dropout", 0.0))
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
            sector_config_path=str(PROJECT_ROOT / "configs/assets.yaml"),
        )
        graph_metadata = {
            "source": "close_to_close_rolling_correlation_plus_sector_edges",
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

    def checkpoint_metadata(epoch: int, val_metrics: dict[str, float]) -> dict[str, Any]:
        reused_historical_test = args.test_start == TemporalSplitter.DEFAULT_TEST_START
        return {
            "epoch": epoch,
            "checkpoint_status": "untrained_fallback" if epoch == 0 else "trained",
            "reward_beta": reward_beta,
            "reward_scaling": reward_scaling,
            "reward_baseline": args.reward_baseline,
            "return_convention": "adjusted_close_scaled_open_close_total_return",
            "sharpe_convention": FinancialMetrics.SHARPE_CONVENTION,
            "universe_methodology": "fixed_current_asset_list_without_point_in_time_membership",
            "test_period_status": (
                "reused_exploratory_period_not_independent_holdout"
                if reused_historical_test
                else "custom_period_freshness_unverified"
            ),
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
            "action_alpha": args.action_alpha,
            "tickers": tickers,
            "feature_names": full_tensor.feature_names,
            "scaler_path": str(args.scaler_output),
            "seed": args.seed,
            "scaler_params": args._scaler_params,
            "feature_parameters": args._feature_parameters,
            "resolved_model_config": configured_model_config,
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
            "training_parameters": {
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
            else f"[{lookback}, {n_assets}, {n_features}]→h_{args.hidden_dim}"
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
            training_rows.append(training_row)
            epoch_metrics = {
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
                f"| train return={train_eval_metrics['cumulative_return']:.2%} "
                f"| train profit={train_eval_metrics['profit']:+,.2f} "
                f"| train total={train_eval_metrics['final_value']:,.2f} "
                f"| train Sharpe={train_eval_metrics['sharpe_ratio']:.3f} "
                f"(Δ1/N {train_sharpe_delta_1n:+.3f}, Δ1/N+cash {train_sharpe_delta_1n_cash:+.3f}) "
                f"| val return={val_metrics['cumulative_return']:.2%} "
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
            "Checkpoint tốt nhất là epoch 0 (policy chưa train). Kết quả test sẽ đo policy khởi tạo, "
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
        "best_validation_metrics": best_validation_metrics,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_status": checkpoint_status,
        "matched_cash_ratio": matched_cash,
        "run_dir": args._run_dir,
    }


def evaluate_test(
    args: argparse.Namespace,
    model: nn.Module,
    context: dict[str, Any],
) -> pd.DataFrame:
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
    if args.test_start == TemporalSplitter.DEFAULT_TEST_START:
        warnings.warn(
            "Giai đoạn test mặc định 2024–2025 đã được xem qua trong nhiều vòng phân tích; "
            "hãy coi đây là kết quả exploratory, không phải holdout độc lập.",
            RuntimeWarning,
            stacklevel=2,
        )
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
    results = {
        "PPO": ppo_result,
        matched_name: matched_baseline_result,
        "Equal Weight (1/N)": baseline_backtester.run_strategy(
            EqualWeightBaseline(
                open_prices=context["test_open"],
                num_assets=n_assets,
                transaction_cost_rate=float(env_config["transaction_cost_rate"]),
                risk_free_rate_annual=float(env_config["risk_free_rate"]),
                cash_ratio=0.0,
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
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    atomic_csv(metrics, args.metrics_output)
    if "run_dir" in context:
        atomic_csv(metrics, context["run_dir"] / "metrics.csv")
        daily = pd.DataFrame({name: result.daily_net_returns for name, result in results.items()}, index=context["test_tensor"].dates[1:])
        daily.index.name = "date"
        atomic_csv(daily, context["run_dir"] / "daily_returns.csv")
        finish_run_manifest(args, "completed", best_epoch=context["checkpoint_epoch"])
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


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.untrained_null_seeds:
        if args.seeds:
            raise ValueError("Dùng riêng --untrained-null-seeds hoặc --seeds.")
        null_seeds = [
            int(value.strip())
            for value in args.untrained_null_seeds.split(",")
            if value.strip()
        ]
        evaluate_untrained_null(args, null_seeds)
        return
    if not args.seeds:
        model, context = train(args)
        evaluate_test(args, model, context)
        return

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) < 2:
        raise ValueError("--seeds cần ít nhất 2 seed, ví dụ: --seeds 7,17,27.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds không được chứa seed trùng lặp.")
    if args.resume is not None:
        raise ValueError("Resume từng seed riêng bằng --seed và --resume, không dùng --seeds.")
    if args.run_dir is not None and args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise ValueError("Thư mục ensemble đã có artifact; chọn --run-dir mới.")
    ensemble_dir = args.run_dir.resolve() if args.run_dir is not None else new_run_directory(PROJECT_ROOT / "models/runs", "ensemble")
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    defaults = build_parser().parse_args([])
    if args.metrics_output == defaults.metrics_output:
        args.metrics_output = ensemble_dir / "metrics.csv"
    atomic_json(ensemble_dir / "ensemble_manifest.json", {"seeds": seeds, "status": "running"})

    per_seed_metrics: dict[int, pd.DataFrame] = {}
    for seed in seeds:
        run_args = copy.copy(args)
        run_args.seed = seed
        run_args.run_dir = ensemble_dir / f"seed_{seed}"
        if args.checkpoint != defaults.checkpoint:
            run_args.checkpoint = args.checkpoint.parent / f"seed_{seed}" / args.checkpoint.name
        run_args.metrics_output = args.metrics_output.with_name(
            f"{args.metrics_output.stem}_seed{seed}{args.metrics_output.suffix}"
        )
        if args.training_log != defaults.training_log:
            run_args.training_log = args.training_log.with_name(f"{args.training_log.stem}_seed{seed}{args.training_log.suffix}")
        if args.tensorboard_dir != defaults.tensorboard_dir:
            run_args.tensorboard_dir = args.tensorboard_dir / f"seed_{seed}"
        print(f"\n===== PPO run seed={seed} =====")
        model, context = train(run_args)
        per_seed_metrics[seed] = evaluate_test(run_args, model, context)

    summary, seed_metadata = summarize_seed_metrics(per_seed_metrics)
    summary_path = args.metrics_output.with_name(
        f"{args.metrics_output.stem}_seeds_summary{args.metrics_output.suffix}"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path)
    metadata_path = args.metrics_output.with_name(
        f"{args.metrics_output.stem}_seeds_metadata{args.metrics_output.suffix}"
    )
    seed_metadata.to_csv(metadata_path, index=False)
    atomic_json(ensemble_dir / "ensemble_manifest.json", {"seeds": seeds, "status": "completed", "summary_path": str(summary_path), "metadata_path": str(metadata_path)})
    print("\nTest mean ± std across seeds:")
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"Đã lưu tổng hợp nhiều seed: {summary_path}")
    print(f"Đã lưu metadata từng seed: {metadata_path}")


if __name__ == "__main__":
    main()

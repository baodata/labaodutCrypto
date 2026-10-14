"""Measure feature rank-IC and a train-fitted linear portfolio on validation only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import TemporalSplitter
from src.env.cost import resolve_transaction_cost_components
from src.evaluation.backtester import DeterministicBacktester
from src.evaluation.metrics import FinancialMetrics, block_bootstrap_excess_returns
from src.evaluation.baselines import EqualWeightBaseline
from src.features.pipeline import FeaturePipeline
from src.features.scaler import MarketFeatureScaler
from src.utils.artifacts import atomic_json, file_hashes
from src.training.train_ppo import (
    close_price_matrix,
    create_env,
    metric_summary,
    open_price_matrix,
)


LINEAR_FEATURES = ("xs_rank_ret_20d", "mom_12_1", "excess_ret_20d")


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)


def rank_ic_table(
    features: np.ndarray,
    feature_names: list[str],
    open_prices: np.ndarray,
    close_prices: np.ndarray,
    dates: list[str],
    start: int,
    end: int,
    horizons: tuple[int, ...] = (1, 5, 20),
    block_size: int = 20,
    bootstrap_samples: int = 2000,
) -> pd.DataFrame:
    """Cross-sectional Spearman IC; label endpoints stay inside their time split."""
    rows: list[dict[str, float | int | str]] = []
    label_specs = [("next_session_intraday", 1)] + [
        ("forward_close_to_close", horizon) for horizon in horizons
    ] + [("forward_open_to_close", horizon) for horizon in horizons if horizon > 1]
    for label_name, horizon in label_specs:
        daily_indices = range(start, max(start, end - horizon))
        for feature_index, feature_name in enumerate(feature_names):
            ics: list[float] = []
            for t in daily_indices:
                if label_name == "next_session_intraday":
                    asset_returns = close_prices[t + 1] / open_prices[t + 1] - 1.0
                elif label_name == "forward_open_to_close":
                    asset_returns = close_prices[t + horizon] / open_prices[t + 1] - 1.0
                else:
                    asset_returns = close_prices[t + horizon] / close_prices[t] - 1.0
                x = features[t, :, feature_index]
                if not np.isfinite(x).all() or not np.isfinite(asset_returns).all():
                    continue
                x_rank = _rank(x)
                y_rank = _rank(asset_returns)
                if np.std(x_rank) <= 1e-12 or np.std(y_rank) <= 1e-12:
                    continue
                ic = float(np.corrcoef(x_rank, y_rank)[0, 1])
                if np.isfinite(ic):
                    ics.append(ic)
            values = np.asarray(ics, dtype=np.float64)
            std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            ci = block_bootstrap_excess_returns(
                values, np.zeros_like(values), block_size=max(block_size, horizon),
                num_samples=bootstrap_samples, seed=1729 + feature_index + horizon,
            ) if len(values) >= max(block_size, horizon) else None
            rows.append(
                {
                    "horizon_days": horizon,
                    "target": label_name,
                    "feature": feature_name,
                    "n_dates": len(values),
                    "mean_rank_ic": float(values.mean()) if len(values) else float("nan"),
                    "std_rank_ic": std,
                    "block_ci_lower_95": ci["ci_lower_95"] / 252.0 if ci else float("nan"),
                    "block_ci_upper_95": ci["ci_upper_95"] / 252.0 if ci else float("nan"),
                    "block_size_sessions": max(block_size, horizon),
                    "label_purge_sessions": horizon,
                    "naive_t_stat": (
                        float(values.mean() / (std / np.sqrt(len(values))))
                        if len(values) > 1 and std > 0
                        else float("nan")
                    ),
                    "positive_ic_fraction": (
                        float(np.mean(values > 0)) if len(values) else float("nan")
                    ),
                    "start_date": dates[start],
                    "last_label_date": dates[end - 1],
                }
            )
    return pd.DataFrame(rows)


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, float]:
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    x_centered = x - x_mean
    y_centered = y - y_mean
    covariance = x_centered.T @ x_centered / len(x_centered)
    cross_covariance = x_centered.T @ y_centered / len(x_centered)
    coefficients = np.linalg.solve(
        covariance + alpha * np.eye(x.shape[1], dtype=np.float64),
        cross_covariance,
    )
    return coefficients, x_mean, y_mean


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    weights = np.exp(np.clip(shifted, -50.0, 50.0))
    return weights / weights.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--model-config", type=Path, default=PROJECT_ROOT / "configs/model.yaml")
    parser.add_argument("--env-config", type=Path, default=PROJECT_ROOT / "configs/env.yaml")
    parser.add_argument("--train-end", default=TemporalSplitter.DEFAULT_TRAIN_END)
    parser.add_argument("--val-start", default=TemporalSplitter.DEFAULT_VAL_START)
    parser.add_argument("--val-end", default=TemporalSplitter.DEFAULT_VAL_END)
    parser.add_argument("--test-start", default=TemporalSplitter.DEFAULT_TEST_START)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--action-alpha", type=float, default=0.25)
    parser.add_argument("--cash-ratio", type=float, default=0.04)
    parser.add_argument("--reward-beta", type=float, default=None)
    parser.add_argument("--rank-ic-output", type=Path, default=PROJECT_ROOT / "experiments/feature_rank_ic.csv")
    parser.add_argument("--linear-output", type=Path, default=PROJECT_ROOT / "experiments/linear_signal_validation.csv")
    parser.add_argument("--coefficients-output", type=Path, default=PROJECT_ROOT / "experiments/linear_signal_coefficients.csv")
    args = parser.parse_args()

    if not np.isfinite(args.ridge_alpha) or args.ridge_alpha < 0:
        raise ValueError("--ridge-alpha ต้อง hữu hạn và không âm.")
    if not 0.0 < args.action_alpha <= 1.0:
        raise ValueError("--action-alpha phải thuộc (0, 1].")
    if not 0.0 <= args.cash_ratio < 1.0:
        raise ValueError("--cash-ratio phải thuộc [0, 1).")

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
    scaler = MarketFeatureScaler().fit(split.train)
    scaled_train = scaler.transform(split.train)
    scaled_val = scaler.transform(split.val)
    scaled_full = scaler.transform(full_tensor)
    tickers = list(full_tensor.tickers)
    close = close_price_matrix(processed, tickers, full_tensor.dates)
    open_all = open_price_matrix(processed, tickers)
    dates = list(full_tensor.dates)
    date_to_index = {date: index for index, date in enumerate(dates)}
    train_start = date_to_index[split.train.dates[0]]
    train_end = date_to_index[split.train.dates[-1]] + 1
    val_start = date_to_index[split.val.dates[0]]
    val_end = date_to_index[split.val.dates[-1]] + 1

    feature_names = list(full_tensor.feature_names)
    feature_indices = [feature_names.index(name) for name in LINEAR_FEATURES]
    all_scaled = scaled_full.tensor.astype(np.float64, copy=False)
    ic_parts = []
    ic_parts.append(
        rank_ic_table(
            all_scaled,
            feature_names,
            open_all,
            close,
            dates,
            train_start,
            train_end,
        ).assign(split="train")
    )
    ic_parts.append(
        rank_ic_table(
            all_scaled,
            feature_names,
            open_all,
            close,
            dates,
            val_start,
            val_end,
        ).assign(split="validation")
    )
    ic_table = pd.concat(ic_parts, ignore_index=True)
    args.rank_ic_output.parent.mkdir(parents=True, exist_ok=True)
    ic_table.to_csv(args.rank_ic_output, index=False)

    # Fit on train dates only. The action executes at next open, so use only
    # the next session's open-to-close return as its immediate asset-selection label.
    train_positions = np.arange(train_start, train_end - 1)
    train_x = all_scaled[train_positions][:, :, feature_indices].reshape(-1, len(feature_indices))
    train_forward_returns = (
        close[train_positions + 1] / open_all[train_positions + 1] - 1.0
    )
    train_y = (
        train_forward_returns - train_forward_returns.mean(axis=1, keepdims=True)
    ).reshape(-1)
    coefficients, x_mean, y_mean = _fit_ridge(train_x, train_y, args.ridge_alpha)
    train_predictions = (train_x - x_mean) @ coefficients + y_mean
    train_pred_panel = train_predictions.reshape(len(train_positions), len(tickers))
    train_score_scale = float(train_pred_panel.std())
    if not np.isfinite(train_score_scale) or train_score_scale <= 1e-12:
        raise RuntimeError("Mô hình tuyến tính tạo score train gần như hằng số.")

    coefficients_frame = pd.DataFrame(
        {
            "feature": LINEAR_FEATURES,
            "coefficient": coefficients,
            "train_feature_mean": x_mean,
            "ridge_alpha": args.ridge_alpha,
            "train_prediction_scale": train_score_scale,
        }
    )
    args.coefficients_output.parent.mkdir(parents=True, exist_ok=True)
    coefficients_frame.to_csv(args.coefficients_output, index=False)

    val_positions = np.arange(val_start, val_end - 1)
    val_x = all_scaled[val_positions][:, :, feature_indices]
    val_scores = (val_x - x_mean) @ coefficients + y_mean
    targets = np.zeros((len(val_positions), len(tickers) + 1), dtype=np.float32)
    for row_index, scores in enumerate(val_scores):
        centered = scores - scores.mean()
        targets[row_index, :-1] = (1.0 - args.cash_ratio) * _softmax(
            centered / train_score_scale
        )
        targets[row_index, -1] = args.cash_ratio

    def linear_strategy(_obs: dict, info: dict | None = None) -> np.ndarray:
        step = int((info or {}).get("step", 0))
        return targets[min(step, len(targets) - 1)]

    open_prices = open_all[val_start:val_end]
    close_val = close[val_start:val_end]
    with args.env_config.open("r", encoding="utf-8") as stream:
        env_raw = yaml.safe_load(stream)["environment"]
    val_history_start = val_start
    cost_rate = resolve_transaction_cost_components(env_raw)[2]
    risk_free = float(env_config["risk_free_rate"])

    def make_validation_env():
        return create_env(
            split.val,
            open_prices,
            args.env_config,
            reward_beta,
            reward_scaling=float(env_config.get("reward_scaling", 1.0)),
            history_tensor=scaled_full,
            history_start_index=val_history_start,
            reward_baseline="equal_weight",
            lookback_window=20,
            action_alpha=args.action_alpha,
            close_prices=close_val,
        )

    linear_result = DeterministicBacktester(make_validation_env()).run_strategy(linear_strategy)
    matched_result = DeterministicBacktester(make_validation_env()).run_strategy(
        EqualWeightBaseline(
            open_prices=open_prices,
            num_assets=len(tickers),
            transaction_cost_rate=cost_rate,
            risk_free_rate_annual=risk_free,
            cash_ratio=args.cash_ratio,
        )
    )
    equal_weight_result = DeterministicBacktester(
        create_env(
            split.val,
            open_prices,
            args.env_config,
            reward_beta,
            reward_scaling=float(env_config.get("reward_scaling", 1.0)),
            history_tensor=scaled_full,
            history_start_index=val_history_start,
            reward_baseline="equal_weight",
            lookback_window=20,
            action_alpha=1.0,
            close_prices=close_val,
        )
    ).run_strategy(
        EqualWeightBaseline(
            open_prices=open_prices,
            num_assets=len(tickers),
            transaction_cost_rate=cost_rate,
            risk_free_rate_annual=risk_free,
            cash_ratio=0.0,
        )
    )

    def summarize(name: str, result) -> dict[str, float | str]:
        metrics = metric_summary(result, risk_free)
        return {
            "strategy": name,
            "sharpe_convention": FinancialMetrics.SHARPE_CONVENTION,
            "cumulative_return": metrics["cumulative_return"],
            "annualized_return": metrics["annualized_return"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "total_turnover": metrics["total_turnover"],
            "total_costs": metrics["total_costs"],
        }

    linear_metrics = summarize("Linear train-fit", linear_result)
    matched_metrics = summarize("Matched 1/N + cash", matched_result)
    equal_weight_metrics = summarize("Equal Weight 1/N", equal_weight_result)
    linear_metrics["sharpe_delta_vs_matched_baseline"] = (
        float(linear_metrics["sharpe_ratio"]) - float(matched_metrics["sharpe_ratio"])
    )
    linear_metrics["return_delta_vs_matched_baseline"] = (
        float(linear_metrics["cumulative_return"])
        - float(matched_metrics["cumulative_return"])
    )
    summary = pd.DataFrame([linear_metrics, matched_metrics, equal_weight_metrics])
    args.linear_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.linear_output, index=False)
    atomic_json(args.coefficients_output.with_suffix(".manifest.json"), {
        "feature_parameters": FeaturePipeline().parameters, "scaler_params": scaler.to_dict(),
        "fit_dates": [split.train.dates[0], split.train.dates[-1]],
        "validation_dates": [split.val.dates[0], split.val.dates[-1]],
        "linear_features": LINEAR_FEATURES, "ridge_alpha": args.ridge_alpha,
        "action_alpha": args.action_alpha, "cash_ratio": args.cash_ratio,
        "label": "next_session_open_to_close_cross_sectional_excess",
        "label_endpoints_purged_at_split_boundaries": True,
        "source_hashes": file_hashes(list((PROJECT_ROOT / "src").rglob("*.py")) + [Path(__file__)], PROJECT_ROOT),
        "data_hashes": file_hashes(args.raw_dir.glob("*.parquet"), PROJECT_ROOT),
        "config_hashes": file_hashes([args.env_config, args.model_config, PROJECT_ROOT / "configs/assets.yaml"], PROJECT_ROOT),
    })

    print("Rank-IC (daily cross-sectional Spearman; naive t-stat ignores serial dependence):")
    print(
        ic_table[
            (ic_table["target"] == "next_session_intraday")
            | ic_table["horizon_days"].isin((5,))
        ]
        .pivot(
            index="feature",
            columns=["split", "target", "horizon_days"],
            values="mean_rank_ic",
        )
        .to_string(float_format=lambda value: f"{value:+.4f}")
    )
    print("\nTrain-only ridge coefficients:")
    print(coefficients_frame.to_string(index=False, float_format=lambda value: f"{value:+.6f}"))
    print("\nValidation portfolio metrics:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Saved rank IC: {args.rank_ic_output}")
    print(f"Saved coefficients: {args.coefficients_output}")
    print(f"Saved validation metrics: {args.linear_output}")


if __name__ == "__main__":
    main()

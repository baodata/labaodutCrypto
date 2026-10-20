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
from src.data.labels import LABEL_DEFINITION, decision_positions, executable_forward_returns
from src.env.cost import resolve_transaction_cost_components
from src.evaluation.backtester import DeterministicBacktester
from src.evaluation.metrics import FinancialMetrics, block_bootstrap_excess_returns
from src.evaluation.baselines import EqualWeightBaseline
from src.features.pipeline import FeaturePipeline
from src.features.scaler import MarketFeatureScaler
from src.utils.artifacts import atomic_json, file_hashes
from src.training.data_preparation import close_price_matrix, open_price_matrix
from src.training.env_factory import create_env
from src.evaluation.policy_evaluator import metric_summary


from src.evaluation.feature_signal import _rank, rank_ic_table, _fit_ridge, _softmax

LINEAR_FEATURES = ("xs_rank_ret_20d", "mom_12_1", "excess_ret_20d")










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
    parser.add_argument("--research-protocol", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.research_protocol is not None:
        from src.evaluation.signal_research import run_signal_research
        run_signal_research(args.research_protocol, args.raw_dir, args.env_config,
                            output_dir=args.output_dir, model_config=args.model_config)
        return

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
        end_date=args.val_end,
    )
    split = TemporalSplitter(
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
    ).split_market_data_tensor(full_tensor, include_test=False)
    scaler = MarketFeatureScaler().fit(split.train)
    scaled_train = scaler.transform(split.train)
    scaled_val = scaler.transform(split.val)
    scaled_full = scaler.transform(full_tensor)
    tickers = list(full_tensor.tickers)
    close = close_price_matrix(processed, tickers, full_tensor.dates)
    open_all = open_price_matrix(processed, tickers)
    dates = list(full_tensor.dates)
    date_to_index = {date: index for index, date in enumerate(dates)}
    train_start = max(date_to_index[split.train.dates[0]], full_tensor.warmup_sessions)
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
    train_positions = decision_positions(len(dates), train_start, train_end, 1)
    if not len(train_positions):
        raise ValueError("Train has no executable labels after feature warm-up.")
    train_x = all_scaled[train_positions][:, :, feature_indices].reshape(-1, len(feature_indices))
    train_forward_returns = executable_forward_returns(open_all, close, train_positions, 1)
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
        "label_definition": LABEL_DEFINITION, "horizon_unit": "trading_sessions",
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

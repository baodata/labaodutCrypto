"""Nested historical development folds for executable IC and bounded 1/N tilts."""

from pathlib import Path
import math

import numpy as np
import pandas as pd
import yaml

from src.data.labels import LABEL_DEFINITION, decision_positions, executable_forward_returns
from src.env.cost import resolve_transaction_cost_components
from src.evaluation.backtester import DeterministicBacktester
from src.evaluation.baselines import EqualWeightBaseline
from src.evaluation.feature_signal import _fit_ridge
from src.evaluation.metrics import block_bootstrap_excess_returns
from src.evaluation.policy_evaluator import metric_summary
from src.evaluation.research_protocol import load_protocol
from src.features.pipeline import FeaturePipeline
from src.features.scaler import MarketFeatureScaler
from src.training.data_preparation import close_price_matrix, open_price_matrix
from src.training.env_factory import create_env
from src.utils.artifacts import atomic_csv, atomic_json, file_hashes, new_run_directory
from src.utils.data_types import MarketDataTensor

ROOT = Path(__file__).resolve().parents[2]


def _slice(tensor, start, end):
    if end - start < 2:
        raise ValueError("Fold phải có ít nhất hai phiên sau feature warm-up.")
    return MarketDataTensor(tensor.tensor[start:end].copy(), list(tensor.tickers),
                            list(tensor.feature_names), list(tensor.dates[start:end]))


def _ic(features, names, opening, closing, dates, start, end, horizons, bootstrap):
    rows = []
    for horizon in horizons:
        positions = decision_positions(len(opening), start, end, horizon)
        returns = executable_forward_returns(opening, closing, positions, horizon)
        yr = pd.DataFrame(returns).rank(axis=1).to_numpy(copy=True)
        yr -= yr.mean(axis=1, keepdims=True)
        for j, name in enumerate(names):
            x = features[positions, :, j]
            ranks = pd.DataFrame(x).rank(axis=1).to_numpy(copy=True)
            xr = ranks - ranks.mean(axis=1, keepdims=True)
            denominator = np.sqrt((xr * xr).sum(axis=1) * (yr * yr).sum(axis=1))
            valid = np.isfinite(denominator) & (denominator > 1e-12)
            values = (xr[valid] * yr[valid]).sum(axis=1) / denominator[valid]
            block = max(horizon, bootstrap["block_size"])
            if len(values) < block:
                raise ValueError(f"Không đủ IC dates cho {name}, horizon={horizon}.")
            ci = block_bootstrap_excess_returns(values, np.zeros_like(values), block_size=block,
                 num_samples=bootstrap["samples"], seed=bootstrap["seed"] + j + horizon)
            k = max(1, x.shape[1] // 5)
            order = np.argsort(x[valid], axis=1, kind="stable")
            top = np.take_along_axis(returns[valid], order[:, -k:], axis=1).mean(axis=1)
            bottom = np.take_along_axis(returns[valid], order[:, :k], axis=1).mean(axis=1)
            rows.append({
                "feature": name, "horizon_days": horizon, "target": "forward_open_to_close",
                "horizon_sessions": horizon,
                "executable": True, "label_definition": LABEL_DEFINITION,
                "n_dates": len(values), "mean_rank_ic": float(values.mean()),
                "std_rank_ic": float(values.std(ddof=1)),
                "block_ci_lower_95": ci["ci_lower_95"] / 252,
                "block_ci_upper_95": ci["ci_upper_95"] / 252,
                "positive_ic_fraction": float((values > 0).mean()),
                "mean_top_minus_bottom_return": float((top - bottom).mean()),
                "top_minus_bottom_positive_fraction": float((top > bottom).mean()),
                "block_size_sessions": block, "label_purge_sessions": horizon,
                "first_decision_date": dates[start], "last_decision_date": dates[end - horizon - 1],
                "last_label_date": dates[end - 1],
            })
    return pd.DataFrame(rows)


def _fit(tensor, opening, closing, start, end, horizon, feature_indices, alpha):
    scaler = MarketFeatureScaler().fit(_slice(tensor, start, end))
    full = scaler.transform(tensor)
    positions = decision_positions(len(opening), start, end, horizon)
    if not len(positions):
        raise ValueError("Không đủ ngày để fit nhãn đã purge.")
    x = full.tensor[positions][:, :, feature_indices].astype(np.float64).reshape(-1, len(feature_indices))
    forward = executable_forward_returns(opening, closing, positions, horizon)
    y = (forward - forward.mean(axis=1, keepdims=True)).reshape(-1)
    coefficients, center, intercept = _fit_ridge(x, y, alpha)
    predictions = (x - center) @ coefficients + intercept
    scale = max(float(predictions.std()), 1e-12)
    scores = (full.tensor[:, :, feature_indices].astype(np.float64) - center) @ coefficients + intercept
    return full, scores, scale, coefficients, scaler, center, intercept


def _targets(scores, scale, tilt, cash):
    normalized = np.clip(scores / scale, -1, 1)
    normalized -= normalized.mean(axis=1, keepdims=True)
    normalized /= np.maximum(np.abs(normalized).max(axis=1, keepdims=True), 1.0)
    weights = (1 - cash) / scores.shape[1] * (1 + tilt * normalized)
    return np.column_stack((weights, np.full(len(weights), cash))).astype(np.float32)


def run_signal_research(protocol_path, raw_dir, env_config, output_dir=None, model_config=None,
                        assets_config=None):
    protocol, digest = load_protocol(protocol_path)
    raw_dir, env_config = Path(raw_dir), Path(env_config)
    assets_config = Path(assets_config or ROOT / "configs/core/assets.yaml")
    output = Path(output_dir) if output_dir else new_run_directory(ROOT / "models/runs", "sprint5_signal")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output đã có artifact; chọn thư mục mới.")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running", "protocol": protocol, "protocol_hash": digest,
        "test_evaluated": False, "data_cutoff": protocol["development_end"],
        "period_status": protocol["period_status"],
        "source_hashes": file_hashes(list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")), ROOT),
        "data_hashes": file_hashes(raw_dir.glob("*.parquet"), ROOT),
        "config_hashes": file_hashes([Path(protocol_path), env_config, assets_config] +
                                      ([Path(model_config)] if model_config else []), ROOT),
        "execution": protocol["execution"], "selection_uses": "inner_only",
    }
    atomic_json(output / "manifest.json", manifest)
    try:
        return _run(protocol, raw_dir, env_config, assets_config, output, manifest)
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        atomic_json(output / "manifest.json", manifest)
        raise


def _run(protocol, raw_dir, env_path, assets_path, output, manifest):
    pipeline = FeaturePipeline(assets_config_path=str(assets_path))
    processed, tensor = pipeline.run_from_raw(raw_dir, export_parquet=False, end_date=protocol["development_end"])
    dates = list(tensor.dates)
    calendar = pd.DatetimeIndex(pd.to_datetime(dates))
    opening = open_price_matrix(processed, list(tensor.tickers))
    closing = close_price_matrix(processed, list(tensor.tickers), dates)
    with env_path.open() as stream:
        config = yaml.safe_load(stream)["environment"]
    cost = resolve_transaction_cost_components(config)[2]
    rf = float(config["risk_free_rate"])
    execution, signal = protocol["execution"], protocol["signal"]
    feature_indices = [tensor.feature_names.index(name) for name in signal["features"]]
    lookback = int(config.get("lookback_window", 20))
    start = tensor.warmup_sessions
    ic_parts, metrics_rows, selections, coefficients_rows, daily_rows = [], [], [], [], []

    def bounds(begin, end):
        a = start if begin is None else int(calendar.searchsorted(pd.Timestamp(begin)))
        b = int(calendar.searchsorted(pd.Timestamp(end), side="right"))
        if a < start or b - a < max(signal["ic_horizons"]) + protocol["bootstrap"]["block_size"]:
            raise ValueError("Fold thiếu dữ liệu đã warm-up.")
        if b == 0 or calendar[b - 1] < pd.Timestamp(end) - pd.Timedelta(days=7):
            raise ValueError("Dữ liệu không phủ đủ cuối fold; không tự thay mốc.")
        return a, b

    def backtest(full, begin, end, targets=None, alpha=None, cash=None):
        env = create_env(_slice(full, begin, end), opening[begin:end], env_path,
                        float(config["reward_beta"]), close_prices=closing[begin:end],
                        history_tensor=full, history_start_index=begin, lookback_window=lookback,
                        action_alpha=execution["action_alpha"] if alpha is None else alpha)
        if targets is None:
            strategy = EqualWeightBaseline(opening[begin:end], len(tensor.tickers), cost, rf,
                         cash_ratio=execution["cash_ratio"] if cash is None else cash)
        else:
            strategy = lambda obs, info: targets[int(info["step"])]
        return DeterministicBacktester(env).run_strategy(strategy)

    for fold in protocol["folds"]:
        print(f"Signal research: {fold['name']} — fit/inner selection/outer", flush=True)
        _, train_end = bounds(None, fold["train_end"])
        inner_start, inner_end = bounds(fold["inner_start"], fold["inner_end"])
        outer_start, outer_end = bounds(fold["outer_start"], fold["outer_end"])
        train_ic = _ic(tensor.tensor, tensor.feature_names, opening, closing, dates, start, train_end,
                       signal["ic_horizons"], protocol["bootstrap"])
        outer_ic = _ic(tensor.tensor, tensor.feature_names, opening, closing, dates, outer_start, outer_end,
                       signal["ic_horizons"], protocol["bootstrap"])
        directions = train_ic.set_index(["feature", "horizon_days"])["mean_rank_ic"]
        for phase, frame in (("train", train_ic), ("outer", outer_ic)):
            frame["fold"] = fold["name"]
            frame["split"] = phase
            signs = np.array([1.0 if directions.loc[(r.feature, r.horizon_days)] >= 0 else -1.0
                              for r in frame.itertuples()])
            frame["train_direction"] = signs
            frame["aligned_mean_rank_ic"] = frame.mean_rank_ic * signs
            frame["aligned_ci_lower_95"] = np.where(signs > 0, frame.block_ci_lower_95, -frame.block_ci_upper_95)
            ic_parts.append(frame)

        best = None
        for ridge in signal["ridge_alphas"]:
            full, scores, scale, _, _, _, _ = _fit(tensor, opening, closing, start, train_end,
                         signal["label_horizon"], feature_indices, ridge)
            for tilt in signal["tilt_fractions"]:
                result = backtest(full, inner_start, inner_end,
                                   _targets(scores[inner_start:inner_end-1], scale, tilt, execution["cash_ratio"]))
                score = metric_summary(result, rf)[signal["selection_metric"]]
                selections.append({"fold": fold["name"], "ridge_alpha": ridge, "tilt_fraction": tilt,
                                   "inner_score": score, "selection_period": "inner_only"})
                if math.isfinite(score) and (best is None or score > best[0] + 1e-12):
                    best = (score, ridge, tilt)
        if best is None:
            raise ValueError("Không có candidate có inner score hữu hạn.")
        _, ridge, tilt = best
        # After hyperparameter selection, refit on train + inner; no outer labels.
        full, scores, scale, coefficients, scaler, center, intercept = _fit(tensor, opening, closing, start, inner_end,
                         signal["label_horizon"], feature_indices, ridge)
        results = {
            "Linear 1/N core tilt": backtest(full, outer_start, outer_end,
                   _targets(scores[outer_start:outer_end-1], scale, tilt, execution["cash_ratio"])),
            "Matched 1/N + Cash": backtest(full, outer_start, outer_end),
            "Equal Weight (1/N)": backtest(full, outer_start, outer_end, alpha=1.0, cash=0.0),
        }
        matched = metric_summary(results["Matched 1/N + Cash"], rf)
        for name, result in results.items():
            summary = metric_summary(result, rf)
            summary.update(fold=fold["name"], strategy=name, ridge_alpha=ridge, tilt_fraction=tilt,
                           evaluation_scope="outer_development", selection_period="inner_only",
                           sharpe_delta_vs_matched_baseline=summary["sharpe_ratio"]-matched["sharpe_ratio"],
                           return_delta_vs_matched_baseline=summary["cumulative_return"]-matched["cumulative_return"],
                           mean_realized_cash=float(result.weights_history[:, -1].mean()))
            metrics_rows.append(summary)
        candidate, baseline = results["Linear 1/N core tilt"], results["Matched 1/N + Cash"]
        excess = candidate.daily_net_returns - baseline.daily_net_returns
        daily_rows.append(pd.DataFrame({"date": dates[outer_start+1:outer_end], "fold": fold["name"],
            "candidate": candidate.daily_net_returns, "matched": baseline.daily_net_returns, "excess": excess}))
        for name, value in zip(signal["features"], coefficients):
            coefficients_rows.append({"fold": fold["name"], "feature": name, "coefficient": value,
                "ridge_alpha": ridge, "tilt_fraction": tilt, "prediction_scale": scale,
                "last_fit_label_date": dates[inner_end-1], "first_outer_decision_date": dates[outer_start]})
        atomic_json(output / f"{fold['name']}_model.json", {"scaler": scaler.to_dict(),
                    "coefficients": coefficients.tolist(), "ridge_alpha": ridge, "tilt_fraction": tilt,
                    "feature_center": center.tolist(), "intercept": intercept,
                    "prediction_scale": scale, "feature_names": signal["features"],
                    "last_fit_label_date": dates[inner_end-1], "label_purge_sessions": signal["label_horizon"],
                    "label_definition": LABEL_DEFINITION, "prediction_unit": "decimal_cross_sectional_excess_return",
                    "horizon_unit": "trading_sessions"})
        atomic_csv(pd.concat(ic_parts), output / "signal_by_fold.csv", index=False)
        atomic_csv(pd.DataFrame(metrics_rows), output / "linear_baseline_by_fold.csv", index=False)
        atomic_csv(pd.DataFrame(selections), output / "inner_search.csv", index=False)
        print(f"  selected alpha={ridge:g}, tilt={tilt:g}; outer ΔSharpe="
              f"{metrics_rows[-3]['sharpe_delta_vs_matched_baseline']:+.4f}", flush=True)

    daily = pd.concat(daily_rows, ignore_index=True)
    stats = block_bootstrap_excess_returns(daily.candidate.to_numpy(), daily.matched.to_numpy(),
             block_size=protocol["bootstrap"]["block_size"], num_samples=protocol["bootstrap"]["samples"],
             seed=protocol["bootstrap"]["seed"], groups=daily.fold.to_numpy())
    metrics = pd.DataFrame(metrics_rows)
    candidate = metrics[metrics.strategy == "Linear 1/N core tilt"]
    matched = metrics[metrics.strategy == "Matched 1/N + Cash"]
    gate = protocol["gate"]
    ic = pd.concat(ic_parts, ignore_index=True)
    relevant = ic[(ic.split == "outer") & ic.feature.isin(signal["features"]) &
                  (ic.horizon_days == signal["label_horizon"])]
    counts = relevant[relevant.aligned_ci_lower_95 > 0].groupby("feature").size()
    conditions = {
        "positive_folds": int((candidate.return_delta_vs_matched_baseline > 0).sum()) >= gate["minimum_positive_folds"],
        "net_excess": stats["mean_excess_annualized"] >= gate["minimum_annualized_net_excess"],
        "confidence": stats["ci_lower_95"] > gate["minimum_ci_lower"],
        "sharpe": float(candidate.sharpe_delta_vs_matched_baseline.mean()) > gate["minimum_mean_sharpe_delta"],
        "drawdown": float((matched.max_drawdown.to_numpy()-candidate.max_drawdown.to_numpy()).mean()) <= gate["maximum_drawdown_degradation"],
        "signal_stability": int(counts.max()) if len(counts) else 0,
    }
    conditions["signal_stability"] = conditions["signal_stability"] >= gate["minimum_signal_folds"]
    decision = {"decision": "GO" if all(conditions.values()) else "NO-GO", "conditions": conditions,
                "bootstrap_vs_matched": stats, "signal_positive_ci_folds": {k: int(v) for k,v in counts.items()},
                "purpose": "permission_for_further_alpha_research_not_independent_alpha_confirmation",
                "test_evaluated": False}
    atomic_csv(daily, output / "outer_daily_returns.csv", index=False)
    atomic_csv(pd.DataFrame(coefficients_rows), output / "coefficients_by_fold.csv", index=False)
    atomic_json(output / "decision.json", decision)
    manifest.update(status="completed", decision=decision["decision"],
                    feature_parameters=pipeline.parameters, resolved_environment=config,
                    tickers=tensor.tickers, feature_names=tensor.feature_names)
    atomic_json(output / "manifest.json", manifest)
    print(f"Decision: {decision['decision']}; artifacts: {output}", flush=True)
    return decision, output

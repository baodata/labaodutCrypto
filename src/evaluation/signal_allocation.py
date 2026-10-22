"""Compare frozen signal/allocator candidates on nested development folds."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.labels import ExecutableLabelPanel
from src.env.cost import resolve_transaction_cost_components
from src.env.orders import PortfolioOrder
from src.evaluation.backtester import DeterministicBacktester
from src.evaluation.cost_aware_allocator import asset_bounds, bounded_tilt, cost_aware_target
from src.evaluation.metrics import block_bootstrap_excess_returns
from src.evaluation.policy_evaluator import metric_summary
from src.evaluation.prediction_bank import PredictionBank
from src.evaluation.predictor_contribution import predictor_contributions
from src.features.liquidity import build_liquidity_bank
from src.features.pipeline import FeaturePipeline
from src.risk.covariance import build_covariance_bank
from src.training.data_preparation import close_price_matrix, open_price_matrix
from src.training.env_factory import create_env
from src.utils.artifacts import atomic_csv, atomic_json, file_hashes, new_run_directory
from src.utils.data_types import MarketDataTensor


ROOT = Path(__file__).resolve().parents[2]


def load_allocation_config(path):
    from src.utils.config import resolve_config_path
    resolved_path = resolve_config_path(path)
    with Path(resolved_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    expected = {"version", "models", "horizons", "tilt_fraction", "risk_aversion",
                "cost_multiplier", "risk", "execution"}
    if not isinstance(config, dict) or set(config) != expected or config["version"] != "signal-allocation-v1":
        raise ValueError("Invalid fixed signal allocation configuration.")
    if (not config["models"] or len(set(config["models"])) != len(config["models"])
            or set(config["models"]) - {"ridge_fixed", "momentum", "mean_reversion", "gru", "gnn_gru", "gnn_gru_no_edges"}):
        raise ValueError("Specify unique supported signal families.")
    if (not config["horizons"] or len(set(config["horizons"])) != len(config["horizons"])
            or any(type(h) is not int or h <= 0 for h in config["horizons"])):
        raise ValueError("Horizons must be unique positive integer session counts.")
    if (not np.isfinite([config[k] for k in ("tilt_fraction", "risk_aversion", "cost_multiplier")]).all()
            or not 0 <= config["tilt_fraction"] <= 1
            or config["risk_aversion"] < 0 or config["cost_multiplier"] < 0):
        raise ValueError("Invalid allocator tilt/risk/cost settings.")
    if config["execution"] != {"accounting": "cash_funded", "impact_eta": 0., "rebalance_every_sessions": 1}:
        raise ValueError("This comparison fixes daily cash-funded commission/slippage, no impact/controller.")
    if set(config["risk"]) != {"method", "window", "min_periods", "decay", "jitter"}:
        raise ValueError("Risk parameters must be explicit.")
    return config


def _holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    maximum = 0.
    for rank, index in enumerate(order):
        maximum = max(maximum, float(p_values[index]) * (len(p_values) - rank))
        adjusted[index] = min(1., maximum)
    return adjusted


def compare_signal_allocators(bank_dir, raw_dir, assets_config, env_config, allocation_config, output_dir=None,
                              *, development_only=False):
    from src.utils.config import resolve_config_path
    assets_config = resolve_config_path(assets_config)
    env_config = resolve_config_path(env_config)
    allocation_config = resolve_config_path(allocation_config)
    bank = PredictionBank.load(bank_dir)
    config = load_allocation_config(allocation_config)
    if file_hashes(Path(raw_dir).glob("*.parquet"), ROOT) != bank.metadata["data_hashes"]:
        raise ValueError("Raw data snapshot differs from the prediction bank.")
    asset_hash = file_hashes([Path(assets_config)], ROOT)
    if any(bank.metadata["config_hashes"].get(k) != value for k, value in asset_hash.items()):
        raise ValueError("Assets config differs from the prediction bank.")
    models = bank.metadata.get("models", {})
    if not set(config["models"]).issubset(models):
        raise ValueError("Bank lacks required signal families; rebuild it with the baseline config.")
    protocol = bank.metadata["protocol"]
    if bank.metadata.get("development_only") and not development_only:
        raise ValueError("A development-only prediction bank cannot evaluate outer.")
    if protocol["execution"]["rebalance_every_sessions"] != 1:
        raise ValueError("Comparison requires the same daily protocol schedule.")
    output = Path(output_dir) if output_dir else new_run_directory(ROOT / "models/runs", "signal_allocation")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Comparison output already contains artifacts; choose a new run.")
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(status="running", test_evaluated=False, selection_uses="fixed_configuration_no_search",
        prediction_bank=str(Path(bank_dir).resolve()), prediction_protocol_hash=bank.metadata["protocol_hash"],
        prediction_source_hashes=bank.metadata["source_hashes"], comparison_config=config,
        source_hashes=file_hashes(list((ROOT / "src").rglob("*.py")) + [ROOT / "scripts/research/compare_signal_allocators.py"], ROOT),
        config_hashes=file_hashes([Path(env_config), Path(allocation_config), Path(assets_config)], ROOT),
        data_hashes=bank.metadata["data_hashes"], period_status=protocol["period_status"],
        objective="forecast_h_vs_tracking_risk_hSigma_and_Close_estimated_turnover_cost",
        cost_model="commission_plus_slippage_cash_funded_actual_Open_stock_legs",
        covariance_horizon_approximation="h_times_daily_covariance_no_annualization")
    atomic_json(output / "manifest.json", manifest)
    try:
        if development_only:
            manifest.update(outer_evaluated=False, development_only=True)
            atomic_json(output / "manifest.json", manifest)
        summary = _compare(bank, config, raw_dir, assets_config, env_config, output,
                           development_only=development_only)
        manifest.update(status="completed", cases=len(summary),
                        portfolio_gate_passes=int(summary.passes_portfolio_gate.sum()))
        atomic_json(output / "manifest.json", manifest)
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        atomic_json(output / "manifest.json", manifest)
        raise
    print(summary[["model", "allocator", "horizon_sessions", "mean_sharpe_delta_vs_matched_baseline",
                   "excess_annual", "ci_lower_95", "ci_upper_95", "holm_p_value", "passes_portfolio_gate"]].to_string(index=False), flush=True)
    print(f"Comparison artifacts: {output}", flush=True)
    return summary, output


def _compare(bank, config, raw_dir, assets_path, env_path, output, *, development_only=False):
    protocol = bank.metadata["protocol"]
    pipeline = FeaturePipeline(assets_config_path=str(assets_path))
    processed, tensor = pipeline.run_from_raw(raw_dir, export_parquet=False, end_date=protocol["development_end"])
    if (tensor.tickers != bank.metadata["tickers"] or tensor.dates != bank.metadata["session_calendar"]
            or pipeline.parameters != bank.metadata["feature_parameters"]):
        raise ValueError("Feature calendar/universe/parameters differ from the prediction bank.")
    opening = open_price_matrix(processed, tensor.tickers)
    closing = close_price_matrix(processed, tensor.tickers, tensor.dates)
    labels = ExecutableLabelPanel(tensor.dates, tensor.tickers, opening, closing)
    with Path(env_path).open(encoding="utf-8") as stream:
        environment = yaml.safe_load(stream)["environment"]
    cost_rate = resolve_transaction_cost_components(environment)[2]
    cash, alpha = protocol["execution"]["cash_ratio"], protocol["execution"]["action_alpha"]
    cap = float(environment.get("max_asset_weight", 1.))
    core, _, _ = asset_bounds(len(tensor.tickers), cash, config["tilt_fraction"], cap)
    risk = build_covariance_bank(pd.DataFrame(closing, index=labels.calendar, columns=tensor.tickers).pct_change(fill_method=None),
                                 **config["risk"], cache_dir=output / "banks")
    liquidity = build_liquidity_bank(processed, tensor.dates, tensor.tickers)
    tensor.quant_context = dict(dates=tensor.dates, tickers=tensor.tickers, risk=risk, liquidity=liquidity,
        config={"impact": {"accounting": "cash_funded", "eta": 0.}, "execution": {"enabled": False}})
    atomic_csv(risk.diagnostics, output / "risk_diagnostics.csv", index=False)
    atomic_json(output / "resolved_comparison.json", dict(environment=environment, execution=protocol["execution"],
        risk=risk.metadata, liquidity=liquidity.metadata, initial_allocation="blend_from_cash",
        accounting_precision="float64", equal_constraints_for_all_signals=True))
    records, daily_parts, decision_records, ic_records = [], [], [], []
    for fold in protocol["folds"]:
        for phase in (("inner",) if development_only else ("inner", "outer")):
            a = int(labels.calendar.searchsorted(pd.Timestamp(fold[f"{phase}_start"])))
            b = int(labels.calendar.searchsorted(pd.Timestamp(fold[f"{phase}_end"]), side="right"))
            episode = MarketDataTensor(tensor.tensor[a:b].copy(), tensor.tickers, tensor.feature_names, tensor.dates[a:b])
            for horizon in config["horizons"]:
                print(f"Compare: {fold['name']} / {phase} / h={horizon}", flush=True)
                positions = labels.positions(a, b, horizon)
                expected_dates = labels.calendar[positions].strftime("%Y-%m-%d").tolist()
                panels, scales = {}, {}
                for family in config["models"]:
                    panel = bank.panel(model_id=models_id(bank, family), fold_id=fold["name"], split=phase, horizon=horizon)
                    if list(panel.index) != expected_dates:
                        raise ValueError("Forecast coverage differs across models or misses a phase decision date.")
                    panels[family] = panel.to_numpy()
                    rows = bank.predictions
                    rows = rows[(rows.model_id == models_id(bank, family)) & (rows.fold_id == fold["name"])
                                & (rows.split == phase) & (rows.horizon_sessions == horizon)]
                    fit_by_date = rows.drop_duplicates("decision_date").set_index("decision_date").fit_id
                    scales[family] = np.array([bank.fits[fit_by_date.loc[d]]["prediction_scale"] for d in expected_dates])
                    forecast_ranks = pd.DataFrame(panels[family]).rank(axis=1).to_numpy(copy=True)
                    outcome_ranks = pd.DataFrame(labels.returns(positions, horizon)).rank(axis=1).to_numpy(copy=True)
                    forecast_ranks -= forecast_ranks.mean(axis=1, keepdims=True)
                    outcome_ranks -= outcome_ranks.mean(axis=1, keepdims=True)
                    denominator = np.sqrt((forecast_ranks**2).sum(axis=1)*(outcome_ranks**2).sum(axis=1))
                    ic = np.divide((forecast_ranks*outcome_ranks).sum(axis=1), denominator,
                                   out=np.full(len(positions), np.nan), where=denominator > 1e-12)
                    valid = ic[np.isfinite(ic)]
                    block = max(horizon, protocol["bootstrap"]["block_size"])
                    uncertainty = block_bootstrap_excess_returns(valid, np.zeros_like(valid), block_size=block,
                        num_samples=protocol["bootstrap"]["samples"], seed=protocol["bootstrap"]["seed"]) if len(valid) >= block else None
                    ic_records.append(dict(fold_id=fold["name"], split=phase, model=family, horizon_sessions=horizon,
                        valid_dates=len(valid), constant_prediction_dates=int(np.count_nonzero(~np.isfinite(ic))),
                        mean_rank_ic=float(valid.mean()) if len(valid) else np.nan,
                        ci_lower_95=uncertainty["ci_lower_95"]/252 if uncertainty else np.nan,
                        ci_upper_95=uncertainty["ci_upper_95"]/252 if uncertainty else np.nan,
                        block_size=block, first_decision=expected_dates[0], last_label=tensor.dates[b-1]))
                if not len(positions) or not risk.ready[positions].all() or not liquidity.ready[a:b-1].all():
                    raise ValueError("Insufficient forecasts/risk/liquidity readiness; no implicit fallback.")

                def simulate(family, allocator):
                    env = create_env(episode, opening[a:b], env_path, float(environment["reward_beta"]),
                        history_tensor=tensor, history_start_index=a, lookback_window=1,
                        action_alpha=alpha, close_prices=closing[a:b])

                    def strategy(obs, info):
                        step = int(info["step"])
                        # Every branch holds through the same horizon-purged tail.
                        if step >= len(positions):
                            return PortfolioOrder.hold()
                        if allocator == "matched_1n":
                            return np.append(core, cash)
                        prediction = panels[family][step]
                        if allocator == "bounded_tilt":
                            return bounded_tilt(prediction, scales[family][step], cash_ratio=cash,
                                tilt_fraction=config["tilt_fraction"], max_asset_weight=cap)
                        target, diagnostic = cost_aware_target(prediction, risk.covariance[a+step],
                            env.portfolio.weights.copy(), horizon=horizon, action_alpha=alpha, cash_ratio=cash,
                            tilt_fraction=config["tilt_fraction"], max_asset_weight=cap,
                            risk_aversion=config["risk_aversion"], cost_rate=cost_rate,
                            cost_multiplier=config["cost_multiplier"])
                        decision_records.append(dict(fold_id=fold["name"], split=phase, model=family,
                            horizon_sessions=horizon, decision_date=tensor.dates[a+step], **diagnostic))
                        return target

                    return DeterministicBacktester(env).run_strategy(strategy)

                baseline = simulate("none", "matched_1n")
                baseline_metrics = metric_summary(baseline, float(environment["risk_free_rate"]))
                cases = [("none", "matched_1n", baseline)]
                for family in config["models"]:
                    for allocator in ("bounded_tilt", "cost_aware"):
                        cases.append((family, allocator, simulate(family, allocator)))
                for family, allocator, result in cases:
                    metrics = metric_summary(result, float(environment["risk_free_rate"]))
                    excess = result.daily_net_returns - baseline.daily_net_returns
                    records.append(dict(metrics, fold_id=fold["name"], split=phase, model=family, allocator=allocator,
                        horizon_sessions=horizon, sharpe_delta_vs_matched_baseline=metrics["sharpe_ratio"]-baseline_metrics["sharpe_ratio"],
                        return_delta_vs_matched_baseline=metrics["cumulative_return"]-baseline_metrics["cumulative_return"],
                        drawdown_degradation=baseline_metrics["max_drawdown"]-metrics["max_drawdown"],
                        mean_realized_cash=float(result.weights_history[:, -1].mean()),
                        tracking_error=float(excess.std(ddof=1)*np.sqrt(252)),
                        total_commission=float(result.daily_commission.sum()), total_slippage=float(result.daily_slippage.sum()),
                        total_impact=float(result.daily_impact.sum()), hold_fraction=float(result.daily_hold.mean()),
                        first_session=tensor.dates[a], last_session=tensor.dates[b-1]))
                    daily_parts.append(pd.DataFrame(dict(date=tensor.dates[a+1:b], fold_id=fold["name"], split=phase,
                        model=family, allocator=allocator, horizon_sessions=horizon,
                        net_return=result.daily_net_returns, matched_return=baseline.daily_net_returns, excess=excess,
                        turnover=result.daily_turnover, costs=result.daily_costs)))
                atomic_csv(pd.DataFrame(records), output / "metrics_by_fold.csv", index=False)
                atomic_csv(pd.DataFrame(decision_records), output / "allocator_decisions.csv", index=False)
                atomic_csv(pd.DataFrame(ic_records), output / "prediction_ic_by_fold.csv", index=False)
    metrics, daily = pd.DataFrame(records), pd.concat(daily_parts, ignore_index=True)
    peer_keys = ["fold_id", "split", "model", "horizon_sessions"]
    peers = metrics[metrics.allocator == "bounded_tilt"].set_index(peer_keys)
    for column in ("sharpe_ratio", "cumulative_return", "total_costs"):
        metrics[f"{column}_delta_vs_bounded_tilt"] = [
            float(row[column] - peers.loc[tuple(row[k] for k in peer_keys), column]) if row["model"] != "none" else np.nan
            for _, row in metrics.iterrows()]
    atomic_csv(metrics, output / "metrics_by_fold.csv", index=False)
    atomic_csv(daily, output / "daily_returns.csv", index=False)
    if development_only:
        # Inner is used to develop features; never label these comparisons as
        # passing the portfolio gate intended for the frozen outer protocol.
        summaries = []
        for (family, allocator, horizon), rows in metrics[metrics.model != "none"].groupby(
                ["model", "allocator", "horizon_sessions"], sort=False):
            subset = daily[(daily.model == family) & (daily.allocator == allocator)
                           & (daily.horizon_sessions == horizon)]
            stats = block_bootstrap_excess_returns(subset.net_return.to_numpy(), subset.matched_return.to_numpy(),
                block_size=max(protocol["bootstrap"]["block_size"], int(horizon)),
                num_samples=protocol["bootstrap"]["samples"], seed=protocol["bootstrap"]["seed"],
                groups=subset.fold_id.to_numpy())
            summaries.append(dict(model=family, allocator=allocator, horizon_sessions=int(horizon),
                mean_sharpe_delta_vs_matched_baseline=float(rows.sharpe_delta_vs_matched_baseline.mean()),
                excess_annual=stats["mean_excess_annualized"], ci_lower_95=stats["ci_lower_95"],
                ci_upper_95=stats["ci_upper_95"], p_value=stats["p_value_positive_excess"],
                passes_portfolio_gate=False, scope="inner_development_only"))
        summary = pd.DataFrame(summaries)
        summary["holm_p_value"] = _holm_adjust(summary.p_value.to_numpy())
        atomic_csv(summary, output / "inner_summary.csv", index=False)
        atomic_json(output / "decision.json", dict(decision="DEVELOPMENT_DIAGNOSTIC_ONLY",
            outer_evaluated=False, test_evaluated=False, portfolio_gate_evaluated=False,
            purpose="feature_research_on_reused_inner_not_independent_alpha_confirmation"))
        return summary
    summaries = []
    for (family, allocator, horizon), rows in metrics[(metrics.split == "outer") & (metrics.model != "none")].groupby(
            ["model", "allocator", "horizon_sessions"], sort=False):
        subset = daily[(daily.split == "outer") & (daily.model == family) & (daily.allocator == allocator)
                       & (daily.horizon_sessions == horizon)]
        stats = block_bootstrap_excess_returns(subset.net_return.to_numpy(), subset.matched_return.to_numpy(),
            block_size=max(protocol["bootstrap"]["block_size"], int(horizon)),
            num_samples=protocol["bootstrap"]["samples"], seed=protocol["bootstrap"]["seed"], groups=subset.fold_id.to_numpy())
        summaries.append(dict(model=family, allocator=allocator, horizon_sessions=int(horizon),
            mean_sharpe_delta_vs_matched_baseline=float(rows.sharpe_delta_vs_matched_baseline.mean()),
            mean_return_delta_vs_matched_baseline=float(rows.return_delta_vs_matched_baseline.mean()),
            positive_folds=int((rows.return_delta_vs_matched_baseline > 0).sum()),
            mean_drawdown_degradation=float(rows.drawdown_degradation.mean()),
            mean_turnover=float(rows.total_turnover.mean()), mean_costs=float(rows.total_costs.mean()),
            excess_annual=stats["mean_excess_annualized"], ci_lower_95=stats["ci_lower_95"],
            ci_upper_95=stats["ci_upper_95"], p_value=stats["p_value_positive_excess"]))
    summary = pd.DataFrame(summaries)
    contribution_folds, contributions = predictor_contributions(metrics, daily, protocol["bootstrap"])
    p_values = np.r_[summary.p_value.to_numpy(), contributions.p_value.to_numpy() if not contributions.empty else []]
    adjusted = _holm_adjust(p_values)
    summary["holm_p_value"] = adjusted[:len(summary)]
    if not contributions.empty:
        contributions["holm_p_value"] = adjusted[len(summary):]
        contributions["positive_contribution_supported"] = ((contributions.holm_p_value < .05)
            & (contributions.ci_lower_95 > 0) & (contributions.mean_sharpe_delta > 0))
        atomic_csv(contribution_folds, output / "predictor_contribution_by_fold.csv", index=False)
        atomic_csv(contributions, output / "predictor_contribution_summary.csv", index=False)
    gate = protocol["gate"]
    summary["passes_portfolio_gate"] = ((summary.positive_folds >= gate["minimum_positive_folds"])
        & (summary.excess_annual >= gate["minimum_annualized_net_excess"])
        & (summary.ci_lower_95 > gate["minimum_ci_lower"]) & (summary.holm_p_value < .05)
        & (summary.mean_sharpe_delta_vs_matched_baseline > gate["minimum_mean_sharpe_delta"])
        & (summary.mean_drawdown_degradation <= gate["maximum_drawdown_degradation"]))
    atomic_csv(summary, output / "outer_summary.csv", index=False)
    atomic_json(output / "decision.json", dict(decision="GO_FOR_FURTHER_RESEARCH" if summary.passes_portfolio_gate.any() else "NO-GO",
        test_evaluated=False, purpose="portfolio_gate_only_not_independent_alpha_confirmation",
        hypotheses=len(p_values), matched_hypotheses=len(summary), predictor_contrasts=len(contributions),
        multiplicity="Holm_over_matched_and_paired_predictor_centered_block_bootstrap_p_values",
        confidence_intervals="pointwise_95_not_simultaneous", gate=gate,
        fixed_configuration=True, signal_stability_gate_evaluated=False))
    return summary


def models_id(bank, family):
    return bank.metadata["models"][family]

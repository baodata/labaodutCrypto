"""QNT-026/027/029 data readiness and predictor ablations, train OOF/inner only."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.external_alpha import ExternalDataUnavailable
from src.evaluation.metrics import block_bootstrap_excess_returns
from src.evaluation.prediction_research import load_prediction_config, run_prediction_research
from src.evaluation.research_protocol import load_protocol
from src.evaluation.signal_allocation import compare_signal_allocators, load_allocation_config, _holm_adjust
from src.features.alpha_context import GROUPS, augment_alpha_tensor, build_alpha_context
from src.features.pipeline import FeaturePipeline
from src.training.data_preparation import close_price_matrix
from src.utils.artifacts import atomic_csv, atomic_json, file_hashes, new_run_directory


def rank_ic(bank, outcomes, variant):
    # Outcomes are explicitly joined only for offline diagnostics, never predictors.
    frame = bank.evaluation_frame(outcomes)
    keys = ["fold_id", "split", "model_id", "horizon_sessions", "decision_date"]
    rows = []
    for key, panel in frame.groupby(keys, sort=False):
        forecast = panel.prediction.rank(method="average").to_numpy()
        realized = panel.realized_cross_sectional_excess_return.rank(method="average").to_numpy()
        forecast -= forecast.mean()
        realized -= realized.mean()
        denominator = float(np.linalg.norm(forecast) * np.linalg.norm(realized))
        rows.append(dict(zip(keys, key), variant=variant,
            rank_ic=float(forecast @ realized / denominator) if denominator > 1e-12 else np.nan,
            mse=float(np.mean((panel.prediction - panel.realized_cross_sectional_excess_return)**2)),
            zero_forecast_mse=float(np.mean(panel.realized_cross_sectional_excess_return**2)),
            forecast_cross_asset_std=float(panel.prediction.std(ddof=0))))
    daily = pd.DataFrame(rows)
    reverse = {value: name for name, value in bank.metadata["models"].items()}
    daily["model"] = daily.model_id.map(reverse)
    summary = daily.groupby(["variant", "fold_id", "split", "model", "horizon_sessions"], sort=False).agg(
        mean_rank_ic=("rank_ic", "mean"), valid_sessions=("rank_ic", "count"),
        total_sessions=("rank_ic", "size"), mse=("mse", "mean"),
        zero_forecast_mse=("zero_forecast_mse", "mean"),
        mean_forecast_std=("forecast_cross_asset_std", "mean")).reset_index()
    # Overlapping horizon labels are not independent. No naive sqrt(T) t-stat.
    return daily, summary


def paired_variants(metrics, daily, bootstrap):
    keys = ["fold_id", "split", "model", "allocator", "horizon_sessions"]
    controls = metrics[metrics.variant == "control"]
    folds, summaries = [], []
    for variant in metrics.variant.unique():
        if variant == "control":
            continue
        candidates = metrics[(metrics.variant == variant) & (metrics.model != "none")]
        paired = candidates.merge(controls, on=keys, suffixes=("_candidate", "_control"),
                                  how="left", validate="one_to_one")
        if paired.sharpe_ratio_control.isna().any():
            raise ValueError("Missing control cases; cannot compare different evaluation samples.")
        paired["candidate"] = variant
        paired["reference"] = "control"
        paired["sharpe_delta"] = paired.sharpe_ratio_candidate - paired.sharpe_ratio_control
        folds.append(paired)
        for (model, allocator, horizon), rows in paired.groupby(["model", "allocator", "horizon_sessions"], sort=False):
            subset = daily[(daily.model == model) & (daily.allocator == allocator) & (daily.horizon_sessions == horizon)]
            join_keys = ["fold_id", "split", "date"]
            a = subset[subset.variant == variant][join_keys + ["net_return"]]
            b = subset[subset.variant == "control"][join_keys + ["net_return"]]
            matched = a.merge(b, on=join_keys, suffixes=("_candidate", "_control"),
                             how="outer", validate="one_to_one")
            if matched.isna().any().any():
                raise ValueError("Unequal strategy dates across feature ablations.")
            stats = block_bootstrap_excess_returns(matched.net_return_candidate.to_numpy(), matched.net_return_control.to_numpy(),
                block_size=max(bootstrap["block_size"], int(horizon)), num_samples=bootstrap["samples"],
                seed=bootstrap["seed"], groups=matched.fold_id.to_numpy())
            summaries.append(dict(candidate=variant, reference="control", model=model, allocator=allocator,
                horizon_sessions=int(horizon), mean_sharpe_delta=float(rows.sharpe_delta.mean()),
                annual_net_excess=stats["mean_excess_annualized"], ci_lower_95=stats["ci_lower_95"],
                ci_upper_95=stats["ci_upper_95"], p_value=stats["p_value_positive_excess"],
                scope="reused_inner_development_only", alpha_confirmed=False))
    summary = pd.DataFrame(summaries)
    if not summary.empty:
        summary["holm_p_value"] = _holm_adjust(summary.p_value.to_numpy())
    return (pd.concat(folds, ignore_index=True) if folds else pd.DataFrame()), summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-config", type=Path, default=ROOT / "configs/research/alpha_features.yaml")
    parser.add_argument("--prediction-config", type=Path, default=ROOT / "configs/research/prediction_bank.yaml")
    parser.add_argument("--allocation-config", type=Path, default=ROOT / "configs/research/signal_allocation.yaml")
    parser.add_argument("--research-protocol", type=Path, default=ROOT / "configs/core/quant_research_protocol.yaml")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--assets-config", type=Path, default=ROOT / "configs/core/assets.yaml")
    parser.add_argument("--env-config", type=Path, default=ROOT / "configs/core/env.yaml")
    parser.add_argument("--groups", default="vix,fama_french,blackout")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-ablation", action="store_true", help="Fit predictors and backtest inner after data readiness; default only builds input banks.")
    args = parser.parse_args(argv)
    groups = [value.strip() for value in args.groups.split(",") if value.strip()]
    if not groups or len(set(groups)) != len(groups) or set(groups) - GROUPS:
        parser.error("Choose unique groups: vix,fama_french,blackout.")
    protocol, digest = load_protocol(args.research_protocol)
    base = load_prediction_config(args.prediction_config)
    if "external_alpha" in base:
        parser.error("Runner expects a base prediction config; it resolves per-arm external features itself.")
    allocation = load_allocation_config(args.allocation_config)
    families = {"ridge_fixed", *base.get("baselines", {}), *base.get("neural", {}).get("models", [])}
    if not set(allocation["models"]).issubset(families) or allocation["horizons"] != base["horizons"]:
        parser.error("Allocation families/horizons must be present in the base prediction config.")
    output = args.output_dir or new_run_directory(ROOT / "models/runs", "alpha_features")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Research directory already contains artifacts; choose a new run.")
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(status="running", purpose="external_feature_research", protocol_hash=digest,
        outer_evaluated=False, test_evaluated=False, groups=groups, alpha_confirmed=False,
        source_hashes=file_hashes(list((ROOT / "src").rglob("*.py")) + [Path(__file__)], ROOT),
        config_hashes=file_hashes([args.alpha_config, args.prediction_config, args.allocation_config,
                                 args.research_protocol, args.assets_config, args.env_config], ROOT),
        data_hashes=file_hashes(args.raw_dir.glob("*.parquet"), ROOT))
    atomic_json(output / "manifest.json", manifest)
    try:
        pipeline = FeaturePipeline(assets_config_path=str(args.assets_config))
        processed, tensor = pipeline.run_from_raw(args.raw_dir, export_parquet=False, end_date=protocol["development_end"])
        closing = close_price_matrix(processed, tensor.tickers, tensor.dates)
        banks, statuses = build_alpha_context(tensor, closing, args.alpha_config, groups, output / "input_banks")
        manifest["external_inputs"] = {name: result[4] for name, result in banks.items()}
        print(statuses.to_string(index=False), flush=True)
        if not statuses.status.eq("ready").all():
            raise ExternalDataUnavailable("Selected external groups are missing usable historical data; see input_banks/data_status.csv.")
        calendar = pd.DatetimeIndex(pd.to_datetime(tensor.dates))
        required_end = max(int(calendar.searchsorted(pd.Timestamp(fold["inner_end"]), side="right"))
                           for fold in protocol["folds"]) - min(base["horizons"])
        common = augment_alpha_tensor(tensor, banks, groups, end_index=required_end)
        start = common.warmup_sessions
        manifest.update(common_warmup_sessions=start, common_ready_start=tensor.dates[start],
            last_required_feature_date=tensor.dates[required_end-1], feature_groups={name: banks[name][1] for name in groups})
        if not args.run_ablation:
            manifest.update(status="inputs_ready_not_trained")
            atomic_json(output / "manifest.json", manifest)
            print(f"Inputs ready. Use --run-ablation to fit OOF/inner predictors. Artifacts: {output}")
            return 0
        variants = {"control": [], **{name: [name] for name in groups}}
        if len(groups) > 1:
            variants["combined"] = groups
        all_metrics, all_daily, all_ic = [], [], []
        for name, selected in variants.items():
            resolved = copy.deepcopy(base)
            resolved["external_alpha"] = dict(config=str(args.alpha_config.resolve()), groups=selected,
                                             minimum_warmup_sessions=start)
            resolved_path = output / f"prediction_{name}.yaml"
            resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
            bank, directory = run_prediction_research(args.research_protocol, resolved_path, args.raw_dir,
                args.assets_config, output / name / "prediction_bank", development_only=True)
            if bank.metadata["data_hashes"] != manifest["data_hashes"]:
                raise ValueError("Raw snapshot changed during the feature ablation.")
            for group in selected:
                if bank.metadata["external_features"][group]["input_hashes"] != banks[group][4]["input_hashes"]:
                    raise ValueError(f"External snapshot changed during ablation: {group}")
            outcomes = pd.read_csv(directory / "labels.csv")
            daily_ic, summary_ic = rank_ic(bank, outcomes, name)
            atomic_csv(daily_ic, output / name / "daily_rank_ic.csv", index=False)
            atomic_csv(summary_ic, output / name / "rank_ic_by_fold.csv", index=False)
            all_ic.append(summary_ic)
            _, compared = compare_signal_allocators(directory, args.raw_dir, args.assets_config, args.env_config,
                args.allocation_config, output / name / "allocation", development_only=True)
            all_metrics.append(pd.read_csv(compared / "metrics_by_fold.csv").assign(variant=name))
            all_daily.append(pd.read_csv(compared / "daily_returns.csv").assign(variant=name))
            del bank, outcomes
        metrics, daily = pd.concat(all_metrics, ignore_index=True), pd.concat(all_daily, ignore_index=True)
        atomic_csv(metrics, output / "inner_metrics.csv", index=False)
        atomic_csv(daily, output / "inner_daily_returns.csv", index=False)
        atomic_csv(pd.concat(all_ic, ignore_index=True), output / "rank_ic_by_fold.csv", index=False)
        folds, pairs = paired_variants(metrics, daily, protocol["bootstrap"])
        atomic_csv(folds, output / "feature_contribution_by_fold.csv", index=False)
        atomic_csv(pairs, output / "feature_contribution_summary.csv", index=False)
        if (file_hashes(args.raw_dir.glob("*.parquet"), ROOT) != manifest["data_hashes"]
                or file_hashes(list((ROOT / "src").rglob("*.py")) + [Path(__file__)], ROOT) != manifest["source_hashes"]
                or file_hashes([args.alpha_config, args.prediction_config, args.allocation_config,
                                args.research_protocol, args.assets_config, args.env_config], ROOT) != manifest["config_hashes"]):
            raise ValueError("Source, data or config snapshot changed during the feature ablation.")
        manifest.update(status="completed", variants=variants, portfolio_gate_evaluated=False,
                        decision="INNER_RESEARCH_ONLY_NO_AUTOMATIC_PREDICTOR_LOCK")
    except ExternalDataUnavailable as exc:
        manifest.update(status="needs_data", reason=str(exc))
        atomic_json(output / "manifest.json", manifest)
        print(f"NEEDS_DATA: {exc}\nArtifacts: {output}", flush=True)
        return 2 if args.run_ablation else 0
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        atomic_json(output / "manifest.json", manifest)
        raise
    atomic_json(output / "manifest.json", manifest)
    print(f"Artifacts: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

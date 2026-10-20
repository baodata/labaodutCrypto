"""Common, causal forecast/risk/execution inputs for decision model comparisons."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.env.cost import resolve_transaction_cost_components
from src.evaluation.cost_aware_allocator import asset_bounds
from src.features.liquidity import build_liquidity_bank
from src.features.pipeline import FeaturePipeline
from src.risk.covariance import build_covariance_bank
from src.training.data_preparation import open_price_matrix, close_price_matrix
from src.training.env_factory import create_env
from src.utils.artifacts import atomic_json, atomic_csv, file_hashes
from src.utils.data_types import MarketDataTensor


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ForecastEpisode:
    start: int
    end: int  # exclusive; last session marks final NAV
    horizon: int
    forecasts: dict  # global date position -> (return vector, past fit scale)


class DecisionData:
    def __init__(self, bank, lock, allocation, raw_dir, assets_path, env_path, output):
        self.bank, self.lock, self.allocation = bank, lock, allocation
        self.protocol = bank.metadata["protocol"]
        self.env_path = Path(env_path)
        if file_hashes(Path(raw_dir).glob("*.parquet"), ROOT) != bank.metadata["data_hashes"]:
            raise ValueError("Data snapshot differs from the locked prediction bank.")
        for key, value in file_hashes([Path(assets_path)], ROOT).items():
            if bank.metadata["config_hashes"].get(key) != value:
                raise ValueError("Asset configuration differs from the locked bank.")
        pipeline = FeaturePipeline(assets_config_path=str(assets_path))
        processed, original = pipeline.run_from_raw(raw_dir, export_parquet=False,
            end_date=self.protocol["development_end"])
        if (original.tickers != lock["tickers"] or original.dates != bank.metadata["session_calendar"]
                or pipeline.parameters != bank.metadata["feature_parameters"]):
            raise ValueError("Market calendar/features/universe differ from the locked bank.")
        self.calendar = pd.DatetimeIndex(pd.to_datetime(original.dates))
        self.opening = open_price_matrix(processed, original.tickers)
        self.closing = close_price_matrix(processed, original.tickers, original.dates)
        self.environment = yaml.safe_load(self.env_path.read_text())["environment"]
        if self.environment.get("allow_short_selling", False):
            raise ValueError("Comparison requires long-only accounting.")
        self.cost_rate = resolve_transaction_cost_components(self.environment)[2]
        self.cash = self.protocol["execution"]["cash_ratio"]
        self.alpha = self.protocol["execution"]["action_alpha"]
        self.cap = float(self.environment.get("max_asset_weight", 1.))
        self.core, self.lower, self.upper = asset_bounds(len(original.tickers), self.cash,
            allocation["tilt_fraction"], self.cap)
        self.risk = build_covariance_bank(pd.DataFrame(self.closing, index=self.calendar,
            columns=original.tickers).pct_change(fill_method=None), **allocation["risk"], cache_dir=output / "banks")
        self.liquidity = build_liquidity_bank(processed, original.dates, original.tickers)
        # Policies never read the environment's market tensor. No raw features
        # or predictor hidden states enter any learned allocation branch.
        self.tensor = MarketDataTensor(np.zeros((len(original.dates), len(original.tickers), 1)),
            original.tickers, ["unused_market_placeholder"], original.dates)
        self.tensor.quant_context = dict(dates=original.dates, tickers=original.tickers,
            risk=self.risk, liquidity=self.liquidity,
            config={"impact": {"accounting": "cash_funded", "eta": 0.}, "execution": {"enabled": False}})
        assets = yaml.safe_load(Path(assets_path).read_text())
        assignments = {row["ticker"]: row["agent_group"] for row in assets["assets"]}
        names = list(assets["agent_groups"])
        if any(assignments.get(ticker) not in names for ticker in original.tickers):
            raise ValueError("Every asset must belong to exactly one configured agent group.")
        self.group_names = [name for name in names if any(assignments[t] == name for t in original.tickers)]
        self.groups = [[i for i, t in enumerate(original.tickers) if assignments[t] == name]
                       for name in self.group_names]
        self.group_prior = np.array([len(g)/len(original.tickers) for g in self.groups])
        atomic_csv(self.risk.diagnostics, output / "risk_diagnostics.csv", index=False)
        atomic_json(output / "resolved_execution.json", dict(environment=self.environment,
            allocation=allocation, execution=self.protocol["execution"], groups=dict(zip(self.group_names, self.groups)),
            sector_budget_prior=self.group_prior.tolist(), covariance=self.risk.metadata,
            liquidity=self.liquidity.metadata, state="forecast_return_and_fit_scale_full_hCov_close_weights_known_costs",
            price_timeline="Close_t_decision_Open_tplus1_execution_Close_tplus1_NAV",
            target_bounds="fixed_cash_and_per_asset_1N_core_plus_minus_tilt_cap",
            blended_weights_note="bounds_apply_to_policy_targets_not_initial_cash_ramp_or_passive_price_drift"))

    def episode(self, selection, split):
        fold = next(f for f in self.protocol["folds"] if f["name"] == selection["fold_id"])
        horizon = selection["horizon_sessions"]
        panel = self.bank.panel(model_id=selection["model_id"], fold_id=selection["fold_id"],
                               split=split, horizon=horizon)
        positions = self.calendar.get_indexer(pd.to_datetime(panel.index))
        if not len(positions) or np.any(np.diff(positions) != 1) or np.any(positions < 0):
            raise ValueError("Selected forecasts need nonempty contiguous session coverage.")
        rows = self.bank.predictions
        rows = rows[(rows.model_id == selection["model_id"]) & (rows.fold_id == selection["fold_id"])
                    & (rows.split == split) & (rows.horizon_sessions == horizon)]
        fits = rows.drop_duplicates("decision_date").set_index("decision_date").fit_id
        forecasts = {int(p): (values, max(float(self.bank.fits[fits.loc[date]]["prediction_scale"]), 1e-12))
                     for p, date, values in zip(positions, panel.index, panel.to_numpy())}
        if split == "train_oof":
            start, end = int(positions[0]), int(positions[-1])+2
            if self.calendar[end-1] > pd.Timestamp(fold["train_end"]):
                raise ValueError("OOF allocation rewards extend beyond train_end.")
        else:
            start = int(self.calendar.searchsorted(pd.Timestamp(fold[f"{split}_start"])))
            end = int(self.calendar.searchsorted(pd.Timestamp(fold[f"{split}_end"]), side="right"))
            if not np.array_equal(positions, np.arange(start, end-horizon)):
                raise ValueError("Forecast coverage must match the same horizon-purged evaluation tail.")
        if not self.risk.ready[positions].all() or not self.liquidity.ready[start:end-1].all():
            raise ValueError("Risk/liquidity not ready; no hidden fallback or unequal dates.")
        return ForecastEpisode(start, end, horizon, forecasts)

    def env(self, episode):
        a, b = episode.start, episode.end
        tensor = MarketDataTensor(self.tensor.tensor[a:b], self.tensor.tickers,
                                  self.tensor.feature_names, self.tensor.dates[a:b])
        return create_env(tensor, self.opening[a:b], self.env_path, 1., reward_scaling=1.,
            history_tensor=self.tensor, history_start_index=a, lookback_window=1,
            action_alpha=self.alpha, close_prices=self.closing[a:b])

    def state(self, episode, position, weights, matched_weights=None):
        prediction, scale = episode.forecasts[position]
        # Fixed unit conversions, no fitting over validation or outer state.
        matched_weights = weights if matched_weights is None else matched_weights
        values = np.r_[prediction*100, np.clip(prediction/scale, -10, 10),
            (episode.horizon*self.risk.covariance[position]*1e4).ravel(), weights, matched_weights,
            self.group_prior, [episode.horizon/20, self.cost_rate*100, self.alpha, self.cash]]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite forecast-only policy state.")
        return values.astype(np.float32)

    @property
    def core_target(self):
        return np.r_[self.core, self.cash]

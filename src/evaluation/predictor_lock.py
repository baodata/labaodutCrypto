"""Fold-local predictor selection using inner outcomes only, before allocation research."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.prediction_bank import PredictionBank
from src.utils.artifacts import atomic_json


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_predictor_lock(bank_dir, output, *, candidates, horizon=5):
    """Choose a family per fold; never pool later folds to choose earlier predictors.

    This is nested OFFLINE model selection. The winning family's historical OOF
    forecasts stay causal, but the family choice was not known at those past dates.
    The lock freezes specifications and phase-specific bank fits, not one global
    set of weights. Outer fits were refitted through inner_end by the bank builder.
    """
    output, bank_dir = Path(output), Path(bank_dir).resolve()
    if output.exists():
        raise ValueError("Predictor lock already exists; choose a new path.")
    bank = PredictionBank.load(bank_dir)
    if (type(horizon) is not int or horizon <= 0 or not candidates
            or len(candidates) != len(set(candidates))
            or not set(candidates).issubset(bank.metadata["models"])):
        raise ValueError("Specify unique bank families and a fixed positive horizon.")
    # Explicit offline access. Only inner rows enter any selection calculation.
    frame = bank.evaluation_frame_from_directory(bank_dir)
    frame = frame[(frame.split == "inner") & (frame.horizon_sessions == horizon)]
    selections = []
    for fold in bank.metadata["protocol"]["folds"]:
        scores, common_dates = [], None
        for family in candidates:
            model_id = bank.metadata["models"][family]
            rows = frame[(frame.fold_id == fold["name"]) & (frame.model_id == model_id)]
            if rows.empty or (rows.label_end_date > fold["inner_end"]).any():
                raise ValueError("Missing inner forecasts or labels extending past selection cutoff.")
            dates = sorted(rows.decision_date.unique())
            if common_dates is not None and dates != common_dates:
                raise ValueError("Candidate predictors need identical inner coverage.")
            common_dates = dates
            daily_ic, constant_predictions = [], 0
            for _, panel in rows.groupby("decision_date", sort=True):
                x = panel.prediction.rank().to_numpy()
                y = panel.realized_cross_sectional_excess_return.rank().to_numpy()
                if np.std(y) == 0:
                    continue  # identical outcome dates are excluded for every candidate
                if np.std(x) == 0:
                    constant_predictions += 1
                    daily_ic.append(0.)  # a constant forecast contains no ranking information
                else:
                    daily_ic.append(float(np.corrcoef(x, y)[0, 1]))
            if not daily_ic:
                raise ValueError("No nonconstant cross-sectional inner outcomes for selection.")
            scores.append(dict(family=family, model_id=model_id, mean_rank_ic=float(np.mean(daily_ic)),
                sessions=len(dates), scored_sessions=len(daily_ic), constant_prediction_sessions=constant_predictions,
                last_label_date=str(rows.label_end_date.max())))
        # Stable configured order breaks exact ties; no outer metric is consulted.
        winner = max(scores, key=lambda row: row["mean_rank_ic"])
        selections.append(dict(fold_id=fold["name"], family=winner["family"], model_id=winner["model_id"],
            horizon_sessions=horizon, selection_cutoff=fold["inner_end"],
            selected_mean_rank_ic=winner["mean_rank_ic"], positive_inner_ic=winner["mean_rank_ic"] > 0,
            candidates=scores))
    payload = dict(version="predictor-lock-v1", prediction_bank=str(bank_dir),
        bank_manifest_sha256=hashlib.sha256((bank_dir / "prediction_bank_manifest.json").read_bytes()).hexdigest(),
        selection_labels_sha256=hashlib.sha256((bank_dir / "labels.csv").read_bytes()).hexdigest(),
        protocol_hash=bank.metadata["protocol_hash"], tickers=bank.metadata["tickers"],
        rule="per_fold_max_inner_daily_spearman_fixed_horizon_constant_forecast_zero_IC_config_order_ties",
        candidates=list(candidates), horizon_sessions=horizon, selections=selections,
        selection_uses="inner_only", test_evaluated=False,
        purpose="exploratory_nested_offline_selection_not_independent_alpha_confirmation",
        weights_policy="bank_phase_specific_causal_fits_no_allocator_updates",
        historical_oof_family_selection_known_at="inner_end_not_at_historical_oof_dates")
    payload["lock_digest"] = digest(payload)
    atomic_json(output, payload)
    return payload


def load_predictor_lock(path, bank_dir=None):
    payload = json.loads(Path(path).read_text())
    checksum = payload.pop("lock_digest")
    if payload.get("version") != "predictor-lock-v1" or digest(payload) != checksum:
        raise ValueError("Predictor lock contents changed.")
    payload["lock_digest"] = checksum
    root = Path(bank_dir or payload["prediction_bank"]).resolve()
    if hashlib.sha256((root / "prediction_bank_manifest.json").read_bytes()).hexdigest() != payload["bank_manifest_sha256"]:
        raise ValueError("Prediction bank differs from the locked snapshot.")
    bank = PredictionBank.load(root)  # also checks forecasts, fits and neural weights
    if bank.metadata["protocol_hash"] != payload["protocol_hash"] or bank.metadata["tickers"] != payload["tickers"]:
        raise ValueError("Lock/bank protocol or ticker order mismatch.")
    expected = {f["name"]: f for f in bank.metadata["protocol"]["folds"]}
    if len(payload["selections"]) != len(expected) or {s["fold_id"] for s in payload["selections"]} != set(expected):
        raise ValueError("Lock must cover each fold exactly once.")
    for selection in payload["selections"]:
        fold = expected[selection["fold_id"]]
        if (selection["selection_cutoff"] != fold["inner_end"]
                or selection["horizon_sessions"] != payload["horizon_sessions"]
                or bank.metadata["models"].get(selection["family"]) != selection["model_id"]):
            raise ValueError("Invalid fold-local predictor selection.")
    return payload, bank

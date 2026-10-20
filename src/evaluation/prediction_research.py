"""Create causal Ridge forecasts for allocator train/inner/outer development.

Hyperparameters are fixed in a separate config, never selected on outer or
used retroactively to produce allocator training forecasts. The frozen
research protocol supplies temporal folds only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.labels import ExecutableLabelPanel, LABEL_DEFINITION, LABEL_TARGET, RETURN_BASIS
from src.evaluation.feature_signal import _fit_ridge
from src.evaluation.prediction_bank import PredictionBank, SCHEMA_VERSION
from src.evaluation.research_protocol import load_protocol
from src.features.pipeline import FeaturePipeline
from src.features.scaler import MarketFeatureScaler
from src.models.statistical_signals import rank_signal, calibrate_rank_signal
from src.training.data_preparation import close_price_matrix, open_price_matrix
from src.utils.artifacts import atomic_csv, atomic_json, file_hashes, new_run_directory


ROOT = Path(__file__).resolve().parents[2]


def load_prediction_config(path) -> dict:
    from src.utils.config import resolve_config_path
    resolved_path = resolve_config_path(path)
    with Path(resolved_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    expected = {"version", "model", "ridge_alpha", "features", "horizons",
                "min_train_sessions", "retrain_every_sessions"}
    if not isinstance(config, dict) or not expected.issubset(config) or set(config) - expected - {"baselines", "neural", "external_alpha"}:
        raise ValueError("Prediction config must explicitly specify exactly the supported fields.")
    if config["version"] != "prediction-research-v1" or config["model"] != "ridge_fixed":
        raise ValueError("Unsupported prediction research version/model.")
    if not np.isfinite(config["ridge_alpha"]) or config["ridge_alpha"] <= 0:
        raise ValueError("Fixed Ridge alpha must be finite and positive.")
    features, horizons = config["features"], config["horizons"]
    if (not isinstance(features, list) or not features
            or any(not isinstance(f, str) or not f for f in features) or len(set(features)) != len(features)):
        raise ValueError("Prediction features must be unique nonempty names.")
    if (not isinstance(horizons, list) or not horizons or len(set(horizons)) != len(horizons)
            or any(type(h) is not int or h <= 0 for h in horizons)):
        raise ValueError("Prediction horizons must be unique positive integer session counts.")
    for name in ("min_train_sessions", "retrain_every_sessions"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if config["min_train_sessions"] <= max(horizons):
        raise ValueError("min_train_sessions must exceed the longest forecast horizon.")
    baselines = config.get("baselines", {})
    if not isinstance(baselines, dict) or set(baselines) - {"momentum", "mean_reversion"}:
        raise ValueError("Only fixed momentum and mean_reversion baseline definitions are supported.")
    for name, spec in baselines.items():
        expected_direction = 1 if name == "momentum" else -1
        if (not isinstance(spec, dict) or set(spec) != {"feature", "direction"}
                or not isinstance(spec["feature"], str) or not spec["feature"]
                or type(spec["direction"]) is not int or spec["direction"] != expected_direction):
            raise ValueError(f"Invalid fixed-direction signal definition: {name}.")
    if "neural" in config:
        validate_neural_settings(config["neural"])
    if "external_alpha" in config:
        spec = config["external_alpha"]
        if (not isinstance(spec, dict) or set(spec) != {"config", "groups", "minimum_warmup_sessions"}
                or not isinstance(spec["config"], str) or not spec["config"]
                or not isinstance(spec["groups"], list)
                or len(set(spec["groups"])) != len(spec["groups"])
                or set(spec["groups"]) - {"vix", "fama_french", "blackout"}
                or type(spec["minimum_warmup_sessions"]) is not int or spec["minimum_warmup_sessions"] < 0):
            raise ValueError("Invalid external alpha feature specification.")
    return config


def validate_neural_settings(settings):
    from src.models.supervised_predictor import NEURAL_FAMILIES
    expected = {"models", "seed", "lookback", "hidden_dim", "epochs", "batch_size", "learning_rate",
                "weight_decay", "device", "cpu_threads", "graph"}
    if not isinstance(settings, dict) or set(settings) != expected:
        raise ValueError("Neural training/graph settings must be explicit.")
    if (not settings["models"] or len(set(settings["models"])) != len(settings["models"])
            or set(settings["models"]) - NEURAL_FAMILIES):
        raise ValueError("Neural models must be unique GRU/GNN-GRU families.")
    for name in ("lookback", "hidden_dim", "epochs", "batch_size", "cpu_threads"):
        if type(settings[name]) is not int or settings[name] <= 0:
            raise ValueError(f"neural.{name} must be a positive integer.")
    if type(settings["seed"]) is not int or settings["seed"] < 0:
        raise ValueError("Neural seed must be a nonnegative integer.")
    if (not np.isfinite([settings["learning_rate"], settings["weight_decay"]]).all()
            or settings["learning_rate"] <= 0 or settings["weight_decay"] < 0):
        raise ValueError("Invalid neural learning rate/weight decay.")
    if not isinstance(settings["device"], str) or not (settings["device"] == "cpu" or settings["device"].startswith("cuda")):
        raise ValueError("Supervised training supports CPU or CUDA devices.")
    graph = settings["graph"]
    if (not isinstance(graph, dict) or set(graph) != {"window", "threshold", "mode", "top_k"}
            or type(graph["window"]) is not int or graph["window"] < 2
            or not np.isfinite(graph["threshold"]) or not 0 <= graph["threshold"] <= 1
            or graph["mode"] not in {"threshold", "knn", "dynamic_threshold"}
            or type(graph["top_k"]) is not int or graph["top_k"] <= 0):
        raise ValueError("Invalid causal neural graph configuration.")


def build_ridge_prediction_bank(tensor, labels: ExecutableLabelPanel, protocol: dict, config: dict,
                                *, artifact_dir=None, assets_config=ROOT / "configs/core/assets.yaml",
                                development_only=False):
    """Build a reusable bank in memory; all label endpoints are purged by split.

    train_oof: expanding fit, refit at block starts, never uses inner labels.
    inner: frozen fit through train_end. outer: refit through inner_end.
    The first min_train_sessions ready observations have no OOF forecasts.
    """
    tensor.validate()
    if list(tensor.tickers) != labels.tickers or not pd.DatetimeIndex(pd.to_datetime(tensor.dates)).equals(labels.calendar):
        raise ValueError("Feature and label calendars/ticker orders must match exactly.")
    if protocol.get("evaluation_scope") != "validation":
        raise ValueError("Prediction research supports development/validation folds only.")
    calendar, start = labels.calendar, tensor.warmup_sessions
    if calendar[-1] > pd.Timestamp(protocol["development_end"]):
        raise ValueError("Input includes sessions beyond the development cutoff.")
    feature_indices = [tensor.feature_names.index(name) for name in config["features"]]
    features = tensor.tensor[:, :, feature_indices]
    date_strings = calendar.strftime("%Y-%m-%d").to_numpy()
    fits, pieces = {}, []
    # Configuration hash in the model ID prevents confusing different fixed-alpha banks.
    config_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    model_id = f"ridge_fixed_{config_digest[:12]}"
    model_ids = {"ridge_fixed": model_id, **{name: f"{name}_{config_digest[:12]}" for name in config.get("baselines", {})}}
    neural = config.get("neural")
    graph, graph_metadata = None, None
    if neural:
        validate_neural_settings(neural)
        if artifact_dir is None:
            raise ValueError("Neural banks require artifact_dir to persist fitted weights.")
        import torch
        torch.set_num_threads(neural["cpu_threads"])
        model_ids.update({name: f"{name}_{config_digest[:12]}" for name in neural["models"]})
        if "gnn_gru" in neural["models"]:
            from src.graph.dynamic_graph import build_dynamic_graph_sequence
            from src.graph.multi_relation_graph import GRAPH_RELATIONS
            graph_settings = neural["graph"]
            graph, mask = build_dynamic_graph_sequence(
                pd.DataFrame(labels.closing, index=calendar, columns=labels.tickers).pct_change(fill_method=None),
                tensor.dates, tensor.tickers, window_size=graph_settings["window"],
                threshold=graph_settings["threshold"], graph_mode=graph_settings["mode"],
                top_k=graph_settings["top_k"], sector_config_path=str(assets_config))
            graph *= mask
            graph_metadata = dict(graph_settings, relations=list(GRAPH_RELATIONS),
                                  alignment="through_Close_t_for_next_Open_execution",
                                  source="adjusted_close_returns_and_configured_sector_edges")

    def emit(prediction, provenance, fit_id, fitted_state, begin, end, horizon):
        fits[fit_id] = dict(provenance, **fitted_state)
        n = len(labels.tickers)
        frame = pd.DataFrame({
            "decision_date": np.repeat(date_strings[begin:end], n),
            "execution_date": np.repeat(date_strings[begin + 1:end + 1], n),
            "label_end_date": np.repeat(date_strings[begin + horizon:end + horizon], n),
            "ticker": np.tile(labels.tickers, end - begin), "prediction": prediction.reshape(-1),
            "fit_id": fit_id, "prediction_kind": "expected_cross_sectional_excess_return",
            "target": LABEL_TARGET, "label_definition": LABEL_DEFINITION, "return_basis": RETURN_BASIS,
        })
        for name, value in provenance.items():
            frame[name] = value
        pieces.append(frame)

    def bounds(begin, end):
        a = start if begin is None else int(calendar.searchsorted(pd.Timestamp(begin)))
        b = int(calendar.searchsorted(pd.Timestamp(end), side="right"))
        if (a < start or b - a <= max(config["horizons"])
                or b == 0 or calendar[b - 1] < pd.Timestamp(end) - pd.Timedelta(days=7)):
            raise ValueError("Insufficient ready history/session coverage for a configured fold.")
        return a, b

    def forecast(fold_id, split, horizon, cutoff, begin, end):
        # cutoff is exclusive: features and training label endpoints end by cutoff-1.
        if cutoff > begin or cutoff - start < config["min_train_sessions"] or end <= begin:
            raise ValueError("Insufficient or noncausal fit/prediction interval.")
        positions = labels.positions(start, cutoff, horizon)
        if not len(positions):
            raise ValueError("No purged training labels for this horizon.")
        scaler = MarketFeatureScaler().fit(features[start:cutoff], feature_names=config["features"])
        train_x = scaler.transform(features[positions]).astype(np.float64).reshape(-1, len(feature_indices))
        train_y = labels.returns(positions, horizon, excess=True).reshape(-1)
        coefficients, center, intercept = _fit_ridge(train_x, train_y, config["ridge_alpha"])
        x = scaler.transform(features[begin:end]).astype(np.float64)
        prediction = (x - center) @ coefficients + intercept
        prediction -= prediction.mean(axis=1, keepdims=True)
        fit_id = f"{fold_id}/{split}/h{horizon}/decision_{begin}"
        provenance = {
            "model_id": model_id, "fold_id": fold_id, "split": split, "horizon_sessions": horizon,
            "training_start": date_strings[start], "training_cutoff": date_strings[cutoff - 1],
            "last_training_label_date": date_strings[positions[-1] + horizon],
            "selection_cutoff": date_strings[cutoff - 1],
        }
        fitted_state = dict(model="ridge_fixed", ridge_alpha=config["ridge_alpha"],
            feature_names=config["features"], scaler_params=scaler.to_dict(),
            scaler_fit_end=date_strings[cutoff - 1], coefficients=coefficients.tolist(),
            feature_center=center.tolist(), intercept=intercept,
            training_decision_count=len(positions), training_asset_rows=len(train_y),
            cross_sectional_center_predictions=True, hyperparameter_selection="fixed_configuration_no_search",
            first_prediction_date=date_strings[begin], last_prediction_date=date_strings[end - 1])
        in_sample = (train_x - center) @ coefficients + intercept
        in_sample = in_sample.reshape(len(positions), len(labels.tickers))
        in_sample -= in_sample.mean(axis=1, keepdims=True)
        fitted_state["prediction_scale"] = max(float(in_sample.std()), 1e-12)
        emit(prediction, provenance, fit_id, fitted_state, begin, end, horizon)
        for name, spec in config.get("baselines", {}).items():
            index = tensor.feature_names.index(spec["feature"])
            train_scores = rank_signal(tensor.tensor[positions, :, index], spec["direction"])
            slope = calibrate_rank_signal(train_scores, train_y.reshape(train_scores.shape), config["ridge_alpha"])
            predictions = slope * rank_signal(tensor.tensor[begin:end, :, index], spec["direction"])
            state = dict(model=name, feature_names=[spec["feature"]], direction=spec["direction"],
                calibration_slope=slope, ridge_alpha=config["ridge_alpha"],
                prediction_scale=max(float((slope * train_scores).std()), 1e-12),
                hyperparameter_selection="fixed_configuration_no_search", scaler_fit_end=date_strings[cutoff - 1],
                training_decision_count=len(positions), training_asset_rows=len(train_y),
                first_prediction_date=date_strings[begin], last_prediction_date=date_strings[end - 1])
            emit(predictions, dict(provenance, model_id=model_ids[name]), f"{name}/{fit_id}", state, begin, end, horizon)
        if neural:
            from src.models.supervised_predictor import fit_neural, predict_neural, save_neural_checkpoint
            scaled_features = scaler.transform(features)
            target_panel = train_y.reshape(len(positions), len(labels.tickers))
            forecast_positions = np.arange(begin, end)
            for family in neural["models"]:
                print(f"  {family}: {split}, train={len(positions)} dates, forecast={end-begin}, h={horizon}", flush=True)
                model, state = fit_neural(scaled_features, positions, target_panel, ready_start=start,
                                         settings=neural, family=family, graph=graph)
                predictions = predict_neural(model, scaled_features, forecast_positions, ready_start=start,
                    settings=neural, graph=graph, target_scale=state["target_scale"])
                fit_predictions = predict_neural(model, scaled_features, positions, ready_start=start,
                    settings=neural, graph=graph, target_scale=state["target_scale"])
                state.update(feature_names=config["features"], tickers=tensor.tickers, scaler_params=scaler.to_dict(),
                    scaler_fit_end=date_strings[cutoff-1], prediction_scale=max(float(fit_predictions.std()), 1e-12),
                    training_decision_count=len(positions), training_asset_rows=len(train_y),
                    hyperparameter_selection="fixed_configuration_no_search", graph_config=graph_metadata,
                    first_prediction_date=date_strings[begin], last_prediction_date=date_strings[end-1])
                neural_fit_id = f"{family}/{fit_id}"
                weights_name = f"neural_models/{family}_{hashlib.sha256(neural_fit_id.encode()).hexdigest()[:20]}.pth"
                provenance_neural = dict(provenance, model_id=model_ids[family])
                save_neural_checkpoint(Path(artifact_dir) / weights_name, model,
                    json.loads(json.dumps(dict(provenance_neural, **state), default=str)))
                state["weights_artifact"] = weights_name
                emit(predictions, provenance_neural, neural_fit_id, state, begin, end, horizon)

    label_parts = []
    for horizon in config["horizons"]:
        label_parts.append(labels.frame(start, len(calendar), horizon))
        for fold in protocol["folds"]:
            _, train_end = bounds(None, fold["train_end"])
            inner_start, inner_end = bounds(fold["inner_start"], fold["inner_end"])
            outer_start, outer_end = bounds(fold["outer_start"], fold["outer_end"])
            if not (train_end <= inner_start < inner_end <= outer_start < outer_end):
                raise ValueError("Fold intervals must be chronological and disjoint.")
            first_oof = start + config["min_train_sessions"]
            if first_oof >= train_end - horizon:
                raise ValueError(f"No allocator train OOF history: {fold['name']}, horizon={horizon}.")
            print(f"Prediction bank: {fold['name']}, horizon={horizon} sessions", flush=True)
            for begin in range(first_oof, train_end - horizon, config["retrain_every_sessions"]):
                end = min(begin + config["retrain_every_sessions"], train_end - horizon)
                forecast(fold["name"], "train_oof", horizon, begin, begin, end)
            forecast(fold["name"], "inner", horizon, train_end, inner_start, inner_end - horizon)
            if not development_only:
                forecast(fold["name"], "outer", horizon, inner_end, outer_start, outer_end - horizon)

    metadata = dict(labels.metadata(), schema_version=SCHEMA_VERSION, target=LABEL_TARGET,
        prediction_kind="expected_cross_sectional_excess_return", prediction_config=config,
        prediction_config_digest=config_digest, session_calendar=date_strings.tolist(),
        model_ids=list(model_ids.values()), models=model_ids, evaluation_scope="validation", test_evaluated=False,
        hyperparameter_selection="fixed_configuration_no_search",
        phase_policy={"train_oof": "expanding_past_only", "inner": "frozen_at_train_end",
                      "outer": "refit_through_inner_end_fixed_hyperparameters"},
        allocator_training_split="train_oof", warmup_sessions=start,
        calibration_status="not_established_by_schema_or_in_sample_fit",
        neural_graph=graph_metadata,
        protocol=protocol, row_order="horizon_fold_phase_decision_configured_ticker_order")
    if development_only:
        metadata.update(outer_evaluated=False, development_only=True)
        metadata["phase_policy"].pop("outer")
    bank = PredictionBank(pd.concat(pieces, ignore_index=True), metadata, fits)
    return bank, pd.concat(label_parts, ignore_index=True)


def run_prediction_research(protocol_path, prediction_config, raw_dir, assets_config, output_dir=None,
                            *, development_only=False):
    """Materialize a new immutable development bank, without backtesting test."""
    from src.utils.config import resolve_config_path
    protocol_path = resolve_config_path(protocol_path)
    prediction_config = resolve_config_path(prediction_config)
    assets_config = resolve_config_path(assets_config)
    protocol, protocol_digest = load_protocol(protocol_path)
    config = load_prediction_config(prediction_config)
    output = Path(output_dir) if output_dir is not None else new_run_directory(ROOT / "models/runs", "prediction_bank")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory already contains artifacts; choose a new run.")
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(status="running", test_evaluated=False, protocol_hash=protocol_digest,
        period_status=protocol["period_status"], data_cutoff=protocol["development_end"],
        source_hashes=file_hashes(list((ROOT / "src").rglob("*.py")) + [ROOT / "scripts/build_prediction_bank.py"], ROOT),
        config_hashes=file_hashes([Path(protocol_path), Path(prediction_config), Path(assets_config)], ROOT),
        data_hashes=file_hashes(Path(raw_dir).glob("*.parquet"), ROOT))
    if development_only:
        manifest.update(outer_evaluated=False, development_only=True)
    atomic_json(output / "manifest.json", manifest)
    try:
        pipeline = FeaturePipeline(assets_config_path=str(assets_config))
        processed, tensor = pipeline.run_from_raw(raw_dir, export_parquet=False, end_date=protocol["development_end"])
        for ticker in tensor.tickers:
            columns = {str(column).lower().replace(" ", "_") for column in processed[ticker].columns}
            if not columns.intersection({"adj_close", "adjusted_close"}):
                raise ValueError(f"{ticker}: prediction-bank total-return labels require Adj Close; no price-only fallback.")
        labels = ExecutableLabelPanel(tensor.dates, tensor.tickers,
            open_price_matrix(processed, tensor.tickers), close_price_matrix(processed, tensor.tickers, tensor.dates))
        external_metadata = None
        if "external_alpha" in config:
            from src.features.alpha_context import build_alpha_context, augment_alpha_tensor, ROOT as FEATURE_ROOT
            spec = config["external_alpha"]
            external_path = Path(spec["config"])
            if not external_path.is_absolute():
                external_path = FEATURE_ROOT / external_path
            phase = "inner" if development_only else "outer"
            required_end = max(int(labels.calendar.searchsorted(pd.Timestamp(fold[f"{phase}_end"]), side="right"))
                               for fold in protocol["folds"]) - min(config["horizons"])
            if spec["groups"]:
                banks, statuses = build_alpha_context(tensor, labels.closing, external_path, spec["groups"], output / "external_features")
                tensor = augment_alpha_tensor(tensor, banks, spec["groups"],
                    minimum_warmup=spec["minimum_warmup_sessions"], end_index=required_end)
                config["features"] = list(dict.fromkeys(list(config["features"]) +
                    [name for group in spec["groups"] for name in banks[group][1]]))
                external_metadata = {group: banks[group][4] for group in spec["groups"]}
            else:
                tensor.warmup_sessions = max(tensor.warmup_sessions, spec["minimum_warmup_sessions"])
                external_metadata = {"control_common_warmup_sessions": tensor.warmup_sessions}
            manifest["external_config_hashes"] = file_hashes([external_path], ROOT)
            manifest["external_features"] = external_metadata
        bank, outcomes = build_ridge_prediction_bank(tensor, labels, protocol, config,
                                                    artifact_dir=output, assets_config=assets_config,
                                                    development_only=development_only)
        bank.metadata.update(source_hashes=manifest["source_hashes"], data_hashes=manifest["data_hashes"],
            config_hashes=manifest["config_hashes"], protocol_hash=protocol_digest,
            feature_parameters=pipeline.parameters, period_status=protocol["period_status"])
        if external_metadata is not None:
            bank.metadata.update(external_features=external_metadata, external_config_hashes=manifest["external_config_hashes"],
                                 predictor_feature_names=config["features"])
        if development_only:
            manifest.update(outer_evaluated=False, development_only=True)
        bank.save(output)
        atomic_csv(outcomes, output / "labels.csv", index=False)
        atomic_json(output / "labels_manifest.json", dict(labels.metadata(),
            horizons=config["horizons"], offline_evaluation_only=True,
            data_hashes=manifest["data_hashes"], protocol_hash=protocol_digest,
            artifact_hashes=file_hashes([output / "labels.csv"], output.resolve())))
        manifest.update(status="completed", rows=len(bank.predictions), fits=len(bank.fits),
                        model_ids=bank.metadata["model_ids"], allocator_training_split="train_oof")
        atomic_json(output / "manifest.json", manifest)
    except BaseException as exc:
        from src.data.external_alpha import ExternalDataUnavailable
        status = "needs_data" if isinstance(exc, ExternalDataUnavailable) else "failed"
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else status, error=str(exc))
        atomic_json(output / "manifest.json", manifest)
        raise
    print(f"Prediction bank saved: {output}", flush=True)
    return bank, output

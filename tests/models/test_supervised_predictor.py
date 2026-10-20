"""Supervised predictor masks, graph controls and bank provenance."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.data.labels import ExecutableLabelPanel
from src.evaluation.prediction_bank import PredictionBank, fitted_weight_paths
from src.evaluation.prediction_research import build_ridge_prediction_bank, load_prediction_config
from src.features.scaler import MarketFeatureScaler
from src.models.supervised_predictor import SupervisedReturnPredictor, load_neural_checkpoint, predict_neural
from src.utils.data_types import MarketDataTensor


@pytest.mark.parametrize("family", ["gru", "gnn_gru", "gnn_gru_no_edges"])
def test_left_padded_history_matches_unpadded_real_sequence(family):
    torch.manual_seed(7)
    model = SupervisedReturnPredictor(2, 8, family).eval()
    real = torch.randn(2, 3, 3, 2)
    padded = torch.cat((torch.full((2, 2, 3, 2), 1e6), real), dim=1)
    graph = torch.rand(2, 3, 3, 3, 3)
    padded_graph = torch.cat((torch.rand(2, 2, 3, 3, 3), graph), dim=1)
    with torch.no_grad():
        expected = model(real, torch.ones(2, 3, dtype=torch.bool), graph)
        actual = model(padded, torch.tensor([[False, False, True, True, True]]).expand(2, -1), padded_graph)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual.sum(dim=1), torch.zeros(2), atol=1e-6, rtol=0)


def test_zero_edge_control_has_same_common_initialization_and_output_as_gru():
    torch.manual_seed(19)
    gru = SupervisedReturnPredictor(2, 8, "gru")
    torch.manual_seed(19)
    graph_model = SupervisedReturnPredictor(2, 8, "gnn_gru")
    torch.manual_seed(19)
    control = SupervisedReturnPredictor(2, 8, "gnn_gru_no_edges")
    x = torch.randn(2, 4, 3, 2)
    mask = torch.ones(2, 4, dtype=torch.bool)
    expected = gru(x, mask)
    torch.testing.assert_close(graph_model(x, mask, torch.zeros(2, 4, 3, 3, 3)), expected, atol=0, rtol=0)
    torch.testing.assert_close(control(x, mask), expected, atol=0, rtol=0)
    with_edges = graph_model(x, mask, torch.rand(2, 4, 3, 3, 3))
    assert not torch.allclose(with_edges, expected)
    with_edges.square().mean().backward()
    assert all(transform.weight.grad is not None and torch.isfinite(transform.weight.grad).all()
               for transform in graph_model.relation_transforms)


@pytest.mark.parametrize("mask", [[[False, False, False]], [[True, False, True]]])
def test_invalid_padding_is_rejected(mask):
    model = SupervisedReturnPredictor(2, 8, "gru")
    with pytest.raises(ValueError, match="left-padding"):
        model(torch.zeros(1, 3, 2, 2), torch.tensor(mask))


def toy_bank_inputs(tmp_path):
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2018-01-01", periods=60).strftime("%Y-%m-%d").tolist()
    tensor = MarketDataTensor(rng.normal(size=(60, 3, 2)).astype(np.float32), ["A", "B", "C"], ["x", "z"], dates, warmup_sessions=1)
    close = 100 * np.cumprod(1 + rng.normal(0, .01, (60, 3)), axis=0)
    opening = close * (1 + rng.normal(0, .002, (60, 3)))
    labels = ExecutableLabelPanel(dates, tensor.tickers, opening, close)
    protocol = dict(evaluation_scope="validation", development_end=dates[-1], folds=[dict(name="fold", train_end=dates[24],
        inner_start=dates[25], inner_end=dates[39], outer_start=dates[40], outer_end=dates[59])])
    config = dict(version="prediction-research-v1", model="ridge_fixed", ridge_alpha=1., features=["x", "z"],
        horizons=[2], min_train_sessions=8, retrain_every_sessions=16,
        neural=dict(models=["gru", "gnn_gru", "gnn_gru_no_edges"], seed=7, lookback=4, hidden_dim=8,
            epochs=1, batch_size=8, learning_rate=.001, weight_decay=.0001, device="cpu", cpu_threads=1,
            graph=dict(window=4, threshold=.5, mode="knn", top_k=1)))
    assets = tmp_path / "assets.yaml"
    assets.write_text(yaml.safe_dump(dict(assets=[dict(ticker=t, sector="same") for t in tensor.tickers])))
    return tensor, labels, protocol, config, assets


def test_neural_bank_is_oof_and_future_edits_do_not_change_past_predictions(tmp_path):
    tensor, labels, protocol, config, assets = toy_bank_inputs(tmp_path)
    bank, _ = build_ridge_prediction_bank(tensor, labels, protocol, config, artifact_dir=tmp_path/"original", assets_config=assets)
    changed_tensor = deepcopy(tensor)
    changed_tensor.tensor[14:] += 100
    opening, closing = labels.opening.copy(), labels.closing.copy()
    opening[14:] *= 1.1
    closing[14:] *= 1.1
    changed_labels = ExecutableLabelPanel(tensor.dates, tensor.tickers, opening, closing)
    changed, _ = build_ridge_prediction_bank(changed_tensor, changed_labels, protocol, config,
                                            artifact_dir=tmp_path/"changed", assets_config=assets)
    before = bank.predictions.decision_date < tensor.dates[14]
    np.testing.assert_array_equal(bank.predictions.loc[before, "prediction"], changed.predictions.loc[before, "prediction"])
    assert set(bank.metadata["models"]) == {"ridge_fixed", "gru", "gnn_gru", "gnn_gru_no_edges"}
    assert (bank.predictions.last_training_label_date < bank.predictions.decision_date).all()
    # The no-edge model is a capacity control with unused relation parameters.
    for phase in ("train_oof", "inner", "outer"):
        gru = bank.panel(model_id=bank.metadata["models"]["gru"], fold_id="fold", split=phase, horizon=2)
        control = bank.panel(model_id=bank.metadata["models"]["gnn_gru_no_edges"], fold_id="fold", split=phase, horizon=2)
        np.testing.assert_array_equal(gru.to_numpy(), control.to_numpy())


def test_checkpoint_reload_and_bank_hash_include_neural_weights(tmp_path):
    tensor, labels, protocol, config, assets = toy_bank_inputs(tmp_path)
    directory = tmp_path / "bank"
    bank, _ = build_ridge_prediction_bank(tensor, labels, protocol, config, artifact_dir=directory, assets_config=assets)
    bank.save(directory)
    loaded = PredictionBank.load(directory)
    assert len(loaded.predictions) == len(bank.predictions)
    fit = next(fit for fit in loaded.fits.values() if fit.get("model") == "gru" and fit["split"] == "outer")
    model, payload = load_neural_checkpoint(directory / fit["weights_artifact"])
    scaler = MarketFeatureScaler.from_dict(payload["scaler_params"])
    selected = loaded.predictions[loaded.predictions.fit_id == next(k for k,v in loaded.fits.items() if v is fit)]
    dates = sorted(selected.decision_date.unique())
    positions = labels.calendar.get_indexer(pd.to_datetime(dates))
    prediction = predict_neural(model, scaler.transform(tensor.tensor), positions, ready_start=tensor.warmup_sessions,
        settings=payload["training_settings"], target_scale=payload["target_scale"])
    expected = loaded.panel(model_id=fit["model_id"], fold_id="fold", split="outer", horizon=2).to_numpy()
    original = bank.panel(model_id=fit["model_id"], fold_id="fold", split="outer", horizon=2).to_numpy()
    np.testing.assert_array_equal(prediction, original)
    # CSV parsing can differ by one float64 ULP; checkpoint replay itself is exact.
    np.testing.assert_allclose(prediction, expected, atol=1e-15, rtol=1e-12)
    with (directory / fit["weights_artifact"]).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="hashes differ"):
        PredictionBank.load(directory)


def test_neural_profile_preserves_allocation_settings_and_statistical_features():
    from src.evaluation.signal_allocation import load_allocation_config
    base = load_allocation_config("configs/research/signal_allocation.yaml")
    neural = load_allocation_config("configs/test/signal_allocation_neural.yaml")
    assert {k:v for k,v in base.items() if k != "models"} == {k:v for k,v in neural.items() if k != "models"}
    base_prediction = load_prediction_config("configs/research/prediction_bank.yaml")
    neural_prediction = load_prediction_config("configs/test/prediction_bank_neural.yaml")
    assert base_prediction["features"] == neural_prediction["features"]
    assert base_prediction["baselines"] == neural_prediction["baselines"]


def test_neural_weights_cannot_escape_bank_directory(tmp_path):
    with pytest.raises(ValueError, match="inside"):
        fitted_weight_paths(tmp_path, {"fit": {"weights_artifact": "../outside.pth"}})

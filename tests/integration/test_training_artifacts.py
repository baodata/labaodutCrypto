from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.training import train_ppo
from src.training.ppo_core import PPOUpdater
from src.utils.config import AssetsConfig


@pytest.fixture
def training_fixture(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    dates = pd.bdate_range("2014-01-02", periods=355)
    rng = np.random.default_rng(19)
    config = AssetsConfig()
    for ticker in config.get_tickers() + [config.get_benchmark_ticker()]:
        close = 100 * np.exp(np.cumsum(rng.normal(0.0002, .008, len(dates))))
        opening = close * np.exp(rng.normal(0, .003, len(dates)))
        frame = pd.DataFrame({
            "date": dates, "open": opening, "high": np.maximum(opening, close) * 1.002,
            "low": np.minimum(opening, close) * .998, "close": close, "adj_close": close,
            "volume": rng.integers(100000, 1000000, len(dates)),
        })
        frame.to_parquet(raw / f"{ticker}.parquet", index=False)
    model = tmp_path / "model.yaml"
    model.write_text(yaml.safe_dump({
        "gnn": {"architecture": "GCN", "hidden_channels": 4, "out_channels": 4, "heads": 1,
                "correlation_window": 5, "graph_mode": "knn", "top_k": 2},
        "rl_agent": {"learning_rate": .0001, "target_kl": 100., "gamma": .95, "dirichlet_kappa": 200., "entropy_coef": 0.},
    }))
    flags = ["--raw-dir", str(raw), "--model-config", str(model),
             "--train-end", dates[309].strftime("%Y-%m-%d"),
             "--val-start", dates[310].strftime("%Y-%m-%d"),
             "--val-end", dates[334].strftime("%Y-%m-%d"),
             "--test-start", dates[335].strftime("%Y-%m-%d"),
             "--lookback", "2", "--hidden-dim", "8", "--min-epochs", "0", "--patience", "0",
             "--ppo-iters", "1", "--batch-size", "4", "--episodes-per-update", "1",
             "--window-min-days", "5", "--window-max-days", "5"]
    return tmp_path, flags


@pytest.mark.parametrize("encoder,architecture", [("gru", "gcn"), ("gnn", "gcn"), ("gnn", "gat")])
def test_real_training_path_embeds_scaler_and_saves_reproducible_state(training_fixture, encoder, architecture):
    tmp, flags = training_fixture
    run = tmp / f"run_{encoder}"
    args = train_ppo.build_parser().parse_args(flags + ["--encoder", encoder, "--gnn-architecture", architecture, "--epochs", "1", "--run-dir", str(run)])
    model, context = train_ppo.train(args)
    last = torch.load(run / "checkpoints/last.pth", weights_only=False)
    assert last["epoch"] == 1
    assert last["scaler_params"]["is_fitted"]
    assert last["scheduler_state_dict"] and last["rng_state"]
    assert last["run_manifest"]["data_hashes"]
    assert len(last["training_state"]["training_rows"]) == 1
    assert pd.read_csv(run / "training_log.csv").epoch.tolist() == [1.0]
    assert context["split"].train.dates[0] > context["full_tensor"].dates[252]
    assert (run / "scaler.json").exists()
    assert args.scaler_output.parent == run


def test_interrupted_run_resumes_identical_model_optimizer_and_rng(training_fixture, monkeypatch):
    tmp, flags = training_fixture
    common = flags + ["--encoder", "mlp", "--epochs", "3"]
    uninterrupted = tmp / "uninterrupted"
    train_ppo.train(train_ppo.build_parser().parse_args(common + ["--run-dir", str(uninterrupted)]))

    original = PPOUpdater.update
    calls = 0

    def interrupt_on_second_update(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt()
        return original(self, *args, **kwargs)

    interrupted = tmp / "interrupted"
    monkeypatch.setattr(PPOUpdater, "update", interrupt_on_second_update)
    with pytest.raises(KeyboardInterrupt):
        train_ppo.train(train_ppo.build_parser().parse_args(common + ["--run-dir", str(interrupted)]))
    assert pd.read_csv(interrupted / "training_log.csv").epoch.tolist() == [1.0]
    assert '"interrupted"' in (interrupted / "manifest.json").read_text()
    monkeypatch.setattr(PPOUpdater, "update", original)
    train_ppo.train(train_ppo.build_parser().parse_args(common + ["--run-dir", str(interrupted), "--resume", str(interrupted / "checkpoints/last.pth")]))
    a = torch.load(uninterrupted / "checkpoints/last.pth", weights_only=False)
    b = torch.load(interrupted / "checkpoints/last.pth", weights_only=False)
    for key in a["model_state_dict"]:
        torch.testing.assert_close(a["model_state_dict"][key], b["model_state_dict"][key], atol=0, rtol=0)
    assert a["scheduler_state_dict"] == b["scheduler_state_dict"]
    torch.testing.assert_close(a["rng_state"]["torch"], b["rng_state"]["torch"])


@pytest.mark.parametrize("encoder", ["mlp", "gru", "lstm", "gnn"])
def test_null_entrypoint_runs_all_encoders_without_training(training_fixture, encoder):
    tmp, flags = training_fixture
    args = train_ppo.build_parser().parse_args(flags + ["--encoder", encoder, "--run-dir", str(tmp / f"null_{encoder}")])
    result = train_ppo.evaluate_untrained_null(args, [7, 17])
    assert len(result) == 2
    assert np.isfinite(result.sharpe_ratio).all()
    assert set(result.checkpoint_status) == {"untrained"}
    assert result.initial_concentration_mean.gt(0).all()


def test_reported_sharpe_delta_and_bootstrap_share_the_same_baseline(training_fixture):
    tmp, flags = training_fixture
    args = train_ppo.build_parser().parse_args(flags + ["--encoder", "mlp", "--epochs", "1",
        "--matched-cash-ratio", "0.07", "--bootstrap-block-size", "5", "--bootstrap-samples", "99",
        "--run-dir", str(tmp / "metrics")])
    model, context = train_ppo.train(args)
    metrics = train_ppo.evaluate_test(args, model, context)
    baseline = metrics.loc["PPO", "cbb_baseline"]
    assert metrics.loc["PPO", "sharpe_delta_vs_matched_baseline"] == pytest.approx(
        metrics.loc["PPO", "sharpe_ratio"] - metrics.loc[baseline, "sharpe_ratio"])
    daily = pd.read_csv(args._run_dir / "daily_returns.csv")
    assert metrics.loc["PPO", "cbb_excess_annual_vs_1n_cash"] == pytest.approx(
        (daily.PPO - daily[baseline]).mean() * 252)
    assert metrics.loc["PPO", "matched_baseline_cash_ratio"] == 0.07
    assert '"completed"' in (args._run_dir / "manifest.json").read_text()


def test_resume_rejects_changed_inputs_before_overwriting_artifacts(training_fixture):
    tmp, flags = training_fixture
    run = tmp / "reproducible"
    args = train_ppo.build_parser().parse_args(flags + ["--encoder", "mlp", "--epochs", "1", "--run-dir", str(run)])
    train_ppo.train(args)
    manifest = (run / "manifest.json").read_bytes()
    scaler = (run / "scaler.json").read_bytes()
    frame_path = args.raw_dir / "AAPL.parquet"
    frame = pd.read_parquet(frame_path)
    frame.loc[0, "volume"] += 1
    frame.to_parquet(frame_path, index=False)
    resume = train_ppo.build_parser().parse_args(flags + ["--encoder", "mlp", "--epochs", "1",
        "--resume", str(run / "checkpoints/last.pth")])
    with pytest.raises(ValueError, match="data_hashes"):
        train_ppo.train(resume)
    assert (run / "manifest.json").read_bytes() == manifest
    assert (run / "scaler.json").read_bytes() == scaler


def test_actual_multiseed_cli_writes_numeric_summary_and_separate_metadata(training_fixture):
    tmp, flags = training_fixture
    run = tmp / "ensemble"
    train_ppo.main(flags + ["--encoder", "mlp", "--epochs", "1", "--seeds", "7,17",
        "--bootstrap-block-size", "5", "--bootstrap-samples", "99", "--run-dir", str(run)])
    summary = pd.read_csv(run / "metrics_seeds_summary.csv", header=[0, 1], index_col=0)
    metadata = pd.read_csv(run / "metrics_seeds_metadata.csv")
    assert ("sharpe_delta_vs_matched_baseline", "mean") in summary.columns
    assert set(metadata.seed) == {7, 17}
    assert "checkpoint_status" in metadata.columns
    for seed in (7, 17):
        assert (run / f"seed_{seed}/checkpoints/last.pth").exists()
        assert (run / f"seed_{seed}/scaler.json").exists()

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from scripts.evaluate_equal_weight_tilt import make_tilt_policy
from scripts.evaluate_tilt_robustness import validate_gate
from scripts.analyze_feature_signal import rank_ic_table
from src.data.split import TemporalSplitter
from src.env.marl_env import MultiAgentTradingEnv
from src.env.trading_env import TradingEnv
from src.evaluation.baselines import BuyAndHoldBaseline, EqualWeightBaseline
from src.evaluation.backtester import DeterministicBacktester
from src.evaluation.metrics import block_bootstrap_excess_returns
from src.features.pipeline import FeaturePipeline
from src.features.risk import calculate_risk_features
from src.features.scaler import MarketFeatureScaler
from src.graph.correlation import compute_rolling_correlation
from src.graph.dynamic_graph import build_dynamic_graph_sequence
from src.graph.multi_relation_graph import MultiRelationGraphBuilder
from src.models.actor_critic import build_actor_critic
from src.models.market_cache import prepare_market_cache
from src.models.policy_adapter import CheckpointPolicy, relation_history_from_correlations
from src.training.ppo_core import PPOBuffer, PPOUpdater, graph_minibatch
from src.training.train_ppo import observation_vector, graph_tensors_from_obs, history_mask_from_obs
from src.utils.data_types import MarketDataTensor


def tensor_fixture(length=30, n=3, f=2):
    return MarketDataTensor(np.random.default_rng(42).normal(size=(length, n, f)).astype(np.float32),
                            ["AAPL", "MSFT", "NVDA"][:n], [f"f{i}" for i in range(f)],
                            pd.bdate_range("2024-01-01", periods=length).strftime("%Y-%m-%d").tolist())


@pytest.mark.parametrize("encoder", ["mlp", "gru", "lstm", "gnn"])
def test_public_policy_diagnostics_works_for_all_encoders(encoder):
    model = build_actor_critic(encoder=encoder, n_assets=3, n_features=2, lookback=2, action_dim=4, hidden_dim=8,
                              gnn_architecture="gcn", gnn_hidden_channels=4, gnn_out_channels=4)
    graph = {"adjacency_history": torch.zeros(2, 3, 3, 3), "edge_mask_history": torch.zeros(2, 3, 3, 3)} if encoder == "gnn" else None
    stats = model.diagnostics(torch.randn(16), graph)
    assert all(np.isfinite(v) for v in stats.values())


@pytest.mark.parametrize("encoder", ["gru", "lstm"])
def test_policy_rejects_dropout_even_with_multiple_layers(encoder):
    with pytest.raises(ValueError, match="dropout=0"):
        build_actor_critic(encoder=encoder, n_assets=3, n_features=2, lookback=2, action_dim=4,
                          rnn_layers=2, rnn_dropout=0.5)


def test_history_date_offset_is_rejected(trading_env_config_path):
    full = tensor_fixture()
    part = MarketDataTensor(full.tensor[5:15], full.tickers, full.feature_names, full.dates[5:15])
    with pytest.raises(ValueError, match="Dates"):
        TradingEnv(part, np.ones((10, 3)), history_tensor=full, history_start_index=6,
                   config_path=str(trading_env_config_path))


def test_pipeline_rejects_same_length_but_reversed_dates():
    pipeline = FeaturePipeline()
    frame = pd.DataFrame(np.zeros((3, len(pipeline.FEATURE_NAMES))), index=pd.bdate_range("2024-01-01", periods=3), columns=pipeline.FEATURE_NAMES)
    with pytest.raises(ValueError, match="Lịch"):
        pipeline.build_market_data_tensor({"AAPL": frame, "MSFT": frame.iloc[::-1]})


def test_pipeline_rejects_invalid_ohlcv_before_filling_features():
    frame = pd.DataFrame({"open": [100., 300.], "high": [101., 101.], "low": [99., 99.], "close": [100., 100.], "volume": [100., 100.]}, index=pd.bdate_range("2024-01-01", periods=2))
    with pytest.raises(ValueError, match="Validation failed"):
        FeaturePipeline().process_asset(frame)


def test_warmup_is_excluded_from_split_and_scaler():
    tensor = tensor_fixture(length=30)
    tensor.warmup_sessions = 5
    tensor.tensor[:5] = 1000
    split = TemporalSplitter(train_end=tensor.dates[14], val_start=tensor.dates[15], val_end=tensor.dates[24], test_start=tensor.dates[25]).split_market_data_tensor(tensor)
    assert split.train.dates[0] == tensor.dates[5]
    scaler = MarketFeatureScaler().fit(tensor)
    assert scaler.means["f0"] == pytest.approx(tensor.tensor[5:, :, 0].mean())
    assert MarketFeatureScaler.from_dict(scaler.to_dict()).to_dict() == scaler.to_dict()


def test_downside_measure_includes_frequency_and_zero_threshold():
    returns = np.array([0., -0.1, -0.1, 0.1, 0.1])
    close = pd.Series(100 * np.cumprod(1 + returns))
    risk = calculate_risk_features(close, close * 1.01, close * 0.99)
    assert risk.downside_vol_20d.iloc[-1] == pytest.approx(np.sqrt(0.02 / 4))


def test_graph_missing_ticker_and_order_are_rejected():
    returns = pd.DataFrame(np.random.default_rng(3).normal(size=(20, 2)), index=pd.bdate_range("2024-01-01", periods=20), columns=["AAPL", "MSFT"])
    with pytest.raises(ValueError, match="thiếu ticker"):
        build_dynamic_graph_sequence(returns, returns.index, ["AAPL", "MISSING"], window_size=5)
    corr = compute_rolling_correlation(returns, 5)
    with pytest.raises(ValueError, match="Thứ tự ticker"):
        MultiRelationGraphBuilder(["MSFT", "AAPL"], corr)


def test_centered_bootstrap_distinguishes_null_test_from_descriptive_tail():
    positive = block_bootstrap_excess_returns(np.full(100, 0.001), np.zeros(100), num_samples=999, block_size=10)
    negative = block_bootstrap_excess_returns(np.full(100, -0.001), np.zeros(100), num_samples=999, block_size=10)
    assert positive["p_value_positive_excess"] == pytest.approx(0.001)
    assert negative["p_value_positive_excess"] == 1.0
    assert "bootstrap_fraction_excess_le_0" in positive
    with pytest.raises(ValueError, match="cùng độ dài"):
        block_bootstrap_excess_returns(np.zeros(5), np.zeros(6))


def test_manifest_score_weights_and_orientation_change_actual_orders():
    features = np.array([[[0.1, 0.5], [0.9, 0.5]]], dtype=np.float32)
    kwargs = dict(features=features, feature_indices={"rsi_14": 0, "mom_12_1": 1}, n_assets=2, cash_ratio=0., tilt_fraction=.5, rebalance_every=1, no_trade_l1=0.)
    momentum = make_tilt_policy(**kwargs, rsi_orientation=1, score_weights=(0., 1.))({}, {"step": 0})
    contrarian = make_tilt_policy(**kwargs, rsi_orientation=-1, score_weights=(0., 1.))({}, {"step": 0})
    assert momentum.target_weights[1] > momentum.target_weights[0]
    assert contrarian.target_weights[1] < contrarian.target_weights[0]


@pytest.mark.parametrize("baseline_type", [EqualWeightBaseline, BuyAndHoldBaseline])
def test_standalone_baseline_and_env_backtest_use_identical_engine(baseline_type, trading_env_config_path):
    tensor = tensor_fixture(n=2)
    prices = np.column_stack((100 * 1.01 ** np.arange(30), 100 * .99 ** np.arange(30)))
    close = prices * np.array([1.002, 0.998])
    baseline = baseline_type(prices, 2, transaction_cost_rate=.001, risk_free_rate_annual=.02)
    env = TradingEnv(tensor, prices, close_prices=close, config_path=str(trading_env_config_path))
    expected = DeterministicBacktester(env).run_strategy(baseline).daily_net_returns
    np.testing.assert_array_equal(baseline.run(close_prices=close), expected)


def test_marl_requires_all_actions_and_exposes_normalized_histories(tmp_path, trading_env_config_path):
    assets = tmp_path / "assets.yaml"
    assets.write_text(yaml.safe_dump({"assets": [{"ticker": "AAPL", "agent_group": "tech"}, {"ticker": "MSFT", "agent_group": "other"}]}))
    tensor = tensor_fixture(n=2)
    graphs = np.zeros((30, 3, 2, 2), dtype=np.float32)
    env = TradingEnv(tensor, np.ones((30, 2)), config_path=str(trading_env_config_path), lookback_window=3,
                     graph_adjacency_sequence=graphs, graph_edge_mask_sequence=graphs)
    marl = MultiAgentTradingEnv(env, {"tech": {}, "other": {}, "empty": {}}, tensor.tickers, str(assets))
    obs, _ = marl.reset()
    assert set(obs) == {"tech", "other"}
    for observation in obs.values():
        assert observation["market_history"].shape == (3, 1, 2)
        assert observation["graph_adjacency_history"].shape == (3, 3, 1, 1)
        assert np.isclose(observation["portfolio_weights"].sum() + observation["cash_ratio"][0], 1.)
    with pytest.raises(ValueError, match="tất cả"):
        marl.step({"tech": np.array([1., 0.])})
    with pytest.raises(ValueError, match="Action"):
        marl.step({"tech": np.array([np.nan, 0.]), "other": np.array([1., 0.])})
    obs, *_ = marl.step({"tech": np.array([1., 0.]), "other": np.array([1., 0.])})
    assert np.isclose(env.portfolio.asset_weights.sum(), 1.)


@pytest.mark.parametrize("architecture", ["gcn", "gat"])
def test_indexed_graph_buffer_and_market_cache_match_dense_and_keep_gradients(trading_env_config_path, architecture):
    torch.manual_seed(11)
    tensor = tensor_fixture()
    returns = pd.DataFrame(tensor.tensor[:, :, 0], index=pd.to_datetime(tensor.dates), columns=tensor.tickers)
    adjacency, mask = build_dynamic_graph_sequence(returns, tensor.dates, tensor.tickers, window_size=5, graph_mode="knn", top_k=2)
    model = build_actor_critic(encoder="gnn", n_assets=3, n_features=2, lookback=3, action_dim=4, hidden_dim=8, gnn_architecture=architecture, gnn_hidden_channels=4, gnn_out_channels=4)
    env = TradingEnv(tensor, np.ones((30, 3)), config_path=str(trading_env_config_path), lookback_window=3, graph_adjacency_sequence=adjacency, graph_edge_mask_sequence=mask)
    cache = prepare_market_cache(model, env, batch_size=7)
    buffer = PPOBuffer(3, 22, 4, graph_dim=3, graph_lookback=3, graph_relations=3, graph_adjacency_sequence=adjacency, graph_edge_mask_sequence=mask, history_lookback=3)
    obs, _ = env.reset()
    for day in range(3):
        state = torch.as_tensor(observation_vector(obs))
        graph = graph_tensors_from_obs(obs, torch.device("cpu"))
        history_mask = history_mask_from_obs(obs, torch.device("cpu"))
        with torch.no_grad():
            dense = model.act(state, deterministic=True, graph=graph, history_mask=history_mask)
            cached = model.act(state, deterministic=True, graph={"market_embedding": cache.get(model, day)}, history_mask=history_mask)
            np.testing.assert_allclose(cached[0].numpy(), dense[0].numpy(), rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(cached[2], dense[2], rtol=1e-5, atol=1e-6)
            action, logp, value = model.act(state, graph={"market_embedding": cache.get(model, day)}, history_mask=history_mask)
            evaluated_logp, _, evaluated_value = model.evaluate(state, action, graph=graph, history_mask=history_mask)
            torch.testing.assert_close(logp.reshape(-1), evaluated_logp.reshape(-1), rtol=1e-5, atol=1e-4)
            torch.testing.assert_close(value.reshape(-1), evaluated_value.reshape(-1), rtol=1e-5, atol=1e-6)
        buffer.store(state.numpy(), action.numpy(), .01 * (day + 1), value.item(), logp.item(), graph_day_index=day, history_mask=obs["history_mask"])
        obs, *_ = env.step(action.numpy())
    buffer.finish_path()
    batch = buffer.get()
    gathered = graph_minibatch(batch, torch.tensor([0, 2]))
    assert gathered["adjacency_history"].shape == (2, 3, 3, 3, 3)
    assert torch.count_nonzero(gathered["edge_mask_history"][0, :2]) == 0
    assert buffer.graph_adjacency_history_buf is None
    updater = PPOUpdater(target_kl=100., entropy_coef=0.)
    stats = updater.update(model, torch.optim.Adam(model.parameters(), lr=1e-3), batch, train_iters=1, batch_size=3)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.graph_encoders[0].parameters())
    with pytest.raises(RuntimeError, match="đã cũ"):
        cache.get(model, 0)
    assert abs(stats["FirstKL"]) < 1e-4


def test_live_relation_adapter_and_embedded_scaler_reproduce_checkpoint(tmp_path):
    tensor = tensor_fixture(length=5, n=2)
    builder = MultiRelationGraphBuilder(tensor.tickers, None, graph_mode="knn", top_k=1)
    matrices = np.tile(np.array([[1., -.8], [-.8, 1.]]), (3, 1, 1))
    graph_obs = relation_history_from_correlations(builder, matrices, tensor.tickers)
    config = dict(encoder="gnn", n_assets=2, n_features=2, lookback=3, action_dim=3, hidden_dim=8, gnn_architecture="gcn", gnn_hidden_channels=4, gnn_out_channels=4)
    model = build_actor_critic(**config)
    scaler = MarketFeatureScaler().fit(tensor)
    obs = {"market_history": tensor.tensor[:3], "portfolio_weights": np.array([0., 0.]), "cash_ratio": np.array([1.]), **graph_obs}
    path = tmp_path / "checkpoint.pth"
    torch.save({"model_state_dict": model.state_dict(), "model_config": config, "scaler_params": scaler.to_dict(), "tickers": tensor.tickers, "feature_names": tensor.feature_names, "lookback": 3}, path)
    policy = CheckpointPolicy(path, tickers=tensor.tickers, feature_names=tensor.feature_names)
    prepared = dict(obs, market_history=scaler.transform(obs["market_history"]))
    with torch.no_grad():
        expected = model.act(torch.as_tensor(observation_vector(prepared)), deterministic=True, graph=graph_tensors_from_obs(prepared, torch.device("cpu")))[0].numpy()
    np.testing.assert_allclose(policy(obs), expected, atol=1e-7)
    assert np.all(graph_obs["graph_adjacency_history"][:, 1, 0, 1] > 0)


def test_rank_ic_execution_labels_use_next_open_and_purge_the_boundary():
    n, length = 3, 30
    features = np.tile(np.arange(n, dtype=np.float32)[None, :, None], (length, 1, 1))
    close = np.full((length, n), 100.)
    opening = np.tile([110., 100., 90.], (length, 1))
    dates = pd.bdate_range("2024-01-01", periods=length).strftime("%Y-%m-%d").tolist()
    result = rank_ic_table(features, ["signal"], opening, close, dates, 0, 25, horizons=(5,), bootstrap_samples=99)
    executable = result[result.target == "forward_open_to_close"].iloc[0]
    assert executable.mean_rank_ic == pytest.approx(1)
    assert executable.n_dates == 20
    assert executable.label_purge_sessions == 5
    assert result[result.target == "forward_close_to_close"].iloc[0].n_dates == 0
    assert executable.block_ci_lower_95 == pytest.approx(1)


@pytest.mark.parametrize("field,value", [("rebalance_every_sessions", 0), ("initial_action_alpha", float("nan")), ("rsi_orientation", "unknown")])
def test_frozen_gate_rejects_invalid_parameters(field, value):
    with open("configs/core/tilt_robustness_gate.yaml", encoding="utf-8") as stream:
        gate = yaml.safe_load(stream)
    gate["candidate"][field] = value
    with pytest.raises(ValueError):
        validate_gate(gate)

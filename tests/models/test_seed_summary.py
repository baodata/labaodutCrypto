import numpy as np
import pandas as pd
import pytest

from src.evaluation.seed_summary import summarize_seed_metrics
from src.training import train_ppo


def metrics_for_seed(seed):
    return pd.DataFrame({
        "sharpe_ratio": [1.0 if seed == 7 else 2.0, 0.5],
        "sharpe_delta_vs_matched_baseline": [0.1 if seed == 7 else 0.3, np.nan],
        "checkpoint_epoch": [4 if seed == 7 else 9, np.nan],
        "checkpoint_status": ["trained" if seed == 7 else "untrained_fallback", "none"],
    }, index=pd.Index(["PPO", "Baseline"], name="Strategy"))


def test_numeric_seed_summary_preserves_checkpoint_metadata():
    summary, metadata = summarize_seed_metrics({7: metrics_for_seed(7), 17: metrics_for_seed(17)})
    assert summary.loc["PPO", ("sharpe_ratio", "mean")] == pytest.approx(1.5)
    assert summary.loc["PPO", ("sharpe_ratio", "std")] == pytest.approx(np.sqrt(0.5))
    assert summary.loc["PPO", ("sharpe_delta_vs_matched_baseline", "mean")] == pytest.approx(0.2)
    assert "checkpoint_status" not in summary.columns.get_level_values(0)
    assert "checkpoint_epoch" not in summary.columns.get_level_values(0)
    ppo_metadata = metadata[metadata.strategy == "PPO"].set_index("seed")
    assert ppo_metadata.loc[7, "checkpoint_epoch"] == 4
    assert ppo_metadata.loc[17, "checkpoint_status"] == "untrained_fallback"


def test_multiseed_entrypoint_writes_summary_and_metadata_without_training(monkeypatch, tmp_path):
    monkeypatch.setattr(train_ppo, "train", lambda args: (None, {"seed": args.seed}))
    monkeypatch.setattr(train_ppo, "evaluate_test", lambda args, model, context: metrics_for_seed(context["seed"]))
    train_ppo.main(["--seeds", "7,17", "--metrics-output", str(tmp_path / "metrics.csv"), "--run-dir", str(tmp_path / "ensemble")])
    summary = pd.read_csv(tmp_path / "metrics_seeds_summary.csv", header=[0, 1], index_col=0)
    metadata = pd.read_csv(tmp_path / "metrics_seeds_metadata.csv")
    assert summary.loc["PPO", ("sharpe_ratio", "mean")] == pytest.approx(1.5)
    assert set(metadata.seed) == {7, 17}
    assert "checkpoint_status" in metadata


def test_duplicate_seeds_are_rejected_before_training(monkeypatch):
    monkeypatch.setattr(train_ppo, "train", lambda args: pytest.fail("Training must not start"))
    with pytest.raises(ValueError, match="trùng"):
        train_ppo.main(["--seeds", "7,7"])

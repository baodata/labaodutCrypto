"""Compatibility and dependency boundaries of the modular PPO entry point."""

import ast
import importlib
from pathlib import Path

import pandas as pd

from src.training import train_ppo
from src.training.multi_seed import train_multi_seed


MODULE_FUNCTIONS = {
    "src.training.cli": ("build_parser",),
    "src.training.env_factory": ("create_env",),
    "src.training.data_preparation": (
        "_total_return_adjustment_factor", "open_price_matrix", "close_price_matrix", "prepare_data",
    ),
    "src.evaluation.policy_evaluator": (
        "action_for_model", "evaluate_policy", "evaluate_policy_with_actions",
        "policy_weight_summary", "metric_summary", "evaluate_equal_weight",
    ),
    "src.evaluation.ppo_evaluation": ("save_or_append_metrics", "evaluate_test", "_evaluate_test"),
    "src.training.run_artifacts": (
        "runtime_versions", "configure_run_artifacts", "record_effective_training_config", "finish_run_manifest",
    ),
    "src.training.null_evaluation": ("evaluate_untrained_null", "_evaluate_untrained_null"),
    "src.training.ppo_trainer": ("train", "_train"),
    "src.training.multi_seed": ("train_multi_seed",),
}


def test_legacy_entrypoint_reexports_the_relocated_functions():
    for module_name, names in MODULE_FUNCTIONS.items():
        module = importlib.import_module(module_name)
        for name in names:
            assert getattr(train_ppo, name) is getattr(module, name)


def test_shared_modules_do_not_depend_on_the_cli_entrypoint():
    for module_name in MODULE_FUNCTIONS:
        module = importlib.import_module(module_name)
        tree = ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "src.training.train_ppo"
                if node.module == "src.training":
                    assert all(alias.name != "train_ppo" for alias in node.names)
            elif isinstance(node, ast.Import):
                assert all(alias.name != "src.training.train_ppo" for alias in node.names)


def test_multiseed_runner_accepts_train_and_evaluation_callbacks(tmp_path):
    args = train_ppo.build_parser().parse_args([
        "--seeds", "7,17", "--run-dir", str(tmp_path / "ensemble"),
        "--metrics-output", str(tmp_path / "metrics.csv"),
    ])
    calls = []

    def train(run_args):
        calls.append((run_args.seed, run_args.run_dir.name))
        return None, {"seed": run_args.seed}

    def evaluate(run_args, model, context):
        assert context["seed"] == run_args.seed
        return pd.DataFrame({
            "sharpe_ratio": [float(run_args.seed)],
            "checkpoint_epoch": [1],
            "checkpoint_status": ["trained"],
        }, index=pd.Index(["PPO"], name="Strategy"))

    train_multi_seed(args, train_fn=train, evaluate_fn=evaluate)
    assert calls == [(7, "seed_7"), (17, "seed_17")]
    assert args.seed == 7
    summary = pd.read_csv(tmp_path / "metrics_seeds_summary.csv", header=[0, 1], index_col=0)
    assert summary.loc["PPO", ("sharpe_ratio", "mean")] == 12.0
    metadata = pd.read_csv(tmp_path / "metrics_seeds_metadata.csv")
    assert metadata.seed.tolist() == [7, 17]

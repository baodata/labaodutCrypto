"""End-to-end single-agent PPO training and historical-period evaluation.

Run from the repository root with::

    python -m src.training.train_ppo --reward-beta 2.0

The policy receives engineered, train-normalized features. Raw OHLCV prices are
used to build those features and to simulate order execution in TradingEnv.

CLI compatibility entry point. Implementations live in the focused modules
imported below; existing scripts may keep importing these functions here."""

from __future__ import annotations

import argparse
import copy
import sys
import time
import warnings
import shutil
import random
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import TemporalSplitter
from src.env.cost import resolve_transaction_cost_components
from src.env.trading_env import TradingEnv
from src.evaluation.backtester import BacktestResult, DeterministicBacktester
from src.evaluation.baselines import BuyAndHoldBaseline, CashBaseline, EqualWeightBaseline
from src.evaluation.metrics import FinancialMetrics, block_bootstrap_excess_returns
from src.evaluation.seed_summary import summarize_seed_metrics
from src.features.pipeline import FeaturePipeline
from src.features.scaler import MarketFeatureScaler
from src.graph.dynamic_graph import build_dynamic_graph_sequence
from src.graph.multi_relation_graph import GRAPH_RELATIONS
from src.models.actor_critic import build_actor_critic
from src.models.policy_adapter import observation_vector, graph_tensors_from_obs, history_mask_from_obs
from src.models.market_cache import prepare_market_cache
from src.training.logger import TrainerLogger
from src.training.ppo_core import PPOBuffer, PPOUpdater
from src.training.config import (
    DEFAULT_TRAINING_CONFIG,
    effective_training_config,
    resolve_training_config,
)
from src.utils.config import AssetsConfig
from src.utils.data_types import MarketDataTensor
from src.utils.artifacts import atomic_csv, atomic_json, file_hashes, new_run_directory, rng_state, restore_rng_state
from src.utils.progress import PhaseProgress

# Backward-compatible function imports.
from src.training.cli import build_parser as build_parser
from src.training.env_factory import create_env as create_env
from src.training.data_preparation import _total_return_adjustment_factor as _total_return_adjustment_factor
from src.training.data_preparation import open_price_matrix as open_price_matrix
from src.training.data_preparation import close_price_matrix as close_price_matrix
from src.training.data_preparation import prepare_data as prepare_data
from src.evaluation.policy_evaluator import action_for_model as action_for_model
from src.evaluation.policy_evaluator import evaluate_policy as evaluate_policy
from src.evaluation.policy_evaluator import evaluate_policy_with_actions as evaluate_policy_with_actions
from src.evaluation.policy_evaluator import policy_weight_summary as policy_weight_summary
from src.evaluation.policy_evaluator import metric_summary as metric_summary
from src.evaluation.policy_evaluator import evaluate_equal_weight as evaluate_equal_weight
from src.evaluation.ppo_evaluation import save_or_append_metrics as save_or_append_metrics
from src.evaluation.ppo_evaluation import evaluate_test as evaluate_test
from src.evaluation.ppo_evaluation import _evaluate_test as _evaluate_test
from src.training.run_artifacts import runtime_versions as runtime_versions
from src.training.run_artifacts import configure_run_artifacts as configure_run_artifacts
from src.training.run_artifacts import record_effective_training_config as record_effective_training_config
from src.training.run_artifacts import finish_run_manifest as finish_run_manifest
from src.training.null_evaluation import evaluate_untrained_null as evaluate_untrained_null
from src.training.null_evaluation import _evaluate_untrained_null as _evaluate_untrained_null
from src.training.ppo_trainer import train as train
from src.training.ppo_trainer import _train as _train
from src.training.multi_seed import train_multi_seed
from src.evaluation.validation_evaluator import evaluate_validation


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    resolve_training_config(args)
    if args.untrained_null_seeds:
        if args.seeds:
            raise ValueError("Dùng riêng --untrained-null-seeds hoặc --seeds.")
        null_seeds = [
            int(value.strip())
            for value in args.untrained_null_seeds.split(",")
            if value.strip()
        ]
        evaluate_untrained_null(args, null_seeds)
        return
    if not args.seeds:
        model, context = train(args)
        evaluator = evaluate_validation if args.evaluation_scope == "validation" else evaluate_test
        evaluator(args, model, context)
        return

    evaluator = evaluate_validation if args.evaluation_scope == "validation" else evaluate_test
    train_multi_seed(args, train_fn=train, evaluate_fn=evaluator)


if __name__ == "__main__":
    main()

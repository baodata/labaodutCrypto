"""Resolve PPO run settings once: explicit CLI > YAML > fallback defaults."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TRAINING_CONFIG = Path(__file__).resolve().parents[2] / "configs/core/training.yaml"

# (YAML key, Namespace attribute, value type, fallback). Architecture and loss
# hyperparameters retain their existing model.yaml/CLI resolution.
SETTINGS = {
    "training": (
        ("epochs", "epochs", int, 50),
        ("ppo_iters", "ppo_iters", int, 10),
        ("episodes_per_update", "episodes_per_update", int, 32),
        ("batch_size", "batch_size", int, 256),
        ("microbatch_size", "microbatch_size", int, None),
        ("window_min_days", "window_min_days", int, 126),
        ("window_max_days", "window_max_days", int, 252),
    ),
    "early_stopping": (
        ("min_epochs", "min_epochs", int, 10),
        ("patience", "patience", int, 10),
    ),
    "checkpoint_selection": (
        ("metric", "selection_metric", str, "sharpe"),
        ("window", "selection_window", int, 5),
        ("margin", "selection_margin", float, 0.0),
    ),
    "evaluation": (
        ("scope", "evaluation_scope", str, "validation"),
        ("train_eval_every", "train_eval_every", int, 5),
        ("bootstrap_block_size", "bootstrap_block_size", int, 20),
        ("bootstrap_samples", "bootstrap_samples", int, 10000),
        ("matched_cash_ratio", "matched_cash_ratio", float, None),
    ),
    "runtime": (
        ("cpu_threads", "cpu_threads", int, 4),
        ("cache_batch_size", "cache_batch_size", int, 64),
        ("cache_temporal_batch_size", "cache_temporal_batch_size", int, 32),
        ("progress_every", "progress_every", float, 10.0),
    ),
}


def _validate(value: Any, name: str, kind: type) -> Any:
    if value is None and name in {"microbatch_size", "matched_cash_ratio"}:
        return None
    if kind is int:
        valid = type(value) is int
    elif kind is float:
        valid = type(value) in {int, float} and math.isfinite(value)
    else:
        valid = isinstance(value, str)
    if not valid:
        raise ValueError(f"Cấu hình {name} phải có kiểu {kind.__name__} hợp lệ: {value!r}")
    if name == "evaluation_scope":
        if value not in {"validation", "test"}:
            raise ValueError("evaluation_scope phải là validation hoặc test.")
    elif name == "selection_metric":
        if value not in {"sharpe", "cumulative_return", "excess_sharpe", "excess_return"}:
            raise ValueError(f"selection_metric không hợp lệ: {value!r}")
    elif name == "matched_cash_ratio":
        if not 0 <= value < 1:
            raise ValueError("matched_cash_ratio phải thuộc [0, 1).")
    elif name in {"patience", "min_epochs", "selection_margin"}:
        if value < 0:
            raise ValueError(f"{name} phải không âm.")
    elif value <= 0:
        raise ValueError(f"{name} phải lớn hơn 0.")
    return float(value) if kind is float else value


def resolve_training_config(args: argparse.Namespace) -> dict[str, Any]:
    """Fill unset CLI arguments, including for callers using train(args) directly.

    Resolution is idempotent so an ensemble can copy the resolved namespace
    without interpreting its YAML-derived values as fresh CLI overrides.
    """
    from src.evaluation.research_protocol import apply_research_protocol
    apply_research_protocol(args)
    if hasattr(args, "_training_config_inputs"):
        return args._training_config_inputs
    raw_training_config = Path(getattr(args, "training_config", DEFAULT_TRAINING_CONFIG))
    if not raw_training_config.exists():
        from src.utils.config import resolve_config_path
        raw_training_config = resolve_config_path(raw_training_config)
    args.training_config = raw_training_config
    with args.training_config.open(encoding="utf-8") as stream:
        configured = yaml.safe_load(stream)
    if configured is None:
        configured = {}
    if not isinstance(configured, dict):
        raise ValueError("training.yaml phải là mapping theo nhóm cấu hình.")
    unknown = set(configured) - set(SETTINGS)
    if unknown:
        raise ValueError(f"Nhóm cấu hình training không hỗ trợ: {sorted(unknown, key=str)}")

    resolved: dict[str, Any] = {}
    for section, fields in SETTINGS.items():
        values = configured.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"Nhóm {section} phải là mapping.")
        unknown = set(values) - {key for key, _, _, _ in fields}
        if unknown:
            raise ValueError(f"Khóa không hỗ trợ trong {section}: {sorted(unknown, key=str)}")
        resolved[section] = {}
        for key, name, kind, fallback in fields:
            override = getattr(args, name, None)
            value = values.get(key, fallback) if override is None else override
            resolved[section][key] = _validate(value, name, kind)

    # Apply only after the entire file has passed validation.
    for section, fields in SETTINGS.items():
        for key, name, _, _ in fields:
            setattr(args, name, resolved[section][key])
    args._training_config_inputs = resolved
    return resolved


def effective_training_config(args: argparse.Namespace, **overrides: Any) -> dict[str, Any]:
    """Snapshot settings after data/device-dependent auto values are known."""
    resolved = copy.deepcopy(resolve_training_config(args))
    known = set()
    for section, fields in SETTINGS.items():
        for key, name, kind, _ in fields:
            if name in overrides:
                resolved[section][key] = _validate(overrides[name], name, kind)
                known.add(name)
    if set(overrides) != known:
        raise ValueError(f"Unknown effective training settings: {set(overrides) - known}")
    return resolved

"""Run provenance, configuration snapshots and resume validation."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from pathlib import Path
from typing import Any
import copy
import platform

import numpy as np
import torch

from src.training.cli import build_parser
from src.training.config import effective_training_config
from src.training.config import resolve_training_config
from src.utils.artifacts import atomic_json
from src.utils.artifacts import file_hashes
from src.utils.artifacts import new_run_directory

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "torch", "torch-geometric", "gymnasium"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def configure_run_artifacts(args, kind="ppo"):
    resolve_training_config(args)
    if hasattr(args, "_run_dir"):
        return
    if min(args.cpu_threads, args.cache_batch_size, args.cache_temporal_batch_size) <= 0 or (args.microbatch_size is not None and args.microbatch_size <= 0):
        raise ValueError("Thread count và batch sizes phải lớn hơn 0.")
    if not np.isfinite(args.progress_every) or args.progress_every <= 0:
        raise ValueError("--progress-every phải hữu hạn và lớn hơn 0.")
    if torch.device(args.device).type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    if args.resume is not None and args.run_dir is None:
        run_dir = args.resume.resolve().parent.parent
    elif args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        if run_dir.exists() and any(run_dir.iterdir()) and args.resume is None:
            raise ValueError("run-dir đã có artifact; chọn thư mục mới hoặc dùng --resume.")
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = new_run_directory(PROJECT_ROOT / "models/runs", f"{kind}_{args.encoder}_seed{args.seed}")
    hashes = {
        "source_hashes": file_hashes(list((PROJECT_ROOT / "src").rglob("*.py")) + list((PROJECT_ROOT / "scripts").glob("*.py")), PROJECT_ROOT),
        "data_hashes": file_hashes(args.raw_dir.glob("*.parquet"), PROJECT_ROOT),
        "config_hashes": file_hashes([
            args.model_config, args.env_config, args.training_config,
            Path(getattr(args, "assets_config", PROJECT_ROOT / "configs/assets.yaml")),
        ], PROJECT_ROOT),
    }
    if args.resume is not None:
        previous = torch.load(args.resume, map_location="cpu", weights_only=False)
        for key, current in hashes.items():
            if previous.get("run_manifest", {}).get(key) != current:
                raise ValueError(f"Resume yêu cầu cùng {key}; artifact cũ được giữ nguyên.")
        if previous.get("run_manifest", {}).get("runtime") != runtime_versions():
            raise ValueError("Resume yêu cầu cùng phiên bản Python/thư viện để giữ quỹ đạo train.")
        if previous.get("run_manifest", {}).get("training_config_inputs") != args._training_config_inputs:
            raise ValueError("Resume yêu cầu cùng cấu hình training sau khi ghi đè CLI.")
        args._resume_payload = previous
    args._run_dir = run_dir
    defaults = build_parser().parse_args([])
    paths = {
        "checkpoint": run_dir / "checkpoints/best.pth",
        "training_log": run_dir / "training_log.csv", "metrics_output": run_dir / "metrics.csv",
        "null_output": run_dir / "untrained_null_validation.csv",
        "scaler_output": run_dir / "scaler.json", "processed_output": run_dir / "features.parquet",
        "tensorboard_dir": run_dir / "tensorboard",
    }
    for name, path in paths.items():
        if getattr(args, name) == getattr(defaults, name):
            setattr(args, name, path)
    args._manifest = {
        "status": "running", "kind": kind, "run_dir": str(run_dir),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith("_")},
        "runtime": runtime_versions(),
        "training_config_inputs": copy.deepcopy(args._training_config_inputs),
        **hashes,
    }
    record_effective_training_config(args)


def record_effective_training_config(args, **overrides) -> dict[str, Any]:
    """Persist resolved settings, including automatic values once available."""
    resolved = effective_training_config(args, **overrides)
    args._manifest["resolved_training_config"] = resolved
    atomic_json(args._run_dir / "training_config.json", resolved)
    atomic_json(args._run_dir / "manifest.json", args._manifest)
    return resolved


def finish_run_manifest(args, status, **fields):
    args._manifest.update(status=status, **fields)
    atomic_json(args._run_dir / "manifest.json", args._manifest)

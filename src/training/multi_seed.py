"""Sequential multi-seed experiment orchestration and aggregation. Owner: B."""

from __future__ import annotations

from pathlib import Path
import copy

import pandas as pd

from src.evaluation.seed_summary import summarize_seed_metrics
from src.training.cli import build_parser
from src.training.config import resolve_training_config
from src.utils.artifacts import atomic_json
from src.utils.artifacts import new_run_directory

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def train_multi_seed(args, *, train_fn=None, evaluate_fn=None) -> None:
    """Train and summarize seeds; callbacks allow alternative trainers/evaluators."""
    resolve_training_config(args)
    if train_fn is None:
        from src.training.ppo_trainer import train as train_fn
    if evaluate_fn is None:
        if args.evaluation_scope == "validation":
            from src.evaluation.validation_evaluator import evaluate_validation as evaluate_fn
        else:
            from src.evaluation.ppo_evaluation import evaluate_test as evaluate_fn

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) < 2:
        raise ValueError("--seeds cần ít nhất 2 seed, ví dụ: --seeds 7,17,27.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds không được chứa seed trùng lặp.")
    if args.resume is not None:
        raise ValueError("Resume từng seed riêng bằng --seed và --resume, không dùng --seeds.")
    if args.run_dir is not None and args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise ValueError("Thư mục ensemble đã có artifact; chọn --run-dir mới.")
    ensemble_dir = args.run_dir.resolve() if args.run_dir is not None else new_run_directory(PROJECT_ROOT / "models/runs", "ensemble")
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    defaults = build_parser().parse_args([])
    if args.metrics_output == defaults.metrics_output:
        args.metrics_output = ensemble_dir / "metrics.csv"
    atomic_json(ensemble_dir / "ensemble_manifest.json", {
        "seeds": seeds, "status": "running",
        "training_config_inputs": args._training_config_inputs,
    })

    per_seed_metrics: dict[int, pd.DataFrame] = {}
    for seed in seeds:
        run_args = copy.copy(args)
        run_args.seed = seed
        run_args.run_dir = ensemble_dir / f"seed_{seed}"
        if args.checkpoint != defaults.checkpoint:
            run_args.checkpoint = args.checkpoint.parent / f"seed_{seed}" / args.checkpoint.name
        run_args.metrics_output = args.metrics_output.with_name(
            f"{args.metrics_output.stem}_seed{seed}{args.metrics_output.suffix}"
        )
        if args.training_log != defaults.training_log:
            run_args.training_log = args.training_log.with_name(f"{args.training_log.stem}_seed{seed}{args.training_log.suffix}")
        if args.tensorboard_dir != defaults.tensorboard_dir:
            run_args.tensorboard_dir = args.tensorboard_dir / f"seed_{seed}"
        print(f"\n===== PPO run seed={seed} =====")
        model, context = train_fn(run_args)
        per_seed_metrics[seed] = evaluate_fn(run_args, model, context)

    summary, seed_metadata = summarize_seed_metrics(per_seed_metrics)
    summary_path = args.metrics_output.with_name(
        f"{args.metrics_output.stem}_seeds_summary{args.metrics_output.suffix}"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path)
    metadata_path = args.metrics_output.with_name(
        f"{args.metrics_output.stem}_seeds_metadata{args.metrics_output.suffix}"
    )
    seed_metadata.to_csv(metadata_path, index=False)
    atomic_json(ensemble_dir / "ensemble_manifest.json", {
        "seeds": seeds, "status": "completed",
        "training_config_inputs": args._training_config_inputs,
        "summary_path": str(summary_path), "metadata_path": str(metadata_path),
    })
    print(f"\n{args.evaluation_scope.title()} mean ± std across seeds:")
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"Đã lưu tổng hợp nhiều seed: {summary_path}")
    print(f"Đã lưu metadata từng seed: {metadata_path}")

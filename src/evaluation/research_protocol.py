"""Frozen development periods, execution contracts and learning gates."""

from pathlib import Path
import hashlib
import math

import pandas as pd
import yaml


def load_protocol(path):
    path = Path(path)
    data = path.read_bytes()
    protocol = yaml.safe_load(data)
    if not isinstance(protocol, dict) or protocol.get("evaluation_scope") != "validation":
        raise ValueError("Research protocol phải dùng evaluation_scope: validation.")
    cutoff = pd.Timestamp(protocol["development_end"])
    names, intervals = set(), []
    for fold in protocol["folds"]:
        if fold["name"] in names:
            raise ValueError("Fold name bị trùng.")
        names.add(fold["name"])
        dates = [pd.Timestamp(fold[key]) for key in (
            "train_end", "inner_start", "inner_end", "outer_start", "outer_end")]
        if not (dates[0] < dates[1] <= dates[2] < dates[3] <= dates[4] <= cutoff):
            raise ValueError(f"Fold không causal hoặc vượt development_end: {fold['name']}")
        if any(dates[3] <= end and start <= dates[4] for start, end in intervals):
            raise ValueError("Outer periods phải không chồng lấn.")
        intervals.append((dates[3], dates[4]))
    if not names:
        raise ValueError("Protocol cần ít nhất một fold.")
    signal = protocol["signal"]
    if signal["label_horizon"] not in signal["ic_horizons"]:
        raise ValueError("label_horizon phải được đo IC.")
    if any(type(h) is not int or h <= 0 for h in signal["ic_horizons"]):
        raise ValueError("IC horizons phải là số nguyên dương.")
    if len(signal["features"]) != len(set(signal["features"])):
        raise ValueError("Signal features không được trùng.")
    if len(signal["ridge_alphas"]) * len(signal["tilt_fractions"]) != signal["search_budget_per_fold"]:
        raise ValueError("Grid không khớp search budget đã đóng băng.")
    if any(not math.isfinite(x) or x <= 0 for x in signal["ridge_alphas"]):
        raise ValueError("Ridge alpha phải hữu hạn, dương.")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in signal["tilt_fractions"]):
        raise ValueError("Tilt fractions phải thuộc [0,1].")
    execution = protocol["execution"]
    if not (0 < execution["action_alpha"] <= 1 and 0 <= execution["cash_ratio"] < 1):
        raise ValueError("Execution alpha/cash không hợp lệ.")
    if execution["initial_allocation"] != "blend_from_cash" or execution["rebalance_every_sessions"] != 1:
        raise ValueError("Sprint 5 protocol hỗ trợ daily rebalance, blend_from_cash.")
    if signal["selection_metric"] != "sharpe_ratio":
        raise ValueError("Sprint 5 inner selection hiện dùng sharpe_ratio.")
    seeds = protocol["seeds"]
    if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int for s in seeds):
        raise ValueError("Protocol seeds phải là các số nguyên khác nhau.")
    if protocol["bootstrap"]["samples"] <= 0 or protocol["bootstrap"]["block_size"] <= 0:
        raise ValueError("Bootstrap settings phải dương.")
    for key, value in protocol["checkpoint_gate"].items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Checkpoint gate không hợp lệ: {key}")
    for key in ("minimum_positive_folds", "minimum_signal_folds"):
        if not 1 <= protocol["gate"][key] <= len(names):
            raise ValueError(f"Gate {key} vượt số fold.")
    return protocol, hashlib.sha256(data).hexdigest()


def apply_research_protocol(args):
    """Activate an optional locked PPO development protocol before resolution."""
    path = getattr(args, "research_protocol", None)
    if path is None or hasattr(args, "_research_protocol"):
        return
    protocol, digest = load_protocol(path)
    if getattr(args, "evaluation_scope", None) not in {None, "validation"}:
        raise ValueError("Research protocol không cho phép đánh giá test.")
    args.evaluation_scope = "validation"
    fold_name = getattr(args, "protocol_fold", None) or protocol["folds"][-1]["name"]
    fold = next((fold for fold in protocol["folds"] if fold["name"] == fold_name), None)
    if fold is None:
        raise ValueError(f"Không có protocol fold: {fold_name}")
    for name, expected in {"train_end": fold["train_end"], "val_start": fold["inner_start"],
                           "val_end": fold["inner_end"], "selection_metric": "sharpe"}.items():
        current = getattr(args, name, None)
        if current is not None and current != expected:
            raise ValueError(f"{name} xung đột protocol fold {fold_name}.")
        setattr(args, name, expected)
    args.protocol_fold = fold_name
    args._research_protocol_fold = fold
    seeds = ([int(s.strip()) for s in args.seeds.split(",") if s.strip()]
             if getattr(args, "seeds", None) else [args.seed])
    if not set(seeds).issubset(protocol["seeds"]):
        raise ValueError("Seed nằm ngoài protocol đã đóng băng.")
    # Fix explicit protocol choices; CLI conflicts are errors, not silent changes.
    for name, expected in {"action_alpha": protocol["execution"]["action_alpha"],
                           "matched_cash_ratio": protocol["execution"]["cash_ratio"]}.items():
        current = getattr(args, name, None)
        if current is not None and current != expected:
            raise ValueError(f"{name} xung đột research protocol.")
        setattr(args, name, expected)
    args._research_protocol = protocol
    args._research_protocol_hash = digest


def checkpoint_learning_gate(epoch, metrics, reference, matched, smoothed_delta, protocol=None):
    requirements = protocol["checkpoint_gate"] if protocol else None
    delta = float(metrics["sharpe_ratio"] - reference["sharpe_ratio"])
    matched_delta = float(metrics["sharpe_ratio"] - matched["sharpe_ratio"])
    passed = epoch > 0 and math.isfinite(smoothed_delta)
    if requirements:
        passed = passed and smoothed_delta >= requirements["minimum_improvement_over_untrained"]
        passed = passed and matched_delta > requirements["minimum_sharpe_delta_vs_matched"]
    return {
        "status": "untrained_reference" if epoch == 0 else "passed" if passed else "not_met",
        "passed": bool(passed), "sharpe_delta_vs_untrained": delta,
        "sharpe_delta_vs_matched_baseline": matched_delta,
        "return_delta_vs_untrained": float(metrics["cumulative_return"] - reference["cumulative_return"]),
        "smoothed_selection_delta_vs_untrained": float(smoothed_delta),
        "requirements": requirements,
    }

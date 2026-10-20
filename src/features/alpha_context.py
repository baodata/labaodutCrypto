"""Optional predictor-only external feature bank; no changes to base OHLCV features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.calendar_events import build_event_features
from src.data.external_alpha import ExternalDataUnavailable
from src.features.factor_risk import build_factor_features
from src.features.macro_volatility import build_macro_features
from src.utils.artifacts import atomic_csv, atomic_json, file_hashes
from src.utils.data_types import MarketDataTensor


ROOT = Path(__file__).resolve().parents[2]
GROUPS = {"vix", "fama_french", "blackout"}


def load_alpha_config(path):
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (not isinstance(config, dict) or set(config) != {"version", "strict_pit", *GROUPS}
            or config["version"] != "external-alpha-v1" or type(config["strict_pit"]) is not bool):
        raise ValueError("Invalid external-alpha config schema.")
    expected = {"vix": {"path", "max_age_days"},
                "fama_french": {"path", "window", "residual_window", "max_age_days", "ridge"},
                "blackout": {"events_path", "coverage_path", "pre_sessions", "post_sessions"}}
    for name, keys in expected.items():
        if not isinstance(config[name], dict) or set(config[name]) != keys:
            raise ValueError(f"Invalid external-alpha.{name} settings.")
        for field in keys & {"path", "events_path", "coverage_path"}:
            candidate = Path(config[name][field])
            config[name][field] = str(candidate if candidate.is_absolute() else ROOT / candidate)
    return config, file_hashes([path], ROOT)


def build_alpha_context(tensor, closing, config_path, groups, output=None):
    if not groups or len(set(groups)) != len(groups) or set(groups) - GROUPS:
        raise ValueError("Choose unique external alpha groups: vix, fama_french, blackout.")
    config, hashes = load_alpha_config(config_path)
    banks, statuses = {}, []
    for name in groups:
        try:
            kwargs = config[name]
            if name == "vix":
                result = build_macro_features(tensor, strict_pit=config["strict_pit"], **kwargs)
            elif name == "fama_french":
                result = build_factor_features(tensor, closing, strict_pit=config["strict_pit"], **kwargs)
            else:
                result = build_event_features(tensor, **kwargs)
            values, names, ready, diagnostics, metadata = result
            banks[name] = result
            statuses.append(dict(group=name, status="ready" if ready.any() else "needs_data",
                ready_sessions=int(ready.sum()), total_sessions=len(ready),
                availability=metadata["availability"], reason="" if ready.any() else "no_available_history"))
            if output is not None:
                output = Path(output)
                output.mkdir(parents=True, exist_ok=True)
                atomic_csv(diagnostics, output / f"{name}_readiness.csv", index=False)
                atomic_json(output / f"{name}_metadata.json", dict(metadata, config_hashes=hashes))
                panel = pd.DataFrame(values.reshape(-1, len(names)), columns=names)
                panel.insert(0, "ticker", np.tile(tensor.tickers, len(tensor.dates)))
                panel.insert(0, "date", np.repeat(tensor.dates, len(tensor.tickers)))
                atomic_csv(panel, output / f"{name}_features.csv", index=False)
        except ExternalDataUnavailable as exc:
            statuses.append(dict(group=name, status="needs_data", ready_sessions=0,
                total_sessions=len(tensor.dates), availability="unknown", reason=str(exc)))
    statuses = pd.DataFrame(statuses)
    if output is not None:
        Path(output).mkdir(parents=True, exist_ok=True)
        atomic_csv(statuses, Path(output) / "data_status.csv", index=False)
    return banks, statuses


def augment_alpha_tensor(tensor, banks, groups, *, minimum_warmup=0, end_index=None):
    """Require one common contiguous ready interval; never train on missing zeros.

    End index allows event calendars to require coverage beyond the evaluation
    cutoff. Leading unready cells are padding, explicitly excluded by warmup.
    """
    if type(minimum_warmup) is not int or minimum_warmup < 0:
        raise ValueError("Common warm-up must be a nonnegative integer.")
    end = len(tensor.dates) if end_index is None else int(end_index)
    if not 1 <= end <= len(tensor.dates):
        raise ValueError("Invalid feature readiness interval.")
    ready = tensor.feature_ready.copy()
    for group in groups:
        if group not in banks:
            raise ExternalDataUnavailable(f"Cannot enable {group}: missing external input.")
        ready &= banks[group][2]
    candidates = np.flatnonzero(ready[:end] & (np.arange(end) >= minimum_warmup))
    if not len(candidates):
        raise ExternalDataUnavailable("No common ready history for the requested alpha groups.")
    start = int(candidates[0])
    if not ready[start:end].all():
        missing = [tensor.dates[i] for i in np.flatnonzero(~ready[start:end])[:5] + start]
        raise ExternalDataUnavailable(f"Gaps in external input readiness: {missing}; no zero fill/backfill allowed.")
    values = [tensor.tensor.copy()]
    names = list(tensor.feature_names)
    for group in groups:
        extra, feature_names = banks[group][:2]
        extra = extra.copy()
        extra[:start] = 0.
        # Outside the requested evaluation interval is not used for fitting/prediction.
        extra[end:] = 0.
        values.append(extra)
        names.extend(feature_names)
    augmented = MarketDataTensor(np.concatenate(values, axis=2).astype(np.float32),
        list(tensor.tickers), names, list(tensor.dates), warmup_sessions=start)
    augmented.validate()
    return augmented

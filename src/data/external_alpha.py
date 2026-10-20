"""Availability-aware external observations, separate from the tradable universe."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.artifacts import file_hashes


class ExternalDataUnavailable(ValueError):
    """An ablation cannot run without the declared historical input contract."""


def utc_times(values, name):
    # Reject naive strings: interpreting an undocumented time zone as UTC is unsafe.
    result = []
    for value in values:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError(f"{name} requires nonmissing timezone-aware timestamps.")
        result.append(stamp.tz_convert("UTC"))
    return pd.DatetimeIndex(result, tz="UTC")


def decision_closes(dates):
    """Conservative 13:00 NY cutoff, including early-close equity sessions.

    A regular-day 16:00 observation is delayed to the next session. This avoids
    treating 13:00 holiday closes as 16:00 without an exchange calendar package.
    """
    days = pd.DatetimeIndex(pd.to_datetime(dates))
    if days.tz is not None or days.has_duplicates or not days.is_monotonic_increasing:
        raise ValueError("Expected unique ordered timezone-naive session dates.")
    return (days.normalize() + pd.Timedelta(hours=13)).tz_localize("America/New_York").tz_convert("UTC")


def load_external(path, value_columns, *, strict_pit=True):
    path = Path(path)
    manifest_path = path.with_suffix(".json")
    if not path.is_file() or not manifest_path.is_file():
        raise ExternalDataUnavailable(f"Missing external CSV/JSON pair: {path}")
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata.get("availability") not in {"point_in_time", "current_vintage_research_only"}:
        raise ValueError(f"Unknown availability contract: {path}")
    if strict_pit and metadata["availability"] != "point_in_time":
        raise ExternalDataUnavailable(f"Historical vintage evidence required: {path}; current-vintage data is research-only.")
    if metadata["availability"] == "point_in_time" and not metadata.get("availability_evidence"):
        raise ValueError("Point-in-time claims require availability_evidence in the manifest.")
    frame = pd.read_csv(path)
    expected = {"date", "available_at", "source", "vintage_id", *value_columns}
    if set(frame.columns) != expected or frame.empty or frame.isna().any().any():
        raise ExternalDataUnavailable(f"Expected nonempty complete external schema {sorted(expected)}: {path}")
    frame["date"] = pd.to_datetime(frame.date, errors="raise")
    if frame.date.dt.tz is not None or not frame.date.eq(frame.date.dt.normalize()).all():
        raise ValueError("External date must be a timezone-naive observation date.")
    frame["available_at"] = utc_times(frame.available_at, "available_at")
    if frame.duplicated(["date", "available_at"]).any():
        raise ValueError("Ambiguous revisions at the same date/availability timestamp.")
    if any(not frame[c].astype(str).str.strip().ne("").all() for c in ("source", "vintage_id")):
        raise ValueError("Every row needs a source and vintage ID.")
    values = frame[list(value_columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite external values; missing data must remain unavailable.")
    observation_day = frame.date.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    if (frame.available_at < observation_day).any():
        raise ValueError("External daily values cannot be available before their observation date.")
    return frame.sort_values(["date", "available_at"]), dict(metadata,
        input_hashes=file_hashes([path, manifest_path], path.parent.resolve()))


def known_history(frame, cutoff):
    """Latest version of every historical observation actually available at cutoff."""
    return (frame.loc[frame.available_at <= cutoff]
            .sort_values(["date", "available_at"]).drop_duplicates("date", keep="last")
            .sort_values("date"))

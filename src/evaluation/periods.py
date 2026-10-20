"""Describe actual evaluated sessions without claiming a fresh holdout.

2024–2025 has already been inspected in this project's research history. A
larger interval that contains those sessions is also exploratory. Unknown
periods remain unverified; changing a CLI date cannot make them independent.
"""
from __future__ import annotations

import warnings

import pandas as pd


EXPLORED_PERIODS = (("2024-01-01", "2025-12-31"),)


def describe_test_period(test_tensor, evaluation_scope: str) -> dict:
    if evaluation_scope not in {"validation", "test"}:
        raise ValueError("Unknown evaluation scope.")
    if evaluation_scope == "validation":
        return {"status": "not_evaluated_validation_only", "start": None, "end": None,
                "overlaps_explored_period": False, "independent_holdout_verified": False}
    if test_tensor is None:
        raise ValueError("Test evaluation requires a prepared test tensor.")
    dates = pd.DatetimeIndex(pd.to_datetime(test_tensor.dates))
    if len(dates) == 0 or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Test dates must be nonempty, unique and increasing.")
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    overlaps = any(((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))).any()
                   for start, end in EXPLORED_PERIODS)
    return {
        "status": ("reused_exploratory_period_not_independent_holdout" if overlaps
                   else "custom_period_freshness_unverified"),
        "start": dates[0].date().isoformat(), "end": dates[-1].date().isoformat(),
        "overlaps_explored_period": bool(overlaps), "independent_holdout_verified": False,
    }


def warn_for_explored_test(period: dict) -> None:
    if period["overlaps_explored_period"]:
        warnings.warn(
            f"Test {period['start']} → {period['end']} chồng lấn giai đoạn 2024–2025 "
            "đã được khảo sát; kết quả là exploratory, không phải holdout độc lập.",
            RuntimeWarning, stacklevel=2,
        )

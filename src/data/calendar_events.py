"""QNT-027: revision-aware event calendars and explicit coverage snapshots.

An empty event list does not establish that a day is event-free. Coverage is a
separate, timestamped assertion from the calendar vendor/archive.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.external_alpha import ExternalDataUnavailable, decision_closes, utc_times
from src.utils.artifacts import file_hashes


EVENT_COLUMNS = {"event_id", "event_type", "ticker", "scheduled_at", "known_at", "status", "source", "vintage_id"}
COVERAGE_COLUMNS = {"ticker", "event_type", "covered_from", "covered_through", "known_at", "source", "vintage_id"}


def build_event_features(tensor, events_path, coverage_path, *, pre_sessions=2, post_sessions=1):
    if any(type(v) is not int or v < 0 for v in (pre_sessions, post_sessions)):
        raise ValueError("Blackout windows must be nonnegative session counts.")
    paths = [Path(events_path), Path(coverage_path), Path(events_path).with_suffix(".json")]
    if any(not path.is_file() for path in paths):
        raise ExternalDataUnavailable("Missing event CSV, coverage CSV or event manifest.")
    events, coverage = pd.read_csv(paths[0], keep_default_na=False), pd.read_csv(paths[1], keep_default_na=False)
    metadata = json.loads(paths[2].read_text(encoding="utf-8"))
    if metadata.get("availability") != "point_in_time" or not metadata.get("availability_evidence"):
        raise ExternalDataUnavailable("Blackout requires archived schedule and coverage availability evidence.")
    if set(events.columns) != EVENT_COLUMNS or set(coverage.columns) != COVERAGE_COLUMNS:
        raise ValueError("Event and coverage CSV schemas must match the documented contract.")
    if coverage.empty:
        raise ExternalDataUnavailable("No timestamped event coverage; cannot infer event-free days.")
    for frame in (events, coverage):
        if frame.isna().any().any() or any(frame[c].astype(str).str.strip().eq("").any() for c in frame.columns):
            raise ValueError("Missing event/coverage fields.")
        frame["known_at"] = utc_times(frame.known_at, "known_at")
        if not frame.event_type.isin(["earnings", "fomc"]).all():
            raise ValueError("Supported event types are earnings and fomc.")
        valid_ticker = ((frame.event_type == "fomc") & (frame.ticker == "*")) | (
            (frame.event_type == "earnings") & frame.ticker.isin(tensor.tickers))
        if not valid_ticker.all():
            raise ValueError("FOMC must use '*'; earnings must use a configured ticker.")
    events["scheduled_at"] = utc_times(events.scheduled_at, "scheduled_at")
    if (not events.status.isin(["scheduled", "cancelled"]).all()
            or events.duplicated(["event_id", "known_at"]).any()):
        raise ValueError("Invalid event status or ambiguous revision.")
    if not events.empty and (events.groupby("event_id")[["ticker", "event_type"]].nunique() > 1).any().any():
        raise ValueError("Ticker/type cannot change within an event ID.")
    for column in ("covered_from", "covered_through"):
        coverage[column] = pd.to_datetime(coverage[column], errors="raise")
        if coverage[column].dt.tz is not None or not coverage[column].eq(coverage[column].dt.normalize()).all():
            raise ValueError("Coverage endpoints must be session dates without time zones.")
    if (coverage.covered_from > coverage.covered_through).any():
        raise ValueError("Reversed calendar coverage interval.")
    if coverage.duplicated(["ticker", "event_type", "known_at", "covered_from", "covered_through"]).any():
        raise ValueError("Duplicate coverage assertion.")
    calendar = pd.DatetimeIndex(pd.to_datetime(tensor.dates))
    names = ["earnings_blackout", "fomc_blackout", "blackout_x_mom_12_1"]
    values = np.full((*tensor.tensor.shape[:2], len(names)), np.nan)
    ready, diagnostic = np.zeros(len(calendar), dtype=bool), []
    momentum = tensor.tensor[:, :, tensor.feature_names.index("mom_12_1")]
    for t, cutoff in enumerate(decision_closes(tensor.dates)):
        # Require coverage for the full pre/post event neighborhood; do not truncate
        # the future edge and mistakenly call the last decisions event-free.
        if t < post_sessions or t + pre_sessions >= len(calendar):
            diagnostic.append(dict(date=tensor.dates[t], ready=False, reason="calendar_boundary"))
            continue
        begin, end = calendar[t-post_sessions], calendar[t+pre_sessions]
        available = coverage.loc[(coverage.known_at <= cutoff) & (coverage.covered_from <= begin)
                                 & (coverage.covered_through >= end)]
        covered = all(((available.event_type == "earnings") & (available.ticker == ticker)).any()
                      for ticker in tensor.tickers)
        covered = covered and ((available.event_type == "fomc") & (available.ticker == "*")).any()
        if not covered:
            diagnostic.append(dict(date=tensor.dates[t], ready=False, reason="unknown_calendar_coverage"))
            continue
        latest = (events.loc[events.known_at <= cutoff].sort_values("known_at")
                  .drop_duplicates("event_id", keep="last"))
        active = latest.loc[latest.status == "scheduled"]
        flags = np.zeros((len(tensor.tickers), 2))
        for row in active.itertuples():
            day = row.scheduled_at.tz_convert("America/New_York").tz_localize(None).normalize()
            event_position = int(calendar.searchsorted(day))
            # A weekend event maps to the next executable equity session.
            if event_position - pre_sessions <= t <= event_position + post_sessions:
                if row.event_type == "fomc":
                    flags[:, 1] = 1.
                else:
                    flags[tensor.tickers.index(row.ticker), 0] = 1.
        values[t, :, :2] = flags
        values[t, :, 2] = flags.max(axis=1) * momentum[t]
        ready[t] = True
        diagnostic.append(dict(date=tensor.dates[t], ready=True,
            earnings_assets=int(flags[:, 0].sum()), fomc=bool(flags[:, 1].any()),
            known_revisions=len(latest)))
    metadata.update(feature_names=names, pre_sessions=pre_sessions, post_sessions=post_sessions,
        input_hashes=file_hashes(paths, paths[0].parent.resolve()),
        window="session_date_inclusive_pre_through_post; no_earnings_result_features",
        decision_cutoff="13:00_America_New_York_conservative")
    return values, names, ready, pd.DataFrame(diagnostic), metadata


def blackout_target(target, current, blocked, *, asset_cap=.02):
    """Close-time overlay: prohibit increases, cap blocked names, move excess to cash.

    This returns a target, not guaranteed next-Open fills. Apply identically to
    strategy and matched baseline when researching the overlay's contribution.
    """
    target, current = np.asarray(target, dtype=np.float64), np.asarray(current, dtype=np.float64)
    blocked = np.asarray(blocked)
    if (target.ndim != 1 or target.size < 2 or target.shape != current.shape
            or blocked.shape != (len(target)-1,) or blocked.dtype != np.bool_
            or not np.isfinite([*target, *current, asset_cap]).all() or not 0 <= asset_cap <= 1
            or (target < 0).any() or (current < 0).any()
            or not np.isclose(target.sum(), 1., atol=1e-10, rtol=0)
            or not np.isclose(current.sum(), 1., atol=1e-10, rtol=0)):
        raise ValueError("Invalid blackout weights/mask/cap.")
    result = target.copy()
    stocks = np.minimum(result[:-1], np.minimum(current[:-1], asset_cap))
    result[:-1] = np.where(blocked, stocks, result[:-1])
    result[-1] = max(0., 1. - result[:-1].sum())
    return result

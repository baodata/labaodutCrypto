"""Executable research labels on the same adjusted-price basis as TradingEnv.

Decision at Close[t], entry at Open[t+1], exit mark at Close[t+h]. Horizons
count observed trading sessions, not calendar days. Labels exclude costs;
portfolio execution, overnight holdings and costs remain the env's job.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


LABEL_DEFINITION = "next_open_to_horizon_close_v1"
RETURN_BASIS = "adjusted_close_scaled_open_close_total_return"
LABEL_TARGET = "cross_sectional_excess_return"


def session_calendar(dates) -> pd.DatetimeIndex:
    calendar = pd.DatetimeIndex(pd.to_datetime(dates))
    if (not len(calendar) or calendar.tz is not None or calendar.hasnans
            or calendar.has_duplicates or not calendar.is_monotonic_increasing
            or not calendar.equals(calendar.normalize())):
        raise ValueError("Session dates must be nonempty, timezone-free, unique increasing dates.")
    return calendar


def decision_positions(length: int, start: int, end: int, horizon: int) -> np.ndarray:
    """End is exclusive; every label endpoint must remain below end."""
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
           for value in (length, start, end)):
        raise ValueError("Label interval bounds must be integer session positions.")
    if isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, (int, np.integer)) or horizon <= 0:
        raise ValueError("Label horizon must be a positive integer number of sessions.")
    if not (0 <= start <= end <= length):
        raise ValueError("Invalid label interval [start,end).")
    return np.arange(start, max(start, end - horizon), dtype=np.int64)


def executable_forward_returns(opening, closing, positions, horizon: int) -> np.ndarray:
    """Shared numerical label calculation; never clip or fill a missing endpoint."""
    opening, closing = np.asarray(opening, dtype=np.float64), np.asarray(closing, dtype=np.float64)
    if opening.ndim != 2 or opening.shape != closing.shape or min(opening.shape) == 0:
        raise ValueError("Aligned Open/Close matrices must have the same nonempty [T,N] shape.")
    decision_positions(len(opening), 0, len(opening), horizon)
    positions = np.asarray(positions)
    if positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer):
        raise ValueError("Decision positions must be a one-dimensional integer array.")
    if np.any(positions < 0) or np.any(positions + horizon >= len(opening)):
        raise ValueError("A label endpoint falls outside the available sessions.")
    entry, exit_mark = opening[positions + 1], closing[positions + horizon]
    if (not np.isfinite(entry).all() or not np.isfinite(exit_mark).all()
            or np.any(entry <= 0) or np.any(exit_mark <= 0)):
        raise ValueError("Executable label endpoints must be finite positive prices.")
    values = exit_mark / entry - 1.0
    if not np.isfinite(values).all():
        raise ValueError("Executable returns overflowed; inspect price adjustment data.")
    return values


@dataclass
class ExecutableLabelPanel:
    """Offline outcome store with explicit ticker order and session calendar.

    Price matrices must already be aligned to this exact calendar/ticker order
    and adjusted on RETURN_BASIS by the data pipeline. Raw OHLCV is not accepted
    as an interchangeable total-return basis.
    """

    dates: list
    tickers: list[str]
    opening: np.ndarray
    closing: np.ndarray
    return_basis: str = RETURN_BASIS

    def __post_init__(self):
        self.calendar = session_calendar(self.dates)
        self.tickers = list(self.tickers)
        if not self.tickers or len(set(self.tickers)) != len(self.tickers) or any(not isinstance(t, str) or not t for t in self.tickers):
            raise ValueError("Tickers must be unique nonempty strings in matrix column order.")
        if self.return_basis != RETURN_BASIS:
            raise ValueError("Unsupported price adjustment basis for executable labels.")
        self.opening = np.asarray(self.opening, dtype=np.float64)
        self.closing = np.asarray(self.closing, dtype=np.float64)
        expected = (len(self.calendar), len(self.tickers))
        if self.opening.shape != expected or self.closing.shape != expected:
            raise ValueError("Price matrices do not match the label calendar/ticker order.")
        if (not np.isfinite(self.opening).all() or not np.isfinite(self.closing).all()
                or np.any(self.opening <= 0) or np.any(self.closing <= 0)):
            raise ValueError("Labels require finite positive aligned Open/Close prices.")

    def positions(self, start: int, end: int, horizon: int) -> np.ndarray:
        return decision_positions(len(self.calendar), start, end, horizon)

    def returns(self, positions, horizon: int, *, excess: bool = False) -> np.ndarray:
        values = executable_forward_returns(self.opening, self.closing, positions, horizon)
        return values - values.mean(axis=1, keepdims=True) if excess else values

    def frame(self, start: int, end: int, horizon: int) -> pd.DataFrame:
        """Offline labels, deliberately separate from the prediction bank."""
        positions = self.positions(start, end, horizon)
        gross = self.returns(positions, horizon)
        excess = gross - gross.mean(axis=1, keepdims=True)
        n = len(self.tickers)
        return pd.DataFrame({
            "decision_date": np.repeat(self.calendar[positions].strftime("%Y-%m-%d"), n),
            "execution_date": np.repeat(self.calendar[positions + 1].strftime("%Y-%m-%d"), n),
            "label_end_date": np.repeat(self.calendar[positions + horizon].strftime("%Y-%m-%d"), n),
            "ticker": np.tile(self.tickers, len(positions)), "horizon_sessions": horizon,
            "realized_return": gross.reshape(-1),
            "realized_cross_sectional_excess_return": excess.reshape(-1),
            "label_definition": LABEL_DEFINITION, "return_basis": self.return_basis,
        })

    def metadata(self) -> dict:
        return {"label_definition": LABEL_DEFINITION, "return_basis": self.return_basis,
                "decision_time": "close", "entry_time": "next_session_open",
                "exit_time": "horizon_session_close", "horizon_unit": "trading_sessions",
                "return_unit": "decimal_simple_return_not_annualized", "costs_in_label": False,
                "excess_reference": "same_session_cross_section_equal_weight_no_cash_no_costs",
                "universe": "fixed_configured_universe_not_point_in_time", "tickers": self.tickers}

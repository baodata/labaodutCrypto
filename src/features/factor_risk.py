"""QNT-029: FF3 excess-return regressions using only released factor vintages."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.external_alpha import decision_closes, known_history, load_external


def build_factor_features(tensor, closing, path, *, strict_pit=True, window=60,
                          residual_window=20, max_age_days=90, ridge=1e-8):
    if (type(window) is not int or window < 5 or type(residual_window) is not int
            or residual_window < 2 or type(max_age_days) is not int or max_age_days < 1
            or not np.isfinite(ridge) or ridge <= 0):
        raise ValueError("Invalid causal factor regression windows/ridge/age limit.")
    factors = ["mkt_rf", "smb", "hml"]
    frame, metadata = load_external(path, factors + ["rf"], strict_pit=strict_pit)
    unit = metadata.get("unit")
    if unit not in {"percent_return", "decimal_return"}:
        raise ValueError("FF3 input must declare percent_return or decimal_return.")
    if frame[factors + ["rf"]].isin([-99.99, -999.]).any().any():
        raise ValueError("Missing factor sentinel; refuse to treat it as a return.")
    if unit == "percent_return":
        frame[factors + ["rf"]] /= 100.
    if (frame[factors + ["rf"]].abs() >= 1).any().any():
        raise ValueError("FF3 values outside daily decimal return range; check missing sentinels/units.")
    calendar = pd.DatetimeIndex(pd.to_datetime(tensor.dates))
    prices = np.asarray(closing, dtype=np.float64)
    if prices.shape != tensor.tensor.shape[:2] or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Factor features require aligned positive total-return closing marks.")
    returns = pd.DataFrame(prices, index=calendar).pct_change(fill_method=None)
    names = ["ff_beta_mkt", "ff_beta_smb", "ff_beta_hml", f"ff_residual_mom_{residual_window}d", f"ff_residual_vol_{residual_window}d"]
    values = np.full((*prices.shape, len(names)), np.nan)
    ready = np.zeros(len(calendar), dtype=bool)
    diagnostics = []
    penalty = np.diag([0., ridge, ridge, ridge])
    for t, cutoff in enumerate(decision_closes(tensor.dates)):
        known = known_history(frame, cutoff).set_index("date")
        # Restrict to past closes even if an importer claims unusually early availability.
        known = known.loc[known.index.isin(calendar[:t])].sort_index()
        aligned = returns.reindex(known.index)
        valid = np.isfinite(aligned.to_numpy()).all(axis=1)
        known, aligned = known.loc[valid], aligned.loc[valid]
        age = (calendar[t] - known.index[-1]).days if len(known) else None
        if len(known) < window + residual_window or age is None or age > max_age_days:
            diagnostics.append(dict(date=tensor.dates[t], ready=False, released_rows=len(known), age_days=age))
            continue
        x = np.column_stack([np.ones(len(known)), known[factors].to_numpy()])
        y = aligned.to_numpy() - known.rf.to_numpy()[:, None]
        # Each realized residual uses coefficients fitted strictly before its own date.
        residuals = []
        for j in range(len(known)-residual_window, len(known)):
            xx, yy = x[j-window:j], y[j-window:j]
            coef = np.linalg.solve(xx.T @ xx + penalty, xx.T @ yy)
            residuals.append(y[j] - x[j] @ coef)
        xx, yy = x[-window:], y[-window:]
        coef = np.linalg.solve(xx.T @ xx + penalty, xx.T @ yy)
        residuals = np.asarray(residuals)
        values[t, :, :3] = coef[1:].T
        # Sum of additive regression residuals; not a compounded tradeable return.
        values[t, :, 3] = residuals.sum(axis=0)
        values[t, :, 4] = residuals.std(axis=0, ddof=1)
        ready[t] = True
        diagnostics.append(dict(date=tensor.dates[t], ready=True, released_rows=len(known),
            last_factor_date=str(known.index[-1].date()), age_days=age,
            max_available_at=known.available_at.max().isoformat(),
            condition_number=float(np.linalg.cond(xx.T @ xx + penalty))))
    metadata.update(feature_names=names, window=window, residual_window=residual_window,
        regression="stock_total_return_minus_RF_on_intercept_Mkt_RF_SMB_HML",
        residual_definition="date_u_coefficients_fit_on_released_rows_strictly_before_u",
        exposure_fit="last_window_released_rows_strictly_before_decision_date",
        output_unit="daily_decimal_returns_and_dimensionless_betas", max_age_days=max_age_days,
        limitations="FF3_has_no_momentum_factor_and_does_not_prove_idiosyncratic_alpha")
    return values, names, ready, pd.DataFrame(diagnostics), metadata

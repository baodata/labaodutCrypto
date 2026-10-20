"""QNT-026: causal VIX/VIX3M context and cross-sectional interactions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.external_alpha import decision_closes, known_history, load_external


def build_macro_features(tensor, path, *, strict_pit=True, max_age_days=7):
    if type(max_age_days) is not int or max_age_days < 1:
        raise ValueError("VIX maximum age must be a positive calendar-day count.")
    frame, metadata = load_external(path, ["vix", "vix3m"], strict_pit=strict_pit)
    if metadata.get("unit") != "index_points" or (frame[["vix", "vix3m"]] <= 0).any().any():
        raise ValueError("VIX inputs require positive index points, not decimal returns.")
    conservative_release = (pd.DatetimeIndex(frame.date) + pd.Timedelta(hours=16, minutes=30)).tz_localize("America/New_York").tz_convert("UTC")
    if (frame.available_at.to_numpy() < conservative_release.to_numpy()).any():
        raise ValueError("Daily VIX/VIX3M close cannot be used before conservative 16:30 NY release.")
    names = ["vix_level", "vix_term_ratio", "vix_inverted",
             "vix_term_x_mom_12_1", "vix_term_x_beta_spy_60d"]
    values = np.full((*tensor.tensor.shape[:2], len(names)), np.nan)
    ready = np.zeros(len(tensor.dates), dtype=bool)
    diagnostics = []
    momentum = tensor.tensor[:, :, tensor.feature_names.index("mom_12_1")]
    beta = tensor.tensor[:, :, tensor.feature_names.index("beta_spy_60d")]
    for t, cutoff in enumerate(decision_closes(tensor.dates)):
        known = known_history(frame, cutoff)
        if known.empty:
            diagnostics.append(dict(date=tensor.dates[t], ready=False, reason="no_available_observation"))
            continue
        row = known.iloc[-1]
        age = (pd.Timestamp(tensor.dates[t]) - row.date).days
        ready[t] = 0 <= age <= max_age_days
        if ready[t]:
            ratio = row.vix / row.vix3m
            values[t, :, 0] = row.vix
            values[t, :, 1] = ratio
            values[t, :, 2] = float(ratio > 1.)
            values[t, :, 3] = (ratio - 1.) * momentum[t]
            values[t, :, 4] = (ratio - 1.) * beta[t]
        diagnostics.append(dict(date=tensor.dates[t], ready=bool(ready[t]),
            observation_date=str(row.date.date()), available_at=row.available_at.isoformat(), age_days=age))
    metadata.update(feature_names=names, max_age_days=max_age_days,
        use="predictor_context_only_no_automatic_cash_override", decision_cutoff="13:00_America_New_York_conservative")
    return values, names, ready, pd.DataFrame(diagnostics), metadata

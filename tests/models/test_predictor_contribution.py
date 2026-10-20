import numpy as np
import pandas as pd
import pytest

from src.evaluation.predictor_contribution import predictor_contributions


def comparison_frames():
    metrics, daily = [], []
    for fold in ("a", "b"):
        for name, offset in (("ridge_fixed", 0.), ("gru", .0001), ("gnn_gru", .0002)):
            metrics.append(dict(fold_id=fold, split="outer", allocator="cost_aware", horizon_sessions=1,
                model=name, sharpe_ratio=1+offset*1000, cumulative_return=.1+offset*10,
                total_costs=100-offset*1000, total_turnover=2.))
            daily.extend(dict(fold_id=fold, date=f"date_{i}", split="outer", allocator="cost_aware", horizon_sessions=1,
                              model=name, net_return=.001+offset) for i in range(10))
    return pd.DataFrame(metrics), pd.DataFrame(daily)


def test_contribution_uses_paired_predictors_with_unchanged_allocator():
    metrics, daily = comparison_frames()
    folds, summary = predictor_contributions(metrics, daily, {"block_size":2, "samples":50, "seed":7})
    assert len(folds) == 4
    assert list(summary.candidate) == ["gru", "gnn_gru"]
    assert list(summary.reference) == ["ridge_fixed", "gru"]
    np.testing.assert_allclose(summary.excess_annual, .0001*252)
    np.testing.assert_allclose(summary.mean_sharpe_delta, .1)
    assert set(summary.allocator) == {"cost_aware"}


def test_missing_paired_sessions_are_rejected():
    metrics, daily = comparison_frames()
    daily = daily.drop(daily[(daily.model == "gru") & (daily.date == "date_0")].index)
    with pytest.raises(ValueError, match="same evaluated sessions"):
        predictor_contributions(metrics, daily, {"block_size":2, "samples":50, "seed":7})

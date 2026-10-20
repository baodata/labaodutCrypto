"""Paired predictor contrasts under the same allocator, horizon and sessions."""
from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import block_bootstrap_excess_returns


PAIRS = [("gru", "ridge_fixed"), ("gnn_gru", "gru"), ("gnn_gru", "gnn_gru_no_edges")]


def predictor_contributions(metrics: pd.DataFrame, daily: pd.DataFrame, bootstrap: dict):
    models = set(metrics.model)
    parts, summaries = [], []
    keys = ["fold_id", "split", "allocator", "horizon_sessions"]
    financial = ["sharpe_ratio", "cumulative_return", "total_costs", "total_turnover"]
    for candidate, reference in PAIRS:
        if not {candidate, reference}.issubset(models):
            continue
        a = metrics[metrics.model == candidate]
        b = metrics[metrics.model == reference]
        joined = a[keys + financial].merge(b[keys + financial], on=keys, suffixes=("_candidate", "_reference"),
                                          validate="one_to_one", how="outer", indicator=True)
        if not joined["_merge"].eq("both").all():
            raise ValueError("Predictor contrast has unmatched allocator/fold/horizon cases.")
        joined = joined.drop(columns="_merge")
        joined["candidate"], joined["reference"] = candidate, reference
        for name in financial:
            joined[name + "_delta"] = joined[name + "_candidate"] - joined[name + "_reference"]
        parts.append(joined)
        for (allocator, horizon), rows in joined[joined.split == "outer"].groupby(["allocator", "horizon_sessions"], sort=False):
            common = daily[(daily.split == "outer") & (daily.allocator == allocator) & (daily.horizon_sessions == horizon)]
            a = common[common.model == candidate][["fold_id", "date", "net_return"]]
            b = common[common.model == reference][["fold_id", "date", "net_return"]]
            paired = a.merge(b, on=["fold_id", "date"], suffixes=("_candidate", "_reference"),
                             validate="one_to_one", how="outer")
            if paired.isna().any().any():
                raise ValueError("Predictor contrast requires exactly the same evaluated sessions.")
            stats = block_bootstrap_excess_returns(paired.net_return_candidate.to_numpy(), paired.net_return_reference.to_numpy(),
                block_size=max(bootstrap["block_size"], int(horizon)), num_samples=bootstrap["samples"],
                seed=bootstrap["seed"], groups=paired.fold_id.to_numpy())
            summaries.append(dict(candidate=candidate, reference=reference, allocator=allocator, horizon_sessions=int(horizon),
                mean_sharpe_delta=float(rows.sharpe_ratio_delta.mean()),
                mean_return_delta=float(rows.cumulative_return_delta.mean()),
                mean_cost_delta=float(rows.total_costs_delta.mean()), positive_folds=int((rows.cumulative_return_delta > 0).sum()),
                excess_annual=stats["mean_excess_annualized"], ci_lower_95=stats["ci_lower_95"],
                ci_upper_95=stats["ci_upper_95"], p_value=stats["p_value_positive_excess"]))
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(), pd.DataFrame(summaries))

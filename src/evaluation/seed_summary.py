"""Summarize numeric metrics across seeds while preserving per-run metadata."""

import pandas as pd


def summarize_seed_metrics(
    per_seed_metrics: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not per_seed_metrics:
        raise ValueError("Cần ít nhất một bảng metrics để tổng hợp seed.")
    cleaned = {
        s: df.drop(columns=[c for c in ["seed", "strategy"] if c in df.columns], errors="ignore")
        for s, df in per_seed_metrics.items()
    }
    stacked = pd.concat(cleaned, names=["seed", "strategy"])
    numeric = stacked.select_dtypes(include="number").drop(
        columns=["checkpoint_epoch"], errors="ignore"
    )
    summary = numeric.groupby(level="strategy").agg(["mean", "std"])
    metadata_columns = stacked.columns.difference(numeric.columns, sort=False)
    metadata = stacked.loc[:, metadata_columns].reset_index()
    return summary, metadata

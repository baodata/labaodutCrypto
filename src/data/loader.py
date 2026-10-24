# cái này để load dữ liệu từ file parquet đã tải về, validate và trả về dạng tensor

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

from src.utils.data_types import MarketDataTensor


def load_config(config_path: str = "configs/assets.yaml") -> dict:
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_all_tickers(config: dict) -> List[str]:
    """Extract a flat, ordered list of unique stock tickers from sectors config."""
    tickers = []
    for sector_name, sector_tickers in config.get("sectors", {}).items():
        for ticker in sector_tickers:
            if ticker not in tickers:
                tickers.append(ticker)
    return sorted(tickers)


def load_market_data(
    config_path: str = "configs/assets.yaml",
) -> Tuple[MarketDataTensor, Dict[str, MarketDataTensor]]:
    """Load, validate, align, and split market data from raw parquet files.

    Args:
        config_path: Path to the assets.yaml configuration file.

    Returns:
        A tuple of:
            - full_data: Complete MarketDataTensor for the entire timeline.
            - splits: Dictionary mapping split names (e.g. 'train', 'val', 'test')
                      to their corresponding sliced MarketDataTensor.
    """
    config = load_config(config_path)
    raw_dir = Path(config.get("raw_data_dir", "data/raw"))
    benchmark_ticker = config.get("benchmark", "SPY")
    feature_names = config.get("features", ["open", "high", "low", "close", "adj_close", "volume"])
    tickers = get_all_tickers(config)

    if not tickers:
        raise ValueError("No stock tickers found in configuration.")

    # 1. Load benchmark file
    bench_file = raw_dir / f"{benchmark_ticker}.parquet"
    if not bench_file.is_file():
        raise FileNotFoundError(f"Benchmark parquet not found: {bench_file}")

    bench_df = pd.read_parquet(bench_file)
    if "date" not in bench_df.columns:
        raise ValueError(f"Benchmark data {bench_file} missing 'date' column.")

    bench_df["date"] = pd.to_datetime(bench_df["date"])
    bench_df = bench_df.sort_values("date").reset_index(drop=True)
    common_dates = bench_df["date"].values

    # Check benchmark features
    missing_bench_feats = [col for col in feature_names if col not in bench_df.columns]
    if missing_bench_feats:
        raise ValueError(f"Benchmark missing features: {missing_bench_feats}")

    bench_values = bench_df[feature_names].values.astype(np.float32)

    # 2. Load and align all asset parquet files
    t_len = len(common_dates)
    n_len = len(tickers)
    f_len = len(feature_names)

    tensor_values = np.empty((t_len, n_len, f_len), dtype=np.float32)

    for i, ticker in enumerate(tickers):
        file_path = raw_dir / f"{ticker}.parquet"
        if not file_path.is_file():
            raise FileNotFoundError(f"Parquet file for ticker '{ticker}' not found at {file_path}")

        df = pd.read_parquet(file_path)
        if "date" not in df.columns:
            raise ValueError(f"File {file_path} missing 'date' column.")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Validate date alignment
        if not np.array_equal(df["date"].values, common_dates):
            # If dates do not match identically, merge/align onto common_dates
            merged = pd.DataFrame({"date": common_dates}).merge(df, on="date", how="left")
            merged = merged.ffill().bfill()
            df = merged

        missing_feats = [col for col in feature_names if col not in df.columns]
        if missing_feats:
            raise ValueError(f"Ticker '{ticker}' missing features: {missing_feats}")

        arr = df[feature_names].values.astype(np.float32)
        tensor_values[:, i, :] = arr

    # 3. Check for NaNs or Infs
    if np.isnan(tensor_values).any() or np.isinf(tensor_values).any():
        raise ValueError("Market data tensor contains NaN or Inf values.")
    if np.isnan(bench_values).any() or np.isinf(bench_values).any():
        raise ValueError("Benchmark values contain NaN or Inf values.")

    # 4. Construct full MarketDataTensor
    full_data = MarketDataTensor(
        dates=common_dates,
        tickers=tickers,
        feature_names=feature_names,
        values=tensor_values,
        benchmark_ticker=benchmark_ticker,
        benchmark_values=bench_values,
    )

    # 5. Generate splits from config
    splits: Dict[str, MarketDataTensor] = {}
    splits_cfg = config.get("splits", {})
    for split_name, bounds in splits_cfg.items():
        start_date = bounds.get("start_date")
        end_date = bounds.get("end_date")
        if start_date and end_date:
            splits[split_name] = full_data.slice_by_date(start_date, end_date)

    return full_data, splits

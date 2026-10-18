"""Causal data splits, normalization and execution-price preparation. Owner: A."""

from __future__ import annotations

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import yaml

from src.data.split import TemporalSplitter
from src.features.pipeline import FeaturePipeline
from src.features.scaler import MarketFeatureScaler
from src.utils.config import AssetsConfig
from src.utils.data_types import MarketDataTensor

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _total_return_adjustment_factor(
    frame: pd.DataFrame,
    dates: pd.DatetimeIndex,
    ticker: str,
) -> np.ndarray:
    """Return Yahoo's Adj Close / Close factor aligned to the requested dates.

    Applying the same daily factor to raw Open and Close creates a consistent
    total-return price index. Ratios across dates then include cash distributions
    (and split adjustments) while within-day returns remain price returns.
    """
    close_col = next(
        (column for column in frame.columns if str(column).lower() == "close"),
        None,
    )
    adj_close_col = next(
        (
            column
            for column in frame.columns
            if str(column).lower().replace(" ", "_") in {"adj_close", "adjusted_close"}
        ),
        None,
    )
    if close_col is None:
        raise ValueError(f"Thiếu cột Close để tính hệ số total return: {ticker}")
    if adj_close_col is None:
        # Retain compatibility with price-only data sources, but make the loss
        # of dividend information explicit instead of silently implying total return.
        warnings.warn(
            f"{ticker} không có Adj Close; lợi nhuận sẽ chỉ dựa trên biến động giá, "
            "không bao gồm cổ tức.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.ones(len(dates), dtype=np.float64)

    aligned = frame.reindex(dates)
    close = aligned[close_col].to_numpy(dtype=np.float64)
    adj_close = aligned[adj_close_col].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(close).all()
        or not np.isfinite(adj_close).all()
        or np.any(close <= 0)
        or np.any(adj_close <= 0)
    ):
        raise ValueError(f"Close/Adj Close của {ticker} không hợp lệ hoặc thiếu ngày.")
    factor = adj_close / close
    if not np.isfinite(factor).all() or np.any(factor <= 0):
        raise ValueError(f"Hệ số Adj Close / Close của {ticker} không hợp lệ.")
    return factor


def open_price_matrix(processed_data: dict[str, pd.DataFrame], tickers: list[str]) -> np.ndarray:
    """Build Open marks on a total-return scale for portfolio return simulation."""
    columns: list[np.ndarray] = []
    expected_index = processed_data[tickers[0]].index
    for ticker in tickers:
        frame = processed_data[ticker]
        if not frame.index.equals(expected_index):
            raise ValueError(f"Lịch giao dịch của {ticker} không khớp với các mã còn lại.")
        open_col = next(
            (column for column in frame.columns if str(column).lower() == "open"),
            None,
        )
        if open_col is None:
            raise ValueError(f"Thiếu cột Open cần cho mô phỏng giao dịch: {ticker}")
        raw_open = frame[open_col].to_numpy(dtype=np.float64)
        dates = pd.DatetimeIndex(expected_index)
        factor = _total_return_adjustment_factor(frame, dates, ticker)
        columns.append(raw_open * factor)
    prices = np.column_stack(columns)
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("Giá Open phải hữu hạn và lớn hơn 0.")
    return prices


def close_price_matrix(
    processed_data: dict[str, pd.DataFrame],
    tickers: list[str],
    dates: list,
) -> np.ndarray:
    """Build Close marks on the same total-return scale as Open marks."""
    expected_index = pd.DatetimeIndex(pd.to_datetime(dates))
    columns: list[np.ndarray] = []
    for ticker in tickers:
        frame = processed_data[ticker]
        close_col = next(
            (column for column in frame.columns if str(column).lower() == "close"),
            None,
        )
        if close_col is None:
            raise ValueError(f"Thiếu cột Close để dựng correlation graph: {ticker}")
        aligned = frame.reindex(expected_index)[close_col].to_numpy(dtype=np.float64)
        if not np.isfinite(aligned).all() or np.any(aligned <= 0):
            raise ValueError(f"Giá Close của {ticker} không căn đủ theo lịch feature.")
        factor = _total_return_adjustment_factor(frame, expected_index, ticker)
        columns.append(aligned * factor)
    return np.column_stack(columns)


def prepare_data(args: argparse.Namespace, reward_beta: float):
    if not args.raw_dir.exists() or not any(args.raw_dir.glob("*.parquet")):
        raise FileNotFoundError(
            f"Không tìm thấy OHLCV parquet trong {args.raw_dir}. "
            "Hãy đặt dữ liệu raw vào thư mục này trước khi train."
        )

    print("[1/6] Tính feature, căn lịch và ghi processed parquet...")
    pipeline = FeaturePipeline()
    processed_data, full_tensor = pipeline.run_from_raw(
        raw_dir=args.raw_dir,
        export_parquet=args.resume is None,
        output_file=args.processed_output,
    )
    # Phân chia dữ liệu theo thứ tự ưu tiên: CLI > configs/assets.yaml > Tự động tỷ lệ
    train_end = args.train_end
    val_start = args.val_start
    val_end = args.val_end
    test_start = args.test_start

    assets_config_path = getattr(args, "assets_config", PROJECT_ROOT / "configs/assets.yaml")
    if Path(assets_config_path).exists():
        assets_cfg = AssetsConfig(str(assets_config_path))
        cfg_splits = assets_cfg.get_split_dates()
        train_end = train_end or cfg_splits.get("train_end")
        val_start = val_start or cfg_splits.get("val_start")
        val_end = val_end or cfg_splits.get("val_end")
        test_start = test_start or cfg_splits.get("test_start")

    # Kiểm tra tính tương thích giữa mốc split và dải ngày thực tế của full_tensor
    tensor_start = pd.Timestamp(full_tensor.dates[0])
    tensor_end = pd.Timestamp(full_tensor.dates[-1])

    need_auto = False
    if not (train_end and val_start and val_end and test_start):
        need_auto = True
    else:
        t_end = pd.Timestamp(train_end)
        v_start = pd.Timestamp(val_start)
        v_end = pd.Timestamp(val_end)
        t_start = pd.Timestamp(test_start)
        if not (tensor_start <= t_end < v_start <= v_end < t_start <= tensor_end):
            need_auto = True

    if need_auto:
        print(
            f"[CẢNH BÁO] Các mốc split ({train_end}, {val_start}, {val_end}, {test_start}) "
            f"không khớp với dải ngày dữ liệu thực tế ({tensor_start.date()} -> {tensor_end.date()}). "
            f"Hệ thống tự động phân chia theo tỷ lệ 70% Train / 15% Val / 15% Test."
        )
        splitter = TemporalSplitter.from_dates(full_tensor.dates, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    else:
        splitter = TemporalSplitter(
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
        )

    split = splitter.split_market_data_tensor(full_tensor)
    with args.env_config.open("r", encoding="utf-8") as stream:
        configured_lookback = yaml.safe_load(stream)["environment"].get("lookback_window", 20)
    lookback = int(configured_lookback if args.lookback is None else args.lookback)
    if lookback <= 0:
        raise ValueError("--lookback phải lớn hơn 0.")
    if getattr(full_tensor, "warmup_sessions", 0):
        offset = lookback - 1
        if len(split.train.dates) <= offset + 2:
            raise ValueError("Train không đủ dữ liệu sau feature warm-up và lookback.")
        split.train = MarketDataTensor(
            tensor=split.train.tensor[offset:].copy(), tickers=list(split.train.tickers),
            feature_names=list(split.train.feature_names), dates=list(split.train.dates[offset:]),
        )
        split.dates_info["train_range"] = f"{split.train.dates[0]} -> {split.train.dates[-1]} (T={len(split.train.dates)}; warm-up excluded)"

    scaler = MarketFeatureScaler()
    scaler.fit(split.train)
    if args.resume is not None:
        previous = args._resume_payload
        if previous.get("scaler_params") != scaler.to_dict():
            raise ValueError("Scaler của resume không khớp dữ liệu/split train.")
        if previous.get("split_dates") != split.dates_info:
            raise ValueError("Resume yêu cầu cùng split_dates.")
    train_tensor = scaler.transform(split.train)
    val_tensor = scaler.transform(split.val)
    test_tensor = scaler.transform(split.test)
    scaled_full_tensor = scaler.transform(full_tensor)
    args.scaler_output.parent.mkdir(parents=True, exist_ok=True)
    scaler.save(args.scaler_output)
    args._scaler_params = scaler.to_dict()
    args._feature_parameters = pipeline.parameters
    if hasattr(args, "_run_dir"):
        scaler.save(args._run_dir / "scaler.json")

    tickers = list(full_tensor.tickers)
    all_open_prices = open_price_matrix(processed_data, tickers)
    all_close_prices = close_price_matrix(processed_data, tickers, full_tensor.dates)
    date_to_index = {str(date): index for index, date in enumerate(full_tensor.dates)}

    def prices_for(partition, price_matrix: np.ndarray) -> np.ndarray:
        indices = [date_to_index[str(date)] for date in partition.dates]
        return price_matrix[indices].copy()

    print(split.summary())
    print(f"Features: {', '.join(full_tensor.feature_names)}")
    print(f"Transaction-cost reward beta: {reward_beta:g}")
    print(
        "Research caveat: the configured asset list is fixed retrospectively; "
        "no point-in-time membership or delisted-asset history is supplied."
    )
    print("Portfolio return basis: Adj Close adjusted Open/Close (total return, dividends reinvested).")
    return (
        full_tensor,
        processed_data,
        scaled_full_tensor,
        split,
        train_tensor,
        val_tensor,
        test_tensor,
        prices_for(split.train, all_open_prices),
        prices_for(split.val, all_open_prices),
        prices_for(split.test, all_open_prices),
        prices_for(split.train, all_close_prices),
        prices_for(split.val, all_close_prices),
        prices_for(split.test, all_close_prices),
        tickers,
    )

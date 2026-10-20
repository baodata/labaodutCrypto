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

    # Phân chia dữ liệu theo thứ tự ưu tiên: CLI > configs/assets.yaml > Tự động tỷ lệ
    train_end = args.train_end
    val_start = args.val_start
    val_end = args.val_end
    test_start = args.test_start

    assets_config_path = getattr(args, "assets_config", PROJECT_ROOT / "configs/core/assets.yaml")
    if Path(assets_config_path).exists():
        assets_cfg = AssetsConfig(str(assets_config_path))
        cfg_splits = assets_cfg.get_split_dates()
        train_end = train_end or cfg_splits.get("train_end")
        val_start = val_start or cfg_splits.get("val_start")
        val_end = val_end or cfg_splits.get("val_end")
        test_start = test_start or cfg_splits.get("test_start")

    include_test = getattr(args, "evaluation_scope", "validation") != "validation"
    if not include_test and not (train_end and val_start and val_end):
        raise ValueError("Validation-only yêu cầu mốc train/validation tường minh.")
    protocol = getattr(args, "_research_protocol", None)
    if protocol and pd.Timestamp(val_end) > pd.Timestamp(protocol["development_end"]):
        raise ValueError("Validation vượt development_end của protocol.")
    print("[1/6] Tính feature, căn lịch và ghi processed parquet...")
    pipeline = FeaturePipeline(assets_config_path=str(assets_config_path))
    processed_data, full_tensor = pipeline.run_from_raw(
        raw_dir=args.raw_dir,
        export_parquet=args.resume is None,
        output_file=args.processed_output,
        end_date=(getattr(args, "_research_protocol_fold", {}).get("outer_end", val_end)
                  if not include_test else None),
    )
    # Kiểm tra tính tương thích giữa mốc split và dải ngày thực tế của full_tensor
    tensor_start = pd.Timestamp(full_tensor.dates[0])
    tensor_end = pd.Timestamp(full_tensor.dates[-1])

    # Validation-only needs no test dates. Its end cutoff can fall on a
    # weekend/holiday after the last observed trading session.
    if not include_test and test_start is None:
        test_start = (pd.Timestamp(val_end) + pd.Timedelta(days=1)).date().isoformat()
    need_auto = False
    if not (train_end and val_start and val_end and test_start):
        need_auto = True
    else:
        t_end = pd.Timestamp(train_end)
        v_start = pd.Timestamp(val_start)
        v_end = pd.Timestamp(val_end)
        t_start = pd.Timestamp(test_start)
        if not (tensor_start <= t_end < v_start <= v_end < t_start
                and v_end <= tensor_end + pd.Timedelta(days=7)
                and (not include_test or t_start <= tensor_end)):
            need_auto = True

    if need_auto and not include_test:
        raise ValueError("Validation-only không được tự thay đổi split đã chọn.")
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

    args._resolved_split_dates = {
        name: getattr(splitter, name).date().isoformat()
        for name in ('train_end', 'val_start', 'val_end', 'test_start')
    }
    if hasattr(args, '_manifest'):
        args._manifest['resolved_split_dates'] = args._resolved_split_dates.copy()
    split = splitter.split_market_data_tensor(full_tensor, include_test=include_test)
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

    from src.training.quant_context import load_quant_config
    quant_config = load_quant_config(args)
    normalization = quant_config.get("normalization", {})
    if normalization.get("enabled", False):
        from src.features.normalization import CausalFeatureScaler
        scaler = CausalFeatureScaler(**{k:v for k,v in normalization.items() if k != "enabled"})
        if any(scaler.mode(f) == "rolling_zscore" for f in full_tensor.feature_names):
            first_ready = full_tensor.warmup_sessions + scaler.min_periods + lookback - 1
            if first_ready >= len(full_tensor.dates):
                raise ValueError("Insufficient history for rolling normalization and lookback.")
            keep = np.asarray(split.train.dates, dtype="datetime64[ns]") >= np.datetime64(full_tensor.dates[first_ready])
            split.train = MarketDataTensor(split.train.tensor[keep].copy(), list(full_tensor.tickers),
                list(full_tensor.feature_names), list(np.asarray(split.train.dates)[keep]))
            split.train.validate()
            split.dates_info["train_range"] = f"{split.train.dates[0]} -> {split.train.dates[-1]} (T={len(split.train.dates)}; causal normalization warm-up excluded)"
    else:
        scaler = MarketFeatureScaler()
    shared_warmup = int(getattr(args, 'normalization_warmup', 0))
    if shared_warmup < 0:
        raise ValueError("normalization-warmup must be nonnegative.")
    if shared_warmup:
        first_ready = full_tensor.warmup_sessions + shared_warmup + lookback - 1
        if first_ready >= len(full_tensor.dates):
            raise ValueError("Insufficient shared normalization warm-up.")
        keep = np.asarray(split.train.dates, dtype="datetime64[ns]") >= np.datetime64(full_tensor.dates[first_ready])
        split.train = MarketDataTensor(split.train.tensor[keep].copy(), list(full_tensor.tickers),
            list(full_tensor.feature_names), list(np.asarray(split.train.dates)[keep]))
        split.train.validate()
        split.dates_info['train_range'] = f"{split.train.dates[0]} -> {split.train.dates[-1]} (T={len(split.train.dates)}; shared causal warm-up excluded)"
    scaler.fit(split.train)
    scaled_full_tensor = scaler.transform(full_tensor)
    if normalization.get('enabled',False) and hasattr(args,'_run_dir'):
        from src.utils.artifacts import atomic_csv
        atomic_csv(scaler.last_batch_diagnostics,args._run_dir/'normalization_daily_diagnostics.csv',index=False)
    def scaled_partition(partition):
        if partition is None:
            return None
        index = pd.DatetimeIndex(full_tensor.dates).get_indexer(pd.DatetimeIndex(partition.dates))
        return MarketDataTensor(scaled_full_tensor.tensor[index].copy(), list(partition.tickers),
            list(partition.feature_names), list(partition.dates))
    train_tensor = scaled_partition(split.train)
    val_tensor = scaled_partition(split.val)
    test_tensor = scaled_partition(split.test) if include_test else None
    if normalization.get("enabled", False):
        from src.utils.artifacts import atomic_json
        # Fitted statistics stay frozen; inference starts with causal state through train end.
        train_end_index = pd.DatetimeIndex(full_tensor.dates).get_loc(pd.Timestamp(split.train.dates[-1]))
        prefix = MarketDataTensor(full_tensor.tensor[:train_end_index+1], list(full_tensor.tickers),
            list(full_tensor.feature_names), list(full_tensor.dates[:train_end_index+1]), full_tensor.warmup_sessions)
        scaler.transform(prefix)
        scaler = CausalFeatureScaler.from_dict(scaler.last_batch_state)
        scaler.set_session_calendar(full_tensor.dates)
        if hasattr(args, "_run_dir"):
            atomic_json(args._run_dir / "normalization_train_state.json", scaler.to_dict())
    if args.resume is not None:
        if args._resume_payload.get("scaler_params") != scaler.to_dict() or args._resume_payload.get("split_dates") != split.dates_info:
            raise ValueError("Resume scaler/split mismatch.")
    args.scaler_output.parent.mkdir(parents=True, exist_ok=True)
    scaler.save(args.scaler_output)
    args._scaler_params = scaler.to_dict()
    args._feature_parameters = pipeline.parameters
    if hasattr(args, "_run_dir"):
        scaler.save(args._run_dir / "scaler.json")

    tickers = list(full_tensor.tickers)
    all_open_prices = open_price_matrix(processed_data, tickers)
    all_close_prices = close_price_matrix(processed_data, tickers, full_tensor.dates)
    from src.training.quant_context import prepare_quant_context
    scaled_full_tensor.quant_context = prepare_quant_context(args, processed_data, full_tensor, all_close_prices)
    if normalization.get('enabled',False) and scaler.context_features:
        scaled_full_tensor.quant_context['market_context'] = np.stack([scaler.market_context(x) for x in full_tensor.tensor])
        scaled_full_tensor.quant_context['market_context_features'] = list(scaler.context_features)
        if hasattr(args,'_run_dir'):
            atomic_json(args._run_dir/'market_context_metadata.json',dict(features=scaler.context_features,
                channels_per_feature=['train_scaled_cross_section_mean','cross_section_std_over_train_std'],
                tickers=list(full_tensor.tickers)))
    for partition in (train_tensor,val_tensor,test_tensor):
        if partition is not None:
            partition.quant_context = scaled_full_tensor.quant_context
    args._quant_config = quant_config
    fold = getattr(args, "_research_protocol_fold", None)
    if fold:
        dates = pd.DatetimeIndex(pd.to_datetime(full_tensor.dates))
        a = int(dates.searchsorted(pd.Timestamp(fold["outer_start"])))
        b = int(dates.searchsorted(pd.Timestamp(fold["outer_end"]), side="right"))
        outer = MarketDataTensor(scaled_full_tensor.tensor[a:b].copy(), tickers,
                    list(full_tensor.feature_names), list(full_tensor.dates[a:b]))
        outer.validate()
        if b - a < 2 or dates[b-1] < pd.Timestamp(fold["outer_end"]) - pd.Timedelta(days=7):
            raise ValueError("Không đủ dữ liệu outer development theo protocol.")
        args._outer_validation_context = {"val_tensor": outer, "val_open": all_open_prices[a:b],
            "val_close": all_close_prices[a:b], "val_history_start": a,
            "evaluation_period": "outer_development", "protocol_fold": fold}
    date_to_index = {str(date): index for index, date in enumerate(full_tensor.dates)}

    def prices_for(partition, price_matrix: np.ndarray) -> np.ndarray:
        if partition is None:
            return None
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

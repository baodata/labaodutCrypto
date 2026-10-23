# cái này dùng để tải dữ liệu thị trường và chuẩn hóa, không phải để chạy huấn luyện hay backtest.

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf

from src.data.loader import get_all_tickers, load_config

logger = logging.getLogger(__name__)

STANDARD_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]


# chuẩn hóa dữ liệu OHLCV từ yfinance: 
# - làm sạch các cột
# - đổi tên các cột để thống nhất
# - đảm bảo cột date là tz-naive
# - loại bỏ các hàng trùng lặp và sắp xếp theo ngày
def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    df_clean = df.copy()

    # Làm phẳng cột nếu yfinance trả về dạng MultiIndex (Price, Ticker)
    if isinstance(df_clean.columns, pd.MultiIndex):
        df_clean.columns = [str(col[0]).strip().lower().replace(" ", "_") for col in df_clean.columns]
    else:
        df_clean.columns = [str(col).strip().lower().replace(" ", "_") for col in df_clean.columns]

    # Đưa date về dạng cột nếu đang là index
    if "date" not in df_clean.columns:
        df_clean = df_clean.reset_index()
        first_col = df_clean.columns[0]
        df_clean = df_clean.rename(columns={first_col: "date"})

    # Đổi tên các biến thể của adj_close
    col_mapping = {}
    for col in df_clean.columns:
        if col in ["adjusted_close", "adjclose"]:
            col_mapping[col] = "adj_close"
    df_clean = df_clean.rename(columns=col_mapping)

    # Nếu không có adj_close thì dùng close thay thế
    if "adj_close" not in df_clean.columns and "close" in df_clean.columns:
        df_clean["adj_close"] = df_clean["close"]

    # Đảm bảo cột date không dính múi giờ (tz-naive)
    df_clean["date"] = pd.to_datetime(df_clean["date"])
    if hasattr(df_clean["date"].dt, "tz") and df_clean["date"].dt.tz is not None:
        df_clean["date"] = df_clean["date"].dt.tz_localize(None)
    df_clean["date"] = df_clean["date"].dt.normalize()

    # Ép kiểu dữ liệu số cho các trường giá và khối lượng
    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Giữ đúng các cột tiêu chuẩn và sắp xếp theo ngày
    available_cols = [c for c in STANDARD_COLUMNS if c in df_clean.columns]
    df_clean = df_clean[available_cols]
    df_clean = df_clean.drop_duplicates(subset=["date"], keep="last")
    df_clean = df_clean.sort_values("date").reset_index(drop=True)

    # Loại bỏ các dòng nếu toàn bộ giá bị NaN
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df_clean.columns]
    if price_cols:
        df_clean = df_clean.dropna(subset=price_cols, how="all").reset_index(drop=True)

    return df_clean


# Hàm này tải dữ liệu OHLCV cho một ticker duy nhất
def download_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> pd.DataFrame:
    ticker_clean = ticker.strip().upper()
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            raw_df = yf.download(
                tickers=ticker_clean,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
            )
            if raw_df is not None and not raw_df.empty:
                df = normalize_ohlcv(raw_df)
                if not df.empty:
                    return df
        except Exception as e:
            last_error = e

        if attempt < max_retries:
            time.sleep(retry_delay * attempt)

    if last_error:
        logger.warning(f"Không thể tải {ticker_clean}: {last_error}")
    return pd.DataFrame(columns=STANDARD_COLUMNS)


# Hàm này tải dữ liệu cho tất cả các ticker được chỉ định hoặc từ cấu hình, lưu dưới dạng Parquet.
def download_and_save_all(
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir: str = "data/raw",
    config_path: str = "configs/assets.yaml",
    include_benchmark: bool = True,
) -> Dict[str, bool]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path) if (tickers is None or start_date is None or end_date is None) else {}

    if tickers is None:
        tickers = get_all_tickers(cfg)
        if include_benchmark:
            bm = cfg.get("benchmark", "SPY")
            if bm and bm not in tickers:
                tickers.append(bm)

    final_start = start_date or cfg.get("start_date", "2015-01-01")
    final_end = end_date or cfg.get("end_date", "2025-12-31")

    results: Dict[str, bool] = {}
    for ticker in tickers:
        df = download_ticker(ticker, start_date=final_start, end_date=final_end)
        if not df.empty:
            file_path = out_path / f"{ticker.upper()}.parquet"
            df.to_parquet(file_path, index=False, engine="pyarrow")
            results[ticker] = True
        else:
            results[ticker] = False

    return results

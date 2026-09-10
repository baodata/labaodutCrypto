"""
src/data/downloader.py
Module tải dữ liệu giá lịch sử OHLCV từ Yahoo Finance.

FILE NÀY ĐỂ LÀM GÌ?
- Trái ngược với script `download_data.py` ở vòng ngoài chỉ để làm giao diện gõ lệnh (CLI),
  file này nằm ở tầng "Lõi" (Core).
- Nhiệm vụ chính: Kết nối trực tiếp vào Yahoo Finance, kéo dữ liệu thô về.
- Rất quan trọng: Nó chứa logic chuẩn hóa định dạng dữ liệu, ép kiểu cột, chuyển đổi thời gian 
  (time-zone), và lọc dòng lỗi. Đảm bảo dữ liệu tải về ở tình trạng "Sạch" và chuẩn form nhất
  cho toàn bộ dự án.

Ticket: DATA-002 (P0)
Primary Owner: Thành viên A
Reviewer: Thành viên B
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import yfinance as yf

from src.utils.config import AssetsConfig, get_project_root

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Chuẩn hóa tên cột thống nhất cho toàn bộ dự án
STANDARD_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]


def normalize_ohlcv_dataframe(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """
    Chuẩn hóa DataFrame OHLCV từ yfinance:
    - Xử lý MultiIndex columns nếu có.
    - Đưa 'date' thành cột thông thường (tz-naive).
    - Chuẩn hóa tên cột thành chữ thường: date, open, high, low, close, adj_close, volume.
    - Chuyển đổi kiểu dữ liệu sang float / int hợp lệ.
    - Loại bỏ hàng trùng lặp và sắp xếp theo ngày tăng dần.
    """
    if df.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    df_clean = df.copy()

    # 1. Xử lý MultiIndex columns (thường xuất hiện ở các bản yfinance gần đây)
    if isinstance(df_clean.columns, pd.MultiIndex):
        # MultiIndex có thể có dạng ('Close', 'AAPL') hoặc ('AAPL', 'Close')
        new_cols = []
        for col in df_clean.columns:
            parts = [str(p) for p in col if str(p) != ""]
            # Tìm phần tử đại diện cho tên chỉ số giá
            metric_candidates = [
                p for p in parts if p.lower() in [
                    "open", "high", "low", "close", "adj close", "adj_close", "volume"
                ]
            ]
            if metric_candidates:
                new_cols.append(metric_candidates[0])
            else:
                new_cols.append(parts[0] if parts else "unknown")
        df_clean.columns = new_cols

    # 2. Xử lý cột date / index
    if "Date" in df_clean.columns:
        df_clean = df_clean.rename(columns={"Date": "date"})
    elif "date" not in df_clean.columns:
        df_clean = df_clean.reset_index()
        # Cột index vừa được reset có thể có tên 'Date', 'index', hoặc DatetimeIndex
        first_col = df_clean.columns[0]
        df_clean = df_clean.rename(columns={first_col: "date"})

    # Chuẩn hóa tên toàn bộ cột về chữ thường và bỏ khoảng trắng thừa
    col_mapping = {}
    for col in df_clean.columns:
        clean_name = str(col).strip().lower().replace(" ", "_")
        if clean_name in ["adj_close", "adjusted_close", "adjclose"]:
            col_mapping[col] = "adj_close"
        elif clean_name in ["open", "high", "low", "close", "volume", "date"]:
            col_mapping[col] = clean_name
    df_clean = df_clean.rename(columns=col_mapping)

    # Nếu thiếu adj_close nhưng có close, gán adj_close = close
    if "adj_close" not in df_clean.columns and "close" in df_clean.columns:
        df_clean["adj_close"] = df_clean["close"]

    # 3. Chuẩn hóa cột date về dạng datetime không timezone (YYYY-MM-DD)
    df_clean["date"] = pd.to_datetime(df_clean["date"])
    if hasattr(df_clean["date"].dt, "tz") and df_clean["date"].dt.tz is not None:
        df_clean["date"] = df_clean["date"].dt.tz_localize(None)

    # Giữ ngày ở dạng YYYY-MM-DD
    df_clean["date"] = df_clean["date"].dt.normalize()

    # 4. Ép kiểu dữ liệu số
    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # 5. Lọc và kiểm tra các cột bắt buộc
    available_cols = [c for c in STANDARD_COLUMNS if c in df_clean.columns]
    df_clean = df_clean[available_cols]

    # Loại bỏ hàng trùng lặp ngày, sắp xếp theo ngày tăng dần
    df_clean = df_clean.drop_duplicates(subset=["date"], keep="last")
    df_clean = df_clean.sort_values("date").reset_index(drop=True)

    # Loại bỏ các dòng mà tất cả giá đều là NaN
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df_clean.columns]
    if price_cols:
        df_clean = df_clean.dropna(subset=price_cols, how="all").reset_index(drop=True)

    return df_clean


class OHLCVDownloader:
    """
    Downloader chuyên nghiệp tải và lưu trữ dữ liệu OHLCV.
    """

    def __init__(
        self,
        output_dir: Union[Path, str] = "data/raw",
        save_format: str = "parquet",
    ):
        path = Path(output_dir)
        if not path.is_absolute():
            path = get_project_root() / path
        self.output_dir = path
        self.output_dir.mkdir(parents=True, exist_ok=True)

        fmt = save_format.lower()
        if fmt not in ["parquet", "csv", "both"]:
            raise ValueError(f"Định dạng {save_format} không hợp lệ. Chọn 'parquet', 'csv', hoặc 'both'.")
        self.save_format = fmt

    def download_ticker(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> pd.DataFrame:
        """
        Tải dữ liệu OHLCV của 1 mã cổ phiếu từ yfinance với cơ chế retry.
        """
        ticker_clean = ticker.strip().upper()
        logger.info(f"Đang tải {ticker_clean} ({start_date} -> {end_date})...")

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                # Tải từ yfinance (auto_adjust=False để giữ nguyên cả Close và Adj Close)
                raw_df = yf.download(
                    tickers=ticker_clean,
                    start=start_date,
                    end=end_date,
                    auto_adjust=False,
                    progress=False,
                )

                if raw_df is not None and not raw_df.empty:
                    df = normalize_ohlcv_dataframe(raw_df, ticker=ticker_clean)
                    if len(df) > 0:
                        logger.info(f"  ✓ {ticker_clean}: Tải thành công {len(df)} phiên giao dịch.")
                        return df
                    else:
                        logger.warning(f"  ⚠ {ticker_clean}: Dữ liệu sau chuẩn hóa rỗng.")

                logger.warning(f"  ⚠ Lần thử {attempt}/{max_retries}: {ticker_clean} chưa có dữ liệu trả về.")
            except Exception as e:
                last_error = e
                logger.warning(f"  ⚠ Lỗi lần {attempt}/{max_retries} khi tải {ticker_clean}: {e}")

            if attempt < max_retries:
                time.sleep(retry_delay * attempt)

        logger.error(f"  ✗ {ticker_clean}: Không thể tải dữ liệu sau {max_retries} lần thử. Lỗi cuối: {last_error}")
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    def save_ticker_data(self, df: pd.DataFrame, ticker: str) -> List[Path]:
        """
        Lưu DataFrame vào thư mục output_dir theo định dạng đã cấu hình.
        """
        if df.empty:
            logger.warning(f"Bỏ qua lưu file cho {ticker} do dữ liệu rỗng.")
            return []

        saved_paths: List[Path] = []
        ticker_clean = ticker.strip().upper()

        if self.save_format in ["parquet", "both"]:
            parquet_path = self.output_dir / f"{ticker_clean}.parquet"
            try:
                df.to_parquet(parquet_path, index=False, engine="pyarrow")
                saved_paths.append(parquet_path)
            except Exception as e:
                logger.warning(f"Lỗi khi lưu Parquet cho {ticker_clean} ({e}). Chuyển sang lưu CSV.")
                csv_fallback = self.output_dir / f"{ticker_clean}.csv"
                df.to_csv(csv_fallback, index=False)
                saved_paths.append(csv_fallback)

        if self.save_format in ["csv", "both"] and not any(p.suffix == ".csv" for p in saved_paths):
            csv_path = self.output_dir / f"{ticker_clean}.csv"
            df.to_csv(csv_path, index=False)
            saved_paths.append(csv_path)

        return saved_paths

    def download_and_save(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        max_retries: int = 3,
    ) -> Dict[str, bool]:
        """
        Tải và lưu lần lượt danh sách các mã cổ phiếu.
        """
        results = {}
        for ticker in tickers:
            df = self.download_ticker(ticker, start_date, end_date, max_retries=max_retries)
            if not df.empty:
                paths = self.save_ticker_data(df, ticker)
                results[ticker] = len(paths) > 0
            else:
                results[ticker] = False
        return results


def download_market_data(
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir: Union[Path, str] = "data/raw",
    save_format: str = "parquet",
    include_benchmark: bool = True,
    config_path: str = "configs/assets.yaml",
) -> Dict[str, bool]:
    """
    API hàm cấp cao phục vụ toàn bộ pipeline:
    Tải và lưu trữ dữ liệu OHLCV tự động theo cấu hình assets.yaml.
    """
    config = AssetsConfig(config_path)

    # 1. Xác định danh sách tickers
    target_tickers = list(tickers) if tickers is not None else config.get_tickers()
    if include_benchmark:
        bm = config.get_benchmark_ticker()
        if bm not in target_tickers:
            target_tickers.append(bm)

    # 2. Xác định khoảng thời gian
    cfg_start, cfg_end = config.get_date_range()
    final_start = start_date or cfg_start
    final_end = end_date or cfg_end

    # 3. Khởi tạo Downloader và chạy
    downloader = OHLCVDownloader(output_dir=output_dir, save_format=save_format)
    return downloader.download_and_save(target_tickers, final_start, final_end)


#!/usr/bin/env python3
"""
tải dữ liệu OHLCV (Open, High, Low, Close, Volume)
    python scripts/data/download_data.py
    python scripts/data/download_data.py --tickers AAPL,MSFT,NVDA
    python scripts/data/download_data.py --start-date 2020-01-01 --end-date 2025-12-31
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

# thêm thư mục gốc dự án vào sys.path để chạy ở đâu cũng import được
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader import download_ticker
from src.data.loader import get_all_tickers, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tải dữ liệu thị trường OHLCV lịch sử và lưu vào data/raw/ dạng file Parquet."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/assets.yaml",
        help="Đường dẫn file cấu hình tài sản (mặc định: configs/assets.yaml)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Danh sách mã cổ phiếu phân cách bằng dấu phẩy (ví dụ: AAPL,MSFT,NVDA). Nếu bỏ trống sẽ lấy từ config.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Ngày bắt đầu định dạng YYYY-MM-DD (mặc định: lấy từ configs/assets.yaml)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Ngày kết thúc định dạng YYYY-MM-DD (mặc định: lấy từ configs/assets.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Thư mục lưu file Parquet (mặc định: data/raw)",
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Bỏ qua không tải chỉ số benchmark (SPY)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Đọc cấu hình
    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"[LỖI] Không thể đọc cấu hình từ {args.config}: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Xác định ngày bắt đầu và kết thúc (ưu tiên tham số CLI, nếu không có thì lấy từ config)
    start_date = args.start_date or cfg.get("start_date", "2015-01-01")
    end_date = args.end_date or cfg.get("end_date", "2025-12-31")

    # 3. Tạo thư mục lưu dữ liệu
    output_path = Path(args.output_dir)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    # 4. Xác định danh sách mã cổ phiếu cần tải
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = get_all_tickers(cfg)
        if not args.no_benchmark:
            bm = cfg.get("benchmark", "SPY")
            if bm and bm not in tickers:
                tickers.append(bm)

    print("=" * 70)
    print("QUY TRÌNH TẢI DỮ LIỆU THỊ TRƯỜNG (YAHOO FINANCE -> PARQUET)")
    print("=" * 70)
    print(f" • Số lượng tài sản  : {len(tickers)} ({', '.join(tickers[:6])}{'...' if len(tickers) > 6 else ''})")
    print(f" • Khoảng thời gian  : {start_date} -> {end_date}")
    print(f" • Thư mục lưu trữ   : {output_path}")
    print("-" * 70)

    success_count = 0
    fail_count = 0

    print(f"{'MÃ TÀI SẢN':<12} | {'TRẠNG THÁI':<14} | {'SỐ PHIÊN':<12} | {'DUNG LƯỢNG':<15}")
    print("-" * 70)

    for ticker in tickers:
        df = download_ticker(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            max_retries=3,
        )

        if not df.empty:
            file_path = output_path / f"{ticker}.parquet"
            df.to_parquet(file_path, index=False, engine="pyarrow")
            file_size_kb = file_path.stat().st_size / 1024
            rows = len(df)
            print(f"{ticker:<12} | {'✓ THÀNH CÔNG':<14} | {rows:<12} | {file_size_kb:.1f} KB")
            success_count += 1
        else:
            print(f"{ticker:<12} | {'✗ THẤT BẠI':<14} | {'0':<12} | {'0 KB':<15}")
            fail_count += 1

    print("=" * 70)
    print(f"Kết quả: {success_count}/{len(tickers)} tải thành công, {fail_count} thất bại.")
    if fail_count > 0:
        print("[CẢNH BÁO] Có một số mã không tải được dữ liệu!", file=sys.stderr)
        sys.exit(1)
    else:
        print("✅ Hoàn tất tải toàn bộ dữ liệu.")
        sys.exit(0)


if __name__ == "__main__":
    main()

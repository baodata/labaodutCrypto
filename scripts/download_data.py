#!/usr/bin/env python3
"""
scripts/download_data.py
CLI script tải dữ liệu OHLCV tự động từ Yahoo Finance.

Ticket: DATA-002 (P0)
Usage:
    python scripts/download_data.py
    python scripts/download_data.py --tickers AAPL,MSFT,NVDA --format parquet
    python scripts/download_data.py --start-date 2020-01-01 --end-date 2024-12-31
"""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

from src.data.downloader import OHLCVDownloader
from src.utils.config import AssetsConfig, get_project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tải dữ liệu OHLCV lịch sử từ Yahoo Finance và lưu vào data/raw/."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/assets.yaml",
        help="Đường dẫn file cấu hình vũ trụ cổ phiếu (mặc định: configs/assets.yaml)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Danh sách mã cổ phiếu phân cách bằng dấu phẩy (ví dụ: AAPL,MSFT,NVDA). Nếu không truyền sẽ lấy toàn bộ từ config.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Ngày bắt đầu định dạng YYYY-MM-DD (mặc định: lấy từ config hoặc 2015-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Ngày kết thúc định dạng YYYY-MM-DD (mặc định: lấy từ config hoặc 2025-12-31)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Thư mục lưu dữ liệu tải về (mặc định: data/raw)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["parquet", "csv", "both"],
        default="parquet",
        help="Định dạng file lưu trữ: parquet, csv, hoặc both (mặc định: parquet)",
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Bỏ qua việc tải chỉ số tham chiếu Benchmark (SPY)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Đọc cấu hình
    try:
        config = AssetsConfig(args.config)
    except Exception as e:
        print(f"[ERROR] Không thể đọc cấu hình từ {args.config}: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Xác định danh sách tickers
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = config.get_tickers()

    if not args.no_benchmark:
        bm = config.get_benchmark_ticker()
        if bm not in tickers:
            tickers.append(bm)

    # 3. Xác định ngày
    cfg_start, cfg_end = config.get_date_range()
    start_date = args.start_date or cfg_start
    end_date = args.end_date or cfg_end

    print("=" * 70)
    print(" H-MARL-GNN: QUY TRÌNH TẢI DỮ LIỆU THỊ TRƯỜNG (DATA-002)")
    print("=" * 70)
    print(f" • Số lượng mã:       {len(tickers)} ({', '.join(tickers[:6])}{'...' if len(tickers) > 6 else ''})")
    print(f" • Khoảng thời gian:  {start_date} -> {end_date}")
    print(f" • Thư mục lưu trữ:   {args.output_dir}")
    print(f" • Định dạng lưu:     {args.format.upper()}")
    print("-" * 70)

    # 4. Thực thi tải dữ liệu
    downloader = OHLCVDownloader(output_dir=args.output_dir, save_format=args.format)
    results = downloader.download_and_save(tickers, start_date, end_date)

    # 5. Tổng kết kết quả
    print("\n" + "=" * 70)
    print(f"{'MÃ CP':<10} | {'TRẠNG THÁI':<12} | {'SỐ PHIÊN':<12} | {'KÍCH THƯỚC':<15}")
    print("-" * 70)

    success_count = 0
    fail_count = 0
    out_path = Path(args.output_dir)
    if not out_path.is_absolute():
        out_path = get_project_root() / out_path

    for ticker, success in results.items():
        if success:
            success_count += 1
            # Tìm file vừa lưu
            file_ext = ".parquet" if args.format in ["parquet", "both"] else ".csv"
            saved_file = out_path / f"{ticker}{file_ext}"
            if saved_file.exists():
                size_kb = saved_file.stat().st_size / 1024
                # Đọc nhanh số dòng
                try:
                    if file_ext == ".parquet":
                        df = pd.read_parquet(saved_file)
                    else:
                        df = pd.read_csv(saved_file)
                    rows = len(df)
                    print(f"{ticker:<10} | {'THÀNH CÔNG':<12} | {rows:<12} | {size_kb:.1f} KB")
                except Exception:
                    print(f"{ticker:<10} | {'THÀNH CÔNG':<12} | {'N/A':<12} | {size_kb:.1f} KB")
            else:
                print(f"{ticker:<10} | {'THÀNH CÔNG':<12} | {'-':<12} | {'-':<15}")
        else:
            fail_count += 1
            print(f"{ticker:<10} | {'THẤT BẠI':<12} | {'0':<12} | {'0 KB':<15}")

    print("=" * 70)
    print(f"Tổng kết: {success_count}/{len(tickers)} thành công, {fail_count} thất bại.")
    if fail_count > 0:
        print("[CẢNH BÁO] Có một số mã không thể tải dữ liệu!", file=sys.stderr)
        sys.exit(1)
    else:
        print("✓ Tải dữ liệu toàn bộ hoàn tất mỹ mãn.")
        sys.exit(0)


if __name__ == "__main__":
    main()


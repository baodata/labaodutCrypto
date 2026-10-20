#!/usr/bin/env python3
"""
==============================================================================
TÊN SCRIPT: scripts/compare_gp_allocators.py
MỤC ĐÍCH:
    Huấn luyện và so sánh bộ phân bổ danh mục tiến hóa bằng Lập trình Di truyền
    (Genetic Programming - GP Symbolic Expressions) với mô hình Convex tối ưu
    và baseline 1/N trên các fold out-of-fold (OOF).
    
    BƯỚC TIẾP THEO: Sau khi chạy xong, dùng scripts/audit_gp_run.py để hậu kiểm toán
    tính toàn vẹn mã băm và trích xuất chỉ số tín hiệu.

HƯỚNG DẪN CHẠY (CLI USAGE):
    # 1. Chạy so sánh GP trên tất cả các fold:
    python scripts/compare_gp_allocators.py --predictor-lock models/runs/predictor_lock.json

    # 2. Chạy riêng một fold cụ thể:
    python scripts/compare_gp_allocators.py \\
        --predictor-lock models/runs/predictor_lock.json \\
        --fold development_2021

ĐẦU VÀO CẦN THIẾT (PREREQUISITES):
    - models/runs/predictor_lock.json (Mô hình dự báo đã khóa)
    - configs/research/gp_allocator.yaml, configs/core/env.yaml

ĐẦU RA SINH RA (OUTPUT ARTIFACTS):
    - models/runs/gp_research_<timestamp>/
        + cases/<fold>/<seed>/: metrics.json, daily_returns.csv, decisions.jsonl
        + manifest.json: Lưu mã băm dữ liệu, mã nguồn và runtime
==============================================================================
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threadpoolctl import threadpool_limits
from src.evaluation.gp_research import compare_gp_allocators


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictor-lock', type=Path, required=True)
    parser.add_argument('--gp-config', type=Path, default=ROOT/'configs/research/gp_allocator.yaml')
    parser.add_argument('--allocation-config', type=Path, default=ROOT/'configs/test/signal_allocation_neural.yaml')
    parser.add_argument('--raw-dir', type=Path, default=ROOT/'data/raw')
    parser.add_argument('--assets-config', type=Path, default=ROOT/'configs/core/assets.yaml')
    parser.add_argument('--env-config', type=Path, default=ROOT/'configs/core/env.yaml')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', type=Path, help='Restart unfinished cases from cash in the same immutable configuration.')
    parser.add_argument('--fold', help='Optional single fold, development OOF/inner only.')
    parser.add_argument('--evaluate-development-outer', action='store_true',
        help='Explicit full-fold exploratory outer backtest; never 2024-2025 test.')
    args = parser.parse_args()
    if args.resume and args.output_dir:
        parser.error('Use --resume or --output-dir, not both.')
    with threadpool_limits(limits=1):
        compare_gp_allocators(args)


if __name__ == '__main__':
    main()

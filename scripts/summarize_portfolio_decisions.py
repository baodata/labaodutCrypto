#!/usr/bin/env python3
"""
==============================================================================
TÊN SCRIPT: scripts/summarize_portfolio_decisions.py
MỤC ĐÍCH:
    Tái tổng hợp báo cáo so sánh ghép cặp (paired summary report) từ chuỗi lợi nhuận
    (daily returns) của một lượt chạy compare_portfolio_decisions đã hoàn tất.
    Không huấn luyện lại, không chọn lại mô hình.
    
    LƯU Ý GỘP: Chức năng này đã được tích hợp trực tiếp vào script:
        python scripts/compare_portfolio_decisions.py --summary-only ...
    Script này được duy trì để đảm bảo tính tương thích ngược cho các lệnh CLI cũ.

HƯỚNG DẪN CHẠY (CLI USAGE):
    python scripts/summarize_portfolio_decisions.py \\
        --run-dir models/runs/portfolio_decision_.../ \\
        --output-dir reports/derived_summary_run1/

ĐẦU VÀO CẦN THIẾT (PREREQUISITES):
    - Thư mục run cũ chứa metrics_by_fold.csv, daily_returns.csv, predictor_lock.json

ĐẦU RA SINH RA (OUTPUT ARTIFACTS):
    - paired_comparison.csv: Bảng tổng kết thống kê ghép cặp
    - manifest.json: Nhật ký nguồn gốc dữ liệu
==============================================================================
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_portfolio_decisions import run_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Thư mục kết quả portfolio decisions đã hoàn thành.")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Thư mục mới lưu báo cáo tái tạo (không được trùng thư mục cũ).")
    args = parser.parse_args()
    run_summary(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
==============================================================================
TÊN SCRIPT: scripts/build_prediction_bank.py
MỤC ĐÍCH:
    Xây dựng "Ngân hàng dự báo" (Prediction Bank) chứa các dự báo tỷ suất sinh lời
    của các mô hình (Ridge Regression, Momentum, Mean Reversion, GNN, GRU)
    trên các fold Cross-Validation mà không làm rò rỉ nhãn (label-free forecasts).
    Dữ liệu dự báo này được dùng làm đầu vào cho các thuật toán phân bổ danh mục.
    
    BƯỚC TIẾP THEO: Chạy scripts/lock_predictor.py để khóa mô hình tốt nhất theo Inner-IC.

HƯỚNG DẪN CHẠY (CLI USAGE):
    # 1. Xây dựng ngân hàng dự báo đầy đủ:
    python scripts/build_prediction_bank.py --output-dir models/runs/prediction_bank/

    # 2. Chỉ tạo dự báo trên tập phát triển (Development only, bỏ qua outer fold):
    python scripts/build_prediction_bank.py --development-only

ĐẦU VÀO CẦN THIẾT (PREREQUISITES):
    - Dữ liệu nến thô trong: data/raw/*.parquet
    - configs/research/prediction_bank.yaml
    - configs/core/quant_research_protocol.yaml

ĐẦU RA SINH RA (OUTPUT ARTIFACTS):
    - models/runs/prediction_bank/
        + predictions.parquet: Ma trận dự báo sinh lời theo ngày/mã/mô hình
        + labels.csv: Nhãn lợi nhuận tương lai thực tế (cách ly riêng)
        + prediction_bank_manifest.json: Mã băm SHA-256 xác thực bất biến
==============================================================================
"""


from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.prediction_research import run_prediction_research


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-config", type=Path, default=ROOT / "configs/research/prediction_bank.yaml")
    parser.add_argument("--research-protocol", type=Path, default=ROOT / "configs/core/quant_research_protocol.yaml")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--assets-config", type=Path, default=ROOT / "configs/core/assets.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--development-only", action="store_true", help="Forecast train OOF/inner only; never forecast outer.")
    args = parser.parse_args(argv)
    run_prediction_research(args.research_protocol, args.prediction_config,
                            args.raw_dir, args.assets_config, args.output_dir, development_only=args.development_only)


if __name__ == "__main__":
    main()

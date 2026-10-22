#!/usr/bin/env python3
"""
==============================================================================
TÊN SCRIPT: scripts/compare_signal_allocators.py
MỤC ĐÍCH:
    So sánh các chiến lược phân bổ danh mục dựa trên tín hiệu từ Prediction Bank:
    Ridge Regression Allocator vs Momentum Allocator vs Mean-Reversion Allocator vs 1/N.
    Áp dụng cùng giả định chi phí giao dịch, độ lệch cận biên (bounded tilts)
    và đo lường Sharpe, Drawdown, Turnover và Chi phí thực tế.

HƯỚNG DẪN CHẠY (CLI USAGE):
    # 1. So sánh bằng Prediction Bank đã có sẵn:
    python scripts/compare_signal_allocators.py --prediction-bank models/runs/prediction_bank/

    # 2. Tự động xây dựng bank và so sánh luôn:
    python scripts/compare_signal_allocators.py --output-dir models/runs/signal_allocation_run1/

ĐẦU VÀO CẦN THIẾT (PREREQUISITES):
    - Ngân hàng dự báo (models/runs/prediction_bank/)
    - configs/research/signal_allocation.yaml
    - configs/core/quant_research_protocol.yaml, configs/core/env.yaml

ĐẦU RA SINH RA (OUTPUT ARTIFACTS):
    - models/runs/signal_allocation_<timestamp>/comparison/
        + metrics_by_fold.csv: Bảng Sharpe, Return, Turnover theo từng fold
        + daily_returns.csv: Lợi nhuận hàng ngày của từng chiến lược
        + paired_comparisons.csv: Kiểm định chênh lệch so với baseline 1/N
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
from src.evaluation.signal_allocation import compare_signal_allocators
from src.utils.artifacts import new_run_directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-bank", type=Path, default=None)
    parser.add_argument("--prediction-config", type=Path, default=ROOT / "configs/research/prediction_bank.yaml")
    parser.add_argument("--research-protocol", type=Path, default=ROOT / "configs/core/quant_research_protocol.yaml")
    parser.add_argument("--allocation-config", type=Path, default=ROOT / "configs/research/signal_allocation.yaml")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--assets-config", type=Path, default=ROOT / "configs/core/assets.yaml")
    parser.add_argument("--env-config", type=Path, default=ROOT / "configs/core/env.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output_dir or new_run_directory(ROOT / "models/runs", "signal_allocation")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Output directory already contains artifacts; choose a new run.")
    bank_dir = args.prediction_bank
    if bank_dir is None:
        built_bank, bank_dir = run_prediction_research(args.research_protocol, args.prediction_config,
            args.raw_dir, args.assets_config, output / "prediction_bank")
        del built_bank  # avoid retaining a second full bank during the comparison
    compare_signal_allocators(bank_dir, args.raw_dir, args.assets_config, args.env_config,
                              args.allocation_config, output / "comparison")


if __name__ == "__main__":
    main()

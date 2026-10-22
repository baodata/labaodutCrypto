#!/usr/bin/env python3
"""
==============================================================================
TÊN SCRIPT: scripts/compare_portfolio_decisions.py
MỤC ĐÍCH:
    So sánh đối đầu trực tiếp 4 cấp độ ra quyết định danh mục đầu tư:
        Convex Optimization  -->  PPO Agent  -->  Sector MARL  -->  Hierarchical MARL (H-MARL)
    Sử dụng mô hình dự báo đã được khóa bất biến (Predictor Lock) để đảm bảo tính
    khách quan, đo đạc Sharpe Delta, turnover, chi phí và kiểm định thống kê ghép cặp.
    
    Đồng thời hỗ trợ chế độ `--summary-only` để tái tạo báo cáo so sánh ghép cặp
    từ thư mục kết quả đã chạy mà không cần huấn luyện lại.

HƯỚNG DẪN CHẠY (CLI USAGE):
    # 1. Chạy so sánh đầy đủ 4 mô hình:
    python scripts/compare_portfolio_decisions.py \\
        --predictor-lock models/runs/predictor_lock.json

    # 2. Chạy thử nghiệm chỉ trên tập phát triển (Development only, bỏ qua test):
    python scripts/compare_portfolio_decisions.py \\
        --predictor-lock models/runs/predictor_lock.json \\
        --development-only

    # 3. Chỉ tính toán lại báo cáo tổng hợp từ run cũ (Summary only):
    python scripts/compare_portfolio_decisions.py \\
        --summary-only \\
        --run-dir models/runs/portfolio_decision_20261020/ \\
        --output-dir reports/summary_run1/

ĐẦU VÀO CẦN THIẾT (PREREQUISITES):
    - models/runs/predictor_lock.json (Mô hình dự báo đã khóa)
    - data/raw/*.parquet (Dữ liệu giá nến)
    - configs/research/portfolio_decision.yaml
    - configs/test/signal_allocation_neural.yaml

ĐẦU RA SINH RA (OUTPUT ARTIFACTS):
    - models/runs/portfolio_decision_<timestamp>/
        + metrics_by_fold.csv: Bảng Sharpe, Return, Drawdown theo từng fold
        + daily_returns.csv: Lợi nhuận hàng ngày của 4 mô hình
        + paired_comparison.csv: Kiểm định chênh lệch ghép cặp và p-value
        + decisions.jsonl: Chi tiết tỷ trọng phân bổ từng ngày
==============================================================================
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.evaluation.portfolio_decision import compare_portfolio_decisions, summarize_decisions
from src.utils.artifacts import atomic_csv, atomic_json, file_hashes


def run_summary(run_dir: Path, output_dir: Path) -> None:
    """Tái tổng hợp báo cáo ghép cặp từ thư mục run đã có sẵn."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Thư mục báo cáo đầu ra đã chứa dữ liệu, vui lòng chọn thư mục mới.")
    root = run_dir.resolve()
    lock = json.loads((root / "predictor_lock.json").read_text())
    bank_manifest = Path(lock["prediction_bank"]) / "prediction_bank_manifest.json"
    if hashlib.sha256(bank_manifest.read_bytes()).hexdigest() != lock["bank_manifest_sha256"]:
        raise ValueError("Mã băm của bank manifest không khớp với predictor lock.")
    metadata = json.loads(bank_manifest.read_text())
    metrics = pd.read_csv(root / "metrics_by_fold.csv")
    daily = pd.read_csv(root / "daily_returns.csv")
    report = summarize_decisions(metrics, daily, metadata["protocol"], lock["horizon_sessions"])
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(report, output_dir / "paired_comparison.csv", index=False)
    atomic_json(output_dir / "manifest.json", dict(
        status="completed", source_run=str(root),
        predictor_lock_digest=lock["lock_digest"], policy_or_checkpoint_changes=False,
        transformation="paired_summary_only_float64_noise_tolerance_1e-12_decimal_returns",
        input_hashes=file_hashes([root / "metrics_by_fold.csv", root / "daily_returns.csv",
                                root / "predictor_lock.json", bank_manifest], ROOT),
        source_hashes=file_hashes([ROOT / "src/evaluation/portfolio_decision.py", Path(__file__)], ROOT)))
    print(report.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary-only", action="store_true",
                        help="Chỉ tổng hợp lại báo cáo ghép cặp từ kết quả run cũ (cần --run-dir và --output-dir).")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Đường dẫn thư mục run cũ khi dùng chế độ --summary-only.")
    parser.add_argument("--predictor-lock", type=Path, default=None,
                        help="File JSON predictor lock đã khóa (bắt buộc khi chạy so sánh đầy đủ).")
    parser.add_argument("--prediction-bank", type=Path, default=None,
                        help="Đường dẫn ngân hàng dự báo di chuyển; mã băm phải khớp lock.")
    parser.add_argument("--allocation-config", type=Path, default=ROOT / "configs/test/signal_allocation_neural.yaml")
    parser.add_argument("--decision-config", type=Path, default=ROOT / "configs/research/portfolio_decision.yaml")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--assets-config", type=Path, default=ROOT / "configs/core/assets.yaml")
    parser.add_argument("--env-config", type=Path, default=ROOT / "configs/core/env.yaml")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Thư mục xuất kết quả (tự động tạo mới nếu để trống).")
    parser.add_argument("--development-only", action="store_true",
                        help="Chỉ chạy trên tập phát triển/train/inner; bỏ qua kiểm định outer test.")
    parser.add_argument("--fold", default=None, help="Chạy riêng một fold (ví dụ: development_2021).")
    args = parser.parse_args()

    if args.summary_only:
        if not args.run_dir or not args.output_dir:
            parser.error("Chế độ --summary-only yêu cầu cung cấp cả --run-dir và --output-dir.")
        run_summary(args.run_dir, args.output_dir)
        return

    if not args.predictor_lock:
        parser.error("Cần cung cấp --predictor-lock (hoặc dùng --summary-only).")

    _, output = compare_portfolio_decisions(
        args.predictor_lock, args.allocation_config, args.decision_config,
        args.raw_dir, args.assets_config, args.env_config, args.output_dir, args.prediction_bank,
        development_only=args.development_only, fold_id=args.fold
    )
    print(f"Comparison artifacts: {output}")


if __name__ == "__main__":
    main()

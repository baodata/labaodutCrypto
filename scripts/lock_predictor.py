#!/usr/bin/env python3
"""
==============================================================================
TÊN SCRIPT: scripts/lock_predictor.py
MỤC ĐÍCH:
    Chọn và khóa (Lock) mô hình có hệ số Inner-Fold Rank-IC cao nhất cho từng fold
    từ Ngân hàng dự báo (Prediction Bank) tại một chân trời dự báo xác định (horizon).
    Tạo ra chữ ký số SHA-256 bất biến để ngăn chặn việc cherry-pick mô hình
    trong các bước phân bổ danh mục tiếp theo.

HƯỚNG DẪN CHẠY (CLI USAGE):
    # Khóa mô hình tốt nhất với chân trời dự báo 5 phiên (horizon=5):
    python scripts/lock_predictor.py \\
        --prediction-bank models/runs/prediction_bank/ \\
        --output models/runs/predictor_lock.json \\
        --horizon 5

ĐẦU VÀO CẦN THIẾT (PREREQUISITES):
    - Thư mục Prediction Bank đã tạo từ scripts/build_prediction_bank.py

ĐẦU RA SINH RA (OUTPUT ARTIFACTS):
    - models/runs/predictor_lock.json: File khóa mô hình chứa tên model thắng cuộc,
      Rank-IC, dải ngày và digest băm của prediction bank.
==============================================================================
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.predictor_lock import make_predictor_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidates", default="ridge_fixed,momentum,mean_reversion,gru,gnn_gru")
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()
    lock = make_predictor_lock(args.prediction_bank, args.output,
        candidates=args.candidates.split(","), horizon=args.horizon)
    for row in lock["selections"]:
        print(f"{row['fold_id']}: {row['family']} h={row['horizon_sessions']} "
              f"inner IC={row['selected_mean_rank_ic']:.4f}, cutoff={row['selection_cutoff']}")
    print(f"Locked: {args.output} / {lock['lock_digest']}")


if __name__ == "__main__":
    main()

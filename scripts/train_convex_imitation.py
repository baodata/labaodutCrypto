#!/usr/bin/env python3
"""
==============================================================================
TÊN SCRIPT: scripts/train_convex_imitation.py
MỤC ĐÍCH:
    Huấn luyện mạng PPO Actor học bắt chước (Imitation Learning / Behavioral Cloning)
    theo các mục tiêu phân bổ tỷ trọng tối ưu sinh ra từ bộ giải Convex
    trên các fold Train out-of-fold (OOF).
    Giúp khởi tạo trọng số mạng Actor tốt hơn trước khi bước vào huấn luyện RL đầy đủ.

HƯỚNG DẪN CHẠY (CLI USAGE):
    python scripts/train_convex_imitation.py \\
        --predictor-lock models/runs/predictor_lock.json \\
        --imitation-config configs/research/convex_imitation.yaml

ĐẦU VÀO CẦN THIẾT (PREREQUISITES):
    - models/runs/predictor_lock.json
    - configs/research/convex_imitation.yaml
    - configs/core/assets.yaml, configs/core/env.yaml

ĐẦU RA SINH RA (OUTPUT ARTIFACTS):
    - models/runs/convex_imitation_<timestamp>/
        + actor_checkpoint.pt: Trọng số mô hình Actor đã học bắt chước
        + imitation_metrics.csv: Loss hàm mục tiêu qua từng epoch
==============================================================================
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.convex_imitation import run_convex_imitation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor-lock", required=True, type=Path)
    parser.add_argument("--imitation-config", type=Path, default=ROOT / "configs/research/convex_imitation.yaml")
    parser.add_argument("--allocation-config", type=Path, default=ROOT / "configs/test/signal_allocation_neural.yaml")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--assets-config", type=Path, default=ROOT / "configs/core/assets.yaml")
    parser.add_argument("--env-config", type=Path, default=ROOT / "configs/core/env.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = run_convex_imitation(args.predictor_lock, args.allocation_config, args.imitation_config,
        args.raw_dir, args.assets_config, args.env_config, args.output_dir)
    print(f"Completed imitation: {output}")


if __name__ == "__main__":
    main()

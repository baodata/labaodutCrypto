#!/usr/bin/env python3
"""
==============================================================================
TÊN SCRIPT: scripts/run_sprint5_backtest.py
MỤC ĐÍCH:
    Điểm vào (entrypoint) tương thích ngược từ Sprint 5 để huấn luyện và backtest
    mô hình PPO (Proximal Policy Optimization).
    
    LƯU Ý: Hiện tại bạn có thể sử dụng trực tiếp script chuẩn hóa mới:
        python scripts/train_ppo.py [tham số...]

HƯỚNG DẪN CHẠY (CLI USAGE):
    python scripts/run_sprint5_backtest.py --encoder gnn --seed 7 --epochs 50 --device cpu

ĐẦU VÀO CẦN THIẾT (PREREQUISITES):
    - Dữ liệu nến thô trong: data/raw/*.parquet
    - Các file configs/core/: assets.yaml, env.yaml, model.yaml, training.yaml

ĐẦU RA SINH RA (OUTPUT ARTIFACTS):
    - models/runs/ppo_<encoder>_seed<seed>_<timestamp>_<hash>/
        + best_model.pt, training_log.csv, validation_metrics.csv
==============================================================================
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_ppo import main


if __name__ == "__main__":
    main()

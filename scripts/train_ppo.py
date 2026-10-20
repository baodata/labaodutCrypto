#!/usr/bin/env python3
"""
==============================================================================
TÊN SCRIPT: scripts/train_ppo.py
MỤC ĐÍCH:
    Huấn luyện và kiểm định mô hình Reinforcement Learning PPO (Proximal Policy
    Optimization) cho bài toán phân bổ danh mục đầu tư đa tài sản.
    Hỗ trợ các bộ mã hóa: Graph Neural Network (GNN: GCN, GAT, ChebNet, GraphSAGE)
    hoặc Recurrent (GRU, LSTM) hoặc MLP.

HƯỚNG DẪN CHẠY (CLI USAGE):
    # 1. Chạy huấn luyện PPO với đồ thị GNN (mặc định seed 7, 50 epochs trên CPU):
    python scripts/train_ppo.py --encoder gnn --seed 7 --epochs 50 --device cpu

    # 2. Chạy với bộ mã hóa chuỗi thời gian GRU:
    python scripts/train_ppo.py --encoder gru --seed 7 --epochs 50

    # 3. Chạy với GPU và số epochs tùy chọn:
    python scripts/train_ppo.py --encoder gnn --device cuda --epochs 100 --lr 0.0003

    # 4. Xem toàn bộ các cờ tham số hỗ trợ:
    python scripts/train_ppo.py --help

ĐẦU VÀO CẦN THIẾT (PREREQUISITES):
    - Dữ liệu nến thô trong thư mục: data/raw/*.parquet
    - Các file cấu hình cốt lõi trong configs/core/:
        + configs/core/assets.yaml (Vũ trụ 24 cổ phiếu & dải ngày train/val/test)
        + configs/core/env.yaml (Cấu hình môi trường giao dịch, phí, đòn bẩy)
        + configs/core/model.yaml (Siêu tham số kiến trúc mạng GNN/Actor-Critic)
        + configs/core/training.yaml (Siêu tham số thuật toán PPO: lr, clip, batch)

ĐẦU RA SINH RA (OUTPUT ARTIFACTS):
    - Thư mục lưu kết quả: models/runs/ppo_<encoder>_seed<seed>_<timestamp>_<hash>/
        + best_model.pt / final_model.pt: Checkpoint trọng số mạng nơ-ron
        + training_log.csv: Nhật ký loss, policy entropy qua từng epoch
        + validation_metrics.csv / test_metrics.csv: Các chỉ số Sharpe, Return, Drawdown
        + portfolio_history.csv: Lịch sử tỷ trọng tiền mặt và cổ phiếu theo ngày
        + training_config.json: Lưu lại toàn bộ tham số đã dùng để đảm bảo khả năng tái lập
==============================================================================
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_ppo import main


if __name__ == "__main__":
    main()

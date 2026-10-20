#!/usr/bin/env python3
"""
==============================================================================
TÊN SCRIPT: scripts/train_marl.py
MỤC ĐÍCH:
    Huấn luyện và kiểm định mô hình Học tăng cường Đa tác tử theo Ngành
    (Sector-based Multi-Agent Reinforcement Learning - MARL).
    Mỗi tác tử (agent) phụ trách một nhóm ngành riêng biệt (Công nghệ, Tài chính,
    Y tế, Tiêu dùng...), phối hợp trong một trò chơi hợp tác (cooperative game)
    để tối ưu hóa danh mục đầu tư tổng thể.

HƯỚNG DẪN CHẠY (CLI USAGE):
    # 1. Chạy huấn luyện MARL cơ bản (mặc định seed 7, 50 epochs trên CPU):
    python scripts/train_marl.py --seed 7 --epochs 50 --device cpu

    # 2. Chạy với hàm phần thưởng hợp tác chia sẻ (cooperative reward):
    python scripts/train_marl.py --seed 7 --reward-mode shared_net --epochs 100

    # 3. Xem danh sách tham số chi tiết:
    python scripts/train_marl.py --help

ĐẦU VÀO CẦN THIẾT (PREREQUISITES):
    - Dữ liệu nến thô trong: data/raw/*.parquet
    - configs/core/assets.yaml (Phân nhóm ngành của từng cổ phiếu trong universe)
    - configs/core/env.yaml (Môi trường MultiAgentTradingEnv)
    - configs/core/training.yaml

ĐẦU RA SINH RA (OUTPUT ARTIFACTS):
    - Thư mục lưu kết quả: models/runs/sector_marl_<encoder>_seed<seed>_<timestamp>_<hash>/
        + best_checkpoint.pt: Trọng số của tất cả các sector agents
        + training_log.csv: Nhật ký huấn luyện đa tác tử
        + validation_metrics.csv: Đánh giá danh mục chung và từng agent
        + validation_daily_returns.csv: Lợi nhuận hàng ngày
        + validation_bootstrap.json: Kiểm định thống kê Block-bootstrap
==============================================================================
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_marl import main


if __name__ == "__main__":
    main()

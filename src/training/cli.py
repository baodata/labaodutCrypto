"""End-to-end single-agent PPO training and historical-period evaluation.

Run from the repository root with::

    python -m src.training.train_ppo --reward-beta 2.0

The policy receives engineered, train-normalized features. Raw OHLCV prices are
used to build those features and to simulate order execution in TradingEnv.
"""

from __future__ import annotations

from pathlib import Path
import argparse

from src.training.config import DEFAULT_TRAINING_CONFIG
from src.utils.config import resolve_config_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalization-warmup", type=int, default=0, help="Warm-up causal bổ sung chung cho mọi nhánh ablation, không đổi ngày train giữa các nhánh.")
    parser.add_argument("--quant-config", type=resolve_config_path, default=None, help="Sprint 6 normalization/risk/execution/reward opt-in YAML.")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument(
        "--processed-output", type=Path, default=PROJECT_ROOT / "data/processed/features.parquet"
    )
    parser.add_argument("--model-config", type=resolve_config_path, default=PROJECT_ROOT / "configs/core/model.yaml")
    parser.add_argument("--env-config", type=resolve_config_path, default=PROJECT_ROOT / "configs/core/env.yaml")
    parser.add_argument("--research-protocol", type=resolve_config_path, default=None,
                        help="Protocol định lượng đóng băng; chỉ dùng development validation.")
    parser.add_argument("--protocol-fold", default=None, help="Fold PPO; mặc định fold cuối protocol.")
    parser.add_argument("--evaluation-scope", choices=("validation", "test"), default=None,
                        help="Chỉ backtest validation hoặc test sau train.")
    parser.add_argument(
        "--training-config", type=resolve_config_path, default=DEFAULT_TRAINING_CONFIG,
        help="Cấu hình train/early stopping/checkpoint/evaluation; CLI ghi đè YAML.",
    )
    parser.add_argument(
        "--assets-config",
        type=resolve_config_path,
        default=PROJECT_ROOT / "configs/core/assets.yaml",
        help="Đường dẫn file cấu hình vũ trụ cổ phiếu và phân chia dữ liệu (mặc định: configs/core/assets.yaml).",
    )
    parser.add_argument(
        "--train-end",
        default=None,
        help="Mốc kết thúc tập train (YYYY-MM-DD). Mặc định đọc từ assets.yaml.",
    )
    parser.add_argument(
        "--val-start",
        default=None,
        help="Mốc bắt đầu tập validation (YYYY-MM-DD). Mặc định đọc từ assets.yaml.",
    )
    parser.add_argument(
        "--val-end",
        default=None,
        help="Mốc kết thúc tập validation (YYYY-MM-DD). Mặc định đọc từ assets.yaml.",
    )
    parser.add_argument(
        "--test-start",
        default=None,
        help="Mốc bắt đầu tập test/OOS (YYYY-MM-DD). Mặc định đọc từ assets.yaml.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--ppo-iters", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--microbatch-size", type=int, default=None, help="Số mẫu forward/backward mỗi lần; GNN mặc định 16 CPU / 32 CUDA, tích lũy gradient đủ minibatch.")
    parser.add_argument("--cpu-threads", type=int, default=None, help="Số thread PyTorch trên CPU; tránh dùng quá nhiều thread cho graph nhỏ.")
    parser.add_argument("--cache-batch-size", type=int, default=None, help="Số ngày mã hóa GNN mỗi lượt khi dựng cache.")
    parser.add_argument("--cache-temporal-batch-size", type=int, default=None, help="Số cửa sổ GRU mỗi lượt khi dựng cache.")
    parser.add_argument("--progress-every", type=float, default=None, help="Khoảng giây giữa các dòng tiến độ trong một giai đoạn train.")
    parser.add_argument("--episodes-per-update", type=int, default=None)
    parser.add_argument("--run-dir", type=Path, default=None, help="Thư mục artifact riêng cho run; mặc định tạo thư mục mới.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume từ last.pth với cùng cấu hình train.")
    parser.add_argument("--train-eval-every", type=int, default=None)
    parser.add_argument("--bootstrap-block-size", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--matched-cash-ratio", type=float, default=None, help="Cash cố định cho matched baseline; mặc định 1/(N+1).")
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Ghi đè gamma trong configs/model.yaml.",
    )
    parser.add_argument(
        "--reward-scaling",
        type=float,
        default=None,
        help="Ghi đè reward_scaling trong configs/env.yaml.",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="Ghi đè target_kl trong configs/model.yaml.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Ghi đè learning_rate trong configs/model.yaml.",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=None,
        help="Ghi đè entropy_coef trong configs/model.yaml.",
    )
    parser.add_argument(
        "--value-clip-range",
        type=float,
        default=None,
        help="Ngưỡng riêng cho value clipping; mặc định tắt nếu YAML để null.",
    )
    parser.add_argument("--window-min-days", type=int, default=None)
    parser.add_argument("--window-max-days", type=int, default=None)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument(
        "--action-alpha",
        type=float,
        default=0.25,
        help="Tỷ lệ tiến về danh mục policy mỗi bước; nhỏ hơn giúp giảm turnover.",
    )
    parser.add_argument("--min-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--selection-window", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--encoder",
        choices=("mlp", "lstm", "gru", "gnn", "gcn", "gat"),
        default="gru",
        help="Loại encoder: gru mặc định; cũng hỗ trợ lstm, mlp, gnn động, gcn và gat.",
    )
    parser.add_argument(
        "--gnn-architecture",
        choices=("gcn", "gat"),
        default=None,
        help="Kiến trúc GNN khi dùng --encoder gnn; mặc định đọc configs/model.yaml.",
    )
    parser.add_argument("--gnn-hidden-channels", type=int, default=None)
    parser.add_argument("--gnn-out-channels", type=int, default=None)
    parser.add_argument("--gnn-heads", type=int, default=4, help="Số head attention cho GAT/GNN.")
    parser.add_argument("--gnn-dropout", type=float, default=0.0, help="Dropout cho GNN (mặc định 0.0 cho PPO).")
    parser.add_argument("--gnn-knn", type=int, default=5, help="Số hàng xóm k-NN cho Dynamic Graph (mặc định 5).")
    parser.add_argument(
        "--rnn-layers",
        type=int,
        default=1,
        help="Số lớp RNN cho LSTM/GRU và GRU thời gian trong GNN.",
    )
    parser.add_argument(
        "--rnn-dropout",
        type=float,
        default=0.0,
        help="Dropout giữa các lớp RNN; GNN PPO yêu cầu 0 để giữ log-prob nhất quán.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Tùy chọn chạy nhiều seed, ví dụ: --seeds 7,17,27; xuất thêm mean/std test.",
    )
    parser.add_argument(
        "--untrained-null-seeds",
        default=None,
        help=(
            "Chỉ đánh giá policy chưa train trên validation với các seed phân tách bằng dấu phẩy; "
            "ví dụ: --untrained-null-seeds 7,17,27. Không chạy PPO hoặc đánh giá test."
        ),
    )
    parser.add_argument(
        "--null-output",
        type=Path,
        default=PROJECT_ROOT / "experiments/untrained_null_validation.csv",
        help="CSV kết quả policy khởi tạo ngẫu nhiên và baseline validation.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--reward-beta",
        type=float,
        default=None,
        help="Penalty multiplier for transaction costs; defaults to configs/env.yaml.",
    )
    parser.add_argument(
        "--reward-baseline",
        choices=("none", "equal_weight"),
        default="equal_weight",
        help="Equal-weight excess-return reward baseline (mặc định bật).",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("sharpe", "cumulative_return", "excess_sharpe", "excess_return"),
        default=None,
        help="Validation metric used to select the best checkpoint.",
    )
    parser.add_argument(
        "--selection-margin",
        type=float,
        default=None,
        help="Biên tối thiểu yêu cầu để checkpoint sau vượt checkpoint trước.",
    )
    parser.add_argument(
        "--dirichlet-kappa",
        type=float,
        default=None,
        help="Tổng concentration kappa để tách mean khỏi variance của Dirichlet (ví dụ 200.0).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models/checkpoints/sprint5_best.pth",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=PROJECT_ROOT / "experiments/sprint5_metrics.csv",
    )
    parser.add_argument(
        "--training-log",
        type=Path,
        default=PROJECT_ROOT / "experiments/sprint5_training_log.csv",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=PROJECT_ROOT / "logs/ppo_training",
    )
    parser.add_argument(
        "--scaler-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/scaler_params.json",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Tên định danh cho chiến lược PPO trong file metrics (mặc định: tự sinh từ cấu hình).",
    )
    parser.add_argument(
        "--overwrite-metrics",
        action="store_true",
        help="Ghi đè hoàn toàn file metrics thay vì ghi nối tiếp (append).",
    )
    return parser

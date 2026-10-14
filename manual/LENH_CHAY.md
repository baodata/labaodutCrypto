# Sổ lệnh chạy

Chạy các lệnh từ thư mục gốc dự án, sau khi kích hoạt môi trường:

```bash
cd /home/tuan/AI/AI_roject1
source .venv/bin/activate
```

## 1. GAT + GRU nhỏ — 5 seed

**Mục đích:** đánh giá độ ổn định qua seed và so với matched baseline cash 4%, cùng action alpha 0.25.

Giữ nguyên `configs/model.yaml` và `configs/env.yaml` giữa các lần so sánh. Mỗi lần chạy dùng một thư mục `--run-dir` mới, chưa có artifact.

```bash
python -m src.training.train_ppo \
  --encoder gnn --gnn-architecture gat \
  --lookback 10 --hidden-dim 64 \
  --gnn-hidden-channels 16 --gnn-out-channels 16 --gnn-heads 2 \
  --episodes-per-update 8 --ppo-iters 3 --batch-size 256 \
  --microbatch-size 16 --cpu-threads 4 --device cpu \
  --epochs 20 --min-epochs 5 --patience 5 \
  --action-alpha 0.25 --reward-baseline equal_weight \
  --matched-cash-ratio 0.04 \
  --seeds 7,17,27,37,47 \
  --run-dir models/runs/gnn_5seeds_cpu
```

**Kết quả:**

- `models/runs/gnn_5seeds_cpu/seed_*/`: checkpoint, training log và metrics từng seed.
- `models/runs/gnn_5seeds_cpu/metrics_seeds_summary.csv`: mean/std của metric số; đọc `sharpe_delta_vs_matched_baseline`.
- `models/runs/gnn_5seeds_cpu/metrics_seeds_metadata.csv`: epoch và trạng thái checkpoint.

**Ghi lại sau khi chạy:**

- Ngày chạy:
- Thời gian chạy:
- Thư mục kết quả thực tế:
- Validation so với policy chưa train:
- Test Sharpe delta so với matched baseline, mean ± std:
- Checkpoint epoch/status từng seed:
- Nhận xét:

Test 2024–2025 đã được dùng nhiều lần để khảo sát; kết quả này không phải xác nhận trên holdout độc lập.

## 2. Lần thử tiếp theo

**Tên lần thử:**

**Mục đích:**

**Thay đổi so với lần trước:**

**Lệnh chạy:**

```bash
# Ghi lệnh tại đây.
```

**Ghi lại sau khi chạy:**

- Ngày chạy:
- Thời gian chạy:
- Thư mục kết quả:
- Cấu hình/seed:
- Metric cần so sánh:
- Nhận xét:

Sao chép mục này để ghi thêm các lần thử.

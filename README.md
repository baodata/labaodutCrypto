# H-MARL-GNN: Hierarchical Multi-Agent Reinforcement Learning with Heterogeneous Dynamic Graph Neural Networks for Algorithmic Trading

Đề tài nghiên cứu xây dựng hệ thống phân bổ danh mục đầu tư và giao dịch tự động kết hợp mạng nơ-ron đồ thị động (Dynamic GNN) và học tăng cường đa tác tử phân cấp (Hierarchical Multi-Agent RL).

---

## 1. Cấu trúc dự án

```text
.
├── configs/                # Cấu hình YAML (assets, mô hình, siêu tham số)
├── data/
│   ├── raw/                # Dữ liệu thị trường thô (OHLCV từ Yahoo Finance)
│   ├── interim/            # Dữ liệu sau khi đồng bộ ngày và làm sạch
│   └── processed/          # Dữ liệu cuối cùng kèm các chỉ báo kỹ thuật
├── src/                    # Mã nguồn chính
│   ├── data/               # Tải, làm sạch và kiểm tra dữ liệu
│   ├── features/           # Tính toán các chỉ báo kỹ thuật (Return, RSI, MACD, Volatility...)
│   ├── graph/              # Xây dựng ma trận tương quan động và đồ thị tài chính
│   ├── env/                # Môi trường giao dịch (Gymnasium), phí giao dịch, rủi ro
│   ├── models/             # GNN / GAT Encoders
│   ├── agents/             # High-Level Agent & Low-Level Sector Agents
│   ├── training/           # Huấn luyện mô hình RL và GNN
│   ├── evaluation/         # Backtest engine và các chỉ số (Sharpe, MDD, Calmar...)
│   └── utils/              # Tiện ích bổ trợ (logging, visualization, seed...)
├── tests/                  # Bộ kiểm thử tự động (pytest)
├── notebooks/              # Jupyter Notebooks phục vụ EDA và thử nghiệm
├── experiments/            # Lưu trữ log huấn luyện và checkpoints
├── scripts/                # Các script chạy pipeline độc lập (xem scripts/README.md)
├── manual/                 # Tài liệu hệ thống, kiến trúc, nghiên cứu và phân công (xem manual/README.md)
├── requirements.txt        # Danh sách thư viện phụ thuộc
├── pyproject.toml          # Cấu hình package Python
└── README.md
```

---

## 2. Hướng dẫn cài đặt & Khởi chạy

### Bước 1: Kích hoạt môi trường ảo

```bash
source .venv/bin/activate
```

### Bước 2: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Bước 3: Cài đặt package ở chế độ Editable

Giúp import `src` từ bất kỳ đâu trong dự án mà không cần chỉnh `sys.path`:

```bash
pip install -e .
```

### Bước 4: Chạy kiểm thử tự động

```bash
pytest
```

## 3. Huấn luyện và artifact

```bash
python -m src.training.train_ppo --encoder gnn --run-dir models/runs/my_gnn_run
```

Mặc định mỗi lần chạy tạo một thư mục mới trong `models/runs/`. Thư mục chứa `checkpoints/best.pth`, `checkpoints/last.pth`, `scaler.json`, `manifest.json`, CSV và TensorBoard. Checkpoint nhúng scaler, cấu hình encoder/feature, ngày split, hash source/data/config, scheduler và trạng thái RNG. CSV được ghi sau mỗi epoch; Ctrl+C giữ epoch đã hoàn thành. `best.pth` là policy dùng để đánh giá, `last.pth` là trạng thái để tiếp tục train.

Resume cần giữ các tham số của lệnh ban đầu, gồm tổng `--epochs` để giữ lịch learning rate:

```bash
python -m src.training.train_ppo --encoder gnn --resume models/runs/my_gnn_run/checkpoints/last.pth
```

Dữ liệu, cấu hình hoặc mã nguồn khác sẽ bị từ chối khi resume. `--seeds 7,17,27,37,47` tạo thư mục riêng cho từng seed và xuất riêng bảng metric số với metadata. Nếu checkpoint tốt nhất vẫn là epoch 0, chương trình cảnh báo và ghi `untrained_fallback` trong kết quả.

GNN dùng graph bank `[T,3,N,N]`; buffer chỉ lưu chỉ số ngày, lấy graph history theo minibatch. Rollout/evaluation cache embedding khi trọng số cố định; update PPO tính lại embedding để giữ gradient. `--train-eval-every 5` là mặc định; validation vẫn được đánh giá mỗi epoch. GRU/LSTM/GNN yêu cầu dropout bằng 0 để log-prob nhất quán trong PPO.

### Chạy GAT nhanh hơn trên CPU

GAT mặc định (hidden 64 mỗi head, 4 heads, GRU 128, lookback 20, 32 cửa sổ, 10 lượt PPO) tốn nhiều phép tính dù rollout đã có cache. Mặc định mới giới hạn PyTorch ở 4 thread và GNN forward/backward ở microbatch 16 trên CPU, 32 trên CUDA. Gradient được tích lũy có trọng số theo số mẫu; Adam, clipping gradient và KL guard vẫn áp dụng theo minibatch cấu hình. Mẫu cuối không đủ microbatch được tính đúng tỷ lệ. Đây là giới hạn bộ nhớ, không tự giảm số mẫu rollout/lượt PPO.

Trong update, các ngày trùng nhau trong một microbatch chỉ được mã hóa GNN một lần với gradient còn nguyên. GRU dùng đường trực tiếp cho history đầy đủ; history có padding vẫn dùng packed sequence. `--cache-batch-size` và `--cache-temporal-batch-size` điều chỉnh riêng batch dựng cache. Log in tiến độ cache/rollout/update/evaluation; `--progress-every 10` điều chỉnh khoảng thời gian in.

Cấu hình thăm dò nhanh, vẫn dùng GAT + GRU và đủ ba quan hệ:

```bash
python -m src.training.train_ppo --encoder gnn \
  --lookback 10 --hidden-dim 64 \
  --gnn-hidden-channels 16 --gnn-out-channels 16 --gnn-heads 2 \
  --episodes-per-update 8 --ppo-iters 3 \
  --epochs 20 --min-epochs 5 --patience 5
```

Một update đo trên dữ liệu thật với cấu hình này: cache 0,28 s, rollout 0,89 s, PPO update 9,00 s cho 3 lượt, evaluation 1,59 s; tổng khoảng 12 s/epoch, chưa tính chuẩn bị dữ liệu lúc khởi động. Đây là phép đo một update trên CPU hiện tại; thời gian các epoch khác phụ thuộc KL stopping và lịch eval. Kích thước model, lookback và rollout của cấu hình thăm dò khác mặc định, nên kết quả học cần được đánh giá riêng. Dùng `--gnn-architecture gcn` nếu cần khảo sát kiến trúc GNN nhẹ hơn.

## 4. Kiểm tra tín hiệu trên validation

```bash
python -m src.training.train_ppo --encoder gru --untrained-null-seeds 7,17,27,37,47,57,67,77,87,97
python scripts/features/analyze_feature_signal.py
python scripts/baselines/evaluate_equal_weight_tilt.py
python scripts/baselines/evaluate_tilt_robustness.py
```

Null hỗ trợ cả `mlp`, `gru`, `lstm`, `gnn` và chỉ đánh giá validation. Matched baseline dùng cùng `action_alpha`, cash mặc định `1/(N+1)` hoặc `--matched-cash-ratio`; Sharpe delta, selection excess và bootstrap đều dùng baseline này. P-value là kiểm định bootstrap một phía với phân phối null đã center; block mặc định 20 phiên, có thể chỉnh bằng `--bootstrap-block-size`.

Feature cần 252 phiên warm-up. Các phiên chưa đủ lịch sử bị loại khỏi tập fit scaler/train; PPO còn yêu cầu đủ lookback đã warm-up. Có thể tải dữ liệu trước năm train để giữ nguyên ngày bắt đầu train mong muốn. Rank-IC có nhãn từ Open[t+1] đến Close[t+h], loại nhãn vượt ranh giới split và có CI theo block. Robustness dùng toàn bộ tham số trong `configs/tilt_robustness_gate.yaml`, lưu manifest và yêu cầu đạt cổng ở mọi baseline. Kết quả 2024–2025 đã được xem qua nhiều vòng nên vẫn là khảo sát.

## 5. Backtest và inference

Quy ước giao dịch nằm trong [ENV_CONVENTIONS.md](manual/architecture/ENV_CONVENTIONS.md). Dùng `PortfolioOrder.hold()` để giữ số cổ phiếu và không tái cân bằng. API baseline độc lập yêu cầu `baseline.run(close_prices=...)`; nó dùng cùng TradingEnv với `run_strategy`, gồm T−1 bước.

Inference từ checkpoint PPO hiện tại dùng `src.models.policy_adapter.CheckpointPolicy`, với đúng thứ tự ticker/feature và `market_history` `[L,N,F]`. Adapter dùng scaler nhúng; truyền `features_scaled=True` nếu history đã chuẩn hóa. Với GNN, cấp hai history `[L,3,N,N]` qua `relation_history_from_correlations`, dùng builder có tham số trùng `graph_config` trong checkpoint. `ObservationAdapter` và `GNN_ActorCritic` cũ là API snapshot legacy, không tương thích checkpoint PPO hiện tại.

### Mask lịch sử

Env cung cấp `history_mask` `[L]`: `False` cho left-padding, `True` cho ngày thật, kể cả khi toàn bộ feature của ngày đó bằng 0. Mask được lưu riêng trong PPOBuffer và truyền cùng observation tới `act`, `evaluate`, diagnostics và bootstrap. Vector observation và kích thước trọng số model giữ nguyên.

GRU/LSTM và GRU trong GNN chỉ chạy timestep hợp lệ, giữ thứ tự thời gian bằng cách chuyển phần hợp lệ lên đầu trước khi pack. History không có timestep hợp lệ trả hidden state bằng 0. History đầy đủ dùng đường RNN trực tiếp. Cache GNN xác định availability theo chỉ số ngày, không suy ra từ giá trị feature.

Checkpoint mới ghi `history_mask_convention` và `CheckpointPolicy` yêu cầu caller cấp mask; adapter giữ padding bằng 0 sau chuẩn hóa. Với lời gọi model/adapter legacy không có mask, history được coi là đầy đủ; caller có padding cần truyền mask tường minh.

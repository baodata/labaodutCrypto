# Sổ lệnh chạy

## Predictor alpha: VIX, Blackout, Fama–French

Chuẩn bị banks/readiness (chưa train; thiếu dữ liệu ghi `needs_data`):

```bash
python scripts/research_alpha_features.py
```

Khi đã có lịch sử đầu vào đạt contract trong `configs/research/alpha_features.yaml`:

```bash
python scripts/research_alpha_features.py --groups vix --run-ablation
python scripts/research_alpha_features.py --groups vix,fama_french,blackout --run-ablation
```

Chỉ train OOF/inner, cùng allocator/cost. [Schema, nguồn và cách chuẩn bị dữ liệu](NANG_CAP_ALPHA_VIX_BLACKOUT_FF3.md). Chưa có dữ liệu PIT nên chưa có kết quả alpha mới.

## Thêm GRU/GNN-GRU predictor, giữ nguyên allocator

```bash
python scripts/compare_signal_allocators.py \
  --prediction-config configs/test/prediction_bank_neural.yaml \
  --allocation-config configs/test/signal_allocation_neural.yaml
```

Đọc thêm `comparison/predictor_contribution_summary.csv` cho các cặp GRU−Ridge và GNN-GRU−GRU. [Kiến trúc, causal fit, checkpoint và nghiệm thu](GRU_GNN_GRU_PREDICTOR_ABLATION.md).

## Ridge + momentum/mean-reversion so với matched 1/N và allocator chi phí

```bash
python scripts/compare_signal_allocators.py
```

Tạo prediction bank ba model và chạy đối chiếu inner/outer trên các fold phát triển, horizon 5/20 phiên. Kết quả: `models/runs/signal_allocation_*/comparison/outer_summary.csv`. [Cơ chế, phí, ràng buộc và cách đọc](RIDGE_MOMENTUM_VA_ALLOCATOR_CHI_PHI.md).

## Prediction bank và nhãn giao dịch — 19/10/2026

```bash
python scripts/build_prediction_bank.py \
  --prediction-config configs/research/prediction_bank.yaml \
  --research-protocol configs/core/quant_research_protocol.yaml
```

Tạo run mới chứa dự báo Ridge `train_oof`/`inner`/`outer` và nhãn riêng, horizon 1/5/20 phiên; chỉ dùng giai đoạn phát triển. [Hợp đồng dữ liệu, artifact và API đọc](PREDICTION_BANK_VA_NHAN_GIAO_DICH.md).

Chạy các lệnh từ thư mục gốc dự án, sau khi kích hoạt môi trường:

```bash
cd /home/tuan/AI/AI_roject1
source .venv/bin/activate
```

Nếu Sprint 6 báo `ModuleNotFoundError: No module named 'sklearn'`, cài dependency đã khai báo vào đúng virtualenv rồi chạy lại lệnh:

```bash
.venv/bin/python -m pip install 'scikit-learn>=1.3.0'
```

## Mặc định evaluation — sửa 19/10/2026

CLI và YAML mặc định dùng **validation**, không tạo test tensor. Để backtest giai đoạn test lịch sử, chọn tường minh `--evaluation-scope test`; không dùng cờ này cùng research protocol. Mốc/cảnh báo test lấy từ phiên dữ liệu thực tế đã resolve (CLI → assets → auto split), không chỉ từ tham số CLI. Giai đoạn chồng lấn 2024–2025 luôn được ghi exploratory; period khác chưa tự được xác nhận là final holdout.

```bash
python -m src.training.train_ppo --encoder gru
# Chỉ khi chủ động đánh giá giai đoạn test đã cấu hình:
python -m src.training.train_ppo --encoder gru --evaluation-scope test
```

Run trước/sau bản sửa float64 dùng phiên bản accounting/source khác nhau. Giữ artifact cũ để tham chiếu; dùng run-dir mới cho bản sửa. Provenance guard sẽ từ chối resume qua source/config hash đã thay đổi. Không gộp các seed khác phiên bản vào một so sánh kiểm soát.

## Cấu hình huấn luyện

Entry point giữ nguyên sau khi tách module. Xem [kiến trúc module PPO và phân công A/B](KIEN_TRUC_MODULE_PPO.md) để biết file cần sửa cho từng trách nhiệm.

Sửa [configs/core/training.yaml](../configs/core/training.yaml) để đổi epochs, rollout/minibatch, early stopping, chọn checkpoint, lịch đánh giá và runtime. Kiến trúc/loss PPO ở `configs/core/model.yaml`; môi trường/reward ở `configs/core/env.yaml`. `batch_size` đã chuyển từ `model.yaml` sang `training.yaml`.

Thứ tự ưu tiên: **CLI được truyền rõ → training YAML → mặc định dự phòng**. File YAML tùy chỉnh có thể chỉ chứa các nhóm/khóa cần đổi; khóa bị gõ sai hoặc giá trị không hợp lệ sẽ báo lỗi. `patience: 0` tắt early stopping; `microbatch_size: null` và `matched_cash_ratio: null` chọn tự động theo model/device và số mã.

```bash
# Đọc các tham số từ configs/core/training.yaml.
python -m src.training.train_ppo --encoder gnn

# Chọn một training YAML riêng, đồng thời ghi đè patience bằng CLI.
python -m src.training.train_ppo \
  --training-config configs/core/training.yaml \
  --encoder gnn --patience 0
```

Các cờ mô hình và môi trường hiện có như `--lookback`, `--action-alpha`, `--reward-baseline`, `--gamma` vẫn giữ cách dùng cũ. Seed, device và đường dẫn output tiếp tục chọn bằng CLI.

Mỗi run lưu `training_config.json` và `resolved_training_config` trong manifest/checkpoint. Các giá trị tự động được cập nhật khi biết kích thước dữ liệu/device; manifest giữ thêm `training_config_inputs` trước khi tính tự động. Resume yêu cầu cùng training YAML, cấu hình CLI đã resolve và các điều kiện tái lập hiện có. Khi chạy nhiều seed, mỗi thư mục seed có snapshot riêng.

## Sprint 5 — protocol đóng băng, chỉ development validation

[Protocol](../configs/core/quant_research_protocol.yaml) và [báo cáo nghiệm thu](NGHIEM_THU_SPRINT_5.md). Chạy từ thư mục gốc:

```bash
# QNT-003: IC + inner-fit linear tilt, bốn outer development fold.
python scripts/analyze_feature_signal.py \
  --research-protocol configs/core/quant_research_protocol.yaml

# QNT-002: PPO chọn checkpoint trên inner 2021, báo cáo outer 2022.
python scripts/run_sprint5_backtest.py \
  --research-protocol configs/core/quant_research_protocol.yaml \
  --encoder gru --lookback 10 --hidden-dim 64 \
  --episodes-per-update 8 --ppo-iters 3 \
  --epochs 10 --min-epochs 5 --patience 5 --bootstrap-samples 2000
```

Mặc định protocol chọn fold cuối; đổi fold bằng `--protocol-fold development_2019` (hoặc 2020/2021/2022 theo tên trong YAML). Mốc CLI, alpha/cash, selection metric và seed xung đột protocol sẽ báo lỗi. Không dùng `--evaluation-scope test` cùng protocol.

Nếu chỉ cần validation của split hiện có mà không dùng protocol: `python -m src.training.train_ppo --evaluation-scope validation`. Khi đó validation vẫn là kỳ chọn checkpoint, không tự trở thành outer holdout.

Kết quả Sprint 5 đã chạy: **NO-GO**; cả bốn inner fold chọn tilt 0, PPO chưa vượt biên cải thiện nên dùng fallback epoch 0. Chỉ đổi protocol/gate bằng một phiên bản nghiên cứu mới đã chốt trước lượt chạy.

## 1. GAT + GRU nhỏ — 5 seed

**Mục đích:** đánh giá độ ổn định qua seed và so với matched baseline cash 4%, cùng action alpha 0.25.

Giữ nguyên `configs/core/model.yaml`, `configs/core/env.yaml` và `configs/core/training.yaml` giữa các lần so sánh. Mỗi lần chạy dùng một thư mục `--run-dir` mới, chưa có artifact.

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


## Sprint 6 — A và bàn giao covariance cho B

### GP temporary-cost: development OOF/inner

```bash
python scripts/compare_gp_allocators.py \
  --predictor-lock models/runs/locked_predictor_h5_20261019/predictor_lock.json \
  --gp-config configs/research/gp_allocator.yaml \
  --fold development_2021
```

Bỏ `--fold` để chạy bốn fold OOF/inner, vẫn không outer/test. Xem [contract, lệnh full development outer và resume](GP_CONVENTIONS.md). Mọi nhánh dùng common cash ramp của protocol dẫn xuất; không so metrics như thể giữ nguyên execution của run cũ.

### Đối chứng execution ER và mua/bán riêng

```bash
python scripts/compare_execution_speeds.py \
  --predictor-lock models/runs/locked_predictor_h5_20261019/predictor_lock.json \
  --execution-config configs/research/execution_speed_research.yaml
```

Không train lại predictor/actor; chỉ train OOF + inner, không outer/test. Cùng convex target tape và matched 1/N ở mỗi controller; xem [cách đọc và giới hạn](EXECUTION_ALPHA_THICH_NGHI.md).

### Khóa predictor → đối chứng cơ chế phân bổ

Actor imitation convex targets: [phương pháp, kết quả và lệnh](BAT_CHUOC_CONVEX_TARGETS.md).

```bash
python scripts/train_convex_imitation.py \
  --predictor-lock models/runs/locked_predictor_h5_20261019/predictor_lock.json \
  --imitation-config configs/research/convex_imitation.yaml
```

Learning curves sau khi sửa patience: [quy tắc và lệnh development-only](SUA_PATIENCE_VA_TRAIN_PROBE.md).

Xem [contract và cách đọc kết quả](LOCK_PREDICTOR_VA_SO_SANH_ALLOCATOR.md).

```bash
python scripts/compare_portfolio_decisions.py \
  --predictor-lock models/runs/locked_predictor_h5_20261019/predictor_lock.json \
  --allocation-config configs/test/signal_allocation_neural.yaml \
  --decision-config configs/research/portfolio_decision.yaml
```

Khóa có sẵn chọn bằng inner IC tại horizon 5; cùng 5 seed cho PPO, sector MARL và H-MARL. Runner không train lại predictor và không dùng test.

Xem [phạm vi, cấu hình, lệnh bank/stress, ablation và sector PPO](TRIEN_KHAI_SPRINT_6.md#5-lệnh-triển-khai-thực-nghiệm). Các nhánh bật riêng từng cơ chế, validation-only; các lệnh mới chưa chạy nghiệm thu.

```bash
python scripts/evaluate_sprint6_banks.py --quant-config configs/core/quant_sprint6.yaml --research-protocol configs/core/quant_research_protocol.yaml --evaluation-scope validation
python scripts/run_sprint6_ablation.py --variants gru_control,control
python -m src.training.train_marl --encoder gru --seed 7 --quant-config configs/core/quant_sprint6.yaml --research-protocol configs/core/quant_research_protocol.yaml --evaluation-scope validation
```

# Nghiệm thu triển khai Sprint 5

**Ngày:** 18/10/2026. **Phạm vi:** phần nền Sprint 5 đã có, bổ sung QNT-002 và QNT-003; chạy end-to-end trên dữ liệu thật. **Kết quả triển khai:** hoàn thành. **Gate nghiên cứu:** NO-GO cho cấu hình đã chạy.

## 1. QNT-002 — protocol và validation-only

- [Protocol đóng băng](../configs/quant_research_protocol.yaml): seed, số candidate, folds train/inner/outer, execution và gate có hash trong manifest/checkpoint.
- CLI mới: `--research-protocol`, `--protocol-fold`, `--evaluation-scope validation`. CLI cũ không dùng protocol giữ mặc định test từ training config.
- Với protocol, PPO mặc định chọn fold cuối: train đến 2020, inner 2021 chọn checkpoint, outer 2022 chỉ báo cáo sau train. Không dùng outer để chọn checkpoint.
- Validation-only cắt dữ liệu trước tính feature/alignment theo cutoff cần thiết, không tạo test tensor, không tự đổi sang split tỷ lệ. Factory/backtest tiếp tục dùng timeline, cap và commission/slippage hiện tại.
- Gate checkpoint yêu cầu smoothed Sharpe vượt epoch 0 ít nhất 0.03 và Sharpe hơn matched baseline. Lưu riêng raw delta vs epoch 0/matched, smoothed selection delta và trạng thái gate. Nếu không đạt thì giữ fallback và báo NO-GO.
- Baseline tham chiếu sau masking được lưu trong `baseline_reference.json`; các lần chạy không dùng cash/exposure nhìn lại từ PPO để tối ưu baseline.

## 2. QNT-003 — tín hiệu và baseline tuyến tính qua fold

- 20 feature × horizon 1/5/20 × train/outer × 4 fold = **480 dòng** trong `signal_by_fold.csv`.
- Label là next-Open → Close ở horizon tương ứng. Endpoints bị purge tại từng ranh giới; sign IC học từ train. Có block CI, top-minus-bottom return, độ ổn định dấu và số mẫu.
- Ridge dùng ba feature định trước; chọn ridge/tilt bằng **6 candidate mỗi inner fold**, tổng **24 dòng** `inner_search.csv`.
- Sau khi chọn, refit train + inner; label fit cuối cùng vẫn trước outer. Lưu coefficient, center/intercept, scaler, prediction scale và thứ tự feature để tái lập.
- Danh mục lõi 1/N + cash 4%, tilt có giới hạn; cùng alpha 0.25, cùng initial cash/blend, phí và cap với matched baseline. 1/N thuần dùng alpha 1 làm đối chứng riêng.
- `linear_baseline_by_fold.csv`: **12 dòng**, gồm ba chiến lược × bốn fold. Có costs, turnover, realized cash, delta vs matched.
- Bootstrap tổng hợp giữ block **bên trong từng fold**, không tạo block chạy xuyên ranh giới fold. Dữ liệu daily outer có **1004 phiên**.

| Fold | Tilt được inner chọn | Outer Sharpe | ΔSharpe vs matched | Chi phí |
|---|---:|---:|---:|---:|
| development_2019 | 0.00 | 1.997825 | +0.000000 | 345.64 |
| development_2020 | 0.00 | 0.875461 | +0.000000 | 494.96 |
| development_2021 | 0.00 | 2.627351 | +0.000000 | 412.00 |
| development_2022 | 0.00 | -0.509156 | +0.000000 | 406.26 |

Cả bốn inner fold chọn **tilt = 0**. Excess ròng so matched bằng 0, CI [0, 0], không đạt ngưỡng 2%/năm hoặc điều kiện Sharpe/positive folds. Hai feature `excess_ret_20d` và `xs_rank_ret_20d` chỉ có CI IC theo hướng train dương ở một fold, chưa đạt điều kiện ba fold. **Quyết định: NO-GO.** Đây là đánh giá bộ feature/horizon/predictor/grid đã chốt, không phải kết luận mọi feature đều vô dụng.

## 3. Lượt PPO end-to-end trên dữ liệu thật

- Seed 7, GRU hidden 64, lookback 10, 8 cửa sổ/update, PPO 3 lượt, tối đa 10 epoch, patience 5.
- Train thực tế: 2016-01-15 → 2020-12-31, **1250 phiên** sau warm-up/lookback; inner 2021 có **252 phiên**.
- Chạy **9 epoch**, early stopping theo cấu hình. Max inner raw Sharpe improvement vs epoch 0: **0.003233**; max smoothed improvement: **0.002176**, thấp hơn biên 0.03.
- Checkpoint được chọn **epoch 0 / untrained_fallback**, quyết định **NO-GO**. Bảng outer là đánh giá policy khởi tạo đã được ghi nhãn, không trình bày như policy PPO có alpha.
- Outer 2022: PPO Sharpe −0.509393; matched −0.509156; Δ ≈ −0.000237. Outer chỉ được tính sau chọn checkpoint.
- `test_evaluated=false`; không backtest 2023–2025. Raw Parquet có thể chứa toàn lịch sử và được hash toàn file cho provenance; các hàng sau cutoff không tham gia feature/scaler/graph/backtest của lượt này.

## 4. Artifact và lệnh tái lập

- [Signal manifest](../models/runs/sprint5_signal_20261018T160239Z_ee775920/manifest.json).
- [Signal by fold](../models/runs/sprint5_signal_20261018T160239Z_ee775920/signal_by_fold.csv).
- [Linear baseline by fold](../models/runs/sprint5_signal_20261018T160239Z_ee775920/linear_baseline_by_fold.csv).
- [Inner search](../models/runs/sprint5_signal_20261018T160239Z_ee775920/inner_search.csv).
- [Signal decision](../models/runs/sprint5_signal_20261018T160239Z_ee775920/decision.json).
- [PPO manifest](../models/runs/ppo_gru_seed7_20261018T160241Z_a4838d80/manifest.json).
- [Baseline reference](../models/runs/ppo_gru_seed7_20261018T160241Z_a4838d80/baseline_reference.json).
- [PPO training log](../models/runs/ppo_gru_seed7_20261018T160241Z_a4838d80/training_log.csv).
- [Outer validation report](../models/runs/ppo_gru_seed7_20261018T160241Z_a4838d80/validation_report.json).

```bash
python scripts/analyze_feature_signal.py \
  --research-protocol configs/quant_research_protocol.yaml

python scripts/run_sprint5_backtest.py \
  --research-protocol configs/quant_research_protocol.yaml \
  --encoder gru --lookback 10 --hidden-dim 64 \
  --episodes-per-update 8 --ppo-iters 3 \
  --epochs 10 --min-epochs 5 --patience 5 --bootstrap-samples 2000
```

Mỗi lần chạy tạo thư mục mới; không ghi đè các artifact đã nghiệm thu. Data/config/source hashes của hai run trên đã được đối chiếu với source hiện tại lúc lập báo cáo. Trạng thái NO-GO không được dùng để chỉnh gate cũ hay chọn thêm candidate bằng outer.

## 5. Giới hạn nghiệm thu

- Đã thực hiện hai luồng end-to-end với dữ liệu thật; chưa chạy lại toàn bộ unit tests trong lượt triển khai Sprint 5 này.
- Đây là historical development, chưa phải holdout độc lập. Danh sách 24 cổ phiếu cố định vẫn có survivorship bias.
- Lượt PPO chỉ một seed để nghiệm thu đường chạy. Đánh giá đủ năm seed và kết luận qua thời gian thuộc ma trận thí nghiệm tiếp theo; không suy ra alpha từ lượt này.
- QNT-002/QNT-003 hoàn thành ở mức triển khai và bàn giao kết quả. Gate alpha NO-GO nên chưa mở rộng predictor/QP chỉ để tìm một kết quả đẹp hơn.

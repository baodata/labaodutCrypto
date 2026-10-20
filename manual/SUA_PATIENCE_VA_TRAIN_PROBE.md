# Patience độc lập promotion gate và đánh giá train cố định

Ngày: 19/10/2026.

Bước chẩn đoán tiếp theo đã triển khai: [actor học bắt chước convex targets trên train OOF](BAT_CHUOC_CONVEX_TARGETS.md). Thí nghiệm độc lập, chưa warm-start PPO hoặc thay các checkpoint trong pilot dưới đây.

## 1. Thay đổi

Trong `src/training/decision_trainer.py`, tách ba trạng thái:

| Trạng thái | Tiêu chí | Tác dụng |
|---|---|---|
| Patience | Smoothed inner Sharpe vượt high-water mark ít nhất `patience_min_delta` | Reset số epoch không tiến bộ |
| Best trained | Raw inner Sharpe tốt nhất trong các epoch đã train | Lưu weights chẩn đoán vào `best_trained.pth` |
| Promotion | Giữ điều kiện min_epochs, smoothed inner vượt reference +0.03 và raw score tốt hơn checkpoint được chấp nhận | Chọn `best.pth` để đánh giá |

Patience theo dõi tiến bộ từ epoch 1; chỉ cộng stale sau min_epochs. Một cải thiện nhỏ vượt high-water mark vẫn reset patience dù chưa qua +0.03. `patience_min_delta=0.00001` là tolerance tiến bộ của optimizer, không phải ngưỡng bằng chứng alpha. Gate chấp nhận mô hình vẫn là **+0.03**.

Vẫn có thể dừng ở epoch 9 nếu smoothed score thực sự không lập đỉnh mới trong 5 epoch từ min_epochs trở đi. Việc sửa không ép mọi lượt chạy đủ 20 epoch.

`best_trained.pth` có `checkpoint_role=diagnostic_only_not_promoted`; runner không tự đưa weights đó vào outer. `last.pth` lưu training state mới nhất; `best.pth` giữ vai trò cũ và có thể vẫn là reference epoch 0.

## 2. Probe train deterministic

Module `src/training/decision_probes.py` tạo các cửa sổ đều theo lịch **train OOF**, không dùng inner/outer để chọn cửa sổ và không tiêu thụ rollout RNG.

Mặc định: 3 cửa sổ, mỗi cửa sổ 126 quyết định. Khi lịch train ngắn hơn, độ dài bị giới hạn bởi lịch khả dụng và start trùng được gộp; toàn bộ cửa sổ thực dùng được ghi lại. Mỗi cửa sổ reset về cash, dùng cùng alpha/cost/constraints và benchmark đồng bộ. Cửa sổ không đổi giữa epoch, method và seed trong cùng fold/config.

Đánh giá epoch 0 và sau mỗi update, bằng deterministic policy. Benchmark của mỗi cửa sổ được tính một lần rồi dùng lại. Đây là đánh giá in-sample trên OOF forecast dùng để học allocator, không phải tập ngoài mẫu mới.

Các cửa sổ có thể chồng lấn. Tổng hợp bằng mean metric theo cửa sổ và average daily net active return; không coi chúng là các quan sát độc lập để tính CI/p-value. Không nối chúng thành một đường NAV liên tục giả.

Các cột mới trong `training_log.csv`:

- `train_probe_mean_sharpe`, `train_probe_mean_sharpe_delta_vs_matched`.
- `train_probe_mean_return`, `train_probe_mean_daily_net_active_return`.
- `train_probe_mean_abs_target_tilt`, `train_probe_mean_daily_turnover`.
- `inner_sharpe_delta_vs_matched`, `patience_improved`, `patience_best_score`, `epochs_without_progress`, `best_trained_epoch`.
- Step reward mean/std, return/advantage std trước khi normalize advantage.
- KL first/max/stop, số minibatch completed/attempted, PPO epochs và early-stop flag.

`train_probe_metrics.csv` lưu từng window/epoch. `train_probe_windows.json` lưu ngày bắt đầu/kết thúc. Checkpoint lưu riêng patience state, diagnostic-best state và promoted state. Cột financial cũ trong log vẫn là đánh giá **inner**, không bị thay nghĩa thành train.

## 3. Cấu hình và lệnh

Trong `configs/portfolio_decision.yaml` bổ sung:

```yaml
patience_min_delta: 0.00001
train_probe_windows: 3
train_probe_days: 126
```

Các config cũ chưa có ba trường này được bổ sung default tương ứng lúc resolve. Gamma, LR, action noise, kiến trúc, chi phí và promotion margin không đổi.

Pilot: seed 7, fold `development_2021`, cả PPO/sector MARL/H-MARL, tối đa 20 epoch như profile cũ. Fold này được dùng để kiểm tra learner trên giai đoạn inner có IC dương trong chẩn đoán trước; không xem pilot là đánh giá ngoài mẫu độc lập.

```bash
python scripts/compare_portfolio_decisions.py \
  --predictor-lock models/runs/locked_predictor_h5_20261019/predictor_lock.json \
  --decision-config configs/test/portfolio_decision_development.yaml \
  --development-only \
  --fold development_2021
```

`--development-only` bỏ toàn bộ outer backtest và bootstrap gate, chỉ xuất learning curves và inner metrics. `--fold` chỉ dùng cùng development-only để không tạo báo cáo toàn protocol từ một fold riêng.

Chạy toàn bộ 5 seed/4 fold để nghiên cứu learning curves, vẫn không đánh giá outer:

```bash
python scripts/compare_portfolio_decisions.py \
  --predictor-lock models/runs/locked_predictor_h5_20261019/predictor_lock.json \
  --decision-config configs/portfolio_decision.yaml \
  --development-only
```

Mỗi run tạo thư mục mới, giữ nguyên artifacts cũ. `learning_curves.csv` ở root run gộp log từng epoch, có fold/method/seed. Khi đọc underfit/generalization, ưu tiên **delta vs matched trong từng phase**; không so trực tiếp Sharpe tuyệt đối của cửa sổ train 126 ngày với inner cả năm khác chế độ thị trường.

Chưa chạy lại full 5-seed benchmark trong bước sửa này. Không tự đổi predictor hoặc hạ gate theo learning curves/outer đã xem.

## 4. Kết quả pilot hoàn tất

Run: `models/runs/portfolio_decision_20261019T063319Z_6f6e7ab1/`.

Seed 7, fold 2021, predictor GRU đã khóa. Chỉ train/inner, không có hàng outer trong metrics, manifest `outer_evaluated=false`, `test_evaluated=false`.

| Nhánh | Số epoch thực chạy | ΔSharpe train probe ở epoch cuối | ΔSharpe inner ở epoch cuối | Epoch best trained |
|---|---:|---:|---:|---:|
| PPO | 9 | +0.002372 | −0.000206 | 2 |
| Sector MARL | 11 | +0.001991 | +0.000590 | 9 |
| H-MARL | 11 | +0.001927 | +0.000974 | 5 |

Các delta đều so với matched baseline trong cùng phase/window. Cột epoch cuối mô tả weights đã train ở epoch cuối, không phải checkpoint được promotion. Best trained được chọn bằng raw inner Sharpe; nó chỉ phục vụ chẩn đoán.

PPO vẫn dừng ở 9 vì smoothed inner score không vượt đỉnh trước trong 5 epoch; sector/H-MARL kéo dài đến 11 vì có tiến bộ dưới gate được ghi nhận và reset patience. Cả ba vẫn chọn epoch 0 cho `best.pth`: chưa đạt +0.03.

Đối chiếu epoch 0–9 với run cũ cùng fold/seed: max absolute difference của Sharpe, rollout episode reward, policy/value loss, KL và EV đều **0.0**. Việc thêm deterministic probes không thay đổi trajectory học ở phần epoch chung.

Ba nhánh có `train_probe_windows.json` giống nhau: 126 quyết định bắt đầu tại 2017-01-03, 2018-03-29 và 2019-06-26. Checkpoint diagnostic đều mang role `diagnostic_only_not_promoted`. Source hashes giữ nguyên từ đầu đến cuối pilot.

Train probe có cải thiện nhỏ nhưng inner cải thiện ít hơn, riêng PPO cuối kỳ âm. Điều này cho thấy **đã có một ít tiến bộ in-sample, chưa chuyển thành lợi thế inner rõ ràng**. Không đủ để kết luận tăng epoch sẽ giải quyết được hoặc quy kết overfitting từ một seed/fold. EV rollout cuối vẫn âm ở cả ba nhánh.

Tệp nên đọc:

- `learning_curves.csv`: toàn bộ đường học của cả ba nhánh, gồm epoch 0.
- `learning_curve_readout.csv`: bảng kết quả pilot, epoch best trained và đối chiếu với run cũ.
- `policies/development_2021/<method>_seed7/train_probe_metrics.csv`: từng cửa sổ theo epoch, để xem mức tiến bộ có đồng đều hay tập trung ở một giai đoạn.

Đã kiểm tra cú pháp và chạy pilot thực tế; không thêm/chạy unit tests. Thí nghiệm full 5 seed/4 fold chưa được chạy lại, các kết quả cũ không bị ghi đè.

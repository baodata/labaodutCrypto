# GRU và GNN-GRU: đối chứng predictor với allocator cố định

Ngày triển khai: 19/10/2026.

Bước tiếp theo đã có: [khóa predictor và so sánh Convex → PPO → Sector MARL → H-MARL](LOCK_PREDICTOR_VA_SO_SANH_ALLOCATOR.md). Predictor chọn riêng từng fold từ inner IC, không lấy winner từ outer summary bên dưới.

## 1. Kết quả kiểm tra hệ thống trước thay đổi

`src/models/actor_critic.py` đã có GRU và DynamicGraphActorCritic (graph → GRU), nhưng đó là encoder phục vụ policy/value của PPO. Chúng không được train bằng nhãn lợi suất supervised và chưa xuất dự báo vào prediction bank.

Đã bổ sung `src/models/supervised_predictor.py`, tích hợp vào builder bank và runner so sánh hiện có. Không sửa hàm mục tiêu hay cách giải của `cost_aware_allocator.py`.

## 2. Kiến trúc và dữ liệu

| Predictor | Đường xử lý |
|---|---|
| Ridge | Ba feature hiện tại → hồi quy lợi suất tương đối |
| GRU | Linear projection từng mã → GRU lịch sử → scalar head từng mã |
| GNN-GRU | Linear projection → relation messages từng ngày → cùng GRU/head |
| GNN-GRU no edges (tùy chọn) | Cùng kiến trúc/tham số GNN-GRU, vô hiệu message edges |

- Cùng ba feature của Ridge: `xs_rank_ret_20d`, `mom_12_1`, `excess_ret_20d`. Neural không tự nhận thêm 20 feature hoặc trạng thái danh mục.
- Cùng nhãn executable Open[t+1] → Close[t+h], target cross-sectional excess; forecast thập phân, không annualize và chưa trừ phí.
- Không pooling mất trục tài sản. GRU/head dùng chung trọng số cho các mã; output `[batch, assets]` được center theo từng ngày.
- GRU và GNN-GRU có cùng projection, GRU/head và cùng khởi tạo phần chung với cùng seed. GNN thêm transform riêng cho tương quan dương, độ lớn tương quan âm và quan hệ ngành.
- Dùng dense message passing với N nhỏ; không gọi GAT/PyG lặp riêng từng graph trong minibatch. Source adjacency `[source,target]` được transpose để aggregate incoming neighbors, normalize theo hàng xóm từng relation, giữ residual của node.
- Graph dùng close-adjusted returns đến ngày t và sector config; correlation window 60, kNN=6 theo profile. Không đọc return sau t.
- History 20 phiên với mask left-padding tường minh; packed GRU bỏ timestep giả. Thống kê scaler không fit trên padding/feature warm-up.

GNN có thêm thông tin quan hệ và thêm tham số. So sánh GRU/GNN đo đóng góp của predictor đầy đủ; chưa cô lập riêng graph topology. Control không cạnh giúp kiểm tra bước tiếp theo. Quan hệ ngành vẫn dựa trên taxonomy hiện tại, không tự khắc phục survivorship bias.

## 3. Fit causal và checkpoint

Mỗi block/fold/horizon fit độc lập từ cùng seed, không warm-start bằng model đã thấy nhãn tương lai:

- `train_oof`: expanding window, label endpoints trước ngày đầu block.
- `inner`: fit chỉ đến train_end; không dùng inner labels để chọn epoch.
- `outer`: refit chỉ đến inner_end, cấu hình đã cố định.

Feature scaler fit trên cùng quá khứ của Ridge. Neural chuẩn hóa target bằng std của nhãn fit; khi predict nhân trả về đơn vị return. Loss là MSE chuẩn hóa, Adam, gradient clip=1. Dropout=0; fixed epoch budget, không early stop/chọn checkpoint theo outer.

Profile ban đầu: hidden_dim=16, 5 epoch, batch 64 ngày (giữ nguyên panel các mã trong cùng ngày), lr=0.001, weight_decay=0.0001, seed=7, CPU 1 thread. Đây là ngân sách đối chứng nhỏ, chưa được chọn theo kết quả outer và chưa chứng minh đủ hội tụ cho mọi fold.

Mỗi fit lưu `neural_models/*.pth`, gồm model state/config, scaler, target scale, ticker/feature order và provenance. Bank ghi `weights_artifact`; manifest hash cả weights. Nạp bằng `load_neural_checkpoint()` dùng `weights_only=True`. Di chuyển bank cần sao chép cả thư mục weights, không chỉ CSV.

## 4. Lệnh chạy

```bash
python scripts/compare_signal_allocators.py \
  --prediction-config configs/prediction_bank_neural.yaml \
  --allocation-config configs/signal_allocation_neural.yaml
```

Luồng gồm tạo bank Ridge + rule baselines + GRU + GNN-GRU, rồi chạy cùng allocator `matched_1n`, `bounded_tilt`, `cost_aware` trên cùng inner/outer folds.

Chỉ tạo bank:

```bash
python scripts/build_prediction_bank.py \
  --prediction-config configs/prediction_bank_neural.yaml
```

Để ablation không cạnh, tạo profile mới từ neural profile, thêm `gnn_gru_no_edges` vào cả `neural.models` và `allocation.models`. Không sửa cấu hình một run đã chạy. Default statistical profiles cũ vẫn không train neural.

## 5. Đo đóng góp predictor

`signal_allocation_neural.yaml` chỉ mở rộng danh sách model. So với profile statistical, giữ nguyên horizons, tilt bounds, risk_aversion, cost multiplier, covariance, cash-funded accounting, commission/slippage, daily cadence, cash và alpha từ protocol. Các constraint/execution của directional baseline cũng không đổi.

Ngoài `prediction_ic_by_fold.csv`, `metrics_by_fold.csv`, `outer_summary.csv`, bổ sung:

- `predictor_contribution_by_fold.csv`: ΔSharpe/return/cost/turnover giữa predictor trong cùng allocator/horizon/fold.
- `predictor_contribution_summary.csv`: contrasts `GRU − Ridge`, `GNN-GRU − GRU`; thêm `GNN-GRU − no-edges` nếu có control.

Bootstrap các chuỗi return ròng paired trên đúng cùng phiên; block ít nhất bằng horizon, không trộn inner vào outer. Holm điều chỉnh chung cả các giả thuyết vs matched và paired predictor contrasts của lượt chạy; profile 5 model/2 allocator/2 horizon có 20 + 8 = 28 phép đo. Vì số giả thuyết tăng, adjusted p của Ridge có thể khác run cũ dù raw portfolio metrics không đổi.

`positive_contribution_supported` đòi hỏi adjusted p<0.05, CI excess dương và mean ΔSharpe dương. Đây là bằng chứng phần tăng thêm so với predictor đối chứng, không tương đương alpha độc lập so với thị trường.

## 6. Kiểm thử

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  --disable-warnings --basetemp /tmp/ai-project1-neural-final-20261019
```

**246 passed, 3 warnings, 23.30 giây.** Coverage mới: mask/padding parity, tác dụng graph và gradients, zero-edge initialization/training parity với GRU, future perturbation, checkpoint replay, hash weights/tampering/path traversal, config allocator parity và paired-session contrasts. Checkpoint replay khớp bit trên output trong bộ nhớ; CSV round-trip dùng tolerance cho sai số float64 ULP.

Không dùng test 2024–2025. Full-data experiment một seed không thay thế kiểm chứng nhiều seed hoặc final holdout độc lập.

## 7. Kết quả chạy dữ liệu thật

Run hoàn tất: `models/runs/signal_allocation_20261019T040033Z_ca30a67d/`.

- Bank: 1,051,200 forecast rows, 480 fit records, trong đó 192 fit neural có checkpoint riêng.
- Outer: 2019/2020/2021/2022, horizon 5/20 phiên, seed 7 và 5 epoch mỗi neural fit.
- Hash raw data, allocator, simulator, cost engine và covariance trùng lượt statistical trước.
- Đối chiếu 112 case baseline cũ: max absolute difference của cumulative return, Sharpe, turnover, costs và mean realized cash đều **0.0**. Khác adjusted p là do số giả thuyết tăng, không phải thay thuật toán allocator.

### Cùng allocator cost-aware: so với matched 1/N

| Predictor | h | Mean ΔSharpe | Excess ròng năm hóa | CI pointwise 95% |
|---|---:|---:|---:|---|
| Ridge | 5 | +0.0111 | +1.309% | [-0.322%, +2.838%] |
| GRU | 5 | -0.0138 | +1.394% | [-0.305%, +3.039%] |
| GNN-GRU | 5 | -0.0120 | +1.420% | [-0.226%, +2.990%] |
| Ridge | 20 | -0.0072 | +1.297% | [-0.424%, +2.917%] |
| GRU | 20 | -0.0054 | +1.204% | [-0.390%, +2.768%] |
| GNN-GRU | 20 | -0.0244 | +0.222% | [-1.232%, +1.700%] |

Excess là `mean(net_return_candidate - net_return_reference)*252`, không phải chênh CAGR hoặc factor-model alpha. Mean ΔSharpe là trung bình delta trên bốn fold; excess dương và ΔSharpe âm có thể cùng xảy ra do phân bố return/risk khác nhau.

### Đóng góp tăng thêm, cùng allocator cost-aware

Nguồn: `comparison/predictor_contribution_summary.csv`.

| Contrast | h | Mean ΔSharpe | Excess ròng năm hóa vs predictor đối chứng | CI 95% |
|---|---:|---:|---:|---|
| GRU − Ridge | 5 | -0.0249 | +0.085% | [-0.672%, +0.875%] |
| GRU − Ridge | 20 | +0.0017 | -0.093% | [-0.656%, +0.490%] |
| GNN-GRU − GRU | 5 | +0.0019 | +0.026% | [-0.376%, +0.423%] |
| GNN-GRU − GRU | 20 | -0.0190 | -0.982% | [-2.014%, +0.047%] |

Tất cả 8 predictor contrasts (cả bounded tilt và cost-aware) có Holm p=1 và `positive_contribution_supported=False`. Cả 20 cấu hình vs matched đều không vượt portfolio gate; overall decision **NO-GO**.

IC chưa ổn định: GRU/GNN-GRU h=5 có mean IC dương ở 2019/2020/2022 nhưng âm ở 2021; chỉ CI pointwise 2020 dương cho cả hai. GNN-GRU h=20 yếu hơn GRU ở 2020 và 2022. Chưa có bằng chứng thêm graph cải thiện tín hiệu ngoài mẫu trong profile này.

**Kết luận nghiệm thu:** tích hợp predictor và phép đối chứng đã chạy hoàn chỉnh; lợi ích kinh tế tăng thêm chưa được xác nhận. Giữ các profile neural ở chế độ opt-in. Không kết luận mọi GRU/GNN đều thất bại từ một seed/ngân sách 5 epoch; cũng không chỉnh budget theo các outer vừa xem rồi coi chúng là xác nhận mới. Control no-edge đã kiểm thử trên dữ liệu synthetic, chưa chạy thành một nhánh full-data của cohort này.

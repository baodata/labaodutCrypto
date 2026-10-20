# Ridge, momentum/mean-reversion và allocator có tính chi phí

Triển khai ngày 19/10/2026, trên [prediction bank và nhãn giao dịch](PREDICTION_BANK_VA_NHAN_GIAO_DICH.md). Đây là nghiên cứu phát triển; không đánh giá test 2024–2025.

## 1. Lệnh chạy

```bash
python scripts/compare_signal_allocators.py
```

Mặc định tạo thư mục mới `models/runs/signal_allocation_<timestamp>_<id>/`:

- `prediction_bank/`: tạo bank mới nếu không truyền bank có sẵn.
- `comparison/`: kết quả IC, backtest và đối chiếu.

Để dùng lại bank có đủ ba model và cùng data snapshot/universe:

```bash
python scripts/compare_signal_allocators.py \
  --prediction-bank models/runs/<run>/prediction_bank
```

Config tín hiệu: `configs/prediction_bank.yaml`. Config allocator: `configs/signal_allocation.yaml`. Có thể truyền `--prediction-config`, `--allocation-config`, `--research-protocol`, `--raw-dir`, `--assets-config`, `--env-config`, `--output-dir`.

## 2. Ba nguồn dự báo

| Model | Tín hiệu / target | Học từ dữ liệu nào |
|---|---|---|
| `ridge_fixed` | Ridge trên `xs_rank_ret_20d`, `mom_12_1`, `excess_ret_20d` | Các nhãn đã kết thúc trước cutoff |
| `momentum` | Rank cắt ngang của `mom_12_1`, chiều dương | Biên độ return được calibrate trên quá khứ |
| `mean_reversion` | Rank cắt ngang của `ret_5d`, đảo dấu | Biên độ return được calibrate trên quá khứ |

Rank được center về 0, nằm trong [-1,1], xử lý ties và N=1. Hai rule có slope không âm khi fit lên nhãn lợi suất tương đối: nếu dữ liệu train không ủng hộ chiều đã định, slope về 0; không tự đổi mean-reversion thành momentum để cải thiện kết quả.

Tất cả xuất dự báo return thập phân cùng target và lịch nhãn. Không trực tiếp dùng rank như expected return. Đơn vị đúng không bảo đảm calibration ngoài mẫu tốt. Hệ số `prediction_scale` của directional tilt chỉ tính trên fit quá khứ, không tính trên val/outer.

Lịch fit: `train_oof` expanding; `inner` freeze tại train_end; `outer` refit đến inner_end. Hyperparameter cố định; không search/chọn lại theo outer.

## 3. Các cách phân bổ

| Nhánh | Cơ chế |
|---|---|
| `matched_1n` | Core bằng nhau với cash và action_alpha lấy từ protocol |
| `bounded_tilt` | Tilt có giới hạn theo prediction / historical prediction scale |
| `cost_aware` | Tối ưu dự báo, active risk và chi phí turnover cùng lúc |

Core cổ phiếu `b_i=(1-cash)/N`. Mỗi target cổ phiếu `u_i` nằm trong `[b_i*(1-tilt), min(b_i*(1+tilt), asset_cap)]`; tổng cổ phiếu bằng `1-cash`. Cả directional tilt và optimizer dùng đúng cùng bộ ràng buộc.

Optimizer nhận weights hiện tại `w` tại Close, với execution blending `x=(1-alpha)*w+alpha*u`. Hàm mục tiêu là:

```text
min_u  risk_aversion/2 * (x - x_core)' * (h * Sigma_daily) * (x - x_core)
       - mu_h' * x_stock
       + cost_multiplier * (commission_rate + slippage_rate)
         * sum(abs(x_stock - w_stock))
```

`x_core` là target core sau cùng phép blend. Dùng tracking risk để đo phần tilt, không tự biến bài toán thành market timing bằng tiền mặt. `h*Sigma_daily` là xấp xỉ covariance cho planning horizon dưới giả định iid; chưa mô hình hóa serial covariance hoặc overnight/intraday riêng.

L1 turnover dùng biến phụ và ràng buộc tuyến tính; covariance PSD cho bài toán convex. SciPy SLSQP giải với gradient tường minh, kiểm tra tính khả thi và trạng thái solver. Lỗi solver dừng run, không âm thầm thay bằng baseline rồi báo optimum.

## 4. Fair execution và phí thực tế

- Default horizon so sánh 5 và 20 phiên; rebalance hàng ngày theo protocol. Horizon là planning/label horizon, không phải lịch giao dịch 5/20 ngày.
- Cùng khởi tạo 100% cash, cùng `action_alpha=0.25`, target cash 4% theo protocol hiện tại; cùng asset cap từ env config.
- Cùng simulator và accounting cash-funded. Phí gồm commission + slippage từ env config, tính trên actual stock legs tại Open tiếp theo. Không tính phí cash leg.
- Cost forecast của optimizer chỉ dùng Close-time holdings; không đọc Open[t+1] để quyết định. Gap qua đêm khiến phí thực tế khác dự báo và được ghi nhận bằng simulator.
- Không bật adaptive controller hoặc square-root impact trong phép đối chiếu này; chúng là ablation riêng.
- `h-1` ngày cuối phase không có nhãn forecast đủ horizon: mọi nhánh, kể cả matched, cùng phát `PortfolioOrder.hold()` cho đoạn đuôi. Không lấp dự báo thiếu hoặc tái cân bằng giả trên ngày hold.
- Báo cáo cash thực tế, turnover và phí để phát hiện khác biệt do exposure/ingress.

Đây là benchmark optimizer một bước với planning horizon h và cập nhật hàng ngày, chưa phải multi-period optimizer đầy đủ.

## 5. Kết quả cần đọc

Trong `comparison/`:

| Tệp | Nội dung |
|---|---|
| `prediction_ic_by_fold.csv` | IC Spearman của dự báo, n_dates và block CI, tách inner/outer |
| `metrics_by_fold.csv` | Return/Sharpe/MDD/cash/turnover/cost; delta vs matched và vs directional tilt cùng model |
| `daily_returns.csv` | Chuỗi return ròng, matched return, excess, turnover và phí |
| `allocator_decisions.csv` | Trạng thái solver, iterations, cost dự báo và planned active variance |
| `outer_summary.csv` | Tổng hợp outer disjoint của từng model/allocator/horizon |
| `decision.json` | Portfolio gate, số giả thuyết và giới hạn diễn giải |
| `manifest.json`, `resolved_comparison.json` | Source/data/config hash, risk/liquidity/environment và quy ước thực nghiệm |

IC không xác định khi forecast hoặc nhãn hằng số: ghi NaN và n_dates hợp lệ, không coi IC=0 là phép đo thật. IC CI và excess-return CI dùng block ít nhất bằng horizon, với bootstrap theo fold cho chuỗi outer.

`outer_summary` có p-value đã hiệu chỉnh Holm trên toàn bộ model × allocator × horizon của lượt chạy, bên cạnh CI pointwise 95%. Các CI không phải simultaneous CI. Số cấu hình đã thử ở các lượt trước vẫn là lịch sử nghiên cứu cần ghi nhận; Holm trong một run không xóa selection bias từ những lần trước.

Portfolio gate lấy ngưỡng trong protocol: số fold dương, excess ròng năm hóa, CI, mean ΔSharpe và drawdown degradation; thêm Holm p<0.05. Đây chưa bao gồm nghiệm thu signal-stability của toàn hệ và chưa phải xác nhận alpha độc lập.

Không gộp inner với outer để báo cáo OOS. Không chọn model thắng outer rồi xem kết quả của chính outer như xác nhận mới. Có thể đọc riêng `cost_aware` so với `bounded_tilt` để đánh giá phần đóng góp của allocator với cùng predictor.

## 6. Kết quả thực nghiệm ngày 19/10/2026

- Bank: `models/runs/signal_allocation_20261019T034739Z_fa4db1da/prediction_bank/`.
- So sánh hoàn tất: `models/runs/signal_allocation_20261019T034906Z_e2008c77/comparison/`.
- Bốn outer fold: 2019, 2020, 2021, 2022; inner tương ứng 2018–2021.
- 3 model × 2 allocator × 2 horizon = 12 giả thuyết; 11,496 lần giải allocator có trạng thái hội tụ, tối đa 15 iterations.
- Bank đã được tạo/lưu/đọc trên dữ liệu thật; không chạy unit test trong lượt này.
- Lượt so sánh đầu gặp `ValueError: output array is read-only` của pandas tại tính IC. Đã sửa bằng bản sao NumPy, giữ artifact failed, dùng lại bank đã tạo thành công và chạy lại toàn bộ phép so sánh. Manifest so sánh mới ghi riêng hash predictor và source so sánh.

**Toàn bộ 12 cấu hình NO-GO theo portfolio gate.** Bảng dưới lấy từ `outer_summary.csv`; excess là trung bình chênh lệch return ròng ngày nhân 252, không phải chênh lệch CAGR hoặc regression alpha. ΔSharpe là trung bình delta của bốn fold. CI pointwise 95%; p đã điều chỉnh Holm trên 12 cấu hình.

| Model | Allocator | h | Mean ΔSharpe | Excess ròng năm hóa | CI 95% | Holm p |
|---|---|---:|---:|---:|---|---:|
| Ridge | Bounded tilt | 5 | +0.0034 | +1.215% | [-0.055%, +2.464%] | 0.360 |
| Ridge | Cost-aware | 5 | +0.0111 | +1.309% | [-0.322%, +2.838%] | 0.480 |
| Momentum | Bounded tilt | 5 | -0.0400 | +0.294% | [-0.925%, +1.411%] | 1.000 |
| Momentum | Cost-aware | 5 | -0.0305 | -0.087% | [-1.268%, +0.993%] | 1.000 |
| Mean-reversion | Bounded tilt | 5 | -0.0993 | -1.318% | [-2.692%, -0.046%] | 1.000 |
| Mean-reversion | Cost-aware | 5 | -0.0157 | -0.617% | [-1.497%, +0.294%] | 1.000 |
| Ridge | Bounded tilt | 20 | +0.0047 | +1.138% | [-0.186%, +2.401%] | 0.467 |
| Ridge | Cost-aware | 20 | -0.0072 | +1.297% | [-0.424%, +2.917%] | 0.612 |
| Momentum | Bounded tilt | 20 | -0.0405 | +0.272% | [-0.937%, +1.382%] | 1.000 |
| Momentum | Cost-aware | 20 | -0.0303 | -0.031% | [-1.127%, +0.994%] | 1.000 |
| Mean-reversion | Bounded tilt | 20 | 0.0000 | 0.000% | [0.000%, 0.000%] | 1.000 |
| Mean-reversion | Cost-aware | 20 | -0.0102 | +0.145% | [-0.785%, +1.107%] | 1.000 |

### Đóng góp allocator và tín hiệu

Ridge h=5: phí trung bình mỗi outer fold từ 447.10 ở bounded tilt xuống 390.55 ở cost-aware, so với matched 411.48 (NAV khởi tạo 100,000 mỗi fold). Mean ΔSharpe tăng từ +0.0034 lên +0.0111 nhưng chỉ 2/4 fold có cumulative-return delta dương, CI vẫn chứa 0 và chưa đạt ngưỡng excess 2%/năm.

Mean-reversion h=5: optimizer giảm phí trung bình từ 1,519.92 xuống 371.05 và giảm mức thua matched. Đây là kiểm soát turnover, chưa phải bằng chứng có alpha chọn tài sản.

Cash thực tế trung bình khoảng 5.51% cho cả các nhánh và matched, cao hơn cash target 4% do khởi tạo lại từ cash và blend ở đầu mỗi fold; không diễn giải phần cash này như dự báo market timing.

IC Ridge có tín hiệu theo giai đoạn, chưa ổn định: h=5 mean IC các outer 2019–2022 lần lượt -0.0033, +0.0977, -0.0439, +0.0729. CI dương rõ ở 2022 cho h=5 và ở 2020 cho h=20; đây là pointwise theo nhiều phép đo, không phải xác nhận độc lập.

Momentum có slope calibration bằng 0 ở outer 2019–2020. Mean-reversion h=20 bằng 0 ở cả bốn outer fold, nên IC dự báo không xác định và bounded tilt trùng matched. Không suy ra raw momentum/reversal không có tín hiệu chỉ từ các dòng NaN này: chúng cho biết mô hình calibration cố định đã không ủng hộ chiều giả thuyết trên dữ liệu fit.

**Diễn giải:** Ridge là hướng có thể nghiên cứu tiếp bằng protocol mới, nhưng chưa có cơ sở dùng kết quả này để tuyên bố alpha hoặc nâng độ phức tạp của RL. Không chỉnh config theo các outer vừa xem rồi tiếp tục coi các năm đó là xác nhận độc lập.

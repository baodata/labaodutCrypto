# Actor PPO học bắt chước convex targets trên train OOF

Ngày triển khai: 19/10/2026.

## 1. Phạm vi

Đã thêm `src/training/convex_imitation.py`, CLI `scripts/train_convex_imitation.py` và `configs/convex_imitation.yaml`.

Thí nghiệm dùng **actor PPO forecast-only** (`ForecastAllocatorPolicy.single_actor`) của nhánh decision comparison. Predictor đã khóa, critic không được train, không chạy PPO update và không tự warm-start các thí nghiệm PPO trước. Mục tiêu là kiểm tra actor/action mapping có học được hành vi convex từ supervised targets hay không.

Chưa mở rộng sang sector MARL: ngân sách ngành cố định của nó làm một phần global convex targets không thuộc feasible set. Nếu đưa vào cùng loss mà không tách representation floor sẽ dễ kết luận sai về learner. H-MARL cũng chưa được distill trong lần này.

## 2. Teacher và dữ liệu

- Fold `development_2021`, seed 7, predictor GRU h=5 từ lock hiện tại.
- Teacher là nguyên `cost_aware_target()`: forecast, covariance causal, current Close weights và chi phí đã biết ở t; không nhận Open[t+1].
- Collect state/target trước khi `env.step()`. Prices kế tiếp chỉ cập nhật NAV/holdings để có observation của bước sau.
- Train pairs gồm continuous train OOF trajectory và 3 fixed train windows giống probe hiện tại, mỗi window reset cash. Có 1,127 state/target pairs, **749 ngày train khác nhau**; các ngày lặp lại có holdings context khác nhau, không được coi là mẫu thị trường độc lập.
- Inner gồm 248 decision dates; chỉ dùng đánh giá fidelity và closed-loop, không dùng fit/scaler/early stopping/chọn epoch.
- Giữ input raw state 682 chiều, hidden32, cash/alpha/bounds/cost như comparison. Không thêm state normalizer để thay đổi bài toán mà không ghi nhận.
- Inner từng được dùng chọn predictor family; vì vậy nó nằm ngoài dữ liệu fit actor nhưng không phải holdout độc lập của toàn pipeline.

## 3. Loss và action mapping

Training dùng mean của actor, không sample Gaussian:

```text
state → actor scores → tanh → bounded simplex projection → stock target weights
loss = mean(((actor_target - convex_target) / (core_weight * tilt_fraction))²)
```

MSE tính trên stock legs; cash luôn cố định và không tạo thành tọa độ dễ làm giảm loss. Chia theo radius của tilt đưa loss về đơn vị chuẩn hóa, không thay bounds.

`ForecastAllocatorPolicy.target_tensors()` hỗ trợ batch và gradient qua đúng công thức tanh/projection của execution. `target()` vẫn trả numpy cho một action như trước và gọi chung implementation. Các phương pháp PPO/sector/H-MARL giữ công thức action mapping; thí nghiệm imitation chỉ fit PPO actor.

Critic được đặt `requires_grad=False`; optimizer chỉ nhận `single_actor.parameters()`. Loss không chứa financial return, critic loss hoặc entropy bonus.

Budget cố định trước run: Adam lr=0.001, 200 epoch, batch64, grad clip1, CPU1thread. Không chọn checkpoint theo inner; `imitation.pth` là epoch cuối theo budget đã khai báo. Gate promotion +0.03 của runner PPO không bị thay đổi.

## 4. Các đối chứng quan trọng

### Core 1/N

Actor head khởi tạo 0 nên output deterministic ban đầu là core. MAE so với core cho biết có học thêm được so với khởi tạo hay không.

### Persistence projected

Chiếu **current Close stock weights** về cùng bounds và stock budget 96%. Baseline này không dùng giá trị forecast trong quyết định. Đây là một weight target được tái cân bằng qua cùng alpha, không phải lệnh giữ nguyên số cổ phiếu; nó vẫn có thể phát sinh giao dịch qua overnight drift.

Trên teacher-forced dataset, current holdings đã chứa ảnh hưởng của các quyết định teacher trước đó. Teacher có transaction cost/no-trade behavior nên baseline này có thể rất chính xác. Cần so actor với baseline này, không chỉ với core.

### Action mapping floor

Dùng inverse `atanh` của normalized teacher tilt, clip ±(1−1e−4) để score hữu hạn, rồi đi qua chính action mapping. Đây là kiểm tra biểu diễn trên targets đã có, không phải model dùng khi giao dịch.

MAE reconstruction trên train khoảng **0.0030 bp**. Như vậy action map PPO có thể xấp xỉ các teacher targets rất sát; sai số NN vài chục bp không phải một lower bound bắt buộc do constraints ở trường hợp này. Nó chưa chứng minh hidden32/raw inputs có thể học chính xác toàn bộ hàm teacher trong budget hiện tại.

## 5. Kết quả hoàn tất

Run đầy đủ các baseline:

`models/runs/convex_imitation_20261019T104638Z_c413badb/`.

### Fidelity trên cùng teacher states

MAE của stock target, đơn vị basis point của NAV weight (1 bp = 0.0001 = 0.01 điểm phần trăm):

| Phương pháp | Train | Inner |
|---|---:|---:|
| Core/actor epoch 0 | 74.11 bp | 82.81 bp |
| Actor sau 200 epoch | 17.22 bp | 57.07 bp |
| Persistence projected | **3.56 bp** | **4.20 bp** |

MSE actor giảm 87.53% so với core trên train, nhưng chỉ 16.05% trên inner. Quan trọng hơn, actor **vẫn kém persistence baseline** về fidelity. Chưa thể gọi đây là clone chính xác teacher hoặc đưa ngay vào PPO với giả định initialization đã tốt.

### Closed-loop, actor dùng holdings của chính mình

| Phương pháp | Sharpe train | Sharpe inner |
|---|---:|---:|
| Matched 1/N | 1.2308 | 0.8756 |
| Convex teacher | 1.2227 | 1.0152 |
| Persistence projected | 1.2428 | 0.8990 |
| Actor imitation | 1.2888 | 1.0040 |

Imitation có hành vi khác core và có kết quả inner tốt hơn baseline trong pilot này. Đây là **một seed/một fold**, không có outer/test hay kiểm định alpha; không tự promotion hoặc thay checkpoint benchmark cũ.

Trong closed-loop, so output actor với teacher được query lại **tại learner's own holdings**, MAE là 5.84 bp trên train và 19.76 bp trên inner. Không so trực tiếp chúng với MAE teacher-forced 17.22/57.07 như cùng một tập dữ liệu: state distribution và teacher target phụ thuộc holdings đều đã đổi.

Field `zero_forecast_target_change_bps` trong learning curves là thay đổi mean trên toàn vector stock+cash khi zero hai nhóm forecast input; đây chỉ là sensitivity probe (có thể ngoài phân phối), không phải causal feature importance. MAE/RMSE fidelity chính chỉ tính stock legs.

## 6. Diễn giải và bước tiếp theo

Đã chứng minh actor có nhận gradient và học được allocation khác khởi tạo, nên không phải toàn bộ mapping bị đứng. Tuy nhiên:

1. Fidelity vẫn kém một baseline sao chép holdings đơn giản.
2. Khoảng cách train–inner lớn; train fit tốt hơn không tự chuyển thành clone tốt trên inner.
3. Portfolio inner có cải thiện nhưng không thay thế bằng chứng về fidelity hoặc alpha độc lập.

Bước tiếp theo hợp lý là đối chứng **chuẩn hóa state chỉ fit từ train**, rồi kiểm tra lại fidelity so với persistence. Cũng có thể nghiên cứu residual target quanh persistence, nhưng đó sẽ là một thay đổi parameterization riêng. Không tự quy kết rằng cần thêm agent hoặc chỉ tăng epoch.

Chưa train lại critic, chưa warm-start PPO và chưa thay gate. Nếu dùng checkpoint sau này cần nạp actor có chủ đích và giữ provenance imitation; critic trong checkpoint vẫn là khởi tạo.

## 7. Lệnh và artifacts

```bash
python scripts/train_convex_imitation.py \
  --predictor-lock models/runs/locked_predictor_h5_20261019/predictor_lock.json \
  --imitation-config configs/convex_imitation.yaml
```

Mỗi run ghi thư mục mới:

- `teacher_pairs.npz`, `teacher_samples.csv`, `teacher_windows.json`: inputs/targets, split, dates và episode context.
- `imitation_learning_curves.csv`: fidelity train/inner, core/persistence controls và forecast sensitivity theo epoch.
- `action_mapping_floor.json`: reconstruction error của inverse map.
- `imitation.pth`: state/config/optimizer/RNG, predictor lock digest; role `convex_imitation_diagnostic_not_promoted`, `critic_trained=false`.
- `portfolio_metrics.csv`, `daily_returns.csv`: closed-loop teacher, imitation, matched core và persistence.
- `closed_loop_fidelity.csv`: teacher được query tại holdings của learner.
- `manifest.json`, `decision.json`: hashes, fit/evaluation scope và non-promotion status.

Run ban đầu `convex_imitation_20261019T104346Z_04a0e18c` được giữ nguyên. Sau khi thêm baseline persistence vào báo cáo, đã chạy lại đúng cùng hyperparameters/budget: max absolute difference của toàn bộ model weights giữa hai run là **0.0**. Không chọn/tune lại weights theo inner. Critic được đối chiếu với khởi tạo cùng seed và không thay đổi; checkpoint nạp được với `weights_only=True`.

Đã kiểm tra cú pháp, chạy training và đánh giá thực tế. Không thêm/chạy unit tests trong bước này. Không đánh giá outer hoặc test 2024–2025.

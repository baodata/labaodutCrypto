# Khóa predictor và so sánh các cơ chế phân bổ

Ngày triển khai: 19/10/2026.

Cập nhật learner: [patience độc lập promotion gate và deterministic train probes](SUA_PATIENCE_VA_TRAIN_PROBE.md). Kết quả thực nghiệm ở mục 8 giữ nguyên provenance của phiên bản cũ; các lượt mới có thêm `best_trained.pth` và `learning_curves.csv`.

## 1. Câu hỏi nghiên cứu

Với cùng dự báo lợi suất, covariance causal, trạng thái danh mục và chi phí đã biết ở Close[t], cơ chế phân bổ nào cho kết quả ròng tốt hơn?

```text
Prediction bank có OOF + inner + outer
              ↓ chọn family bằng inner IC trong từng fold
       predictor_lock.json + SHA256
              ↓ cùng forecast/risk/execution contract
   Convex → PPO → Sector MARL → H-MARL
              ↓
  So sánh paired theo ngày/fold và qua 5 seed
```

Không đưa raw feature/graph embedding vào riêng nhánh RL. Không fine-tune predictor trong vòng allocation. Các trainer PPO/MARL cũ vẫn dùng được; runner mới là một thí nghiệm allocation-only riêng.

## 2. Khóa predictor

`scripts/lock_predictor.py` chọn **family riêng từng fold**, horizon được khai báo trước là 5 phiên. Tiêu chí là trung bình Spearman IC từng ngày trên inner; hòa điểm thì theo thứ tự candidate trong lệnh. Forecast không đổi giữa các mã được tính 0 IC; ngày outcome không đổi bị loại chung cho tất cả candidate.

Không dùng outer để chọn và không gộp inner của các fold tương lai để quyết định predictor cho fold cũ. Tệp khóa lưu candidate scores, ngày chốt lựa chọn, model ID, hash bank manifest/labels và digest nội dung. Loader kiểm tra hash forecasts, fit metadata, neural weights, protocol và ticker order; sửa tệp khóa/bank sẽ bị chặn.

Khóa family và các fitted artifacts đã có trong bank, **không phải một bộ weights dùng cho mọi ngày**:

- `train_oof`: mỗi candidate có fit expanding/past-only đã được tạo trước.
- `inner`: predictor fit qua train_end.
- `outer`: predictor được refit qua inner_end theo cùng cấu hình.

Chọn family ở inner_end rồi train allocator trên historical OOF là **nested offline model selection**. Không tuyên bố family được chọn đã được biết ở các ngày OOF trong quá khứ. Runner không đọc labels để suy luận hay huấn luyện allocator; return thực tế chỉ xuất hiện sau mỗi bước mô phỏng.

Khóa thực tế: `models/runs/locked_predictor_h5_20261019/predictor_lock.json`.

| Fold outer | Predictor chọn từ inner | Mean inner IC |
|---|---|---:|
| 2019 | GRU | +0.0524 |
| 2020 | GNN-GRU | +0.0159 |
| 2021 | GRU | +0.1057 |
| 2022 | GNN-GRU | −0.0427 |

Fold 2022 đứng đầu trong các candidate nhưng vẫn IC âm. Việc xếp hạng không chứng minh tín hiệu dương hoặc có ý nghĩa thống kê; runner ghi riêng `positive_inner_ic_folds` và không tự tuyên bố vượt signal gate.

## 3. Bốn cơ chế phân bổ

| Nhánh | Cơ chế | Ngân sách ngành |
|---|---|---|
| Convex | Gọi nguyên `cost_aware_target()`: forecast − tracking risk − estimated turnover cost | Tự do trong cùng giới hạn từng mã |
| PPO | Một actor xuất score tất cả mã, một critic | Tự do trong cùng giới hạn từng mã |
| Sector MARL | Actor riêng từng agent group, critic toàn danh mục | Cố định theo số mã/ngành |
| H-MARL | Manager chọn budget ngành, worker chọn mã trong budget đó, critic toàn danh mục | Học mỗi ngày, giới hạn theo tổng cap/tilt của các mã |

Taxonomy lấy từ assets config: 5 sector kinh tế, 4 agent group; Energy và Consumer chung một group. Với 24 mã và cash 4%, core budgets lần lượt 24%, 20%, 20%, 32% NAV (6/5/5/8 mã), tổng phần cổ phiếu là 96%. Không chia đều 25% cho 4 group rồi gọi đó là asset-level 1/N.

Mọi actor nhận cùng global state: forecast thập phân, forecast chia scale từ fit quá khứ, full horizon covariance, weights/cash hiện tại của chính mình và benchmark đồng bộ, horizon, alpha, phí và core group prior. Worker được thêm budget do manager/fixed prior tạo ra. Đây là đối chứng cách tổ chức action, không phải thí nghiệm giới hạn thông tin giữa agent.

RL dùng Gaussian scores với std cố định 0.10, sau đó tanh và project vào bounded simplex. Log-prob PPO được tính trên **score đã lấy mẫu**, không tính lại trên trọng số đã project. Mean của actor head lúc khởi tạo bằng 0, nên deterministic epoch 0 trùng matched 1/N.

H-MARL dùng likelihood đúng của chuỗi quyết định:

`p(manager | state) × ∏ p(worker_g | state, sampled_budget)`.

PPO update dùng joint likelihood, advantage toàn đội và centralized critic. Khi evaluate minibatch, worker phải được condition bằng **manager action lưu trong rollout**. Mọi cấp nhận cùng daily net active reward; không gán reward ngành riêng hoặc counterfactual credit. Đây là cooperative joint PPO có hierarchy, không phải IPPO độc lập hay một thuật toán multi-timescale. Manager/worker đều quyết định hàng ngày để tránh sai discounted reward/GAE khi thời lượng action thay đổi.

## 4. Điều kiện so sánh chung

- Cash target 4%, action_alpha=0.25, khởi tạo 100% cash.
- Stock target nằm trong core 1/N ±25% và cap từ env config; mọi nhánh cùng projection constraints.
- Sector MARL có thêm ràng buộc budget ngành cố định; PPO/H-MARL được đổi budget. Đây là khác biệt kiến trúc cần đo, không phải chênh lệch cash/fee.
- Bounds áp dụng cho **policy target**. Initial cash ramp, blending và price drift có thể làm tỷ trọng thực tế khác bounds; không tuyên bố ràng buộc target là hard cap NAV ở mọi thời điểm.
- Quyết định Close[t] → khớp Open[t+1] → đánh dấu Close[t+1]. Cash-funded accounting, commission 0.10% + slippage 0.05%; impact/controller tắt như profile đối chứng predictor.
- Tái cân bằng hàng ngày. Tail h−1 phiên cuối phase bị purge forecast thì tất cả cùng true hold, không sinh giao dịch giả.
- Convex giữ nguyên objective và solver hiện tại. RL reward là `100 × (daily net return − synchronized matched daily net return)`, đã bao gồm phí thực; không thêm risk penalty/DSR. Vì vậy đây là đối chứng phương pháp quyết định với cùng thông tin/ràng buộc, không khẳng định chúng tối ưu cùng một hàm utility.
- Mỗi random window reset cả policy và benchmark về cash; bootstrap critic ở truncation nếu còn OOF forecast; cuối OOF phase là terminal.

## 5. Train, checkpoint và thống kê

Profile `configs/portfolio_decision.yaml`: 5 seeds 7/17/27/37/47; tối đa 20 epoch, 8 episode/update, 63–126 quyết định/episode, 5 PPO iterations, hidden32, minibatch256, CPU1thread, gamma0.95, lr0.0003, targetKL0.03. Fixed Gaussian std làm entropy không đổi; không áp dụng entropy bonus.

Allocator chỉ train trên selected candidate's `train_oof`. Inner vừa chọn predictor vừa chọn allocator checkpoint nên có selection optimism; các nhánh dùng cùng inner period và cùng budget. Chỉ chấp nhận checkpoint sau min_epochs=5 khi smoothed inner Sharpe vượt reference ít nhất 0.03 và Sharpe thực của checkpoint hơn best trước. Chỉ số checkpoint lưu là Sharpe thực, không gán rolling average cho weights một epoch.

Nếu chưa đạt gate, xuất epoch 0 với status **`untrained_reference`**. Kết quả đó phải đọc là baseline/không học vượt gate, không gọi là trained PPO có alpha. Không fallback âm thầm. All deterministic initial policies được yêu cầu trùng matched baseline trước train.

Tất cả train/selection trong một fold kết thúc trước khi runner đánh giá outer fold đó. Outer không đổi optimizer/checkpoint/config. Các phase vẫn nằm trong development 2019–2022 đã được khảo sát; không đọc test 2024–2025, không tạo thêm tính độc lập cho lịch sử đã xem.

So sánh gồm 8 contrasts: mỗi nhánh vs matched 1/N, PPO−convex, sector−PPO, H-MARL−sector và H-MARL−convex. Bootstrap paired daily net returns, block≥h và không nối block qua fold; Holm trên cả 8 contrasts. Năm seed được giữ riêng trong bảng chi tiết; bootstrap trên **seed-mean daily returns**, không xem cùng một phiên ở 5 seed là 5 ngày độc lập. CI là pointwise và conditional on fitted policies, chưa bao gồm toàn bộ model-selection uncertainty. Gate drawdown dùng degradation tệ nhất theo fold sau khi lấy mean qua seed.

## 6. Lệnh chạy

Khóa mới (không ghi đè khóa đã tồn tại):

```bash
python scripts/lock_predictor.py \
  --prediction-bank models/runs/signal_allocation_20261019T040033Z_ca30a67d/prediction_bank \
  --horizon 5 \
  --output models/runs/locked_predictor_h5_new/predictor_lock.json
```

So sánh toàn bộ 5 seed từ khóa có sẵn:

```bash
python scripts/compare_portfolio_decisions.py \
  --predictor-lock models/runs/locked_predictor_h5_20261019/predictor_lock.json \
  --allocation-config configs/signal_allocation_neural.yaml \
  --decision-config configs/portfolio_decision.yaml
```

Runner tạo thư mục run mới. `--output-dir` chỉ nhận thư mục mới/rỗng. Nếu di chuyển bank, dùng `--prediction-bank` trỏ snapshot đã sao chép đủ weights; hash phải khớp khóa. Chưa hỗ trợ resume runner này; không dùng `last.pth` với trainer PPO cũ vì state/action contract khác.

## 7. Artifacts để đọc kết quả

- `predictor_lock.json`, `manifest.json`, `resolved_execution.json`: khóa, provenance và cấu hình thực dùng.
- `policies/<fold>/<method>_seed<seed>/best.pth`, `last.pth`, `training_log.csv`: policy/optimizer/RNG/settings, predictor digest, epoch/status và metric inner.
- `metrics_by_fold.csv`: checkpoint status, Sharpe, ΔSharpe matched, return, MDD, phí, turnover, realized cash, constraint violation theo seed/fold.
- `daily_returns.csv`, `allocation_decisions.csv`: paired returns, target từng mã và budget từng group.
- `seed_fold_summary.csv`: mean/std qua seed trong từng fold; `seed_macro_fold_metrics.csv`, `seeds_summary.csv`: macro-average qua fold rồi mean/std qua seed.
- `paired_comparison.csv`: 8 contrasts với annual excess, CI, Holm p, drawdown degradation và gate.
- `decision.json`: số contrast vượt portfolio gate; luôn giữ `independent_alpha_claim=false` và yêu cầu review development results.

Không tự chọn “winner” để triển khai từ một bảng outer. Kiểm tra signal gate và tính ổn định qua seed/fold trước, sau đó mới khóa cấu hình cho holdout chưa dùng.

## 8. Kết quả thực nghiệm hoàn tất

Run: `models/runs/portfolio_decision_locked_h5_20261019/`.

- 4 fold × 3 nhánh RL × 5 seed = **60 lượt train**, mỗi lượt dừng sau 9 epoch theo patience; tổng 540 epoch. Predictor không được cập nhật.
- Đủ 136 hàng metrics (inner/outer; mỗi fold có 15 policy + convex + matched).
- **60/60 checkpoint được chọn là `untrained_reference`, epoch 0.** Các epoch đã train có thay đổi, nhưng không đạt gate ổn định +0.03 Sharpe inner.
- Max raw inner ΔSharpe quan sát qua từng loại: PPO **+0.001897**, sector MARL **+0.001343**, H-MARL **+0.002908**. Đây là max chọn trong nhật ký, không phải ước lượng alpha hoặc điểm của checkpoint được chấp nhận.
- Max policy-target bound violation: **0**.
- Đối chiếu 16 case convex/matched với comparison predictor trước: matched khớp hoàn toàn; max sai khác convex ≈5.52e−11 Sharpe và 1.24e−9 đơn vị tiền phí, chỉ ở mức sai số số học solver.

### Sharpe outer

| Fold | Matched 1/N | Convex | PPO / Sector MARL / H-MARL được chọn |
|---|---:|---:|---:|
| 2019 | 1.9973 | 2.0064 | 1.9973 |
| 2020 | 0.8756 | 1.0210 | 0.8756 |
| 2021 | 2.6256 | 2.3640 | 2.6256 |
| 2022 | −0.5107 | −0.4501 | −0.5107 |

Các nhánh RL trùng baseline vì **weights được chọn là khởi tạo core**, không phải bằng chứng rằng ba mô hình đã học được cùng một chiến lược hoặc ổn định qua seed. Không thể dùng bảng này để tuyên bố convex thắng mọi trained RL controller; chưa có trained checkpoint đạt điều kiện để thay reference.

### Convex so với matched baseline

- Annualized mean daily net excess: **+1.4508%**, CI95% **[−0.2217%, +3.0627%]**.
- Mean ΔSharpe qua fold: **−0.011614**; return tốt hơn trong **3/4 fold**.
- p một phía: **0.04048**; Holm p trên 8 contrasts: **0.32384**. CI95% là hai phía nên có thể chứa 0 dù p một phía dưới 0.05.
- **Không vượt portfolio gate:** annual excess dưới 2%, CI chứa 0, adjusted p lớn hơn 0.05 và mean ΔSharpe âm.

Không nhánh nào vượt gate. Trong profile/ngân sách này chưa có cơ sở nâng cấp từ benchmark sang H-MARL, cũng chưa có bằng chứng alpha độc lập. Không đọc test 2024–2025; các outer development này đã được khảo sát trước đó.

### Báo cáo ổn định số học

Report gốc có thể đếm chênh lệch float64 cỡ 1e−16 giữa các core targets là “fold dương”. Đã sửa summarizer để dùng tolerance **1e−12 đơn vị return** cho các chuỗi tương đương số học và dấu cumulative-return delta. Lựa chọn predictor, policy, checkpoint và cấu hình train giữ nguyên. Các lượt chạy mới dùng summarizer đã sửa.

Run gốc được giữ nguyên. Report dẫn xuất để đọc là:

`models/runs/portfolio_decision_locked_h5_20261019/numeric_report/paired_comparison.csv`.

Report này ghi nhận tất cả RL−matched bằng 0, positive_folds=0, p=1; thống kê convex giữ nguyên. `numeric_report/manifest.json` lưu hashes inputs/source của phép tổng hợp riêng, không thay provenance của lượt train gốc.

Tái tạo báo cáo vào thư mục mới:

```bash
python scripts/summarize_portfolio_decisions.py \
  --run-dir models/runs/portfolio_decision_locked_h5_20261019 \
  --output-dir models/runs/portfolio_decision_locked_h5_20261019/numeric_report_new
```

Đã kiểm tra cú pháp các module mới và chạy toàn bộ thí nghiệm ở trên. Không thêm/chạy unit tests trong lần triển khai này.

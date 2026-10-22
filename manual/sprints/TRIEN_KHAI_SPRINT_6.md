# Sprint 6 — bàn giao mã nguồn A và tích hợp covariance cho B

Ngày 18/10/2026. **Trạng thái: đã triển khai code; chưa nghiệm thu thực nghiệm.** Không chạy unit test, training dài hoặc ablation trong lượt triển khai này. Gate Sprint 5 vẫn **NO-GO**; tài liệu này không đưa ra kết luận có alpha.

QNT-006 thuộc **A**. Phần B nhận bàn giao nằm ở **QNT-007**; đường residual của **QNT-008** cũng được bổ sung để API nhất quán. Các checkbox DoD đòi bằng chứng chạy thực tế vẫn để mở.

## Chuẩn bị dependency — cập nhật 19/10/2026

Các lệnh Sprint 6 dùng `risk.method: ledoit_wolf` cần package **scikit-learn**, tên import Python là `sklearn`. Package đã khai báo trong `requirements.txt`, nhưng virtualenv tạo trước đó có thể chưa cài đủ. Cài vào đúng interpreter đang chạy dự án:

```bash
.venv/bin/python -m pip install 'scikit-learn>=1.3.0'
```

Đã cài thành công scikit-learn **1.9.1** vào `.venv` của workspace ngày 19/10/2026. Chưa chạy lại training/ablation sau khi cài.

Sau đó chạy lại lệnh đã lỗi. Runner tự tạo thư mục mới nếu không chỉ định `--run-dir`; không dùng lại thư mục run thất bại với một lệnh train mới. Các run cũ không bị xóa.

Code đã chuyển sklearn sang import khi thực sự chọn Ledoit–Wolf và kiểm tra dependency trước khi tạo run/tính feature. Sample/EWMA và PPO không dùng bank không đòi import sklearn; không tự đổi estimator khi thiếu dependency. Script báo cáo bank luôn so sánh cả ba estimator nên vẫn cần scikit-learn. Manifest ghi thêm phiên bản scikit-learn để kiểm tra resume.

## Bản sửa lỗi xác nhận — 19/10/2026

- Accounting weights/projection/execution diagnostics/PnL credit dùng float64 nội bộ. Observation policy vẫn float32. Checkpoint/manifest ghi `accounting_precision`.
- Evaluation mặc định validation cả YAML lẫn fallback; test cần chọn tường minh. Chấp nhận validation cutoff rơi vào cuối tuần/ngày nghỉ trong tolerance 7 ngày, không tự thay split trong validation-only.
- Checkpoint/manifest lưu `resolved_split_dates`; test metrics lưu actual evaluation dates và trạng thái reused/freshness. Cảnh báo dựa vào ngày thực tế trong test tensor; test bắt đầu 2023 nhưng chứa 2024–2025 vẫn được ghi exploratory.
- Helper GAT entropy trung bình theo node đích/head, kiểm tra probability mass và hỗ trợ chuẩn hóa log(degree). PPO vẫn dùng entropy Dirichlet; không tự bật attention regularizer.
- Bản sửa không tạo ra dữ liệu point-in-time, alpha, high-level manager hoặc một final holdout mới. Các hạng mục đó vẫn cần bằng chứng/nghiên cứu riêng.
- Source/config hash thay đổi: giữ run cũ làm artifact phiên bản cũ; chạy bản sửa trong run-dir mới, không gộp seed khác phiên bản.

## 1. Phạm vi code

| Ticket | Module và cơ chế | Bằng chứng còn cần chạy |
|---|---|---|
| QNT-004 | `features/normalization.py`: static train, identity, XS zscore, XS rank, Gaussian rank; cấu hình từng feature, ties trung bình, N=1, universe mask và missing. Rank/RSI giữ nguyên mặc định. | Batch/stream parity, trường hợp biên và ablation XS |
| QNT-005 | Rolling theo mã/feature, chỉ dùng **[t-window:t)**; lưu raw history, validity, normalized history, ngày cuối, readiness và calendar. PPO/null/inference/resume cùng serialization v2. | Future-perturbation, parity, resume và rolling/hybrid nhiều fold |
| QNT-006 | `risk/covariance.py`: sample, EWMA có trọng số chuẩn hóa trong cửa sổ, **linear** Ledoit–Wolf qua sklearn; daily covariance, PSD/jitter/shrinkage/condition/readiness; cache theo data/config/implementation/library hash. | PSD/N=1/missing và đánh giá forecast ngoài train |
| QNT-007 (B) | `graph/dynamic_graph.py` nhận covariance bank; covariance → correlation → cùng matrix API ba quan hệ; kNN vẫn theo abs(corr). | Ma trận vs batch, edge churn/connectivity và Pearson vs shrinkage |
| QNT-008 (B) | Regression alpha/beta SPY fit trên **[t-window:t)**, tính residual t sau khi biết Close[t], rồi dựng **bank riêng** cho graph. Thiếu benchmark/Adj Close báo lỗi. | Causality, market exposure và residual vs raw graph |
| QNT-011 | `features/liquidity.py`: trailing raw-dollar ADV và daily total-return sigma; readiness từng mã, zero volume không đủ điều kiện giao dịch. | Xác nhận basis giá/volume của nguồn dữ liệu và split events |
| QNT-012 | Impact tiền = Σ Qᵢ η σᵢ √(Qᵢ/ADVᵢ); commission/slippage/impact tách riêng; Q là chân cổ phiếu thực tế sau giới hạn tiền mặt. | Cost/NAV reconciliation, eta=0 parity và stress NAV/eta |
| QNT-013 | `ExecutionController`: chọn tốc độ, no-trade mask và dollar cap từ Close; Open chỉ drift và thực thi giới hạn đã chốt. Phí trả từ cash, giảm buy nếu thiếu tiền; hold giữ lượng cổ phiếu. | Hold/cap/cash invariants và so sánh cùng controller |
| QNT-014 | DSR với A/B **trước cập nhật**, warm-up, variance floor, clipping; reference risk-free hoặc equal-weight active; beta phải bằng 1. | Constant return/degenerate variance, DSR state và ablation reward |
| MARL-001..004 | Sector mapping từ assets.yaml, obs local + observed/target budget, action local, một lần execution toàn cục; reward shared hoặc sector-active attribution. | Mapping, pooled cash/credit và joint execution reconciliation |
| MARL-002/003 (A), MARL-005 (B) | `training/train_marl.py`: independent PPO cho từng sector với critic/buffer/optimizer/scheduler riêng; checkpoint best/last, epoch-boundary resume, learning gate và validation-only. | Huấn luyện đầy đủ nhiều seed/fold |
| MARL-006 | Baseline sector khớp ngân sách/cash/alpha/controller; thêm asset-level matched 1/N, daily returns và bootstrap. Runner có nhánh GRU control, GNN control và sector MARL. | Bảng so sánh thực tế cùng giai đoạn, seed và chi phí |

## 2. Cấu hình và tính tương thích

- `configs/quant_sprint6.yaml` chỉ có tác dụng khi truyền **`--quant-config`**. Không truyền thì các cơ chế định lượng mới vẫn tắt; mặc định evaluation hiện là validation và accounting nội bộ đã sửa sang float64.
- File mẫu bật việc dựng risk/liquidity bank, nhưng **tắt normalization, adaptive execution, impact và DSR**; graph dùng `legacy`. Có bank không có nghĩa là đã bật overlay/allocator tối ưu.
- Chuẩn hóa XS giữ hai context channel cho mỗi feature bị rank/zscore: mean cắt ngang chuẩn hóa bằng train statistics và std cắt ngang chia train std. Context đi vào actor/critic, không làm thay đổi graph features. DSR/market context làm tăng observation dimension và được lưu trong `model_config.portfolio_context_dim`.
- Checkpoint cũ vẫn dùng `portfolio_context_dim=0` và scaler v1. Resume một run cũ sau khi source thay đổi tiếp tục bị provenance guard chặn theo thiết kế.
- `--normalization-warmup 252` loại cùng số phiên đầu khỏi **mọi** nhánh ablation, kể cả control. Nếu chỉ chạy rolling đơn lẻ, pipeline tự loại rolling warm-up + lookback. Tránh so các scaler trên các tập train khác ngày.
- Normalization fit train một lần; full sequence transform chỉ tính causal statistics, sau đó slice partitions. State nhúng cho inference dừng ở **train end**, không chứa raw history của inner/outer tương lai.
- Lịch phiên trong state chỉ là lịch ngày, không chứa giá/feature tương lai. Inference phải feed đủ mọi phiên mới, không nhảy từ train end sang outer. Mở rộng lịch bằng `set_session_calendar()` trước khi đi qua ngày cuối calendar.
- Raw inference dùng `history_dates` và `history_mask`; overlapping history lấy lại normalized cache, không update rolling state hai lần. `normalization_state_dict()`/`load_normalization_state_dict()` hỗ trợ lưu state đang chạy. Có thể truyền `features_scaled=True` nếu upstream đã chuẩn hóa đúng contract.
- Với DSR, observation thêm `[A, B, count]`. Episode reset moments về 0; random-window bootstrap dùng observation có moments hiện tại. Best/last lưu moments của episode cuối để chẩn đoán; resume tại ranh giới epoch bắt đầu episode mới và reset moments đúng như run liên tục. `TradingEnv.reward_state_dict()`/`load_reward_state_dict()` hỗ trợ host lưu/khôi phục reward state; đây **không** phải snapshot toàn bộ portfolio giữa episode.
- Legacy snapshot encoder `--encoder gat/gcn` chưa nhận context mới; khi có DSR/XS context sẽ báo lỗi rõ. Dùng `--encoder gnn --gnn-architecture gat/gcn` cho graph có GRU.

## 3. Basis và execution

- Risk: daily **simple total-return** close-to-close; covariance không annualize. Muốn annual volatility thì caller lấy `sqrt(252 × w'Σw)` đúng một lần.
- ADV: **Close chưa điều chỉnh cổ tức × Volume cùng split basis**; Yahoo split-adjusted Close/Volume được ghi rõ trong metadata. Không dùng `Adj Close × raw Volume` làm dollar ADV. Nguồn khác phải xác nhận lại basis; không tự suy diễn split data bị sai là hợp lệ.
- Liquidity/risk tại Close[t] chỉ dùng lịch sử đến t. Actual dollar Q được biết ở Open[t+1] để tính phí; volume ngày t+1 không tham gia quyết định.
- Đường legacy giữ nguyên khi **controller off, eta=0, accounting=legacy**. Khi bật controller hoặc eta>0, cash-funded execution tính chân giao dịch thực tế và trừ phí từ cash; cash thiếu thì giảm buy. Vì vậy có nhánh `cash_accounting_control` (eta=0/controller off nhưng accounting=cash_funded) để tách tác động accounting. So impact và adaptive với nhánh này; so nhánh này với legacy control. Stress grid dùng cash-funded trong mọi arm.
- Participation cap cố định theo ADV/Close NAV; Open chỉ dùng nó như giới hạn dollar. Band/hold giữ số lượng, không đưa trọng số Close cũ thành một lệnh cân lại. Vị thế tự drift vượt cap khi hold không bị cưỡng bức bán.
- Cash leg không có impact. Eta=0 không tính impact. Missing liquidity active trade gây lỗi ở impact-only; adaptive controller giữ tài sản chưa ready.

## 4. Sector MARL

- Universe vẫn 24 cổ phiếu, SPY không giao dịch. **Bốn agent groups hiện có** từ assets.yaml: technology, financials, healthcare, energy_consumer; energy/consumer là hai ngành trong một agent group.
- Fixed equal sector budgets; chưa có manager học budget của Sprint 7. Agent quan sát local holdings + pooled cash quy thuộc đều và cả observed budget/target budget.
- `shared`: tất cả nhận reward ròng toàn danh mục, theo cooperative independent PPO. `sector_active`: credit PnL cổ phiếu của nhóm + phần cash PnL chia đều, chuẩn hóa theo target budget, trừ equal-weight return của nhóm. Đây là **credit attribution**, không giả định có sub-account cash độc lập.
- Không bật sector-active cùng DSR. Default mỗi policy có mean cash giống matched baseline bằng cách khởi tạo actor head zero và prior theo fixed kappa; không dùng mean cash nhìn lại từ PPO để chọn baseline.
- Sector trainer hỗ trợ **GRU/LSTM/MLP**, GNN đa quan hệ vẫn chạy riêng ở single-agent control để ablate graph trước. `SectorCheckpointPolicy` normalize toàn universe trước, slice group sau và combine theo đúng budget như training.
- Resume bằng `--resume .../checkpoints/last.pth` với cùng CLI/config/horizon; chưa có resume giữa rollout. Sau interrupt có thể tiếp tục từ epoch đã hoàn thành cuối cùng.

## 5. Lệnh triển khai thực nghiệm

Chạy từ root, đã activate `.venv`. Mỗi run dùng thư mục mới. Các lệnh dưới **chưa được chạy trong lượt này**.

### Bàn giao bank + stress execution (validation, không train)

```bash
python scripts/evaluate_sprint6_banks.py \
  --quant-config configs/quant_sprint6.yaml \
  --research-protocol configs/quant_research_protocol.yaml \
  --protocol-fold development_2022 \
  --evaluation-scope validation
```

Sản phẩm: NPZ bank có dates/tickers/readiness; covariance diagnostics; variance forecast vs outcome t+1; graph counts/churn/components; grid eta `[0, .1, .3]` × NAV `[100k, 10m]` × controller on/off. Đây là stress grid cố định, không chọn cấu hình theo kết quả tốt nhất.

### Control và ablation theo từng nhóm

```bash
# Graph có GRU so với GRU thuần, cùng feature/seed/execution
python scripts/run_sprint6_ablation.py --variants gru_control,control

# Chuẩn hóa: control và các cơ chế mới, cùng warm-up/train dates
python scripts/run_sprint6_ablation.py --variants control,xs,rolling,hybrid

# B nhận covariance: giữ nguyên model và chỉ thay graph source
python scripts/run_sprint6_ablation.py \
  --variants control,graph_sample,graph_ewma,graph_ledoit_wolf,graph_residual

# Execution/reward độc lập; không gộp tất cả thay đổi vào một nhánh
python scripts/run_sprint6_ablation.py \
  --variants control,cash_accounting_control,impact,adaptive_execution,dsr

# So sánh nhóm ngành, kèm GRU một agent làm control
python scripts/run_sprint6_ablation.py \
  --variants gru_control,sector_marl,sector_active_reward
```

Runner mặc định 5 seed `7,17,27,37,47`, fold `development_2022`, validation-only, không chạy historical test. Các outer development đã quan sát vẫn là exploratory; không chọn variant tốt nhất từ outer rồi coi đó là holdout mới. Muốn chạy fold khác chỉ định `--protocol-fold` trước khi chạy, giữ nguyên gate.

Artifact tổng hợp: `ablation_manifest.json`, YAML riêng từng nhánh, `validation_comparison.csv`, `validation_comparison_mean_std.csv`. Delta của MARL có hai cột: vs **sector matched** và vs **asset matched**, để phân biệt hiệu quả phân nhóm với học policy. Không cộng/trung bình p-value qua seed để tuyên bố ý nghĩa thống kê.

### Train sector riêng

```bash
python -m src.training.train_marl \
  --encoder gru --seed 7 \
  --quant-config configs/quant_sprint6.yaml \
  --research-protocol configs/quant_research_protocol.yaml \
  --protocol-fold development_2022 \
  --evaluation-scope validation
```

## 6. Điều kiện đóng ticket

**Cập nhật 19/10/2026:** đã chạy kiểm chứng kỹ thuật và artifact bank/stress, xem [audit](KIEM_CHUNG_NHAN_XET_2026_10_19.md) và [bản sửa](SUA_LOI_KIEM_CHUNG_2026_10_19.md). Nội dung dưới mô tả trạng thái tại lúc bàn giao 18/10.

Code đã nối thành luồng chạy và đã đọc lại các giao diện; kiểm tra cú pháp Python bằng AST đã thực hiện. **Chưa chạy unit/integration tests hoặc training/ablation**, nên chưa có bằng chứng runtime cho DoD causality, parity, PSD, accounting hay resume. Chỉ đóng checkbox sau khi ghi artifact thực nghiệm, chốt khác biệt mong đợi và reviewer kiểm tra. Không đổi default graph/reward/execution hoặc tuyên bố có alpha chỉ vì module mới đã có.

Nguồn thuật toán: [sklearn linear Ledoit–Wolf](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ledoit_wolf.html), [Moody–Saffell: Learning to Trade via Direct Reinforcement](https://people.idsia.ch/~juergen/rnnaissance2003talks/MoodySaffellTNN01.pdf). DSR và impact là mô hình nghiên cứu cần ablation/calibration, không phải cam kết hiệu quả đầu tư.

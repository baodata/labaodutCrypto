# H-MARL-GNN: KẾ HOẠCH HÀNH ĐỘNG RIÊNG CHO THÀNH VIÊN A

### VAI TRÒ: QUANT AI & MACRO ALLOCATOR LEAD

_(DATA + ENVIRONMENT + PPO BASELINE + HIGH-LEVEL MACRO AGENT + EVALUATION)_

> **Mô tả vai trò:**
> Bạn chịu trách nhiệm về toàn bộ chuỗi giá trị cốt lõi: **Data $\rightarrow$ Environment $\rightarrow$ Single-Agent PPO $\rightarrow$ High-Level Macro Agent $\rightarrow$ Backtest**.
>
> Bạn không phải là người chuẩn bị dữ liệu đơn thuần, mà là một **Quant AI Engineer** thực thụ: nắm vững cách thị trường vận hành, mô phỏng môi trường giao dịch thực tế, xây dựng cơ chế phần thưởng (Reward Formulation), làm chủ thuật toán PPO và huấn luyện Agent vĩ mô phân bổ vốn vào các nhóm ngành. Khi bảo vệ đồ án và phỏng vấn, bạn có câu chuyện AI trọn vẹn và độc lập về **Financial RL & Hierarchical Decision Systems**.
>
> ⚖️ **Phân định đóng góp khoa học (Scientific Contributions):**
>
> - **Thành viên A** chịu trách nhiệm chính cho **Financial RL & Hierarchical Allocation Contribution** (Trading Environment, Reward Formulation, PPO Baseline, High-Level Macro Policy, Backtesting Framework).
> - **Thành viên B** chịu trách nhiệm chính cho **Graph-learning Contribution** (Dynamic Financial Graph, GCN/GAT, Low-Level Sector Agents, Graph Communication).
> - Cả hai thành viên **đồng sở hữu Overall Scientific Contribution** của đề tài.
>
> 📖 **Quy chế 70/30 & Lộ trình 5 Level học AI:** Xem chi tiết tại [QUY_CHE_DONG_BO_VA_HOC_AI_70_30.md](file:///home/tuan/AI/AI_roject1/manual/QUY_CHE_DONG_BO_VA_HOC_AI_70_30.md).
> 🤝 **Bảng điều phối chung với B:** Xem tại [BANG_DIEU_PHOI_2_NGUOI.md](file:///home/tuan/AI/AI_roject1/manual/BANG_DIEU_PHOI_2_NGUOI.md).
> **Ticket định lượng đã tích hợp vào Sprint 5–9:** [đặc tả và bảng ánh xạ](TICKET_CAI_THIEN_DINH_LUONG.md). Cập nhật 16/10/2026.

---

## 1. Các thư mục do Thành viên A chịu trách nhiệm trực tiếp

**Phân công module PPO từ 18/10/2026:** A chủ trì `src/training/data_preparation.py`, `src/training/env_factory.py`, `src/evaluation/policy_evaluator.py` và `src/evaluation/ppo_evaluation.py`; B review. Các ticket QNT về data/env/metrics ưu tiên sửa đúng module này. [Chi tiết kiến trúc và điểm bàn giao](KIEN_TRUC_MODULE_PPO.md).

```text
src/
├── data/           # Mã nguồn tải, tiền xử lý và kiểm định dữ liệu
├── features/       # Mã nguồn tính toán chỉ báo kỹ thuật (Return, RSI, MACD...)
├── env/            # Mã nguồn mô phỏng danh mục, phí giao dịch, Gymnasium Env
├── evaluation/     # Mã nguồn Backtest, tính toán Sharpe, MDD, Turnover
│
├── agents/
│   ├── high_level.py   # Agent phân bổ vốn vĩ mô (Sector & Cash Allocation)
│   └── combiner.py     # Bộ ghép nối tỷ trọng phân cấp
│
└── training/
    └── single_agent_ppo.py  # Đồng sở hữu huấn luyện PPO Baseline cùng B
```

---

## 2. Phân tầng Scope cá nhân & 3 Mốc Dừng

- **🎯 MỐC A (Đủ hoàn thành / Bảo vệ tốt):** Hoàn thành toàn bộ các ticket đánh dấu `[CORE]` từ Sprint 1 đến Sprint 10 (Data Pipeline, Features, Trading Env, Backtesting, Single-Agent PPO, High-Level Macro Policy, Combiner). Đủ điều kiện hoàn thành và bảo vệ tốt đồ án tốt nghiệp.
- **🏆 MỐC B (Mục tiêu Xuất sắc):** Hoàn thành thêm các phân tích chuyên sâu `[MỐC B]` (Phân tích độ nhạy phí giao dịch `EXP-009`, Phân tích 3 chế độ thị trường `EXP-010`, chạy 5 seeds kiểm chứng thống kê).
- **🚀 MỐC C (Research Extension — Mở rộng nghiên cứu Paper):** Hỗ trợ B tích hợp các quan hệ phi giá trên đồ thị tri thức (chuỗi cung ứng, sở hữu tổ chức) nếu còn thời gian trước hạn chót.

---

## 3. Checklist công việc chi tiết theo từng Sprint

> **Tích hợp định lượng — cập nhật 16/10/2026:** Ticket `QNT-*` có checklist trực tiếp trong Sprint 5–9 bên dưới. Checklist ở file owner là nguồn trạng thái; [đặc tả định lượng](TICKET_CAI_THIEN_DINH_LUONG.md) giữ công thức và phụ thuộc.
> - Sprint 1–4/4.5 giữ phần nền đã ghi nhận; nâng cấp mới xếp từ Sprint 5 để contract/protocol có trước các task sử dụng chúng.
> - Trong cùng sprint, làm theo phụ thuộc và bàn giao A ↔ B. P2 là `[MỐC C — CÓ ĐIỀU KIỆN]`, không chặn nghiệm thu H-MARL core.
> - Trạng thái theo dõi hiện có được giữ lại khi chuyển mục. “Đã có core runner” là tiến độ một phần QNT-025, chưa phải nghiệm thu toàn ticket.

### 🏁 SPRINT 1: Thiết lập Vũ trụ Cổ phiếu & Tải dữ liệu thô

_Mục tiêu: Có dữ liệu OHLCV đầy đủ của 20–30 mã cổ phiếu lưu vào `data/raw/`._

- [x] **[SETUP-001] Khởi tạo Repository & Skeleton nền tảng (P0 - Chung A+B)**
  - _Đã hoàn thành ở Epic 0:_ Tạo đầy đủ cây thư mục, `.gitignore`, `requirements.txt`, `pyproject.toml`, test nền tảng 4/4 pass.

- [x] **[DATA-001] Chọn vũ trụ cổ phiếu (P0)**
  - _Tệp cần tạo:_ `configs/assets.yaml` (Đã hoàn thành: 24 cổ phiếu thuộc 5 nhóm ngành)
  - _Nội dung:_ Chọn 20–30 cổ phiếu Mỹ có thanh khoản cao thuộc ít nhất 4 nhóm ngành (Công nghệ: AAPL, MSFT, NVDA, AMD; Tài chính: JPM, BAC, GS, MS; Y tế: JNJ, PFE, UNH; Tiêu dùng/Năng lượng: AMZN, XOM, CVX...).
  - _Tiêu chuẩn nghiệm thu:_ File yaml hợp lệ, không hard-code danh sách mã trong mã nguồn Python. 100% tests passed.
  - _Người review:_ Thành viên B.

- [x] **[DATA-002] Viết module OHLCV Downloader (P0)**
  - _Tệp đã tạo:_ `src/data/downloader.py`, `scripts/download_data.py`, `tests/test_downloader.py`
  - _Nội dung:_ Tải tự động dữ liệu 2015–2025 từ Yahoo Finance. Lưu từng mã thành file định dạng `.parquet` trong `data/raw/`. Có cơ chế retry (max 3 lần), chuẩn hóa cột thống nhất (`date`, `open`, `high`, `low`, `close`, `adj_close`, `volume`), khử trùng lặp ngày, chuyển timezone về tz-naive.
  - _Kết quả thực tế:_ Tải thành công 25/25 mã (24 cổ phiếu + SPY benchmark), mỗi mã đạt chính xác 2.765 phiên giao dịch (100% không rò rỉ, không trùng lặp). Đã vượt qua 13/13 unit tests.
  - _Người review:_ Thành viên B.

- [x] **[CONTRACT-001] Thống nhất giao diện dữ liệu linh hoạt với Thành viên B (P0 - Chung)** _(đã chỉnh)_
  - _Tệp đã tạo:_ `src/utils/contracts.py`, `tests/test_contracts.py`
  - _Đã chốt định dạng đầu ra cho B:_ `MarketDataTensor` kích thước `[T, N, F]` (thời gian $\times$ tài sản $\times$ đặc trưng kỹ thuật).
  - _Quy ước:_ **$N = 24$ tài sản giao dịch.** SPY chỉ là benchmark đánh giá hiệu suất, **không được đưa vào** tensor `[T, N, F]` hay action `[N+1]`.
  - _Đã chốt cấu trúc đồ thị tương thích với B (`builder.py`):_ `DynamicGraphData` gồm `node_features` `[N, F]`, `edge_index` `[2, E]`, `edge_weight` `[E]`.
  - _Đã chốt định dạng Observation của Môi trường hỗ trợ 2 chế độ (`MarketObservation`):_
    - Chế độ 1: `raw_features` `[N, F]` là feature đã tính từ OHLCV (return, RSI, MACD...), cho Single-Agent PPO không cần GNN; không phải OHLCV thô.
    - Chế độ 2: `graph_embeddings` `[N, D]` (cho mô hình GNN/H-MARL).
  - _Đã chốt định dạng Action (`PortfolioAction`):_ Trọng số tài sản $w_i \ge 0, w_{cash} \ge 0$ thỏa mãn $\sum w_i + w_{cash} = 1.0$.
  - 100% unit tests kiểm tra hợp đồng pass (`tests/test_contracts.py`).

---

### 🏁 SPRINT 2: Kiểm định, Đồng bộ ngày & Tính đặc trưng ban đầu

_Mục tiêu: Dữ liệu sạch sẽ, không có NaN/giá âm và đã tính xong Return, Volatility, RSI._

- [x] **[DATA-003] Viết bộ kiểm định dữ liệu Raw Data Validator (P0)**
  - _Tệp đã tạo:_ `src/data/validator.py`, `tests/test_validator.py`
  - _Nội dung:_ Kiểm tra tự động 7 luật: Non-empty, Required OHLCV, Timeline sorted/unique, Giá > 0, Volume $\ge$ 0, Logic nến $High \ge Low / Open / Close$, Không chứa NaN/Inf.
  - _Tiêu chuẩn nghiệm thu:_ Đã kiểm định tự động toàn bộ 25 file Parquet thật trong `data/raw/` đều hợp lệ 100%. Đạt 12/12 unit tests pass.

- [x] **[DATA-004] Đồng bộ ngày giao dịch (Trading Calendar Alignment) (P0)**
  - _Tệp đã tạo:_ `src/data/alignment.py`, `tests/test_alignment.py`
  - _Nội dung:_ Hỗ trợ 2 cơ chế: `intersection` (giữ ngày giao dịch chung) và `union_ffill` (điền giá hôm trước chống look-ahead bias). Hỗ trợ xuất bảng giá đa tài sản `to_price_panel()` `[T, N]`.
  - _Tiêu chuẩn nghiệm thu:_ Đã đồng bộ 25 mã cổ phiếu thật về chung 1 lịch giao dịch gồm 2,765 ngày, 6/6 unit tests pass.

- [x] **[DATA-005] Lưu siêu dữ liệu nguồn (Provenance Metadata) (P1)**
  - _Tệp đã tạo:_ `data/interim/dataset_metadata.json`
  - _Nội dung:_ Lưu lại thời gian tạo, phiên bản dữ liệu, số mã (25), số ngày (2,765), phân loại theo 5 sector, benchmark SPY, phương pháp căn chỉnh, và tổng số 69,125 dòng phục vụ báo cáo khoa học.

- [x] **[FEAT-001] Tính toán Daily Return (P0)**
  - _Tệp đã tạo:_ `src/features/returns.py`, `tests/test_returns.py`
  - _Công thức:_ Simple Return $R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$ và Log Return $r_t = \ln(P_t / P_{t-1})$. Hỗ trợ xuất ma trận $[T, N]$.
  - _Tiêu chuẩn nghiệm thu:_ 5/5 unit tests pass, sai số số học $< 10^{-8}$.

- [x] **[FEAT-002] Tính toán Rolling Volatility (P0)**
  - _Tệp đã tạo:_ `src/features/volatility.py`, `tests/test_volatility.py`
  - _Công thức:_ Độ lệch chuẩn của return trong cửa sổ trượt 20 ngày (`rolling(20).std()`), hỗ trợ niên độ hóa $\sigma_{\text{ann}} = \sigma_{\text{daily}} \times \sqrt{252}$ và `min_periods=1` chống NaN.
  - _Tiêu chuẩn nghiệm thu:_ 5/5 unit tests pass, hỗ trợ xuất ma trận $[T, N]$.
- [x] **[FEAT-003] Tính toán RSI-14 (P1)**
  - _Tệp đã tạo:_ `src/features/rsi.py`, `tests/test_rsi.py`
  - _Nội dung:_ Cài đặt chuẩn xác công thức Wilder RSI chu kỳ 14 ngày (EMA với $\alpha = 1/14$). Hỗ trợ scale $[0, 1]$ cho RL.
  - _Tiêu chuẩn nghiệm thu:_ 6/6 unit tests pass (bull run $\ge 99.9$, bear run $\le 0.1$, flat $= 50.0$, và test trực tiếp trên 25 mã thật).

> 🤝 **Bàn giao:** Chuyển dữ liệu Return đã làm sạch cho Thành viên B để B bắt đầu thử nghiệm dựng ma trận tương quan đồ thị.

---

### 🏁 SPRINT 3: Hoàn thiện Feature Pipeline & Phân chia tập an toàn

_Mục tiêu: Đóng gói toàn bộ đặc trưng vào tensor và chia Train/Val/Test tuyệt đối không rò rỉ dữ liệu._

- [x] **[FEAT-004] Tính toán MACD (P1)**
  - _Tệp đã tạo:_ `src/features/macd.py`, `tests/test_macd.py`
  - _Nội dung:_ Tính đường MACD (EMA12 - EMA26), đường Signal (EMA9 của MACD) và Histogram. 6/6 unit tests pass.

- [x] **[FEAT-005] Chuẩn hóa Volume (P1)**
  - _Tệp đã tạo:_ `src/features/volume.py`, `tests/test_volume.py`
  - _Nội dung:_ Áp dụng $\log(1 + \text{Volume})$ và Rolling Z-score ($W=20$). Tuyệt đối không dùng thông tin tương lai. 7/7 unit tests pass.

- [x] **[FEAT-006] Đóng gói toàn bộ Feature Pipeline (P0)**
  - _Tệp đã tạo:_ `src/features/pipeline.py`, `tests/test_pipeline.py`
  - _Nội dung:_ Gộp 20 đặc trưng kỹ thuật, tương đối cắt ngang, động lượng và rủi ro vào `MarketDataTensor` $[T, N, F]$. SPY chỉ làm chuỗi tham chiếu cho excess return/beta, không nằm trong action space.

- [x] **[SPLIT-001] Phân chia Train / Validation / Test theo thời gian (P0)**
  - _Tệp đã tạo:_ `src/data/split.py`, `tests/test_split.py`
  - _Phân chia:_ Train (2015–2021), Validation (2022–2023), Test (2024–2025).
  - _Tiêu chuẩn nghiệm thu:_ Đảm bảo tuyệt đối `max(train_date) < min(val_date) < min(test_date)`. Hỗ trợ cắt `MarketDataTensor`. 3/3 unit tests pass.

- [x] **[SPLIT-002] Chuẩn hóa chỉ học trên tập Train (Train-only Normalization) (P0)**
  - _Tệp đã tạo:_ `src/features/scaler.py`, `tests/test_scaler.py`
  - _Nội dung:_ Tính $\mu$ và $\sigma$ duy nhất từ tập Train, áp dụng đóng băng biến đổi cho Validation và Test. Lưu trữ tham số tại `data/processed/scaler_params.json`. 4/4 unit tests pass.

---

- [x] **[LEAK-002] Kiểm thử rò rỉ dữ liệu mở rộng (P0)** _(mới)_
  - _Nội dung:_ Test rò rỉ dữ liệu cho MACD, volume_norm, scaler (train-only) và split, bổ sung cho LEAK-001 (chỉ phủ returns, volatility, RSI).

---

### 🏁 SPRINT 4: Xây dựng Môi trường Giao dịch (Trading Environment)

_Mục tiêu: Xây dựng logic tính toán tiền mặt, cổ phiếu, phí giao dịch và lợi nhuận danh mục._

- [x] **[ENV-001] Quản lý trạng thái danh mục (Portfolio State) (P0)**
  - _Tệp cần tạo:_ `src/env/portfolio.py`
  - _Nội dung:_ Class lưu trữ: `cash`, `portfolio_value`, `weights`, `peak_value`.

- [x] **[ENV-002] Bộ máy tính toán lợi nhuận danh mục (Return Engine) (P0)** _(đã chỉnh)_
  - _Tệp cần tạo:_ `src/env/returns.py`, `tests/test_portfolio_returns.py`
  - _Công thức:_ $R_p(t+1) = \sum_{i} w_i(t) \cdot r_i(t+1)$. **Lưu ý: Đây là return GỘP (gross), chưa trừ phí giao dịch.** Có unit test chặt chẽ.

- [x] **[ENV-003] Ràng buộc tỷ trọng đầu tư (Rebalancing Constraints) (P0)**
  - _Tệp cần tạo:_ `src/env/rebalance.py`
  - _Ràng buộc:_ Long-only ($w_i \ge 0$), tổng tỷ trọng cổ phiếu + tiền mặt = 1 ($\sum w_i + w_{cash} = 1$). Có hàm chiếu (projection) nếu agent đưa ra weight không hợp lệ.

- [x] **[ENV-004] Mô hình hóa chi phí giao dịch (Transaction Cost) (P0)** _(đã chỉnh)_
  - _Tệp cần tạo:_ `src/env/cost.py`
  - _Công thức:_
    - Tính tỷ trọng đã trôi (drifted weight) bao gồm cả cash với lãi suất phi rủi ro: $ ilde{w}_i = w_{i,t-1} \cdot (1 + r_i) / (1 + R_p)$
    - Tính tỷ lệ phí giao dịch: $TC_t = c \cdot \sum_i |w_{i,t} - 	ilde{w}_i|$
  - _Lưu ý:_ Không nhân $V_t$ trực tiếp trong env. Giá trị đô la = $TC_t \cdot V_t$ nếu cần báo cáo.

- [x] **[ENV-005] Bộ theo dõi sụt giảm vốn (Drawdown Tracker) (P0)**
  - _Tệp cần tạo:_ `src/env/drawdown.py`
  - _Công thức:_ $\text{DD}_t = \frac{V_t - \text{Peak}_t}{\text{Peak}_t}$; tính Maximum Drawdown (MDD).

- [x] **[ENV-006] Tính toán độ biến động rủi ro danh mục (Risk Calculator) (P1)** _(đã chỉnh)_
  - _Tệp cần tạo:_ `src/env/risk.py`
  - _Nội dung:_ Tính rolling portfolio volatility: $\sigma_p = \sqrt{w^T \Sigma w}$. Cửa sổ ước lượng covariance từ 60 ngày hoặc dùng shrinkage Ledoit-Wolf (Lý do: 20 quan sát cho 24 tài sản làm ma trận suy biến).

- [x] **[ENV-007] Thiết kế hàm Reward V1 (P0)** _(đã chỉnh)_
  - _Tệp cần tạo:_ `src/env/reward.py`
  - _Công thức ban đầu:_ $\text{Reward}_t = R_{p,t} - \beta \cdot TC_t$
  - _Ghi chú:_ $\beta = 1$ là phạt đúng phí thật. Nếu $\beta > 1$ thì đó là reward shaping, và mọi metrics đánh giá vẫn tính trên net return thật ($R_p - TC$).

- [x] **[ENV-009] Chốt quy ước khớp lệnh và thời điểm quyết định (P0)** _(mới)_
  - _Quy ước:_ Quyết định cuối ngày $t$, khớp lệnh tại giá Open ngày $t+1$, return tính theo Open $\rightarrow$ Open.
  - _Nội dung:_ Env đọc thêm ma trận giá open $[T, N]$. Yêu cầu ghi quy ước này vào docs TRƯỚC khi viết `trading_env.py`. Nhắc lại ở RESEARCH-002 (Sprint 9) rằng việc này chỉ còn là kiểm toán, không phải lúc quyết định.

- [x] **[GNN-002] Phối hợp chạy Integration Test cho GNN (P0 - Cùng B)**
  - Kiểm tra xem mạng GNN do Thành viên B viết có nhận đúng định dạng dữ liệu từ Feature Pipeline của bạn hay không.

---

### 🏁 SPRINT 5: Hoàn thiện Gym Environment & Bộ máy Backtest

_Mục tiêu: Hoàn thiện môi trường Gym tiêu chuẩn và chạy được các chiến lược Baseline truyền thống._

#### Ticket định lượng của Sprint 5

**Thứ tự phối hợp:** B/QNT-001 → A/QNT-002 → A/QNT-003. QNT-003 đánh giá feature hiện có; sau QNT-004/005 ở Sprint 6, chạy lại đánh giá theo protocol đã chốt.

<a id="qnt-002"></a>

- [x] **[QNT-002] — Đóng băng protocol, baseline và gate (P0) [ĐỊNH LƯỢNG]**
  - **Theo dõi:** ✅ Triển khai/nghiệm thu dữ liệu thật; gate NO-GO.
  - **Mức độ:** P0 · **Owner:** A · **Reviewer:** B · **Phụ thuộc:** QNT-001
  - **Phạm vi:** Tạo config thí nghiệm chuẩn gồm train/inner-val/outer-val, seed, search budget, checkpoint rule và gate. Thêm chế độ đánh giá validation/fold phát triển mà không tự động đọc test set.
  - **Tệp tác động:** `src/evaluation/research_protocol.py`, `validation_evaluator.py`, `src/training/ppo_trainer.py`, `data_preparation.py`; config `configs/quant_research_protocol.yaml`.
  - **DoD (Nghiệm thu):**
    - Baseline 1/N, matched cash + alpha và untrained policy có cùng timeline, cost, cap. Lịch tái cân bằng và first allocation được ghi rõ.
    - Khi thêm overlay rủi ro, baseline áp dụng cùng quy tắc causal; không lấy cash/exposure nhìn lại từ PPO để tối ưu baseline.
    - Checkpoint lưu cả mức cải thiện so với epoch 0 và matched baseline; trường hợp không vượt margin đã định báo "chưa đạt gate", không trình bày fallback như kết quả đã học.
  - **Sản phẩm bàn giao:** Config có hash, baseline reference mới sau masking, bảng protocol và giới hạn holdout.
  - **Đặc tả bổ sung:** [QNT-002](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-002).
  - **Nghiệm thu 18/10/2026:** đã triển khai và chạy dữ liệu thật; gate nghiên cứu **NO-GO**. [Báo cáo và artifact](NGHIEM_THU_SPRINT_5.md).

<a id="qnt-003"></a>

- [x] **[QNT-003] — Đo tín hiệu ngoài mẫu và baseline tuyến tính (P0) [ĐỊNH LƯỢNG]**
  - **Theo dõi:** ✅ Có IC/linear baseline theo fold; gate NO-GO.
  - **Mức độ:** P0 · **Owner:** A · **Reviewer:** B · **Phụ thuộc:** QNT-002
  - **Phạm vi:** Tính rank-IC ở horizon 1/5/20 trên nhãn thực thi được; hệ số, hướng tín hiệu và strength fit trên train/inner-val. Đánh giá riêng từng outer fold/giai đoạn phát triển, gồm 1/N lõi + tilt và matched baseline.
  - **Tệp tác động:** `scripts/analyze_feature_signal.py`, `scripts/evaluate_equal_weight_tilt.py`, `scripts/evaluate_tilt_robustness.py`.
  - **DoD (Nghiệm thu):** Purge các label train kéo vào kỳ đánh giá; IC có số mẫu và block CI; báo cáo turnover, chi phí, spread lợi suất giữa các nhóm score và độ ổn định dấu IC. Tham số không được chọn bằng outer fold/test.
  - **Sản phẩm bàn giao:** `signal_by_fold.csv`, `linear_baseline_by_fold.csv` và quyết định GO/NO-GO alpha modeling. Kết quả âm vẫn tính là hoàn thành ticket nếu implementation đúng.
  - **Đặc tả bổ sung:** [QNT-003](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-003).
  - **Nghiệm thu 18/10/2026:** đã triển khai và chạy dữ liệu thật; gate nghiên cứu **NO-GO**. [Báo cáo và artifact](NGHIEM_THU_SPRINT_5.md).

#### Công việc nền và H-MARL của sprint

- [x] **[ENV-008] Đóng gói Gymnasium-compatible Trading Environment (P0)**
  - _Tệp cần tạo:_ `src/env/trading_env.py`, `tests/test_trading_env.py`
  - _Nội dung:_ Cài đặt chuẩn Gymnasium: `reset()`, `step(action)`, `observation_space`, `action_space`.

- [x] **[BACKTEST-001] Bộ máy Backtest xác định (Deterministic Backtester) (P0)**
  - _Tệp cần tạo:_ `src/evaluation/backtester.py`
  - _Nội dung:_ Lặp qua từng ngày trong tập Test: ghi nhận Portfolio Value, Return, Phí giao dịch, Drawdown theo thời gian.

- [x] **[BASE-001] Triển khai Baseline Giữ tiền mặt (Cash Baseline) (P0)** _(đã chỉnh)_
  - _Tệp cần tạo:_ `src/evaluation/baselines/cash.py`
  - Giữ 100% Cash để làm mốc so sánh, cho cash sinh lãi bằng risk-free rate thay vì 0%.

- [x] **[BASE-002] Triển khai Baseline Danh mục đều tay (Equal Weight 1/N) (P0)**
  - _Tệp cần tạo:_ `src/evaluation/baselines/equal_weight.py`
  - Mỗi cổ phiếu phân bổ đều $1/N$ tỷ trọng (ví dụ 20 mã thì mỗi mã 5%), tự tự tái cân bằng hàng ngày.

- [x] **[BASE-003] Triển khai Baseline Mua và Nắm giữ (Buy & Hold) (P0)**
  - _Tệp cần tạo:_ `src/evaluation/baselines/buy_and_hold.py`
  - Phân bổ đều lúc đầu kỳ, sau đó để mặc cho giá tự trôi mà không rebalance.

- [x] **[BASE-004] Cùng Thành viên B lập trình và huấn luyện Single-Agent PPO (P0 - Chung A+B)** _(đã chỉnh)_
  - _Tệp cùng thực hiện:_ `src/training/ppo_core.py`, `tests/models/test_ppo_core.py`
  - Đã tách riêng lõi PPO Core vững chắc với `PPOBuffer` và `PPOUpdater`. Sử dụng GAE/advantage và clip loss chuẩn mực.

- [x] **[EXP-001] Cài đặt đầy đủ các chỉ số tài chính chuẩn (P0)** _(đã chỉnh)_
  - _Tệp cần tạo:_ `src/evaluation/metrics.py`
  - Cung cấp tính toán Cumulative Return, Annualized Return, Volatility, Sharpe, MDD, Calmar, Turnover, Costs.

> **Nghiệm thu bổ sung 18/10/2026:** [Sprint 5 đã chạy end-to-end theo protocol train/inner/outer development](NGHIEM_THU_SPRINT_5.md), không tự backtest giai đoạn test. QNT-002/QNT-003 hoàn thành triển khai; gate alpha NO-GO. Lệnh legacy có đánh giá test dưới đây vẫn được giữ để tương thích.

> 🏆 **Definition of Done (DoD) Sprint 5:** Một command duy nhất chạy end-to-end: dữ liệu OHLCV $\rightarrow$ processed features $\rightarrow$ train/validation/test theo thời gian $\rightarrow$ PPO $\rightarrow$ backtest trên test $\rightarrow$ xuất bảng metrics so sánh với Equal Weight, Buy & Hold và Cash. Lệnh đã được triển khai trong `scripts/run_sprint5_backtest.py`; chỉ đánh dấu nghiệm thu sau khi chạy thành công với bộ dữ liệu thật.

---

### 🏁 SPRINT 6: Feature, Risk, Execution & Sector MARL

_Mục tiêu: Đo lường chính xác hiệu quả đầu tư so với mức độ rủi ro gánh chịu._

#### Ticket định lượng của Sprint 6

**Thứ tự phối hợp:** QNT-004 → QNT-005; QNT-006 bàn giao cho B/QNT-007 → QNT-008; QNT-011 → QNT-012 → QNT-013 (cần thêm QNT-006). QNT-014 là ablation reward riêng; không bật đồng thời mọi thay đổi khi đo tác động.

<a id="qnt-004"></a>

- [ ] **[QNT-004] — Chuẩn hóa cross-sectional theo feature (P1) [ĐỊNH LƯỢNG]**
  - **Theo dõi (kế thừa):** 📝 Sẵn sàng.
  - **Mức độ:** P1 · **Owner:** A · **Reviewer:** B · **Phụ thuộc:** QNT-002
  - **Phạm vi:** Thêm chế độ `static_train`, `xs_zscore`, `xs_rank` và tùy chọn gaussian rank. Cấu hình theo tên feature; feature vốn là rank/RSI hữu hạn không tự động chuẩn hóa lần nữa.
  - **Tệp tác động:** `src/features/scaler.py`, `src/features/pipeline.py`, `src/models/policy_adapter.py`; module dự kiến `src/features/normalization.py`.
  - **DoD (Nghiệm thu):** Ngày t chỉ dùng các mã thuộc universe hợp lệ ngày t; ticker order không đổi; xử lý ties, N=1, std=0 và thiếu dữ liệu. Padding giữ bằng 0 sau transform và dùng mask tường minh.
  - **Sản phẩm bàn giao:** Metadata mode/feature/clip/epsilon, batch-stream parity, ablation với scaler hiện tại. Giữ market-level context khi rank làm mất mức tuyệt đối.
  - **Đặc tả bổ sung:** [QNT-004](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-004).

<a id="qnt-005"></a>

- [ ] **[QNT-005] — Causal rolling normalization và state inference (P1) [ĐỊNH LƯỢNG]**
  - **Theo dõi (kế thừa):** 📝 Sẵn sàng.
  - **Mức độ:** P1 · **Owner:** A · **Reviewer:** B · **Phụ thuộc:** QNT-004
  - **Phạm vi:** Rolling z-score theo từng mã/feature, cửa sổ ban đầu 252 phiên; chốt `statistics[t-252:t)` để chuẩn hóa feature t. Các horizon khác chỉ chọn trong inner-val.
  - **Tệp tác động:** `src/features/scaler.py`, `src/utils/data_types.py`, `src/training/train_ppo.py`.
  - **DoD (Nghiệm thu):** Sửa dữ liệu sau t không đổi normalized feature t; batch và inference từng ngày khớp; không fit cả val/test một lần. Lưu rolling state, ngày cuối, ticker/feature order, warm-up và readiness.
  - **Sản phẩm bàn giao:** Checkpoint/resume/walk-forward khôi phục đúng state; tỷ lệ clip/outlier theo fold và ablation static vs xs vs rolling vs hybrid.
  - **Đặc tả bổ sung:** [QNT-005](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-005).

<a id="qnt-006"></a>

- [ ] **[QNT-006] — Risk bank covariance EWMA/Ledoit–Wolf (P1) [ĐỊNH LƯỢNG]**
  - **Theo dõi (kế thừa):** 📝 Bàn giao cho B.
  - **Mức độ:** P1 · **Owner:** A · **Reviewer:** B · **Phụ thuộc:** QNT-001, QNT-002
  - **Phạm vi:** Một bộ ước lượng covariance causal theo ngày dùng chung cho graph (bàn giao cho B), risk overlay và allocator; gồm baseline sample covariance, EWMA và Ledoit–Wolf tuyến tính.
  - **Tệp tác động:** `src/env/risk.py`, `src/graph/correlation.py`, `src/utils/data_types.py`; module dự kiến `src/risk/covariance.py`.
  - **DoD (Nghiệm thu):** Shape N=1 đúng; dữ liệu hữu hạn, ma trận đối xứng và PSD trong tolerance; shrinkage/fallback/jitter được log. Dữ liệu sau t không ảnh hưởng covariance t; annualization chỉ thực hiện một lần.
  - **Sản phẩm bàn giao:** Covariance bank cache theo data/config hash, condition number, eigenvalue minimum và sai số dự báo volatility ngoài mẫu. Bàn giao cho B làm QNT-007.
  - **Đặc tả bổ sung:** [QNT-006](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-006).

<a id="qnt-011"></a>

- [ ] **[QNT-011] — Thanh khoản ADV, volatility và data contract (P1) [ĐỊNH LƯỢNG]**
  - **Theo dõi (kế thừa):** 📝 Sẵn sàng.
  - **Mức độ:** P1 · **Owner:** A · **Reviewer:** B · **Phụ thuộc:** QNT-001
  - **Phạm vi:** Trailing dollar ADV (Average Daily Volume) và daily volatility theo từng mã; thông tin để quyết định ở Close[t] chỉ dùng dữ liệu đến t.
  - **Tệp tác động:** `src/features/pipeline.py`, `src/data/alignment.py`, `src/utils/data_types.py`; module dự kiến `src/features/liquidity.py`.
  - **DoD (Nghiệm thu):** Đơn vị ADV là tiền/ngày, volatility là tỷ lệ/ngày. Chốt giá/volume adjustment basis; không nhân giá total-return đã điều chỉnh cổ tức với volume rồi mặc định đó là notional giao dịch thật. Xử lý zero volume, thiếu dữ liệu, split và ticker order.
  - **Sản phẩm bàn giao:** Liquidity bank theo ngày, metadata source/window/readiness. Volume thực tế ngày t+1 không được dùng tại t.
  - **Đặc tả bổ sung:** [QNT-011](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-011).

<a id="qnt-012"></a>

- [ ] **[QNT-012] — Square-root impact và phân rã chi phí (P1) [ĐỊNH LƯỢNG]**
  - **Theo dõi (kế thừa):** 📝 Sẵn sàng.
  - **Mức độ:** P1 · **Owner:** A · **Reviewer:** B · **Phụ thuộc:** QNT-011, QNT-002
  - **Phạm vi:** Optional impact trên commission/spread baseline: $\text{impact\_rate}_i = \eta \cdot \sigma_i \cdot \sqrt{Q_i / \text{dollar\_ADV}_i}$, với $Q_i$ là notional giao dịch thực tế tại Open kế tiếp.
  - **Tệp tác động:** `src/env/cost.py`, `src/env/trading_env.py`, `src/evaluation/metrics.py`, `configs/env.yaml`.
  - **DoD (Nghiệm thu):** Stock legs chỉ tính một lần, cash leg không tạo impact; hold/zero trade có phí 0; $\eta=0$ khớp đường hiện tại; tổng chi phí tiền khớp NAV. Log riêng commission, spread/slippage và impact, không tính trùng.
  - **Sản phẩm bàn giao:** Stress testing theo $\eta$ và NAV, bảng phân rã chi phí (cost decomposition) và cảnh báo participation rate khi thiếu dữ liệu.
  - **Đặc tả bổ sung:** [QNT-012](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-012).

<a id="qnt-013"></a>

- [ ] **[QNT-013] — Tốc độ execution thích nghi và vùng hold (P1) [ĐỊNH LƯỢNG]**
  - **Theo dõi (kế thừa):** 📝 Sẵn sàng.
  - **Mức độ:** P1 · **Owner:** A · **Reviewer:** B · **Phụ thuộc:** QNT-012, QNT-006
  - **Phạm vi:** Tốc độ tái cân bằng từng mã dựa trên chi phí, thanh khoản và rủi ro causal; vùng không giao dịch (no-trade band) và participation cap.
  - **Tệp tác động:** `src/env/rebalance.py`, `src/env/trading_env.py`.
  - **DoD (Nghiệm thu):** Target chốt từ Close, execution cost dùng drift tại Open; không đọc Open tương lai để chọn tốc độ tại Close. Sau điều chỉnh tốc độ từng mã, weights/cash/cap vẫn hợp lệ.
  - **Sản phẩm bàn giao:** Báo cáo so sánh các baseline cùng execution controller, tỷ lệ hold, turnover, chi phí và signal decay.
  - **Đặc tả bổ sung:** [QNT-013](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-013).

<a id="qnt-014"></a>

- [ ] **[QNT-014] — Reward Differential Sharpe có state (P1) [ĐỊNH LƯỢNG]**
  - **Theo dõi (kế thừa):** 📝 Sẵn sàng.
  - **Mức độ:** P1 · **Owner:** A · **Reviewer:** B · **Phụ thuộc:** QNT-001, QNT-002
  - **Phạm vi:** Reward mode riêng cho Differential Sharpe Ratio (DSR); trước cập nhật EMA, dùng moments $A, B$ của bước trước và delta từ return ròng hiện tại. Chốt rõ excess return so với risk-free hay active return so với benchmark.
  - **Tệp tác động:** `src/env/reward.py`, `src/env/trading_env.py`, `src/training/train_ppo.py`.
  - **DoD (Nghiệm thu):** Mẫu số có $\epsilon$ chống chia 0 và cơ chế warm-up; constant-return không gây reward nổ/NaN. EMA moments, readiness và update convention có state trong observation/reset/checkpoint. Không khởi tạo moments bằng lợi suất tương lai.
  - **Sản phẩm bàn giao:** Chẩn đoán reward scale/clipping, gradient, KL, EV; ablation excess-return vs DSR với seed/fold cố định.
  - **Đặc tả bổ sung:** [QNT-014](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-014).

#### Công việc nền và H-MARL của sprint

- [ ] **[MARL-001] Thống nhất ranh giới phân chia Sector với B (P0 - Chung)**
  - Cung cấp danh sách phân nhóm cổ phiếu cho từng sector agent.

- [ ] **[MARL-002] Huấn luyện Sector Agent phân bổ tỷ trọng (P0)** _(mới)_
  - Huấn luyện Agent phân bổ nội bộ cho nhóm ngành được phân công.

- [ ] **[MARL-003] Huấn luyện cơ chế khen thưởng từng nhóm (P0)** _(mới)_
  - Thiết kế reward riêng cho mỗi Sector Agent.

- [ ] **[MARL-004] Tích hợp mạng MARL với Môi trường giao dịch (P0)** _(mới)_
  - Kết nối Multi-Agent với `portfolio_env.py`.

- [ ] **[MARL-006] Đánh giá mô hình MARL sơ bộ (P0 - Chung)**
  - Dùng Backtester chạy đánh giá so sánh giữa Baseline truyền thống và mô hình MARL do B train.

---

### 🏁 SPRINT 7: Phát triển High-Level Agent & Ghép nối Phân cấp

_Mục tiêu: Làm chủ Agent vĩ mô phân bổ vốn vào các nhóm ngành và tiền mặt, sau đó ghép nối toàn diện._

#### Ticket định lượng của Sprint 7

**Thứ tự phối hợp:** QNT-010 dùng risk bank Sprint 6. QNT-016 có thể dùng alpha ridge fit train; bàn giao allocator thường cho B/QNT-017 trước QNT-018. Nhánh P2 không bắt buộc cho HMARL-004.

<a id="qnt-010"></a>

- [ ] **[QNT-010] — Volatility targeting cho policy và baseline (P1) [ĐỊNH LƯỢNG]**
  - **Theo dõi (kế thừa):** 📝 Sẵn sàng.
  - **Mức độ:** P1 · **Owner:** A · **Reviewer:** B · **Phụ thuộc:** QNT-006, QNT-002
  - **Phạm vi:** Risk overlay causal với hệ số co $k = \min(1, \text{target\_vol} / \text{forecast\_vol})$; giảm phần cổ phiếu theo $k$, chuyển phần còn lại vào cash. Target và forecast dùng cùng đơn vị đã khai báo.
  - **Tệp tác động:** `src/env/risk.py`, `src/env/rebalance.py`, `src/env/trading_env.py`, `src/evaluation/baselines/`.
  - **DoD (Nghiệm thu):** Forecast từ covariance t; cash/zero variance an toàn, không dùng đòn bẩy; weights tổng bằng 1 và cap hợp lệ; thay đổi overlay dẫn đến giao dịch thì tính phí đúng. Explicit hold giữ nguyên cổ phiếu nếu không có lịch/lệnh risk override.
  - **Sản phẩm bàn giao:** Realized vol, target tracking, cash/exposure, return và tail loss. So sánh 1/N dùng cùng overlay, estimator và lịch.
  - **Đặc tả bổ sung:** [QNT-010](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-010).

<a id="qnt-016"></a>

- [ ] **[QNT-016] — Allocator lồi độc lập (Convex Allocator) (P2) [ĐỊNH LƯỢNG · MỐC C — CÓ ĐIỀU KIỆN]**
  - **Theo dõi (kế thừa):** 🔬 Nghiên cứu.
  - **Mức độ:** P2 · **Owner:** A · **Reviewer:** B · **Phụ thuộc:** QNT-006, QNT-003
  - **Phạm vi:** Allocator lồi giải bằng solver thường (trước khi làm QP khả vi); nhận alpha dự báo từ ridge/QNT-015, ma trận covariance, danh mục tại Close và hệ số chi phí.
  - **Tệp tác động:** `src/env/rebalance.py`, `src/env/risk.py`, `src/evaluation/backtester.py`; package dự kiến `src/allocation/`.
  - **Bài toán:** Maximize $\alpha^T w - \frac{\gamma}{2} w^T \Sigma w - \frac{\lambda}{2} \Delta w^T \Lambda \Delta w$ thỏa mãn long-only, max asset cap, sector caps.
  - **DoD (Nghiệm thu):** Xử lý synthetic feasible/infeasible cases; sai số ràng buộc trong tolerance; solver fallback được log khi timeout. Trạng thái $w_{prev}$ là trạng thái biết tại Close.
  - **Sản phẩm bàn giao:** Runtime/solver status, so sánh mean-variance baseline vs alpha + allocator vs simple tilt.
  - **Đặc tả bổ sung:** [QNT-016](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-016).

#### Công việc nền và H-MARL của sprint

- [ ] **[HMARL-000] Tài liệu thiết kế huấn luyện phân cấp (P0)** _(mới)_
  - Chốt 3 điều:
    - (a) Reward của low-level agent là reward danh mục chung hay reward riêng theo sector?
    - (b) Tần suất ra quyết định của high-level (hàng ngày hay hàng tuần)?
    - (c) Lịch cập nhật hai tầng (xen kẽ hoặc đóng băng một tầng khi tầng kia học) để giảm non-stationarity.

- [ ] **[HMARL-001] Xây dựng Trạng thái Quan sát cho High-Level Agent (P0)**
  - _Tệp cần tạo:_ `src/agents/high_level.py`
  - _Nội dung:_ Vector quan sát vĩ mô gồm: Trạng thái tổng hợp thị trường (Global Market Embedding), mức độ biến động rủi ro toàn danh mục, số dư tiền mặt hiện tại và drawdown hiện hành.

- [ ] **[HMARL-002] Xây dựng Chính sách Phân bổ Vốn Vĩ mô (Macro Policy) (P0)**
  - _Tệp cần tạo:_ `src/agents/high_level.py`
  - _Nội dung:_ Agent học chính sách phân bổ ngân sách cho từng nhóm ngành và tiền mặt:
    $$\text{Budget}_{Tech}, \text{Budget}_{Finance}, \text{Budget}_{Healthcare}, \text{Budget}_{Energy\_Consumer}, \text{Ratio}_{Cash}$$ _(đã chỉnh)_
  - _Mục tiêu:_ Tự động co hẹp ngân sách ngành và tăng tỷ lệ tiền mặt khi thị trường biến động mạnh hoặc downtrend.

- [ ] **[HMARL-003] Xây dựng bộ ghép nối tỷ trọng phân cấp (P0)**
  - _Tệp cần tạo:_ `src/agents/combiner.py`
  - _Công thức:_ $\text{Weight}(AAPL) = \text{Budget}(Tech) \times \text{Weight}_{Tech}(AAPL)$.
  - _Tiêu chuẩn nghiệm thu:_ Tổng trọng số toàn bộ cổ phiếu và tiền mặt luôn đúng bằng 1 và không có giá trị âm.

- [ ] **[HMARL-004] Cùng Thành viên B tích hợp Vòng lặp Huấn luyện H-MARL (P0 - Chung)**
  - Kết nối High-Level Agent (của A) và Low-Level Agents (của B) để cùng cập nhật chính sách phân cấp.

---

### 🏁 SPRINT 8: Thí nghiệm Cắt giảm Đặc trưng & Phân tích Chi phí

_Mục tiêu: Thực hiện các thí nghiệm chứng minh sự đóng góp của các thành phần (Ablation Study)._

#### Ticket định lượng của Sprint 8

**Thứ tự phối hợp:** QNT-009 cấp fold cho B/QNT-025. Nếu mở nhánh CVaR, chờ B/QNT-022 rồi làm QNT-023 trước khi chốt báo cáo tail risk trong EXP-004. Feature/cost ablation dùng các phiên bản đã nghiệm thu ở Sprint 6–7.

<a id="qnt-009"></a>

- [ ] **[QNT-009] — Runner walk-forward causal (P1) [ĐỊNH LƯỢNG]**
  - **Theo dõi (kế thừa):** 📝 Sẵn sàng.
  - **Mức độ:** P1 · **Owner:** A · **Reviewer:** B · **Phụ thuộc:** QNT-002
  - **Phạm vi:** Rolling/expanding folds; retrain hoặc fine-tune mỗi 20/60 phiên. Mỗi fit chỉ nhận dữ liệu có sẵn trước ngày deployment; scaler và checkpoint selection thuộc phần train/inner-val của fold.
  - **Tệp tác động:** `src/data/split.py`, `src/training/train_ppo.py`, `src/evaluation/backtester.py`; script dự kiến `scripts/evaluate_walk_forward.py`.
  - **DoD (Nghiệm thu):** Label/rollout overlap được purge; model triển khai chỉ chạy sau thời điểm fit; nối OOS daily returns không trùng/ngắt ngày. NAV, holdings, cash và phí tiếp tục qua điểm retrain, không reset danh mục miễn phí ở mỗi fold.
  - **Sản phẩm bàn giao:** Manifest theo fold chứa fit/deploy dates, parent checkpoint/scaler state, fresh-train vs fine-tune, runtime và artifact. Baseline chạy liên tục trên cùng lịch.
  - **Đặc tả bổ sung:** [QNT-009](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-009).

<a id="qnt-023"></a>

- [ ] **[QNT-023] — Mục tiêu actor CVaR và gate tail risk (P2) [ĐỊNH LƯỢNG · MỐC C — CÓ ĐIỀU KIỆN]**
  - **Theo dõi (kế thừa):** 🔬 Nghiên cứu.
  - **Mức độ:** P2 · **Owner:** A · **Reviewer:** B · **Phụ thuộc:** QNT-022
  - **Phạm vi:** Định nghĩa mục tiêu CVaR trên đuôi dưới (lower-tail return), horizon, tail level $\alpha$ (ví dụ 5%) và mục tiêu Actor; quy tắc risk-sensitive advantage hoặc constrained policy update.
  - **Tệp tác động:** `src/training/ppo_core.py`, `src/env/reward.py`, `src/evaluation/metrics.py`, `src/training/train_ppo.py`.
  - **DoD (Nghiệm thu):** Không chỉ thay Critic bằng quantiles rồi giữ nguyên Actor loss cũ và gọi là tối ưu CVaR. Với 32 quantiles, tail 5% chỉ có 1–2 điểm; cần grid phân vị đủ mịn ở đuôi.
  - **Sản phẩm bàn giao:** Realized tail loss/CVaR, drawdown, net return, cash và baseline rủi ro theo fold; gate giới hạn đánh đổi lợi suất.
  - **Đặc tả bổ sung:** [QNT-023](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-023).

#### Công việc nền và H-MARL của sprint

- [ ] **[EXP-006] Thí nghiệm Feature Ablation (P1)**
  - _Nội dung:_ So sánh hiệu quả mô hình khi chỉ dùng:
    - Nhóm 1: Chỉ dùng giá thô và Volume.
    - Nhóm 2: Đầy đủ chỉ báo kỹ thuật (Return + RSI + MACD + Volatility + VolumeNorm).
  - _Mục đích:_ Chứng minh giá trị đóng góp của Feature Engineering.

- [ ] **[EXP-009] Phân tích độ nhạy với phí giao dịch (Transaction Cost Sensitivity) (P1)**
  - _Nội dung:_ Chạy backtest với các mức phí khác nhau: 0%, 0.05%, 0.1%, 0.2%.
  - _Mục đích:_ Xem mô hình có bị "ăn mòn" lợi nhuận khi phí giao dịch tăng lên hay không.

- [ ] **[EXP-004] Tổng hợp bảng so sánh mô hình chính (P0 - Chung)**
  - Chạy toàn bộ các mô hình qua Backtester để lập bảng kết quả tổng hợp.

---

### 🏁 SPRINT 9: Kiểm toán Rò rỉ Dữ liệu & Thống kê Khoa học

_Mục tiêu: Đảm bảo kết quả trung thực, không bị bias và có ý nghĩa thống kê._

#### Ticket định lượng của Sprint 9

**Thứ tự phối hợp:** QNT-024 nhận walk-forward từ Sprint 8. CPCV ghi rõ research-only nếu train có nhóm ở tương lai so nhóm đánh giá; walk-forward vẫn là phép đánh giá causal chính.

<a id="qnt-024"></a>

- [ ] **[QNT-024] — CPCV research, purge và embargo (P2) [ĐỊNH LƯỢNG · MỐC C — CÓ ĐIỀU KIỆN]**
  - **Theo dõi (kế thừa):** 🔬 Nghiên cứu.
  - **Mức độ:** P2 · **Owner:** A · **Reviewer:** B · **Phụ thuộc:** QNT-009
  - **Phạm vi:** Chia tập tổ hợp (Combinatorial Purged Cross-Validation) phục vụ nghiên cứu; mỗi mẫu có decision time, label interval và rollout interval. Fit scaler/hyperparameter trên train fold theo protocol.
  - **Tệp tác động:** `src/data/split.py`, `scripts/analyze_feature_signal.py`; module dự kiến `src/evaluation/purged_cv.py`.
  - **DoD (Nghiệm thu):** Purge overlap theo interval thực tế; embargo được cấu hình và giải thích logic, không tự động đặt bằng lookback. Tránh episode nối qua nhóm đánh giá.
  - **Sản phẩm bàn giao:** Split visualization, overlap checks và CPCV research report. Lưu ý walk-forward QNT-009 vẫn là phép đánh giá chính theo thời gian thực tế.
  - **Đặc tả bổ sung:** [QNT-024](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-024).

#### Công việc nền và H-MARL của sprint

- [ ] **[RESEARCH-002] Kiểm toán toàn diện Bias & Rò rỉ dữ liệu (P0)**
  - _Nội dung:_ Soát xét toàn bộ mã nguồn để đảm bảo:
    - Không bị Look-ahead Bias (tại thời điểm $t$ không dùng giá của ngày $t+1$).
    - Không bị Data Leakage khi chuẩn hóa dữ liệu.
    - (Nhắc lại từ ENV-009): Kiểm toán xác nhận Env sử dụng đúng giá Open ngày t+1 để khớp lệnh và tính return, không vi phạm quy ước.
    - Giả định khớp lệnh thực tế (quyết định vào cuối ngày $t$, khớp lệnh tại giá mở cửa $t+1$ hoặc đóng cửa $t$).

- [ ] **[EXP-011] Tổng hợp thống kê và xuất biểu đồ trực quan (P1)**
  - _Tệp cần tạo:_ `src/evaluation/plots.py`, `scripts/generate_plots.py`
  - _Nội dung:_ Vẽ biểu đồ đường cong vốn (Equity Curve), phân bổ tỷ trọng theo thời gian (Asset Allocation Plot), biểu đồ Drawdown dưới dạng đồ thị trực quan phục vụ báo cáo.

- [ ] **[EXP-010] Đánh giá theo chế độ thị trường (P1 - Chung)**
  - Đánh giá hiệu quả mô hình riêng biệt trong 3 giai đoạn: Thị trường tăng (Bull), Thị trường giảm (Bear) và Biến động mạnh (High-volatility).

---

### 🏁 SPRINT 10: Viết Báo cáo & Đóng gói Tái lập

_Mục tiêu: Hoàn tất tài liệu nghiên cứu và đảm bảo chạy lại từ đầu thành công._

- [ ] **[DOC-001] Soạn thảo tài liệu kiến trúc Data & Môi trường (P1)**
  - Mô tả chi tiết luồng xử lý: Raw Data $\rightarrow$ Preprocessing $\rightarrow$ Features $\rightarrow$ Environment $\rightarrow$ Backtester.

- [ ] **[DOC-003] Soạn thảo tài liệu thiết lập thí nghiệm & Bảng kết quả (P0)**
  - Tổng hợp QNT đã nghiệm thu ở Sprint 5–9: config/fold/seed, cost, matched baseline, gate PASS/FAIL và giới hạn holdout; chỉ báo cáo nhánh P2 đã thực sự chạy.
  - Trình bày bảng so sánh Sharpe, MDD, Return, Calmar và các biểu đồ phân tích.

- [ ] **[FINAL-001] Tái lập kết quả từ môi trường sạch (P0 - Chung)**
  - Clone repo ra một máy/môi trường mới, chạy script từ bước tải data đến xuất báo cáo kết quả xem có lỗi nào không.

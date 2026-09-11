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

---

## 1. Các thư mục do Thành viên A chịu trách nhiệm trực tiếp

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

- [x] **[CONTRACT-001] Thống nhất giao diện dữ liệu linh hoạt với Thành viên B (P0 - Chung)**
  - _Tệp đã tạo:_ `src/utils/contracts.py`, `tests/test_contracts.py`
  - _Đã chốt định dạng đầu ra cho B:_ `MarketDataTensor` kích thước `[T, N, F]` (thời gian $\times$ tài sản $\times$ đặc trưng kỹ thuật).
  - _Đã chốt cấu trúc đồ thị tương thích với B (`builder.py`):_ `DynamicGraphData` gồm `node_features` `[N, F]`, `edge_index` `[2, E]`, `edge_weight` `[E]`.
  - _Đã chốt định dạng Observation của Môi trường hỗ trợ 2 chế độ (`MarketObservation`):_
    - Chế độ 1: `raw_features` `[N, F]` (cho Single-Agent PPO chạy không cần GNN).
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

- [ ] **[DATA-004] Đồng bộ ngày giao dịch (Trading Calendar Alignment) (P0)**
  - _Tệp cần tạo:_ `src/data/alignment.py`
  - _Nội dung:_ Thị trường có thể có ngày nghỉ bất thường hoặc mã niêm yết lệch. Cần tạo lịch giao dịch chung (Common Dates) sao cho toàn bộ các mã có cùng một Index ngày.
  - _Tiêu chuẩn nghiệm thu:_ `len(df_AAPL) == len(df_MSFT) == len(df_NVDA)` và chung Index `Date`.

- [ ] **[DATA-005] Lưu siêu dữ liệu nguồn (Provenance Metadata) (P1)**
  - _Tệp cần tạo:_ `data/interim/dataset_metadata.json`
  - _Nội dung:_ Lưu lại thời gian tải, phiên bản dữ liệu, số dòng, khoảng thời gian để phục vụ viết báo cáo khoa học.

- [ ] **[FEAT-001] Tính toán Daily Return (P0)**
  - _Tệp cần tạo:_ `src/features/returns.py`, `tests/test_returns.py`
  - _Công thức:_ $R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$
  - _Tiêu chuẩn nghiệm thu:_ Unit test tính toán chính xác sai số $< 10^{-8}$.

- [ ] **[FEAT-002] Tính toán Rolling Volatility (P0)**
  - _Tệp cần tạo:_ `src/features/volatility.py`
  - _Công thức:_ Độ lệch chuẩn của return trong cửa sổ trượt 20 ngày (`rolling(20).std()`).

- [ ] **[FEAT-003] Tính toán RSI-14 (P1)**
  - _Tệp cần tạo:_ `src/features/rsi.py`, `tests/test_rsi.py`
  - _Nội dung:_ Cài đặt công thức Wilder RSI chu kỳ 14 ngày.
  - _Tiêu chuẩn nghiệm thu:_ Giá liên tục tăng thì RSI tiệm cận 100; giá liên tục giảm thì RSI tiệm cận 0.

> 🤝 **Bàn giao:** Chuyển dữ liệu Return đã làm sạch cho Thành viên B để B bắt đầu thử nghiệm dựng ma trận tương quan đồ thị.

---

### 🏁 SPRINT 3: Hoàn thiện Feature Pipeline & Phân chia tập an toàn

_Mục tiêu: Đóng gói toàn bộ đặc trưng vào tensor và chia Train/Val/Test tuyệt đối không rò rỉ dữ liệu._

- [ ] **[FEAT-004] Tính toán MACD (P1)**
  - _Tệp cần tạo:_ `src/features/macd.py`
  - _Nội dung:_ Tính đường MACD (EMA12 - EMA26), đường Signal (EMA9 của MACD) và Histogram.

- [ ] **[FEAT-005] Chuẩn hóa Volume (P1)**
  - _Tệp cần tạo:_ `src/features/volume.py`
  - _Nội dung:_ Áp dụng $\log(1 + \text{Volume})$ và chuẩn hóa (Normalize). Tuyệt đối không dùng thông tin tương lai để tính mean/std.

- [ ] **[FEAT-006] Đóng gói toàn bộ Feature Pipeline (P0)**
  - _Tệp cần tạo:_ `src/features/pipeline.py`
  - _Nội dung:_ Gộp 6 đặc trưng thành tensor đồng nhất cho từng ngày và từng cổ phiếu: `[Return, Volatility, RSI, MACD, MACD_Signal, VolumeNorm]`.
  - _Tiêu chuẩn nghiệm thu:_ Tạo ra file dữ liệu sẵn sàng cho Model tại `data/processed/features.parquet`.

- [ ] **[SPLIT-001] Phân chia Train / Validation / Test theo thời gian (P0)**
  - _Tệp cần tạo:_ `src/data/split.py`
  - _Quy tắc sống còn:_ KHÔNG dùng `train_test_split(shuffle=True)`.
  - _Phân chia:_ Train (2015–2021), Validation (2022–2023), Test (2024–2025).
  - _Tiêu chuẩn nghiệm thu:_ `max(train_date) < min(val_date)` và `max(val_date) < min(test_date)`.

- [ ] **[SPLIT-002] Chuẩn hóa chỉ học trên tập Train (Train-only Normalization) (P0)**
  - _Tệp cần tạo:_ `src/features/scaler.py`
  - _Nội dung:_ Tính $\mu$ và $\sigma$ duy nhất từ tập Train, sau đó áp dụng biến đổi cho tập Validation và Test. Lưu lại tham số scaler ra file.

---

### 🏁 SPRINT 4: Xây dựng Môi trường Giao dịch (Trading Environment)

_Mục tiêu: Xây dựng logic tính toán tiền mặt, cổ phiếu, phí giao dịch và lợi nhuận danh mục._

- [ ] **[ENV-001] Quản lý trạng thái danh mục (Portfolio State) (P0)**
  - _Tệp cần tạo:_ `src/env/portfolio.py`
  - _Nội dung:_ Class lưu trữ: `cash`, `portfolio_value`, `weights`, `peak_value`.

- [ ] **[ENV-002] Bộ máy tính toán lợi nhuận danh mục (Return Engine) (P0)**
  - _Tệp cần tạo:_ `src/env/returns.py`, `tests/test_portfolio_returns.py`
  - _Công thức:_ $R_p(t+1) = \sum_{i} w_i(t) \cdot r_i(t+1)$. Có unit test chặt chẽ.

- [ ] **[ENV-003] Ràng buộc tỷ trọng đầu tư (Rebalancing Constraints) (P0)**
  - _Tệp cần tạo:_ `src/env/rebalance.py`
  - _Ràng buộc:_ Long-only ($w_i \ge 0$), tổng tỷ trọng cổ phiếu + tiền mặt = 1 ($\sum w_i + w_{cash} = 1$). Có hàm chiếu (projection) nếu agent đưa ra weight không hợp lệ.

- [ ] **[ENV-004] Mô hình hóa chi phí giao dịch (Transaction Cost) (P0)**
  - _Tệp cần tạo:_ `src/env/cost.py`
  - _Công thức:_ $\text{Turnover}_t = \sum_i |w_{i,t} - w_{i,t-1}|$; $\text{Cost}_t = c \cdot \text{Turnover}_t \cdot V_t$ (với tỷ lệ phí $c = 0.1\%$ lấy từ cấu hình).

- [ ] **[ENV-005] Bộ theo dõi sụt giảm vốn (Drawdown Tracker) (P0)**
  - _Tệp cần tạo:_ `src/env/drawdown.py`
  - _Công thức:_ $\text{DD}_t = \frac{V_t - \text{Peak}_t}{\text{Peak}_t}$; tính Maximum Drawdown (MDD).

- [ ] **[ENV-006] Tính toán độ biến động rủi ro danh mục (Risk Calculator) (P1)**
  - _Tệp cần tạo:_ `src/env/risk.py`
  - _Nội dung:_ Tính rolling portfolio volatility: $\sigma_p = \sqrt{w^T \Sigma w}$.

- [ ] **[ENV-007] Thiết kế hàm Reward V1 (P0)**
  - _Tệp cần tạo:_ `src/env/reward.py`
  - _Công thức ban đầu:_ $\text{Reward}_t = R_{p,t} - \beta \cdot \text{Cost}_t$ (đơn giản, ổn định trước khi thêm phạt rủi ro).

- [ ] **[GNN-002] Phối hợp chạy Integration Test cho GNN (P0 - Cùng B)**
  - Kiểm tra xem mạng GNN do Thành viên B viết có nhận đúng định dạng dữ liệu từ Feature Pipeline của bạn hay không.

---

### 🏁 SPRINT 5: Hoàn thiện Gym Environment & Bộ máy Backtest

_Mục tiêu: Hoàn thiện môi trường Gym tiêu chuẩn và chạy được các chiến lược Baseline truyền thống._

- [ ] **[ENV-008] Đóng gói Gymnasium-compatible Trading Environment (P0)**
  - _Tệp cần tạo:_ `src/env/trading_env.py`, `tests/test_trading_env.py`
  - _Nội dung:_ Cài đặt chuẩn Gymnasium: `reset()`, `step(action)`, `observation_space`, `action_space`.

- [ ] **[BACKTEST-001] Bộ máy Backtest xác định (Deterministic Backtester) (P0)**
  - _Tệp cần tạo:_ `src/evaluation/backtester.py`
  - _Nội dung:_ Lặp qua từng ngày trong tập Test: ghi nhận Portfolio Value, Return, Phí giao dịch, Drawdown theo thời gian.

- [ ] **[BASE-001] Triển khai Baseline Giữ tiền mặt (Cash Baseline) (P0)**
  - _Tệp cần tạo:_ `src/evaluation/baselines/cash.py`
  - Giữ 100% Cash để làm mốc so sánh tối thiểu (Sanity check).

- [ ] **[BASE-002] Triển khai Baseline Danh mục đều tay (Equal Weight 1/N) (P0)**
  - _Tệp cần tạo:_ `src/evaluation/baselines/equal_weight.py`
  - Mỗi cổ phiếu phân bổ đều $1/N$ tỷ trọng (ví dụ 20 mã thì mỗi mã 5%), tự động tái cân bằng hàng ngày.

- [ ] **[BASE-003] Triển khai Baseline Mua và Nắm giữ (Buy & Hold) (P0)**
  - _Tệp cần tạo:_ `src/evaluation/baselines/buy_and_hold.py`
  - Phân bổ đều lúc đầu kỳ, sau đó để mặc cho giá tự trôi mà không rebalance.

- [ ] **[BASE-004] Cùng Thành viên B lập trình và huấn luyện Single-Agent PPO (P0 - Chung A+B)**
  - _Tệp cùng thực hiện:_ `src/training/single_agent_ppo.py`, `scripts/train_single_agent.py`
  - _Phần việc của A:_
    - Kết nối Trading Environment với thuật toán PPO.
    - Thiết kế hàm Reward động (thưởng Return, phạt Transaction Cost và Volatility).
    - Theo dõi đồ thị học tập (learning curve) và phân tích hành vi đặt trọng số của Agent.
    - Cùng B tinh chỉnh (tune) siêu tham số: `learning_rate`, `clip_range`, `gamma`, `entropy_coef`.

> 🏆 **Definition of Done (DoD) Sprint 5:** Một command duy nhất chạy end-to-end: Dữ liệu processed $\rightarrow$ Trading Environment $\rightarrow$ PPO (A+B) $\rightarrow$ Backtest $\rightarrow$ Xuất bảng metrics so sánh với Equal Weight & Cash. Không chỉ là từng module chạy riêng lẻ!

---

### 🏁 SPRINT 6: Bộ chỉ số đo lường hiệu quả danh mục (Financial Metrics)

_Mục tiêu: Đo lường chính xác hiệu quả đầu tư so với mức độ rủi ro gánh chịu._

- [ ] **[EXP-001] Cài đặt đầy đủ các chỉ số tài chính chuẩn (P0)**
  - _Tệp cần tạo:_ `src/evaluation/metrics.py`, `tests/test_metrics.py`
  - _Các chỉ số bắt buộc:_
    1. **Cumulative Return:** Tổng tỷ suất sinh lời toàn kỳ.
    2. **Annualized Return:** Lợi nhuận quy năm.
    3. **Annualized Volatility:** Biến động danh mục quy năm ($\sigma_{daily} \times \sqrt{252}$).
    4. **Sharpe Ratio:** Tỷ suất sinh lời có điều chỉnh theo rủi ro.
    5. **Maximum Drawdown (MDD):** Mức sụt giảm tài khoản sâu nhất từ đỉnh.
    6. **Calmar Ratio:** Tỷ số giữa Annualized Return và MDD.
    7. **Turnover & Total Costs:** Tỷ lệ đảo danh mục và tổng chi phí phát sinh.

- [ ] **[MARL-001] Thống nhất ranh giới phân chia Sector với B (P0 - Chung)**
  - Cung cấp danh sách phân nhóm cổ phiếu cho từng sector agent.

- [ ] **[MARL-006] Đánh giá mô hình MARL sơ bộ (P0 - Chung)**
  - Dùng Backtester chạy đánh giá so sánh giữa Baseline truyền thống và mô hình MARL do B train.

---

### 🏁 SPRINT 7: Phát triển High-Level Agent & Ghép nối Phân cấp

_Mục tiêu: Làm chủ Agent vĩ mô phân bổ vốn vào các nhóm ngành và tiền mặt, sau đó ghép nối toàn diện._

- [ ] **[HMARL-001] Xây dựng Trạng thái Quan sát cho High-Level Agent (P0)**
  - _Tệp cần tạo:_ `src/agents/high_level.py`
  - _Nội dung:_ Vector quan sát vĩ mô gồm: Trạng thái tổng hợp thị trường (Global Market Embedding), mức độ biến động rủi ro toàn danh mục, số dư tiền mặt hiện tại và drawdown hiện hành.

- [ ] **[HMARL-002] Xây dựng Chính sách Phân bổ Vốn Vĩ mô (Macro Policy) (P0)**
  - _Tệp cần tạo:_ `src/agents/high_level.py`
  - _Nội dung:_ Agent học chính sách phân bổ ngân sách cho từng nhóm ngành và tiền mặt:
    $$\text{Budget}_{Tech}, \text{Budget}_{Finance}, \text{Budget}_{Healthcare}, \text{Budget}_{Energy}, \text{Ratio}_{Cash}$$
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

- [ ] **[RESEARCH-002] Kiểm toán toàn diện Bias & Rò rỉ dữ liệu (P0)**
  - _Nội dung:_ Soát xét toàn bộ mã nguồn để đảm bảo:
    - Không bị Look-ahead Bias (tại thời điểm $t$ không dùng giá của ngày $t+1$).
    - Không bị Data Leakage khi chuẩn hóa dữ liệu.
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
  - Trình bày bảng so sánh Sharpe, MDD, Return, Calmar và các biểu đồ phân tích.

- [ ] **[FINAL-001] Tái lập kết quả từ môi trường sạch (P0 - Chung)**
  - Clone repo ra một máy/môi trường mới, chạy script từ bước tải data đến xuất báo cáo kết quả xem có lỗi nào không.

# H-MARL-GNN: BẢNG ĐIỀU PHỐI VÀ ĐỒNG BỘ CÔNG VIỆC CHO NHÓM 2 NGƯỜI (BẢN CHÍNH THỨC)

> **Định hướng chiến lược:**
>
> - Phân công theo mô hình **70/30 (Primary Owner + Secondary Learner)**: Cả hai thành viên đều làm chủ thuật toán Học tăng cường (RL) và hiểu toàn bộ luồng hệ thống.
> - Tách rõ **CORE (Bắt buộc)** và **ADVANCED (Nâng cao)** để tránh quá tải vào cuối đồ án.
>
> ⚖️ **Phân định đóng góp khoa học (Scientific Contributions):**
> - **Thành viên A** chịu trách nhiệm chính cho **Financial RL & Hierarchical Allocation Contribution** (Trading Environment, Reward Formulation, PPO Baseline, High-Level Macro Policy, Backtesting Framework).
> - **Thành viên B** chịu trách nhiệm chính cho **Graph-learning Contribution** (Dynamic Financial Graph, GCN/GAT, Low-Level Sector Agents, Graph Communication).
> - Cả hai thành viên **đồng sở hữu Overall Scientific Contribution** của đề tài. Khi bảo vệ và phỏng vấn, mỗi người đều có câu chuyện AI độc lập, sâu sắc và thuyết phục.
>
> 📌 _Tài liệu chi tiết riêng của từng người & Quy chế học tập:_
>
> - Chi tiết của A: [NHIEM_VU_THANH_VIEN_A.md](file:///home/tuan/AI/AI_roject1/manual/NHIEM_VU_THANH_VIEN_A.md)
> - Chi tiết của B: [NHIEM_VU_THANH_VIEN_B.md](file:///home/tuan/AI/AI_roject1/manual/NHIEM_VU_THANH_VIEN_B.md)
> - **Quy chế 70/30 & Lộ trình 5 Level học AI:** [QUY_CHE_DONG_BO_VA_HOC_AI_70_30.md](file:///home/tuan/AI/AI_roject1/manual/QUY_CHE_DONG_BO_VA_HOC_AI_70_30.md)

---

## 1. Bản đồ phân chia Module trong Source Code

```text
src/
├── data/           👉 THÀNH VIÊN A (Downloader, Validator, Alignment)
├── features/       👉 THÀNH VIÊN A (Return, Volatility, RSI, MACD, Volume)
├── env/            👉 THÀNH VIÊN A (Portfolio State, Return, Transaction Cost, Gym Env)
├── evaluation/     👉 THÀNH VIÊN A (Backtester, Sharpe, MDD, Turnover, Metrics)
│
├── graph/          👉 THÀNH VIÊN B (Rolling Correlation, Dynamic Graph, Heterogeneous)
├── models/         👉 THÀNH VIÊN B (GCN, GAT, Feature Encoders, Actor-Critic Adapters)
│
├── agents/
│   ├── high_level.py       👉 THÀNH VIÊN A (High-Level Macro & Sector Allocator)
│   ├── combiner.py         👉 THÀNH VIÊN A (Hierarchical Weight Combiner)
│   ├── low_level.py        👉 THÀNH VIÊN B (Low-Level Stock Picking Sector Agents)
│   └── communication.py    👉 THÀNH VIÊN B (GAT Inter-agent Communication)
│
├── training/
│   ├── single_agent_ppo.py 🤝 CHUNG A + B (Pair-programming huấn luyện PPO đầu tiên)
│   ├── checkpoint.py       👉 THÀNH VIÊN B (Logger & Checkpoints)
│   ├── train_marl.py       👉 THÀNH VIÊN B (Multi-Agent Sector Loop)
│   └── train_hmarl.py      🤝 CHUNG A + B (Tích hợp vòng lặp huấn luyện phân cấp)
│
├── utils/          🤝 DÙNG CHUNG (Logger, Seed, Visualization, Config Loader)
└── configs/        🤝 DÙNG CHUNG (assets.yaml, env.yaml, model.yaml)
```

---

## 2. Phân tầng Scope & 3 Mốc Dừng Chiến Lược (Stopping Milestones)

Toàn bộ đồ án được phân định ranh giới rõ ràng giữa **Phần bắt buộc** và **Phần mở rộng**:

```text
CORE — BẮT BUỘC HOÀN THÀNH (ĐỦ BẢO VỆ XUẤT SẮC)
│
├── Data Pipeline & Feature Engineering
├── Trading Environment & Deterministic Backtester
├── Baseline Cash, Equal Weight (1/N), Buy & Hold
├── Single-Agent PPO (Baseline RL)
├── Rolling Correlation & Dynamic Graph Builder
├── Graph Convolutional Network (GCN)
├── Sector Multi-Agent RL
├── Hierarchical Multi-Agent RL (High-Level + Low-Level)
└── Experiments & Core Ablation Matrix
       │
       ▼ (Nếu còn thời gian trước hạn nộp)
ADVANCED — NGHIÊN CỨU MỞ RỘNG (HƯỚNG TỚI BÀI BÁO KHOA HỌC)
│
├── ADV-GRAPH-001..004: True Advanced Heterogeneous Graph
│   (Company + Sector + Institution + Macro + Supply-chain...)
├── ADV-GNN-001: Heterogeneous GNN / Hetero-GAT
├── HMARL-005: Cross-Agent GAT Communication
└── ADV-EXP-001: Advanced Graph Ablation
```

### 🎯 3 MỐC DỪNG CHIẾN LƯỢC

| Mốc Dừng  | Mục tiêu                                               | Phạm vi kỹ thuật hoàn thành                                                                                                                                                                                        | Ý nghĩa nghiệm thu                                                                                                       |
| :-------: | :----------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------- |
| **MỐC A** | **Đủ Hoàn thành / Bảo vệ tốt**<br>_(Nền tảng Core)_    | • Dynamic Correlation Graph<br>• GCN Encoder<br>• Sector MARL<br>• Hierarchical MARL<br>• Backtesting & Core Ablation                                                                                              | **Đủ điều kiện hoàn thành và bảo vệ tốt đồ án.** Hệ thống chứng minh trọn vẹn giá trị của phân cấp H-MARL và đồ thị tương quan. Điểm số thực tế phụ thuộc chất lượng mã nguồn, độ sâu thực nghiệm và phần phản biện trước hội đồng. |
| **MỐC B** | **Mục tiêu Xuất sắc**<br>_(Điểm cao / A+)_             | • Tất cả Mốc A<br>• **GAT Encoder** (Graph Attention Network)<br>• **GRAPH-005:** Multi-relation Graph V1 (`correlation` + `same_sector`)<br>• Đa hạt giống (5 Seeds: Mean $\pm$ Std)<br>• Phân tích 3 Chế độ thị trường (Bull / Bear / High-Volatility)<br>• Báo cáo kiểm toán Leakage & Bias | **Mục tiêu xuất sắc.** Đồ thị đa quan hệ V1 kết hợp cơ chế Attention (GAT) và đánh giá thống kê đa hạt giống (5 seeds) giúp củng cố vững chắc luận điểm khoa học khi phản biện. |
| **MỐC C** | **Research Extension**<br>_(Hướng tới xuất bản Paper)_ | • Tất cả Mốc B<br>• **ADV-GRAPH-\*:** True Advanced Heterogeneous Graph (Company + Sector + Institution + Macro + Supply-chain)<br>• Hetero-GNN / Hetero-GAT (`ADV-GNN-001`)<br>• Cross-Agent GAT Communication (`HMARL-005`)<br>• Advanced Graph Ablation (`ADV-EXP-001`) | **Đạt chuẩn công bố khoa học.** Mô hình khai thác triệt để đồ thị tri thức không đồng nhất toàn diện và cơ chế truyền tin liên tác tử.             |

---

## 3. Bảng điều phối 10 Sprint (Gắn nhãn CORE / ADVANCED)

| Sprint                                               | Thành viên A (Data, Env & High-Level)                                                                                                                                                                                                                                                                                                                                 | Thành viên B (Graph, GNN & Low-Level)                                                                                                                                                                                                                                                                                                                       | Điểm đồng bộ (Chốt chung)                                                                                                             | Definition of Done (DoD - Nghiệm thu Sprint)                                                                                                                                                                                                                                    |
| :--------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Sprint 1**<br>_(Nền tảng & Dữ liệu thô)_           | • **SETUP-001:** Hoàn thiện skeleton `[CORE]`<br>• **DATA-001:** Chọn 20–30 mã cổ phiếu `[CORE]`<br>• **DATA-002:** Viết module tải OHLCV `[CORE]`                                                                                                                                                                                                                    | • **SETUP-001:** Virtual env & pytest `[CORE]`<br>• **CONFIG-001:** Hệ thống cấu hình YAML `[CORE]`<br>• **TEST-001:** Synthetic market fixture `[CORE]`                                                                                                                                                                                                    | 🤝 **SETUP-001 (A+B):** Đã pass tests nền tảng<br>🤝 **CONTRACT-001:** Thống nhất định dạng Tensor                                    | • **Code DoD:** `python -m pytest` PASSED.<br>• Chạy `python scripts/download_data.py` tải đủ 20–30 mã vào `data/raw/`.<br>• **Knowledge DoD:** Cả 2 nắm vững cấu trúc repo và contract.                                                                                        |
| **Sprint 2**<br>_(Làm sạch & Graph Prototype)_       | • **DATA-003:** Kiểm tra dữ liệu hợp lệ `[CORE]`<br>• **DATA-004:** Đồng bộ ngày giao dịch `[CORE]`<br>• **DATA-005:** Lưu metadata nguồn `[CORE]`<br>• **FEAT-001:** Tính Daily Return `[CORE]`<br>• **FEAT-002:** Tính Rolling Volatility `[CORE]`<br>• **FEAT-003:** Tính RSI-14 `[CORE]`                                                                          | • **GRAPH-001:** Ma trận tương quan trượt `[CORE]`<br>• **GRAPH-002:** Đồ thị tương quan `[CORE]`<br>_(B dùng synthetic fixture từ TEST-001 trước, sau đó mới nối dữ liệu Return thật khi A bàn giao)_                                                                                                                                                      | 🤝 A bàn giao chuỗi Daily Return sạch để B kiểm chứng trên dữ liệu thật                                                               | • **Code DoD:** Toàn bộ mã có chung Index ngày, không NaN.<br>• Ma trận correlation tính đúng sai số và tạo được `edge_index`, `edge_weight`.<br>• **Knowledge DoD:** B dạy A về Pearson Correlation matrix.                                                                    |
| **Sprint 3**<br>_(Pipeline đặc trưng & Đồ thị động)_ | • **FEAT-004:** Tính MACD `[CORE]`<br>• **FEAT-005:** Chuẩn hóa Volume `[CORE]`<br>• **FEAT-006:** Gộp Feature Pipeline `[CORE]`<br>• **SPLIT-001:** Chia Train/Val/Test `[CORE]`<br>• **SPLIT-002:** Chuẩn hóa (fit Train) `[CORE]`                                                                                                                                  | • **LEAK-001:** Test chặn rò rỉ dữ liệu `[CORE]`<br>• **GRAPH-003:** Sector Graph `[CORE]`<br>• **GRAPH-004:** Dynamic Graph Builder `[CORE]`<br>• **GRAPH-005:** Multi-relation Graph V1 (correlation + same_sector) `[MỐC B]`                                                                                                                              | 🤝 B chạy bộ kiểm toán Leakage trên dataset do A chuẩn bị                                                                             | • **Code DoD:** Sinh ra `data/processed/features.parquet`.<br>• `pytest tests/test_data_leakage.py` PASSED 100%.<br>• **Knowledge DoD:** A giải thích cho B nguyên tắc chặn look-ahead bias.                                                                                    |
| **Sprint 4**<br>_(Môi trường Trading & Mạng GNN)_    | • **ENV-001:** Trạng thái Portfolio `[CORE]`<br>• **ENV-002:** Lợi nhuận danh mục `[CORE]`<br>• **ENV-003:** Ràng buộc tỷ trọng `[CORE]`<br>• **ENV-004:** Chi phí giao dịch `[CORE]`<br>• **ENV-005:** Theo dõi Drawdown/MDD `[CORE]`<br>• **ENV-006:** Đo lường rủi ro `[CORE]`<br>• **ENV-007:** Thiết kế hàm Reward V1 `[CORE]`<br>• _(A học GCN & viết Toy GCN)_ | • **GNN-001:** GCN Encoder `[CORE]` (Primary: B, Secondary: A)<br>• **GNN-003:** GAT Encoder `[MỐC B]` (Primary: B, Secondary: A)                                                                                                                                                                                                                           | 🤝 **GNN-002:** Hai bên phối hợp test tích hợp GNN nhận dữ liệu thực từ Feature Pipeline                                              | • **Code DoD:** GCN forward pass thành công với real features.<br>• Các hàm tài chính của Env đạt test sai số $< 10^{-7}$.<br>• **Knowledge DoD (Teach-back):** B dạy A Message Passing; A dạy B hàm Reward.                                                                    |
| **Sprint 5**<br>_(End-to-End Backtester & PPO)_      | • **ENV-008:** Gymnasium Trading Env `[CORE]`<br>• **BACKTEST-001:** Bộ máy Backtest xác định `[CORE]`<br>• **BASE-001:** Baseline Cash `[CORE]`<br>• **BASE-002:** Baseline Equal Weight `[CORE]`<br>• **BASE-003:** Baseline Buy & Hold `[CORE]`<br>• **BASE-004: Single-Agent PPO `[CORE]`** (Primary: A, Secondary: B)                                            | • **MODEL-001:** Adapter Obs/Action `[CORE]` (Primary: B)<br>• **TRAIN-001:** Checkpoint & TensorBoard `[CORE]`<br>• **BASE-004:** Hỗ trợ cấu trúc Actor-Critic & Loss `[CORE]` (Secondary: B)                                                                                                                                                              | 🤝 **BASE-004:** Cùng nghiệm thu mốc chạy end-to-end PPO đầu tiên                                                                     | • **Code DoD:** **Chạy một lệnh duy nhất:** Huấn luyện PPO trên train set $\rightarrow$ Backtest trên test set $\rightarrow$ Xuất bảng metrics so sánh với Equal Weight & Cash.<br>• **Knowledge DoD:** Cả 2 giải thích được Actor $\pi(a\|s)$, Critic $V(s)$ và PPO clip loss. |
| **Sprint 6**<br>_(Dynamic GNN & Sector MARL)_        | • **MARL-002:** Observation theo Sector `[CORE]` (Primary: A)<br>• **MARL-003:** Action nội bộ Sector `[CORE]` (Primary: A)<br>• **MARL-004:** Multi-Agent Env Wrapper `[CORE]` (Primary: A)<br>• **EXP-001:** Bộ đo lường tài chính `[CORE]`                                                                                                                         | • **GNN-004:** Nạp Dynamic Graph theo ngày $t$ `[CORE]`<br>• **MARL-005:** Vòng lặp huấn luyện MARL `[CORE]` (Primary: B, Secondary: A)                                                                                                                                                                                                                     | 🤝 **MARL-001:** Thống nhất ranh giới Sector cho các Agent<br>🤝 **MARL-006:** Đánh giá so sánh MARL với Single-Agent PPO             | • **Code DoD:** Huấn luyện đa tác tử theo sector hoàn tất không lỗi OOM; Backtest MARL vs Single-Agent PPO.<br>• **Knowledge DoD:** Cả 2 giải thích được tại sao chia sector giải quyết được non-stationarity.                                                                  |
| **Sprint 7**<br>_(Hierarchical MARL - Phân cấp)_     | • **HMARL-001:** State High-Level `[CORE]` (Primary: A)<br>• **HMARL-002:** Macro Policy `[CORE]` (Primary: A)<br>• **HMARL-003:** Ghép tỷ trọng Combiner `[CORE]` (Primary: A)                                                                                                                                                                                       | • **HMARL-004:** Vòng lặp huấn luyện H-MARL `[CORE]` (Primary: B, Secondary: A)<br>• **HMARL-005:** GAT Communication `[MỐC C - ADVANCED]` (Primary: B, Secondary: A)                                                                                                                                                                                       | 🤝 **HMARL-004:** Ghép High-Level Agent (A) và Low-Level Agents (B) vào vòng lặp huấn luyện H-MARL chung                              | • **Code DoD:** Mô hình phân cấp 2 tầng học được chính sách phân bổ vốn. Tỷ trọng $\sum w_i + w_{cash} = 1$ và $w \ge 0$.<br>• **Knowledge DoD:** Cả 2 vẽ và giải thích được toàn bộ sơ đồ phân cấp từ vĩ mô đến vi mô.                                                         |
| **Sprint 8**<br>_(Ablation Study)_                   | • **EXP-006:** Feature Ablation `[CORE]`<br>• **EXP-009:** Phân tích độ nhạy phí `[MỐC B]`                                                                                                                                                                                                                                                                            | • **EXP-002:** CLI chạy thí nghiệm tự động `[CORE]`<br>• **EXP-003:** 5 random seeds (Mean $\pm$ Std) `[MỐC B]`<br>• **EXP-005:** Graph Ablation (No vs Static vs Dynamic) `[CORE]`<br>• **EXP-007:** Hierarchy Ablation (Single vs MARL vs H-MARL) `[CORE]`<br>• **EXP-008:** GCN vs GAT `[MỐC B]`<br>• **ADV-EXP-001:** Advanced Graph Ablation `[MỐC C]` | 🤝 **EXP-004:** Chốt bảng kết quả so sánh tổng thể giữa các mô hình                                                                   | • **Code DoD:** `python scripts/run_experiments.py` tự động xuất đầy đủ bảng kết quả Ablation.<br>• **Knowledge DoD:** Cả 2 nắm rõ kết luận khoa học của từng thí nghiệm đối chứng.                                                                                             |
| **Sprint 9**<br>_(Kiểm toán khoa học & Độ vững)_     | • **EXP-011:** Bảng số liệu & Đồ thị trực quan `[CORE]`<br>• **RESEARCH-002:** Kiểm toán Leakage & Bias `[CORE]`                                                                                                                                                                                                                                                      | • **RESEARCH-003:** Model Registry & Artifacts `[CORE]`<br>• **ADV-GRAPH-001..004:** True Heterogeneous Graph (Supply-chain / Ownership / Macro) `[MỐC C - ADVANCED]`                                                                                                                                                                                        | 🤝 **EXP-010:** Đánh giá qua 3 chế độ thị trường `[MỐC B]`<br>🤝 **RESEARCH-001:** Kiểm toán so sánh công bằng với Baselines `[CORE]` | • **Code DoD:** `experiments/registry.json` lưu đủ mã commit, config, seed, weights; báo cáo xác nhận 0 lỗi leakage.<br>• **Knowledge DoD:** Cả 2 sẵn sàng trả lời phản biện về độ tin cậy của kết quả.                                                                         |
| **Sprint 10**<br>_(Báo cáo & Tái lập Nghiên cứu)_    | • **DOC-001:** Tài liệu kiến trúc hệ thống `[CORE]`<br>• **DOC-003:** Thiết lập thí nghiệm & kết quả `[CORE]`                                                                                                                                                                                                                                                         | • **DOC-002:** Phương pháp luận mô hình toán & RL `[CORE]`                                                                                                                                                                                                                                                                                                  | 🤝 **DOC-004:** Tổng kết giới hạn đề tài `[CORE]`<br>🤝 **FINAL-001:** Tái lập thành công 100% từ môi trường sạch `[CORE]`            | • **Code DoD:** Tái lập thành công 100% từ môi trường sạch.<br>• **Knowledge DoD:** Cả 2 vượt qua bài khảo thí 10 câu hỏi bảo vệ đồ án.                                                                                                                                         |

---

## 4. Bản giao kèo kỹ thuật (4 Interface Contract linh hoạt)

1. **Feature Input (A bàn giao cho B):** Tensor `[T, N, F]` (Thời gian $\times$ Số mã $\times$ Số đặc trưng).
2. **Dynamic Graph (B xây dựng):** `node_features` `[N, F]`, `edge_index` `[2, E]`, `edge_weight` `[E]`.
3. **Observation Contract:**
   $$\text{Observation}_t = [\text{market\_state}_t, \; \text{portfolio\_weights}_t, \; \text{cash\_ratio}_t]$$
   Hỗ trợ cả `raw_features` `[N, F]` (cho Single-Agent PPO) và `graph_embeddings` `[N, D]` (cho GNN).
4. **Action Output:** Tensor `weights` kích thước `[N + 1]`, thỏa mãn $\sum_{i=1}^{N} w_i + w_{cash} = 1$ và $w \ge 0$.

---

## 5. Nguyên tắc nghiên cứu bất biến (The Ablation Backbone)

$$\text{Equal Weight (1/N)} \longrightarrow \text{Single-Agent PPO} \longrightarrow \text{GCN / GAT} \longrightarrow \text{Sector MARL} \longrightarrow \text{Hierarchical MARL} \longrightarrow \text{H-MARL + GAT}$$

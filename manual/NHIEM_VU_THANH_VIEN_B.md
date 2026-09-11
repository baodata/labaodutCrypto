# H-MARL-GNN: KẾ HOẠCH HÀNH ĐỘNG RIÊNG CHO THÀNH VIÊN B

### VAI TRÒ: GRAPH AI & MULTI-AGENT LEAD

_(DYNAMIC GRAPH + GNN/GAT + PPO BASELINE + LOW-LEVEL SECTOR MARL + GAT COMMUNICATION)_

> **Mô tả vai trò:**
> Bạn là người chịu trách nhiệm về phần **trí tuệ nhân tạo đồ thị và hệ thống đa tác tử**: xây dựng **Đồ thị tài chính động (Dynamic Financial Graph)**, mạng **GCN/GAT** để trích xuất embedding, cùng A làm chủ thuật toán **PPO**, phát triển **Low-Level Sector Agents** và **Cơ chế giao tiếp liên tác tử qua Graph Attention (GAT Communication)**.
>
> ⚖️ **Phân định đóng góp khoa học (Scientific Contributions):**
>
> - **Thành viên B** chịu trách nhiệm chính cho **Graph-learning Contribution** (Dynamic Financial Graph, GCN/GAT, Low-Level Sector Agents, Graph Communication).
> - **Thành viên A** chịu trách nhiệm chính cho **Financial RL & Hierarchical Allocation Contribution** (Trading Environment, Reward Formulation, PPO Baseline, High-Level Macro Policy, Backtesting Framework).
> - Cả hai thành viên **đồng sở hữu Overall Scientific Contribution** của đề tài. Khi bảo vệ đồ án và phỏng vấn, mỗi thành viên đều có câu chuyện AI độc lập, sâu sắc và thuyết phục: B đại diện cho **Graph Representation Learning & Multi-Agent Coordination**, A đại diện cho **Financial RL & Hierarchical Decision Systems**.
>
> 📖 **Quy chế 70/30 & Lộ trình 5 Level học AI:** Xem chi tiết tại [QUY_CHE_DONG_BO_VA_HOC_AI_70_30.md](file:///home/tuan/AI/AI_roject1/manual/QUY_CHE_DONG_BO_VA_HOC_AI_70_30.md).
> 🤝 **Bảng điều phối chung với A:** Xem tại [BANG_DIEU_PHOI_2_NGUOI.md](file:///home/tuan/AI/AI_roject1/manual/BANG_DIEU_PHOI_2_NGUOI.md).

---

## 1. Các thư mục do Thành viên B chịu trách nhiệm trực tiếp

```text
src/
├── graph/          # Xây dựng ma trận tương quan rolling, đồ thị quan hệ động
├── models/         # Các mạng Graph Neural Network: GCN, GAT, Feature Encoder, Adapter
│
├── agents/
│   ├── low_level.py        # Các tác tử chọn cổ phiếu chi tiết trong từng Sector
│   └── communication.py    # Cơ chế giao tiếp liên tác tử qua Graph Attention
│
└── training/
    ├── single_agent_ppo.py # Đồng sở hữu huấn luyện PPO Baseline cùng A
    ├── checkpoint.py       # Bộ ghi log TensorBoard & Checkpoints
    ├── train_marl.py       # Vòng lặp huấn luyện Multi-Agent Sector
    └── train_hmarl.py      # Tích hợp vòng lặp huấn luyện phân cấp cùng A
```

---

## 2. Phân tầng Scope cá nhân & 3 Mốc Dừng

- **🎯 MỐC A (Đủ hoàn thành / Bảo vệ tốt):** Hoàn thành toàn bộ các ticket đánh dấu `[CORE]` từ Sprint 1 đến Sprint 10 (Correlation Graph, Dynamic Graph, GCN Encoder, Single-Agent PPO cùng A, Low-Level Sector Agents, H-MARL Training Loop cùng A, Backtest & Ablation). Đủ điều kiện hoàn thành và bảo vệ tốt đồ án.
- **🏆 MỐC B (Mục tiêu Xuất sắc):** Hoàn thành thêm các module nâng cao: GAT Encoder (`GNN-003`), Multi-relation Graph V1 (`GRAPH-005`), Dynamic GNN training (`GNN-004`), GCN vs GAT Ablation (`EXP-008`), 5 Random Seeds (`EXP-003`), Phân tích 3 chế độ thị trường (`EXP-010`), Model Registry (`RESEARCH-003`).
- **🚀 MỐC C (Research Extension — Mở rộng nghiên cứu Paper):** Khai phá đồ thị tri thức không đồng nhất toàn diện: True Advanced Heterogeneous Graph (`ADV-GRAPH-001..004`: Supply-chain, Ownership, Macro nodes), Hetero-GNN/Hetero-GAT (`ADV-GNN-001`), Cross-Agent GAT Communication (`HMARL-005`), và Advanced Graph Ablation (`ADV-EXP-001`).

---

## 3. Checklist công việc chi tiết theo từng Sprint

### 🏁 SPRINT 1: Thiết lập Hệ thống Cấu hình & Hạ tầng Test

_Mục tiêu: Đảm bảo code chạy qua config (không hard-code tham số) và có khung test tự động._

- [x] **[SETUP-001] Khởi tạo Repository & Skeleton nền tảng (P0 - Chung A+B) `[CORE]`**
  - _Đã hoàn thành ở Epic 0:_ Tạo cấu trúc thư mục, môi trường ảo, requirements.txt, pyproject.toml, chạy pass 4/4 tests ban đầu.

- [x] **[CONFIG-001] Xây dựng hệ thống cấu hình YAML chuẩn (P0)**
  - _Tệp cần tạo:_ `configs/env.yaml`, `configs/model.yaml`, `src/utils/config.py`
  - _Nội dung:_ Viết hàm đọc và validate file cấu hình YAML. Tuyệt đối không để các tham số (learning rate, window size, threshold, transaction fee...) bị hard-code trong mã nguồn.
  - _Người review:_ Thành viên A.

- [x] **[TEST-001] Thiết lập hạ tầng kiểm thử và dữ liệu giả lập (Fixture) (P0)**
  - _Tệp cần tạo:_ `tests/conftest.py`
  - _Nội dung:_ Tạo synthetic market data fixture (dữ liệu thị trường giả định) trong pytest để có thể chạy test các thuật toán ngay cả khi Thành viên A chưa tải xong toàn bộ dữ liệu thật.

- [x] **[CONTRACT-001] Thống nhất giao diện dữ liệu linh hoạt (Data Contract) với A (P0 - Chung)**
  - Chốt cấu trúc dữ liệu đầu vào: Tensor `[T, N, F]`.
  - Chốt định dạng Observation của Môi trường hỗ trợ 2 chế độ:
    - Chế độ 1: `raw_features` `[N, F]` (cho Single-Agent PPO chạy độc lập không cần GNN).
    - Chế độ 2: `graph_embeddings` `[N, D]` (cho các mô hình tích hợp GNN/GAT).

---

### 🏁 SPRINT 2: Xây dựng Đồ thị Tương quan cơ bản (Correlation Graph)

_Mục tiêu: Chuyển đổi dữ liệu chuỗi thời gian thành biểu diễn Đồ thị (Graph)._

- [ ] **[GRAPH-001] Tính ma trận tương quan trượt (Rolling Correlation) (P0)**
  - _Tệp cần tạo:_ `src/graph/correlation.py`, `tests/test_correlation.py`
  - _Lưu ý quan trọng:_ Trước khi A bàn giao dữ liệu Return thật, B dùng **synthetic fixture từ TEST-001** để phát triển và test thuật toán. Sau đó mới nối với Return thật từ A.
  - _Nội dung:_ Sử dụng chuỗi lợi nhuận (Return) để tính Pearson correlation trong cửa sổ trượt (ví dụ: 60 ngày).
  - _Quy tắc an toàn:_ Chỉ dùng dữ liệu $\le t$, tuyệt đối không nhìn trước dữ liệu tương lai.

- [ ] **[GRAPH-002] Xây dựng đồ thị tương quan tĩnh/ngưỡng (P0)**
  - _Tệp cần tạo:_ `src/graph/builder.py`
  - _Nội dung:_ Tạo cấu trúc đồ thị từ ma trận tương quan:
    - Nếu $|Corr(i, j)| \ge \text{threshold}$ (ví dụ $0.6$): Tạo cạnh kết nối giữa asset $i$ và asset $j$.
    - Trọng số cạnh (`edge_weight`): Giá trị tương quan.
  - _Đầu ra:_ `edge_index` kích thước `[2, E]` và `edge_weight` kích thước `[E]`.

---

### 🏁 SPRINT 3: Chặn rò rỉ dữ liệu & Xây dựng Đồ thị Động (Dynamic Graph)

_Mục tiêu: Đồ thị tài chính thay đổi linh hoạt theo từng ngày giao dịch (Dynamic & Heterogeneous)._

- [ ] **[LEAK-001] Viết bài test tự động chặn rò rỉ dữ liệu (Data Leakage Tests) (P0)**
  - _Tệp cần tạo:_ `tests/test_data_leakage.py`
  - _Nội dung:_ Viết test tự động kiểm tra: Đồ thị tại ngày $t$ không chứa bất kỳ cạnh/trọng số nào được tính từ ngày $t+1$. Đây là bước bảo vệ uy tín học thuật của đề tài.

- [ ] **[GRAPH-003] Xây dựng đồ thị nhóm ngành (Sector Graph) (P1)**
  - _Tệp cần tạo:_ `src/graph/sector_graph.py`
  - _Nội dung:_ Tạo thêm quan hệ giữa các cổ phiếu thuộc cùng một nhóm ngành (`same_sector_edge`).

- [ ] **[GRAPH-004] Xây dựng bộ tạo đồ thị động theo thời gian (Dynamic Graph Builder) (P0)**
  - _Tệp cần tạo:_ `src/graph/dynamic_graph.py`
  - _Nội dung:_ Với mỗi bước thời gian $t$, module trả về đồ thị tài chính tương ứng của ngày hôm đó: `Graph_t = (Nodes_t, Edges_t)`.

- [ ] **[GRAPH-005] Đồ thị đa quan hệ Multi-relation Graph V1 (P1) `[MỐC B]`**
  - _Tệp cần tạo:_ `src/graph/multi_relation_graph.py` (hoặc `hetero_graph.py`)
  - _Nội dung:_ Tích hợp 2 loại quan hệ trên cùng một đồ thị: Cạnh tương quan giá (`correlation`) và cạnh cùng ngành (`same_sector`).
  - _Lưu ý học thuật:_ Đây là **Multi-relation Graph V1** ở mức cơ bản, đóng vai trò bước đệm trước khi mở rộng lên Đồ thị không đồng nhất hoàn chỉnh (**True Advanced Heterogeneous Graph** ở Mốc C). Tránh gọi nhầm V1 là đồ thị tri thức hoàn chỉnh trong báo cáo khoa học.

---

### 🏁 SPRINT 4: Xây dựng Bộ mã hóa Mạng nơ-ron đồ thị (GNN & GAT Encoders)

_Mục tiêu: Mạng GNN nhận đồ thị và trả về vector đặc trưng (Embedding) cho từng mã cổ phiếu._

- [ ] **[GNN-001] Cài đặt bộ mã hóa GCN Encoder (P0)**
  - _Tệp cần tạo:_ `src/models/gcn.py`, `tests/test_gcn.py`
  - _Kiến trúc:_ 2 lớp Graph Convolutional Network (GCN).
  - _Đầu vào:_ Node features `[N, F]` (do A chuẩn bị) và `edge_index` `[2, E]`.
  - _Đầu ra:_ Node embeddings `[N, embedding_dim]` (ví dụ: `embedding_dim = 32` hoặc `64`).

- [ ] **[GNN-002] Chạy kiểm thử tích hợp GNN với dữ liệu của Thành viên A (P0 - Chung)**
  - Đảm bảo model forward pass thành công với tensor dữ liệu thật từ Feature Pipeline mà không gặp lỗi lệch shape hay NaN.

- [ ] **[GNN-003] Cài đặt bộ mã hóa Graph Attention Network (GAT Encoder) (P1)**
  - _Tệp cần tạo:_ `src/models/gat.py`
  - _Nội dung:_ Sử dụng cơ chế Attention để học trọng số liên kết giữa các node cổ phiếu (ví dụ: NVDA sẽ chú ý nhiều hơn đến TSM và AMD so với các mã khác).

---

### 🏁 SPRINT 5: Triển khai Mô hình Học tăng cường Đơn tác tử (Single-Agent PPO Baseline)

_Mục tiêu: Chứng minh pipeline RL hoạt động trơn tru trước khi phát triển lên đa tác tử._

- [ ] **[MODEL-001] Xây dựng Adapter chuyển đổi Observation & Action (P0)**
  - _Tệp cần tạo:_ `src/models/adapter.py`
  - _Nội dung:_
    - Chuyển đổi trạng thái từ môi trường của A thành tensor đầu vào cho mạng Actor-Critic.
    - Hỗ trợ nạp trực tiếp `raw_features` `[N, F]` mà không cần qua GNN.
    - Áp dụng lớp Softmax ở đầu ra của Actor để đảm bảo tổng trọng số luôn bằng 1 và không âm:
      $$w_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

- [ ] **[BASE-004] Cùng Thành viên A lập trình và huấn luyện Single-Agent PPO (P0 - Chung A+B)**
  - _Tệp cùng thực hiện:_ `src/training/single_agent_ppo.py`, `scripts/train_single_agent.py`
  - _Phần việc của B:_
    - Thiết kế kiến trúc mạng Actor-Critic (MLP).
    - Cài đặt thuật toán PPO Clip Objective, Value Loss, Entropy Bonus.
    - Đảm bảo đầu ra qua hàm Softmax tuân thủ ràng buộc $\sum w_i + w_{cash} = 1$.
    - Cùng A huấn luyện trên `raw_features` (KHÔNG dùng GNN) để thiết lập mốc chuẩn đối chứng tối thiểu cho Ablation Study.

- [ ] **[TRAIN-001] Xây dựng bộ ghi log huấn luyện & Lưu Checkpoint (P0)**
  - _Tệp cần tạo:_ `src/training/checkpoint.py`, `src/utils/logger.py`
  - _Nội dung:_ Tự động lưu checkpoint mô hình có lợi nhuận/Sharpe tốt nhất trên tập Validation. Ghi log training loss, reward qua TensorBoard.

> 🏆 **Definition of Done (DoD) Sprint 5:** Một command duy nhất chạy end-to-end: Dữ liệu processed $\rightarrow$ Trading Environment $\rightarrow$ PPO (A+B) $\rightarrow$ Backtest $\rightarrow$ Xuất bảng metrics so sánh với Equal Weight & Cash. Không chỉ là từng module chạy riêng lẻ!

---

### 🏁 SPRINT 6: Tích hợp Dynamic GNN & Xây dựng Multi-Agent RL (MARL)

_Mục tiêu: Chia việc ra quyết định cho nhiều Agent chuyên trách theo từng nhóm ngành._

- [ ] **[GNN-004] Tích hợp Dynamic Graph vào quá trình huấn luyện theo thời gian (P1)**
  - _Nội dung:_ Tại mỗi timestep $t$, nạp đúng đồ thị của ngày $t$ vào GNN để cập nhật embedding trước khi đưa vào Agent.

- [ ] **[MARL-001] Thiết kế ranh giới và phân công Agent theo Sector (P0 - Chung)**
  - Chia hệ thống thành các Low-Level Sector Agents:
    - Agent 1: Technology Sector (AAPL, MSFT, NVDA, AMD...)
    - Agent 2: Finance Sector (JPM, BAC, GS, MS...)
    - Agent 3: Healthcare Sector (JNJ, PFE, UNH...)
    - Agent 4: Energy & Consumer Sector (XOM, CVX, AMZN...)

- [ ] **[MARL-002] Xây dựng Observation Space cho từng Low-Level Agent (P0)**
  - _Tệp cần tạo:_ `src/agents/low_level.py`
  - _Nội dung:_ Mỗi Sector Agent chỉ quan sát embedding của các mã thuộc nhóm ngành của mình + ngân sách được cấp.

- [ ] **[MARL-003] Xây dựng Action Space cho từng Low-Level Agent (P0)**
  - _Đầu ra:_ Phân bổ tỷ trọng nội bộ bên trong nhóm ngành ($\sum w_{trong\_sector} = 1$).

- [ ] **[MARL-004] Thiết lập môi trường huấn luyện Multi-Agent (P0)**
  - _Tệp cần tạo:_ `src/env/marl_wrapper.py`
  - _Nội dung:_ Nhận action từ tất cả các Sector Agents và chuyển tiếp cho môi trường của A.

- [ ] **[MARL-005] Xây dựng vòng lặp huấn luyện MARL Loop (P0)**
  - _Tệp cần tạo:_ `src/training/train_marl.py`
  - _Nội dung:_ Huấn luyện đồng thời các Sector Agents phối hợp đưa ra quyết định.

- [ ] **[MARL-006] Đánh giá so sánh MARL với Single-Agent (P0 - Chung)**
  - Chạy Backtest so sánh xem chia nhiều agent có cải thiện hiệu năng so với 1 agent xử lý toàn bộ hay không (trả lời RQ2).

---

### 🏁 SPRINT 7: Tích hợp Hệ thống Học tăng cường Phân cấp (Hierarchical MARL)

_Mục tiêu: Tích hợp High-Level Macro Policy (A) và Low-Level Sector Policies (B) thành một pipeline hoàn chỉnh mà KHÔNG PHỤ THUỘC vào GAT Communication._

#### 🔹 PHẦN CORE (BẮT BUỘC ĐỂ HOÀN THÀNH SPRINT 7):

Pipeline CORE vận hành độc lập:

```text
High-Level Agent (A: HMARL-001/002)
       ↓ (Quyết định phân bổ ngân sách ngành)
Sector Budgets
       ↓ (Cấp ngân sách cho từng nhóm ngành)
Low-Level Sector Agents (B: MARL-002/003)
       ↓ (Tối ưu tỷ trọng nội bộ từng ngành)
Asset Allocation
       ↓ (Ghép nối tỷ trọng toàn cục)
Combiner (A: HMARL-003)
       ↓
Portfolio Weights (Nạp vào Gym Trading Env)
```

- [ ] **[HMARL-004] Tích hợp Vòng lặp Huấn luyện Hierarchical MARL (P0 - Chung A+B) `[CORE]`**
  - _Tệp cùng thực hiện:_ `src/training/train_hmarl.py`, `scripts/train_hmarl.py`
  - _Quy trình phối hợp:_
    1. High-Level Agent (do A xây dựng) ra quyết định phân bổ ngân sách cho từng ngành và tỷ lệ tiền mặt.
    2. Các Low-Level Sector Agents (do B xây dựng) nhận ngân sách ngành và quyết định tỷ trọng các mã cổ phiếu trong ngành đó.
    3. Bộ Combiner (do A xây dựng) tính toán tỷ trọng danh mục đầu tư toàn cục và chuyển cho Trading Environment.
    4. Cả A và B cùng phối hợp tối ưu hóa gradient cho cả 2 tầng tác tử.

#### 🔸 PHẦN OPTIONAL / MỐC C (NGHIÊN CỨU MỞ RỘNG — KHÔNG BẮT BUỘC CHO DEFENSE):

Extension khi còn thời gian hướng tới xuất bản bài báo khoa học:

```text
Low-Level Sector Agents
       ↕
GAT Communication (HMARL-005)
       ↓
Asset Allocation
```

- [ ] **[HMARL-005] Cơ chế giao tiếp liên tác tử qua Graph Attention (P1) `[MỐC C - ADVANCED]`**
  - _Tệp cần tạo:_ `src/agents/communication.py`
  - _Nội dung:_ Cho phép các Sector Agent trao đổi thông tin ẩn (latent messages) qua mạng Graph Attention Network (GAT) trước khi chốt tỷ trọng.
  - _Lưu ý quan trọng:_ Pipeline CORE bắt buộc phải chạy độc lập hoàn hảo mà **KHÔNG CẦN** module này. `HMARL-005` chỉ là thành phần mở rộng bổ sung cho Mốc C.

---

### 🏁 SPRINT 8: Thực hiện Thí nghiệm Nghiên cứu & Ablation Studies

_Mục tiêu: Chạy các bài đo thực nghiệm chuyên sâu để chứng minh đóng góp của từng thành phần._

- [ ] **[EXP-002] Xây dựng CLI chạy thí nghiệm tự động (Experiment Runner) (P0)**
  - _Tệp cần tạo:_ `scripts/run_experiments.py`
  - _Nội dung:_ Chạy thí nghiệm tự động dựa trên file cấu hình `configs/experiment.yaml`.

- [ ] **[EXP-003] Huấn luyện đa hạt giống ngẫu nhiên (Multiple Random Seeds) (P0)**
  - _Nội dung:_ Chạy 5 random seeds (42, 123, 456, 789, 2026) cho từng mô hình để tính Mean $\pm$ Std.

- [ ] **[EXP-005] Thí nghiệm Graph Ablation (P0)**
  - So sánh: **Không dùng Graph** vs **Dùng Static Graph** vs **Dùng Dynamic Graph**.
  - Trả lời trực tiếp câu hỏi nghiên cứu RQ1 và RQ4.

- [ ] **[EXP-007] Thí nghiệm Hierarchy Ablation (P0)**
  - So sánh: **Single-Agent** vs **MARL (phẳng)** vs **H-MARL (phân cấp)**.
  - Trả lời trực tiếp câu hỏi nghiên cứu RQ2 và RQ3.

- [ ] **[EXP-008] So sánh Mạng đồ thị GCN vs GAT (P1)**
  - So sánh hiệu quả embedding giữa GCN thông thường và Graph Attention Network.
  - Trả lời câu hỏi nghiên cứu RQ5.

- [ ] **[EXP-004] Chốt bảng so sánh tổng hợp với Thành viên A (P0 - Chung)**

---

### 🏁 SPRINT 9: Kiểm toán Thuật toán & Quản lý Thí nghiệm (Registry)

_Mục tiêu: Đảm bảo khả năng tái lập và tính trung thực của các kết quả nghiên cứu._

- [ ] **[RESEARCH-001] Kiểm toán so sánh công bằng với Baseline (P0 - Chung)**
  - Kiểm tra xem các mô hình Baseline (PPO, Equal Weight) có được huấn luyện và đánh giá trên cùng điều kiện chi phí và tập dữ liệu hay không.

- [ ] **[RESEARCH-003] Xây dựng Registry quản lý mô hình & Artifacts (P1)**
  - _Tệp cần tạo:_ `experiments/registry.json`
  - _Nội dung:_ Ghi lại chính xác: Git Commit Hash, file cấu hình, seed, đường dẫn lưu model weights `.pt` và kết quả đánh giá tương ứng.

- [ ] **[EXP-010] Phân tích theo chế độ thị trường Bull / Bear / High-Volatility (P1 - Chung)**

---

### 🏁 SPRINT 10: Viết Phương pháp luận & Tái lập Nghiên cứu

_Mục tiêu: Hoàn thiện báo cáo khoa học và đóng gói source code._

- [ ] **[DOC-002] Soạn thảo phần Phương pháp luận (Methodology) cho báo cáo (P0)**
  - _Nội dung:_ Trình bày công thức toán học về Dynamic Graph, kiến trúc GAT, bài toán tối ưu phân cấp H-MARL và hàm phần thưởng.

- [ ] **[DOC-004] Tổng kết các hạn chế và rủi ro của đề tài (P0 - Chung)**
  - Viết về các hạn chế: non-stationarity, chi phí trượt giá (slippage) và tính ổn định của RL.

- [ ] **[FINAL-001] Tái lập toàn bộ đồ án từ môi trường sạch (P0 - Chung)**
  - Kiểm tra lệnh chạy toàn bộ pipeline từ training đến backtesting không phát sinh bất kỳ lỗi nào.

---

## 4. Nhiệm vụ Nghiên cứu Mở rộng (ADVANCED — Mốc C: Còn thời gian mới làm)

> **Lưu ý:** Đây là các module nghiên cứu chuyên sâu hướng tới xuất bản bài báo khoa học (Paper track). Chỉ bắt đầu khi đã hoàn thành vững chắc **Mốc A** và **Mốc B**.

- [ ] **[ADV-GRAPH-001] Xây dựng True Advanced Heterogeneous Graph (P2)**
  - _Tệp cần tạo:_ `src/graph/hetero_v2.py`
  - _Nội dung:_ Tích hợp toàn diện các loại node và quan hệ phức tạp thành đồ thị tri thức tài chính thực thụ: Company + Sector + Institution + Macro + Supply-Chain. Khác biệt rõ rệt với Multi-relation Graph V1 ở Mốc B (chỉ gồm correlation + sector).

- [ ] **[ADV-GRAPH-002] Tích hợp quan hệ Chuỗi cung ứng (Supply-Chain Relationships) (P2)**
  - _Tệp cần tạo:_ `src/graph/supply_chain.py`, `data/raw/supply_chain.json`
  - _Nội dung:_ Thu thập/định nghĩa quan hệ khách hàng - nhà cung cấp (ví dụ: NVDA $\leftrightarrow$ TSM, AAPL $\leftrightarrow$ QCOM). Tạo cạnh có hướng biểu diễn sự lan truyền rủi ro và doanh thu trong chuỗi giá trị.

- [ ] **[ADV-GRAPH-003] Tích hợp quan hệ Sở hữu tổ chức (Institutional Ownership) (P2)**
  - _Tệp cần tạo:_ `src/graph/ownership.py`
  - _Nội dung:_ Khai thác dữ liệu 13F (BlackRock, Vanguard, State Street) để xây dựng cạnh sở hữu chung: 2 cổ phiếu có tỷ lệ quỹ lớn cùng nắm giữ cao sẽ có liên kết với nhau.

- [ ] **[ADV-GRAPH-004] Thiết kế Node Vĩ mô & Ngành (Macro / Sector Nodes) (P2)**
  - _Tệp cần tạo:_ `src/graph/macro_nodes.py`
  - _Nội dung:_ Bổ sung các node trung gian đại diện cho Macro Indicators (Lãi suất Fed, Lạm phát CPI, VIX) và Sector ETF (XLK, XLF, XLV, XLE) kết nối với các node cổ phiếu tương ứng.

- [ ] **[ADV-GNN-001] Cài đặt Heterogeneous GNN / Hetero-GAT (P2)**
  - _Tệp cần tạo:_ `src/models/hetero_gat.py`
  - _Nội dung:_ Cài đặt mạng Heterogeneous Graph Attention Network (sử dụng PyTorch Geometric `HeteroConv` hoặc `HGTConv`) để học trọng số chú ý riêng biệt cho từng loại quan hệ (Correlation, Supply-Chain, Ownership, Macro).

- [ ] **[HMARL-005] Cơ chế giao tiếp liên tác tử qua Graph Attention chuyên sâu (P2)**
  - _Tệp cần tạo:_ `src/agents/communication.py`
  - _Nội dung:_ Cho phép các Sector Agent trao đổi thông tin ẩn (latent messages) qua mạng đồ thị GAT, tạo sự phối hợp hiệp đồng trước khi đưa ra quyết định danh mục cuối cùng.

- [ ] **[ADV-EXP-001] Thí nghiệm bóc tách đồ thị chuyên sâu (Advanced Graph Ablation) (P2)**
  - _Nội dung:_ Đo lường định lượng mức độ đóng góp của từng loại cạnh:
    1. Base: Dynamic Correlation Graph
    2. - Supply-Chain Edges
    3. - Ownership Edges
    4. - Macro Nodes
  - Phân tích xem loại thông tin phi giá nào mang lại Alpha vượt trội nhất cho danh mục đầu tư.

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
> **Ticket định lượng đã tích hợp vào Sprint 5–9:** [đặc tả và bảng ánh xạ](TICKET_CAI_THIEN_DINH_LUONG.md). Cập nhật 16/10/2026.

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

> **Tích hợp định lượng — cập nhật 16/10/2026:** Ticket `QNT-*` có checklist trực tiếp trong Sprint 5–9 bên dưới. Checklist ở file owner là nguồn trạng thái; [đặc tả định lượng](TICKET_CAI_THIEN_DINH_LUONG.md) giữ công thức và phụ thuộc.
> - Sprint 1–4/4.5 giữ phần nền đã ghi nhận; nâng cấp mới xếp từ Sprint 5 để contract/protocol có trước các task sử dụng chúng.
> - Trong cùng sprint, làm theo phụ thuộc và bàn giao A ↔ B. P2 là `[MỐC C — CÓ ĐIỀU KIỆN]`, không chặn nghiệm thu H-MARL core.
> - Trạng thái theo dõi hiện có được giữ lại khi chuyển mục. “Đã có core runner” là tiến độ một phần QNT-025, chưa phải nghiệm thu toàn ticket.

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
    - Chế độ 1: `raw_features` `[N, F]` là feature đã tính từ OHLCV (return, RSI, MACD...), cho Single-Agent PPO không cần GNN; không phải OHLCV thô.
    - Chế độ 2: `graph_embeddings` `[N, D]` (cho các mô hình tích hợp GNN/GAT).

---

### 🏁 SPRINT 2: Xây dựng Đồ thị Tương quan cơ bản (Correlation Graph)

_Mục tiêu: Chuyển đổi dữ liệu chuỗi thời gian thành biểu diễn Đồ thị (Graph)._

- [x] **[GRAPH-001] Tính ma trận tương quan trượt (Rolling Correlation) (P0)**
  - _Tệp đã tạo:_ `src/graph/correlation.py`, `tests/test_graph.py`
  - _Nội dung:_ Sử dụng chuỗi lợi nhuận (Return) để tính Pearson correlation trong cửa sổ trượt (20 ngày). Đã pass unit test.

- [x] **[GRAPH-002] Xây dựng đồ thị tương quan tĩnh/ngưỡng (P0)**
  - _Tệp đã tạo:_ `src/graph/builder.py`, `tests/test_graph.py`
  - _Nội dung:_ Tạo `edge_index` [2, E] và `edge_weight` [E] với ngưỡng threshold=0.5. Đã pass unit test.

---

### 🏁 SPRINT 3: Chặn rò rỉ dữ liệu & Xây dựng Đồ thị Động (Dynamic Graph)

_Mục tiêu: Đồ thị tài chính thay đổi linh hoạt theo từng ngày giao dịch (Dynamic & Heterogeneous)._

- [x] **[LEAK-001] Viết bài test tự động chặn rò rỉ dữ liệu (Data Leakage Tests) (P0)**
  - _Tệp đã tạo:_ `tests/test_data_leakage.py`
  - _Nội dung:_ Viết 3 bài test tự động kiểm tra rò rỉ dữ liệu trên returns, volatility, rsi. 3/3 tests pass.

- [x] **[GRAPH-003] Xây dựng đồ thị nhóm ngành (Sector Graph) (P1)**
  - _Tệp đã tạo:_ `src/graph/sector_graph.py`, `tests/test_graph.py`
  - _Nội dung:_ Tạo quan hệ giữa các cổ phiếu thuộc cùng một nhóm ngành (`same_sector_edge` với trọng số 1.0). Đã pass unit test.

- [x] **[GRAPH-004] Xây dựng bộ tạo đồ thị động theo thời gian (Dynamic Graph Builder) (P0)**
  - _Tệp đã tạo:_ `src/graph/dynamic_graph.py`, `tests/test_graph.py`
  - _Nội dung:_ Lớp `DynamicGraphGenerator` trả về `(edge_index, edge_weight)` của bất kỳ ngày giao dịch t nào. Đã pass unit test.

- [x] **[GRAPH-005] Đồ thị đa quan hệ Multi-relation Graph V1 (P1) `[MỐC B]`**
  - _Tệp đã tạo:_ `src/graph/multi_relation_graph.py`, `tests/test_graph.py`
  - _Nội dung:_ Lớp `MultiRelationGraphBuilder` tích hợp đồng thời cạnh tương quan động và cạnh cùng ngành tĩnh. Đã pass unit test.

---

### 🏁 SPRINT 4: Xây dựng Bộ mã hóa Mạng nơ-ron đồ thị (GNN & GAT Encoders)

_Mục tiêu: Mạng GNN nhận đồ thị và trả về vector đặc trưng (Embedding) cho từng mã cổ phiếu._

- [x] **[GNN-001] Cài đặt bộ mã hóa GCN Encoder (P0)**
  - _Tệp cần tạo:_ `src/models/gcn.py`, `tests/test_gcn.py`
  - _Kiến trúc:_ 2 lớp Graph Convolutional Network (GCN).
  - _Đầu vào:_ Node features `[N, F]` (do A chuẩn bị) và `edge_index` `[2, E]`.
  - _Đầu ra:_ Node embeddings `[N, embedding_dim]` (ví dụ: `embedding_dim = 32` hoặc `64`).

- [x] **[GNN-002] Chạy kiểm thử tích hợp GNN với dữ liệu của Thành viên A (P0 - Chung)**
  - Đảm bảo model forward pass thành công với tensor dữ liệu thật từ Feature Pipeline mà không gặp lỗi lệch shape hay NaN.

- [x] **[GNN-003] Cài đặt bộ mã hóa Graph Attention Network (GAT Encoder) (P1)**
  - _Tệp cần tạo:_ `src/models/gat.py`
  - _Nội dung:_ Sử dụng cơ chế Attention để học trọng số liên kết giữa các node cổ phiếu (ví dụ: NVDA sẽ chú ý nhiều hơn đến TSM và AMD so với các mã khác).

---

### 🏁 SPRINT 4.5: Kiên cố Kiến trúc đồ thị & GNN — Chống Market Correlation Spike

_Lý do tách sprint: Sau khi phân tích kỹ, kiến trúc hiện tại (Threshold cố định + GATv2 thuần) có nguy cơ **Over-smoothing** nghiêm trọng trong giai đoạn thị trường khủng hoảng (market correlation spike) do đồ thị tiệm cận **Fully Connected Graph**. Các ticket dưới đây là các "bản vá kiến trúc" (architectural hardening) bắt buộc trước khi đưa vào vòng lặp RL._

- [x] **[GRAPH-006] Nâng cấp Graph Builder: Thêm chế độ k-NN và Dynamic Threshold (P0) `[MỐC B]`**
  - _Tệp đã nâng cấp:_ `src/graph/builder.py`
  - _Vấn đề giải quyết:_ Với `mode="threshold"` cố định (baseline gốc), khi thị trường hoảng loạn (VD: COVID-2020, VNIndex crash), tương quan Pearson giữa các mã đồng loạt tăng lên $\approx 1.0$. Hầu hết cặp cổ phiếu đều vượt ngưỡng $0.5$, biến đồ thị thành gần như **Fully Connected**. GATv2 sau đó phân bổ Attention gần đều cho mọi hàng xóm → Over-smoothing → mất khả năng phân biệt tài sản.
  - _Giải pháp đã cài đặt:_
    - `mode="knn"` **(KHUYẾN NGHỊ):** Mỗi node chỉ kết nối với đúng `top_k` hàng xóm có `|corr|` cao nhất. Đảm bảo đồ thị luôn thưa ổn định bất kể thị trường.
    - `mode="dynamic_threshold"`: Ngưỡng tự điều chỉnh: $\text{threshold}_t = \mu(|corr|) + \alpha \cdot \sigma(|corr|)$. Khi thị trường spike, $\mu$ tăng → ngưỡng tự tăng theo → lọc bỏ cạnh "giả tạo".
  - _Hàm:_ `generate_correlation_edges(corr_matrix, mode="knn", top_k=5)`.
  - _Unit test:_ Chạy `pytest tests/test_graph.py` — kiểm tra số cạnh đầu ra bằng $N \times K$ cho mode knn.

- [x] **[GNN-005] Nâng cấp GATv2 Encoder: Thêm Residual Connections & Entropy Regularization (P0) `[MỐC B]`**
  - _Tệp đã nâng cấp:_ `src/models/gnn/gat_encoder.py`
  - _Vấn đề giải quyết:_ Kể cả khi đồ thị đã được kiểm soát độ thưa, input features của các cổ phiếu trong giai đoạn khủng hoảng vẫn rất giống nhau (cùng giảm sàn, RSI đỏ rực). GATv2 do đó vẫn có thể học được Attention gần đồng đều → Over-smoothing ở cấp độ đặc trưng.
  - _Giải pháp đã cài đặt:_
    1. **Residual Connections** (tham số `residual_alpha`): $\mathbf{h}_i^{(l+1)} = (1-\alpha)\cdot\text{GATv2}(\mathbf{h}_i^{(l)}) + \alpha \cdot \mathbf{W}\mathbf{h}_i^{(l)}$. Giữ lại "bản sắc riêng" của mỗi cổ phiếu sau khi tổng hợp thông tin hàng xóm. Mặc định: $\alpha = 0.5$.
    2. **Entropy Regularization Loss** (hàm `compute_attention_entropy_loss`): $\mathcal{L}_{\text{ent}} = -\frac{1}{|E|}\sum_{(i,j)\in E} \alpha_{ij} \log \alpha_{ij}$. Ép GATv2 chọn ra một số ít cạnh trọng yếu (sharp attention) thay vì chia đều. Đưa vào tổng Loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RL}} + \lambda_{\text{ent}} \cdot \mathcal{L}_{\text{ent}}$ (với $\lambda_{\text{ent}} \approx 0.01$).
  - _Sử dụng trong training:_
    ```python
    emb, (att_idx, att_alpha) = encoder(x, edge_index, edge_weight, return_attention_weights=True)
    loss_ent = compute_attention_entropy_loss(att_alpha)
    loss_total = loss_rl + 0.01 * loss_ent
    ```

---

### 🏁 SPRINT 5: Triển khai Mô hình Học tăng cường Đơn tác tử (Single-Agent PPO Baseline)

_Mục tiêu: Chứng minh pipeline RL hoạt động trơn tru trước khi phát triển lên đa tác tử._

#### Ticket định lượng của Sprint 5

**Thứ tự phối hợp:** Bàn giao QNT-001 cho A/QNT-002 → QNT-003. Sau QNT-002 có thể dựng sớm hạ tầng QNT-025; checklist nghiệm thu đầy đủ của runner nằm ở Sprint 8.

<a id="qnt-001"></a>

- [x] **[QNT-001] — Nghiệm thu contract hiện tại, history mask và cache (P0) [ĐỊNH LƯỢNG]**
  - **Theo dõi (kế thừa):** ✅ Đã nghiệm thu.
  - **Mức độ:** P0 · **Owner:** B · **Reviewer:** A · **Phụ thuộc:** Không
  - **Đầu vào:** Nguồn hiện có đã truyền `history_mask` qua env, adapter, buffer và model.
  - **Phạm vi:** Kiểm tra GRU/LSTM/GNN với all-valid, left padding, ngày thật có feature bằng 0 và all-invalid; kiểm tra cache, microbatch và buffer cùng dùng mask. Lưu schema version cho observation, graph, model và checkpoint.
  - **Tệp tác động:** `src/models/actor_critic.py`, `src/training/ppo_core.py`, `src/env/trading_env.py`, `src/env/marl_env.py`.
  - **DoD (Nghiệm thu):**
    - Hidden state của chuỗi có padding khớp chuỗi thật trong tolerance; `act/evaluate` có log-prob nhất quán trước update.
    - Gradient hữu hạn, không bị bùng nổ hay triệt tiêu; cache và đường uncached khớp nhau.
    - Masking không chỉ được đánh dấu hoàn thành bằng compile mà phải có unit tests xác thực.
  - **Sản phẩm bàn giao:** Báo cáo regression, schema hiện tại và cấu hình nhỏ cố định cho các ablation tiếp theo.
  - **Đặc tả bổ sung:** [QNT-001](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-001).

#### Công việc nền và H-MARL của sprint

- [x] **[MODEL-001] Xây dựng Adapter chuyển đổi Observation & Action (P0)**
  - _Tệp thực hiện:_ `src/models/actor_critic.py`, `src/training/train_ppo.py`
  - _Nội dung:_
    - Chuyển đổi trạng thái từ môi trường thành tensor đầu vào cho mạng Actor-Critic (hỗ trợ cả MLP và Recurrent GRU/LSTM tự động unflatten qua `_unflatten_obs`).
    - Hỗ trợ nạp trực tiếp feature đã tính và chuẩn hóa `[L, N, F]` mà không cần qua GNN (`raw_features` chuẩn bị mốc baseline).
    - Đầu ra Actor sử dụng phân phối Dirichlet trên không gian Simplex (thay thế Softmax đơn thuần), đảm bảo nghiêm ngặt $\sum w_i + w_{cash} = 1$ và $w_i \ge 0$.

- [x] **[BASE-004] Cùng Thành viên A lập trình và huấn luyện Single-Agent PPO (P0 - Chung A+B)**
  - _Tệp thực hiện:_ `src/training/ppo_core.py`, `src/models/actor_critic.py`, `src/training/train_ppo.py`
  - _Phần việc của B:_
    - Thiết kế kiến trúc mạng Actor-Critic (`SimplexActorCritic`, `GRUActorCritic`, `LSTMActorCritic`).
    - Cài đặt thuật toán PPO Clip Objective, Value Function Clipping (PPO2 style), Entropy Bonus, và GAE-Lambda.
    - Cùng A huấn luyện trên feature đã tính và chuẩn hóa (KHÔNG dùng GNN), áp dụng CosineAnnealingLR để thiết lập mốc chuẩn đối chứng tối thiểu cho Ablation Study.

- [x] **[TRAIN-001] Xây dựng bộ ghi log huấn luyện & Lưu Checkpoint (P0)**
  - _Tệp thực hiện:_ `src/training/logger.py`, `src/training/train_ppo.py`
  - _Nội dung:_ Tự động lưu checkpoint mô hình có Sharpe/Reward tốt nhất trên tập Validation (`best_model.pth`). Ghi log training metrics và TensorBoard qua `TrainerLogger`.

> 🏆 **Definition of Done (DoD) Sprint 5:** Một command duy nhất chạy end-to-end: Dữ liệu processed $\rightarrow$ Trading Environment $\rightarrow$ PPO (A+B) $\rightarrow$ Backtest $\rightarrow$ Xuất bảng metrics so sánh với Equal Weight & Cash. Không chỉ là từng module chạy riêng lẻ!

---

### 🏁 SPRINT 6: Tích hợp Dynamic GNN & Xây dựng Multi-Agent RL (MARL)

_Mục tiêu: Chia việc ra quyết định cho nhiều Agent chuyên trách theo từng nhóm ngành._

#### Ticket định lượng của Sprint 6

**Thứ tự phối hợp:** Chờ A/QNT-006 trước QNT-007 → QNT-008. Tái sử dụng GNN + GRU và graph ba quan hệ hiện tại; graph mới phải qua ablation trước khi đổi cấu hình mặc định.

<a id="qnt-007"></a>

- [ ] **[QNT-007] — Graph từ correlation sau shrinkage (P1) [ĐỊNH LƯỢNG]**
  - **Theo dõi (kế thừa):** 📝 Chờ A giao QNT-006.
  - **Mức độ:** P1 · **Owner:** B · **Reviewer:** A · **Phụ thuộc:** QNT-006 (A bàn giao)
  - **Phạm vi:** Nhận ma trận covariance đã shrink (Ledoit-Wolf) từ QNT-006 của A, chuyển thành correlation rồi dựng đồ thị; giữ nguyên 3 quan hệ (dương, âm, cùng ngành) và k-NN theo `abs(corr)`. Sample Pearson vẫn là baseline đối chứng.
  - **Tệp tác động:** `src/graph/dynamic_graph.py`, `src/graph/multi_relation_graph.py`, `src/graph/builder.py`, `src/training/train_ppo.py`.
  - **DoD (Nghiệm thu):** Đường chéo chuẩn hóa đúng và xử lý variance=0; loại bỏ self-edge động theo contract; $k$ không vượt $N-1$; dấu và độ mạnh cạnh đi đúng kênh; API ma trận trực tiếp và chuỗi batch nhất quán.
  - **Sản phẩm bàn giao:** Số lượng cạnh từng quan hệ, tỷ lệ cạnh thay đổi từng ngày, connected components và ablation Pearson vs Shrinkage với cùng model, seed, feature và cost.
  - **Đặc tả bổ sung:** [QNT-007](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-007).

<a id="qnt-008"></a>

- [ ] **[QNT-008] — Residual graph sau hồi quy SPY (P1) [ĐỊNH LƯỢNG]**
  - **Theo dõi (kế thừa):** 📝 Sẵn sàng.
  - **Mức độ:** P1 · **Owner:** B · **Reviewer:** A · **Phụ thuộc:** QNT-007
  - **Phạm vi:** Hồi quy causal từng cổ phiếu theo SPY trên cửa sổ trailing; dựng correlation của chuỗi phần dư (residual returns). Chốt rõ cách tính beta và residual trong cửa sổ tại $t$; SPY tiếp tục là benchmark, không đưa vào action space.
  - **Tệp tác động:** `src/features/relative.py`, `src/graph/correlation.py`, `src/graph/dynamic_graph.py`, `src/graph/multi_relation_graph.py`.
  - **DoD (Nghiệm thu):** Ticker, date và return basis khớp chuẩn; không fit beta trên giai đoạn tương lai; beta và residual hữu hạn; thiếu SPY có cơ chế fallback rõ ràng. Không ghi đè risk bank từ QNT-006.
  - **Sản phẩm bàn giao:** So sánh raw graph vs residual graph; phân tích mức độ phơi nhiễm thị trường (market exposure) và mức đóng góp của graph so với GRU thuần.
  - **Đặc tả bổ sung:** [QNT-008](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-008).

#### Công việc nền và H-MARL của sprint

- [x] **[GNN-004] Tích hợp Dynamic Graph vào quá trình huấn luyện theo thời gian (P1)**
  - _Tệp thực hiện:_ `src/graph/dynamic_graph.py`, `src/env/trading_env.py`, `src/models/actor_critic.py`, `src/training/ppo_core.py`, `src/training/train_ppo.py`
  - _Nội dung:_ Tại mỗi timestep $t$, tính rolling correlation từ close-to-close returns chỉ đến ngày $t$, ghép cạnh ngành, rồi căn graph theo cùng lịch với feature tensor. Environment đưa đủ adjacency và edge mask của lookback vào observation; PPO lưu toàn bộ graph history của từng transition và dùng lại graph đó trong minibatch update. `DynamicGraphActorCritic` chạy GCN/GAT riêng ở mỗi ngày, đưa embedding của từng mã qua GRU theo thời gian, rồi dùng learned attention pooling cho trạng thái toàn danh mục.
  - _Đồ thị:_ Mặc định dùng k-NN với cửa sổ tương quan 60 phiên và $k=5$, tránh số cạnh biến động mạnh theo ngưỡng tương quan cố định. Có thể đổi `graph_mode`/`top_k` trong `configs/model.yaml`.
  - _Chạy:_ `python -m src.training.train_ppo --encoder gnn` (mặc định lấy GAT, GRU thời gian và tham số graph từ `configs/model.yaml`); có thể chọn `--gnn-architecture gcn`.
  - _Căn thời gian:_ Graph của observation ngày $t$ chỉ dùng returns lịch sử đến ngày $t$; action được thực hiện từ Open($t+1$), không dùng dữ liệu tương lai. Cấu hình graph được lưu trong checkpoint metadata.

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

#### Ticket định lượng của Sprint 7

**Thứ tự phối hợp:** QNT-015 sau signal gate và normalization; nhận A/QNT-016 rồi QNT-017 → QNT-018. Đây là nhánh P2; HMARL-004 core có thể tiếp tục dùng policy hiện tại.

<a id="qnt-015"></a>

- [ ] **[QNT-015] — Predictor alpha có supervision (P2) [ĐỊNH LƯỢNG · MỐC C — CÓ ĐIỀU KIỆN]**
  - **Theo dõi (kế thừa):** 🔬 Nghiên cứu.
  - **Mức độ:** P2 · **Owner:** B · **Reviewer:** A · **Phụ thuộc:** QNT-003, QNT-004; cần gate tín hiệu
  - **Phạm vi:** Thử nghiệm GRU trước, Spatio-Temporal GNN + GRU sau; head xuất ra vector $\hat{\alpha}[N]$, không trực tiếp sinh ra portfolio weights. Baseline Ridge hồi quy từ script có sẵn là đối chứng tối thiểu.
  - **Tệp tác động:** `src/models/actor_critic.py`, `src/models/adapters.py`, `scripts/analyze_feature_signal.py`; module dự kiến `src/models/alpha_predictor.py`.
  - **DoD (Nghiệm thu):** Chốt nhãn next-Open $\rightarrow$ Close tại horizon 1/5/20 ngày và benchmark tương đối; nhãn train kết thúc trước outer-val; hàm loss MSE/Rank được tinh chỉnh trong inner-val. Nếu dùng rank score trong allocator, phải hiệu chỉnh về đơn vị expected return.
  - **Sản phẩm bàn giao:** Báo cáo IC, calibration, turnover theo fold; vector alpha và metadata có batch-stream parity. Gate nghiệm thu phải so với Ridge/Tilt và 1/N ròng.
  - **Đặc tả bổ sung:** [QNT-015](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-015).

<a id="qnt-017"></a>

- [ ] **[QNT-017] — Thiết kế contract Alpha → QP và loss học (P2) [ĐỊNH LƯỢNG · MỐC C — CÓ ĐIỀU KIỆN]**
  - **Theo dõi (kế thừa):** 🔬 Nghiên cứu.
  - **Mức độ:** P2 · **Owner:** B · **Reviewer:** A · **Phụ thuộc:** QNT-015, QNT-016
  - **Phạm vi:** Soạn thảo tài liệu thiết kế kiến trúc trước khi nối loss: lựa chọn giữa (1) supervised alpha + fixed allocator, (2) differentiable utility training, hoặc (3) policy-gradient trên latent action alpha.
  - **Tệp tác động:** `src/models/actor_critic.py`, `src/training/ppo_core.py`, `src/models/adapters.py`; tài liệu dự kiến `manual/THIET_KE_ALPHA_QP.md`.
  - **DoD (Nghiệm thu):** Định nghĩa rõ action space, phân phối xác suất, log-prob, deterministic evaluation và cấu trúc buffer. Nếu PPO lấy mẫu alpha, ratio dùng log-prob của alpha đã lấy mẫu; không đánh giá QP weights dưới Dirichlet.
  - **Sản phẩm bàn giao:** Schema version, parity giữa train và inference, xác định giá trị nào cần gradient. Tài liệu được cả A và B nghiệm thu trước khi lập trình QNT-018.
  - **Đặc tả bổ sung:** [QNT-017](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-017).

<a id="qnt-018"></a>

- [ ] **[QNT-018] — Lớp QP khả vi (Differentiable Convex Optimization Layer) (P2) [ĐỊNH LƯỢNG · MỐC C — CÓ ĐIỀU KIỆN]**
  - **Theo dõi (kế thừa):** 🔬 Nghiên cứu.
  - **Mức độ:** P2 · **Owner:** B · **Reviewer:** A · **Phụ thuộc:** QNT-017
  - **Phạm vi:** Tích hợp backend `cvxpylayers` hoặc solver khả vi được chọn theo thiết kế; đối chiếu nghiệm số học với QNT-016 của A. Giữ lại backend solver thường để fallback/debug.
  - **Tệp tác động:** Package `src/allocation/` dự kiến, `src/models/actor_critic.py`, `src/training/train_ppo.py`, `pyproject.toml`.
  - **DoD (Nghiệm thu):** Cố định phiên bản solver và kiểm tra quy tắc kỷ luật DPP (Disciplined Parametrized Programming); ma trận covariance được tham số hóa đảm bảo PSD. Gradient tính ra hữu hạn và khớp với đạo hàm sai phân hữu hạn (finite difference).
  - **Sản phẩm bàn giao:** Benchmark CPU/GPU/memory overhead, báo cáo gradient và thực nghiệm so sánh với fixed allocator. Chỉ tiếp tục duy trì nếu overhead không làm nghẽn training.
  - **Đặc tả bổ sung:** [QNT-018](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-018).

#### Công việc nền và H-MARL của sprint

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

#### Ticket định lượng của Sprint 8

**Thứ tự phối hợp:** QNT-025 kế thừa hạ tầng bắt đầu ở Sprint 5 và nhận folds A/QNT-009. Nếu mở tail risk: QNT-022 → A/QNT-023 trước báo cáo tail risk EXP-004; không chờ critic phân vị để chạy ablation core.

<a id="qnt-025"></a>

- [ ] **[QNT-025] — Runner ablation, năm seed và báo cáo quyết định (P1) [ĐỊNH LƯỢNG]**
  - **Theo dõi (kế thừa):** ✅ Đã có core runner.
  - **Mức độ:** P1 · **Owner:** B · **Reviewer:** A · **Phụ thuộc:** QNT-001, QNT-002
  - **Đã có:** Lõi `--seeds`, tính Mean $\pm$ Std tự động trong `train_ppo.py`, Circular Block Bootstrap 2000 mẫu, profile đo thời gian (`roll`, `upd`, `eval`).
  - **Phạm vi:** Mở rộng runner tự động chạy bộ 5 seeds chuẩn (`7, 17, 27, 37, 47`), quản lý checkpoint và tự động tổng hợp ma trận Ablation Study cho toàn bộ đồ án.
  - **Tệp tác động:** `src/training/train_ppo.py`, `src/evaluation/seed_summary.py`, `src/utils/artifacts.py`, `scripts/run_quant_ablations.py`.
  - **DoD (Nghiệm thu):** Mỗi run lưu đủ data hash, config hash, seed, checkpoint status, baseline ID, schema tham số và thời gian chạy. Giữ kết quả của cả các seed thất bại để báo cáo trung thực, không chỉ chọn seed đẹp.
  - **Sản phẩm bàn giao:** File CSV tổng hợp long-form, bảng Mean $\pm$ Std, khoảng tin cậy bắt cặp (paired block CI) và quyết định PASS/FAIL từng giả thuyết khoa học.
  - **Đặc tả bổ sung:** [QNT-025](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-025).

<a id="qnt-022"></a>

- [ ] **[QNT-022] — Critic phân vị và quantile loss (P2) [ĐỊNH LƯỢNG · MỐC C — CÓ ĐIỀU KIỆN]**
  - **Theo dõi (kế thừa):** 🔬 Nghiên cứu.
  - **Mức độ:** P2 · **Owner:** B · **Reviewer:** A · **Phụ thuộc:** QNT-001, QNT-002
  - **Phạm vi:** Thử nghiệm Critic dự báo $M$ phân vị (Quantile Regression PPO), áp dụng hàm mất mát Quantile Huber Loss; Actor tạm thời vẫn tối ưu hóa kỳ vọng để bóc tách độc lập tác động của việc học phân phối giá trị.
  - **Tệp tác động:** `src/models/actor_critic.py`, `src/training/ppo_core.py`, `src/training/logger.py`, `src/training/train_ppo.py`.
  - **DoD (Nghiệm thu):** Giá trị vô hướng dùng cho GAE/Advantage là trung bình các phân vị theo quy ước rõ ràng. Kiểm tra hiện tượng quantile crossing, độ chuẩn xác phân phối (calibration) và độ bao phủ (coverage). Checkpoint ghi đủ grid phân vị $\tau$.
  - **Sản phẩm bàn giao:** Báo cáo so sánh Scalar Critic vs Quantile Critic trên cùng policy actor và chi phí giao dịch.
  - **Đặc tả bổ sung:** [QNT-022](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-022).

#### Công việc nền và H-MARL của sprint

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

#### Ticket định lượng của Sprint 9

**Thứ tự phối hợp:** QNT-019 → QNT-021; QNT-020 làm riêng sau covariance/shrinkage bank. Concept nodes phối hợp ADV-GRAPH-* để tránh implementation trùng; chỉ mở nhánh đạt gate và còn budget.

<a id="qnt-019"></a>

- [ ] **[QNT-019] — Lead-lag graph và số quan hệ linh hoạt (P2) [ĐỊNH LƯỢNG · MỐC C — CÓ ĐIỀU KIỆN]**
  - **Theo dõi (kế thừa):** 🔬 Nghiên cứu.
  - **Mức độ:** P2 · **Owner:** B · **Reviewer:** A · **Phụ thuộc:** QNT-001, QNT-007, QNT-003; cần gate graph
  - **Phạm vi:** Bổ sung ma trận tương quan lệch pha $\text{Corr}(r_{i, s-\text{lag}}, r_{j, s})$ với $s \le t$, lag ứng viên 1/2/5 ngày. Quy ước cạnh source $i \rightarrow$ target $j$ nhất quán với message passing; không suy diễn tương quan thành quan hệ nhân quả.
  - **Tệp tác động:** `src/graph/dynamic_graph.py`, `src/graph/multi_relation_graph.py`, `src/models/actor_critic.py`, `src/training/ppo_core.py`.
  - **DoD (Nghiệm thu):** Tính chất bất biến tiền tố (prefix invariance); số mẫu hữu hiệu sau khi trừ lag đúng chuẩn. Số lượng quan hệ $R$ được truyền xuyên suốt qua buffer, model, checkpoint; không hard-code $R=3$.
  - **Sản phẩm bàn giao:** Thống kê độ ổn định và số lượng cạnh; ablation chứng minh lead-lag graph có mang lại alpha ngoài mẫu so với GRU hoặc sector-only graph hay không.
  - **Đặc tả bổ sung:** [QNT-019](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-019).

<a id="qnt-020"></a>

- [ ] **[QNT-020] — Spectral/RMT denoising (P2) [ĐỊNH LƯỢNG · MỐC C — CÓ ĐIỀU KIỆN]**
  - **Theo dõi (kế thừa):** 🔬 Nghiên cứu.
  - **Mức độ:** P2 · **Owner:** B · **Reviewer:** A · **Phụ thuộc:** QNT-006, QNT-007
  - **Phạm vi:** Thử nghiệm cắt lọc trị riêng (eigenvalue clipping/shrinkage) dựa trên dải nhiễu Marchenko-Pastur từ Lý thuyết Ma trận Ngẫu nhiên (RMT). So sánh trực tiếp với Ledoit-Wolf shrinkage.
  - **Tệp tác động:** `src/graph/correlation.py`, `src/graph/dynamic_graph.py`; module dự kiến `src/graph/denoising.py`.
  - **DoD (Nghiệm thu):** Tái cấu trúc ma trận đối xứng và PSD; lưu vết dải trị riêng $\lambda$ và số lượng eigenmodes giữ lại. Nếu loại bỏ mode thị trường (top eigenmode), phải đo đạc tương quan với SPY và chuẩn hóa lại correlation.
  - **Sản phẩm bàn giao:** Phổ trị riêng (eigen spectrum), độ ổn định cạnh đồ thị và báo cáo ablation: Raw vs Ledoit-Wolf vs RMT vs SPY-residual.
  - **Đặc tả bổ sung:** [QNT-020](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-020).

<a id="qnt-021"></a>

- [ ] **[QNT-021] — Concept nodes và node/action mapping (P2) [ĐỊNH LƯỢNG · MỐC C — CÓ ĐIỀU KIỆN]**
  - **Theo dõi (kế thừa):** 🔬 Nghiên cứu.
  - **Mức độ:** P2 · **Owner:** B · **Reviewer:** A · **Phụ thuộc:** QNT-019; cần gate graph
  - **Phạm vi:** Bổ sung $K$ latent concept/sector nodes học được từ feature và cơ chế attention causal. Đồng bộ thiết kế với các ticket `ADV-GRAPH-*` của Mốc C để tránh trùng lặp.
  - **Tệp tác động:** `src/models/actor_critic.py`, `src/graph/multi_relation_graph.py`; module dự kiến `src/models/gnn/concept_encoder.py`.
  - **DoD (Nghiệm thu):** Phân biệt rành mạch giữa số node đồ thị $N_{graph} = N + K$ và số tài sản giao dịch $N_{tradable} = N$; vector action của policy vẫn là $N+1$ (gồm cash). Buffer, edge mask và checkpoint giữ đúng mapping.
  - **Sản phẩm bàn giao:** Báo cáo ablation so sánh giữa: Sector-only graph, Learned concept nodes, và Randomized control.
  - **Đặc tả bổ sung:** [QNT-021](TICKET_CAI_THIEN_DINH_LUONG.md#qnt-021).

#### Công việc nền và H-MARL của sprint

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
  - Tổng hợp QNT đã nghiệm thu ở Sprint 5–9: config/fold/seed, cost, matched baseline, gate PASS/FAIL và giới hạn holdout; chỉ báo cáo nhánh P2 đã thực sự chạy.
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

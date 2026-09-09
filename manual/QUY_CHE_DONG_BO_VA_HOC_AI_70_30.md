# H-MARL-GNN: QUY CHẾ PHỐI HỢP 70/30 & LỘ TRÌNH LÀM CHỦ AI CHO NHÓM 2 NGƯỜI

> **Triết lý cốt lõi:**
> _"Data & Graph chia ownership; AI & RL chia knowledge."_
>
> Nhóm 2 người tuyệt đối không chia theo kiểu "A làm Data, B làm AI". Cách làm đó giúp code nhanh lúc đầu, nhưng khi bảo vệ đồ án:
>
> - Thành viên A hiểu môi trường rất sâu nhưng không giải thích được cơ chế GAT hay thuật toán MARL.
> - Thành viên B hiểu mô hình nhưng không nắm chắc giả định tài chính, rủi ro hay logic tránh rò rỉ dữ liệu (Data Leakage).
>
> Giải pháp chuẩn xác nhất: **Mô hình 70/30 (Primary Owner + Secondary Learner)**.

```text
                 H-MARL-GNN
                      │
            ┌─────────┴─────────┐
            │                   │
            A                   B
     Primary (70%):      Primary (70%):
     Data                Graph
     Features            GCN / GAT
     Environment         Training Pipelines
     Evaluation          GAT Communication
            │                   │
            └─────────┬─────────┘
                      │
               SHARED AI (70/30)
                      │
                  ┌───┴───┐
                  │       │
                 PPO     MARL
                  │       │
                  └───┬───┘
                      │
                    H-MARL
```

---

## 1. Cơ chế phối hợp 70/30 (Primary Owner & Secondary Learner)

Với mỗi Ticket kỹ thuật:

- **Primary Owner (Chịu trách nhiệm chính - 70%):** Thiết kế kiến trúc, trực tiếp code module chính thức (production code), viết unit tests và chuẩn bị tài liệu bàn giao.
- **Secondary Learner (Học & Phản biện - 30%):** Hiểu rõ input/output, tự viết một bản mini (Toy implementation) hoặc integration test, chạy debug trên máy cá nhân và review PR của Primary.

### Bảng phân vai chi tiết các Ticket AI then chốt:

| Ticket AI     | Nội dung kỹ thuật                       | Primary Owner (70% Code) |         Secondary Learner (30% Học & Test)          |
| :------------ | :-------------------------------------- | :----------------------: | :-------------------------------------------------: |
| **BASE-004**  | **Single-Agent PPO**                    |          **A**           |                        **B**                        |
| **GNN-001**   | **GCN Encoder**                         |          **B**           |   **A** _(A viết Toy GCN, hiểu Message Passing)_    |
| **GNN-003**   | **GAT Encoder**                         |          **B**           | **A** _(A hiểu Attention weight giữa các cổ phiếu)_ |
| **MARL-002**  | **Sector Observation Space**            |          **A**           |                        **B**                        |
| **MARL-003**  | **Sector Action Space**                 |          **A**           |                        **B**                        |
| **MARL-004**  | **Multi-Agent Environment**             |          **A**           |                        **B**                        |
| **MARL-005**  | **MARL Training Loop**                  |          **B**           |       **A** _(A chạy debug và theo dõi loss)_       |
| **HMARL-001** | **High-Level State (Macro)**            |          **A**           |                        **B**                        |
| **HMARL-002** | **High-Level Agent (Sector Allocator)** |          **A**           |                        **B**                        |
| **HMARL-003** | **Hierarchical Combiner**               |          **A**           |                        **B**                        |
| **HMARL-004** | **H-MARL Training Pipeline**            |          **B**           |          **A** _(A tích hợp Env & Reward)_          |
| **HMARL-005** | **GAT Inter-agent Communication**       |          **B**           |                        **A**                        |

---

## 2. Quy tắc "Teach-back" bắt buộc (15–30 phút sau mỗi Ticket AI)

Sau khi hoàn thành một Ticket AI, người làm Primary **bắt buộc phải dạy lại cho người Secondary**:

> **Ví dụ khi B hoàn thành `GNN-001` (GCN):**
> B không được chỉ nói: _"Code xong rồi, merge nhé!"_
> B phải trình bày trực tiếp cho A trả lời được 7 câu hỏi:
>
> 1. Input của GCN là gì? (Kích thước tensor $X [N, F]$)
> 2. `edge_index` biểu diễn cái gì? Cách tạo ma trận kề thưa ra sao?
> 3. Message Passing diễn ra như thế nào giữa các node cổ phiếu?
> 4. Output embedding $H [N, D]$ chứa thông tin gì?
> 5. Embedding này được Policy của RL sử dụng như thế nào?
> 6. Tại sao bài toán này cần Đồ thị mà không dùng mạng MLP thông thường?
> 7. Nếu bỏ đồ thị đi thì mô hình sẽ trở thành gì? (Trục đối chứng cho Ablation Study).
>
> 👉 **Sau đó A phải giải thích ngược lại được toàn bộ logic.** Nếu A không hiểu hoặc không debug được, ticket đó **chưa được coi là hoàn thành**.

---

## 3. Hai tầng Definition of Done (DoD) mới

Để một Ticket AI được đóng và merge vào `develop`, bắt buộc phải đạt đủ 2 tầng nghiệm thu:

### 🔹 TẦNG 1: CODE DoD (Kỹ thuật)

- [x] Đã cài đặt hoàn chỉnh tính năng (Implementation).
- [x] 100% Unit tests và Integration tests PASSED.
- [x] Không hard-code tham số, đọc từ file config YAML.
- [x] Viết docstrings và cập nhật README/doc.

### 🔹 TẦNG 2: KNOWLEDGE DoD (Học thuật)

- [x] **Primary** giải thích thông suốt bản chất toán học và thuật toán.
- [x] **Secondary** giải thích được input/output, luồng dữ liệu và ý nghĩa thực tế.
- [x] **Secondary** tự chạy, tự debug và reproduce được kết quả trên máy của mình.
- [x] Cả hai vẽ được vị trí của module này trên sơ đồ kiến trúc tổng thể.

---

## 4. Lộ trình học AI 5 cấp độ (AI Learning Track chạy song song)

Cả hai bạn cùng tiến bước qua 5 level kiến thức:

### 📍 Level 1: Nền tảng Deep Learning (Neural Networks)

- **Khái niệm bắt buộc:** Tensor, Linear Layer, Activation (ReLU, Softmax), Forward Pass, Loss Function, Gradient, Backpropagation, Optimizer (Adam), Training Loop.
- **Bài thực hành:** Mỗi người tự viết một mạng PyTorch nhỏ (MLP) dự đoán xu hướng giá để hiểu sâu cách gradient chạy.

### 📍 Level 2: Reinforcement Learning & PPO

- **Khái niệm bắt buộc:** Bộ tứ $(s_t, a_t, r_t, s_{t+1})$, Agent, Environment, State, Observation, Action, Reward, Episode, Policy $\pi(a|s)$, Value Function $V(s)$.
- **Thuật toán PPO (A là người dẫn dắt chính):**
  $$\text{Actor: } \pi(a|s) \quad \longleftrightarrow \quad \text{Critic: } V(s)$$
  $$\text{Trajectory } \tau \longrightarrow \text{Return } R_t \longrightarrow \text{Advantage } A_t \longrightarrow \text{PPO Clipped Objective} \longrightarrow \text{Gradient Update}$$
- **Bài thực hành:** A trực tiếp viết `BASE-004 Single-Agent PPO`, B cùng tham gia thiết kế mạng Actor-Critic và Softmax.

### 📍 Level 3: Graph Neural Networks (GCN & GAT)

- **Khái niệm bắt buộc (B là người dẫn dắt chính):** Stock = Node, Relationship = Edge, Financial Features = Node Features.
  $$X [N, F] \xrightarrow{\quad \text{Message Passing (GCN / GAT)} \quad} H [N, D]$$
- **Bài thực hành Toy Example cho A:**
  A tự viết một script mini 3 node (`AAPL — MSFT — NVDA`), tự tay tạo `edge_index`, cho chạy qua GCN và in ra embedding.

### 📍 Level 4: Multi-Agent Reinforcement Learning (MARL)

- **Khái niệm bắt buộc:** Sự khác biệt giữa Single-Agent và Multi-Agent, Non-stationarity khi nhiều agent cùng học, phân chia quan sát cục bộ (Local Observation) và hành động ngành (Sector Actions).
- **Bài thực hành:** A đóng gói Multi-Agent Env Wrapper (`MARL-004`), B xây dựng vòng lặp cập nhật chính sách (`MARL-005`).

### 📍 Level 5: Hierarchical MARL (H-MARL) & GAT Communication

- **Khái niệm bắt buộc:** Kiến trúc phân cấp 2 tầng: High-Level quản lý rủi ro vĩ mô, Low-Level tối ưu tỷ trọng vi mô; Cơ chế chú ý đồ thị liên tác tử (Cross-Agent Attention).
- **Bài thực hành:** A phụ trách High-Level Allocator (`HMARL-001/002`), B phụ trách Low-Level GAT Communication (`HMARL-005`), cả hai tích hợp vòng lặp H-MARL (`HMARL-004`).

---

## 5. Bộ 10 câu hỏi "Khảo thí" trước Hội đồng Bảo vệ Đồ án

Trước ngày bảo vệ, cả Thành viên A và Thành viên B **đều phải trả lời lưu loát 10 câu hỏi này**:

1. **Tại sao bài toán quản lý danh mục lại dùng Reinforcement Learning mà không dùng Supervised Learning thông thường?**
   _(Gợi ý: Do bản chất tương tác đa bước, chi phí giao dịch làm thay đổi danh mục tương lai, và mục tiêu tối ưu hàm lợi ích dài hạn Sharpe Ratio thay vì đoán giá ngày mai)._
2. **State, Action và Reward cụ thể của hệ thống này là gì?**
3. **Thuật toán PPO hoạt động về mặt nguyên lý như thế nào? Tại sao cần cơ chế Clipped Objective?**
4. **Tại sao thị trường chứng khoán lại có thể biểu diễn tự nhiên dưới dạng Đồ thị (Graph)?**
5. **Cơ chế Message Passing trong mạng GCN hoạt động ra sao?**
6. **Mạng GAT (Graph Attention Network) khác và vượt trội hơn GCN ở điểm nào trong bài toán tài chính?**
   _(Gợi ý: GAT tự học được trọng số chú ý giữa các cổ phiếu thay vì gán trọng số cố định)._
7. **Multi-Agent RL khác Single-Agent RL ở đâu? Tại sao phân chia theo Sector lại tốt hơn để 1 agent gánh toàn bộ thị trường?**
8. **Tại sao hệ thống cần kiến trúc Phân cấp (Hierarchical)? High-Level Agent giải quyết bài toán gì mà Low-Level Agent không làm được?**
9. **High-Level Agent và Low-Level Sector Agents phối hợp với nhau như thế nào để đưa ra tỷ trọng danh mục cuối cùng?**
10. **Đóng góp thực sự của Dynamic GNN và H-MARL là gì khi so sánh với các đường đối chứng (Ablation Study: Equal Weight, Single-Agent PPO, Static Graph)?**

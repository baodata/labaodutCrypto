with open('manual/NHIEM_VU_THANH_VIEN_A.md', 'r') as f:
    text = f.read()

# 1.1 ENV-002
old_env002 = """- [ ] **[ENV-002] Bộ máy tính toán lợi nhuận danh mục (Return Engine) (P0)**
  - _Tệp cần tạo:_ `src/env/returns.py`, `tests/test_portfolio_returns.py`
  - _Công thức:_ $R_p(t+1) = \sum_{i} w_i(t) \cdot r_i(t+1)$. Có unit test chặt chẽ."""
new_env002 = """- [ ] **[ENV-002] Bộ máy tính toán lợi nhuận danh mục (Return Engine) (P0)** _(đã chỉnh)_
  - _Tệp cần tạo:_ `src/env/returns.py`, `tests/test_portfolio_returns.py`
  - _Công thức:_ $R_p(t+1) = \sum_{i} w_i(t) \cdot r_i(t+1)$. **Lưu ý: Đây là return GỘP (gross), chưa trừ phí giao dịch.** Có unit test chặt chẽ."""
text = text.replace(old_env002, new_env002)

# 1.2 ENV-004
old_env004 = """- [ ] **[ENV-004] Mô hình hóa chi phí giao dịch (Transaction Cost) (P0)**
  - _Tệp cần tạo:_ `src/env/cost.py`
  - _Công thức:_ $\\text{Turnover}_t = \sum_i |w_{i,t} - w_{i,t-1}|$; $\\text{Cost}_t = c \cdot \\text{Turnover}_t \cdot V_t$ (với tỷ lệ phí $c = 0.1\%$ lấy từ cấu hình)."""
new_env004 = """- [ ] **[ENV-004] Mô hình hóa chi phí giao dịch (Transaction Cost) (P0)** _(đã chỉnh)_
  - _Tệp cần tạo:_ `src/env/cost.py`
  - _Công thức:_ 
    - Tính tỷ trọng đã trôi (drifted weight) bao gồm cả cash với lãi suất phi rủi ro: $\tilde{w}_i = w_{i,t-1} \cdot (1 + r_i) / (1 + R_p)$
    - Tính tỷ lệ phí giao dịch: $TC_t = c \cdot \sum_i |w_{i,t} - \tilde{w}_i|$
  - _Lưu ý:_ Không nhân $V_t$ trực tiếp trong env. Giá trị đô la = $TC_t \cdot V_t$ nếu cần báo cáo."""
text = text.replace(old_env004, new_env004)

# 1.3 ENV-007
old_env007 = """- [ ] **[ENV-007] Thiết kế hàm Reward V1 (P0)**
  - _Tệp cần tạo:_ `src/env/reward.py`
  - _Công thức ban đầu:_ $\\text{Reward}_t = R_{p,t} - \\beta \cdot \\text{Cost}_t$ (đơn giản, ổn định trước khi thêm phạt rủi ro)."""
new_env007 = """- [ ] **[ENV-007] Thiết kế hàm Reward V1 (P0)** _(đã chỉnh)_
  - _Tệp cần tạo:_ `src/env/reward.py`
  - _Công thức ban đầu:_ $\\text{Reward}_t = R_{p,t} - \\beta \cdot TC_t$
  - _Ghi chú:_ $\\beta = 1$ là phạt đúng phí thật. Nếu $\\beta > 1$ thì đó là reward shaping, và mọi metrics đánh giá vẫn tính trên net return thật ($R_p - TC$)."""
text = text.replace(old_env007, new_env007)

# 1.4 ENV-006
old_env006 = """- [ ] **[ENV-006] Tính toán độ biến động rủi ro danh mục (Risk Calculator) (P1)**
  - _Tệp cần tạo:_ `src/env/risk.py`
  - _Nội dung:_ Tính rolling portfolio volatility: $\sigma_p = \sqrt{w^T \Sigma w}$."""
new_env006 = """- [ ] **[ENV-006] Tính toán độ biến động rủi ro danh mục (Risk Calculator) (P1)** _(đã chỉnh)_
  - _Tệp cần tạo:_ `src/env/risk.py`
  - _Nội dung:_ Tính rolling portfolio volatility: $\sigma_p = \sqrt{w^T \Sigma w}$. Cửa sổ ước lượng covariance từ 60 ngày hoặc dùng shrinkage Ledoit-Wolf (Lý do: 20 quan sát cho 24 tài sản làm ma trận suy biến)."""
text = text.replace(old_env006, new_env006)

# 1.5 ENV-009
env_009 = """- [ ] **[ENV-009] Chốt quy ước khớp lệnh và thời điểm quyết định (P0)** _(mới)_
  - _Quy ước:_ Quyết định cuối ngày $t$, khớp lệnh tại giá Open ngày $t+1$, return tính theo Open $\\rightarrow$ Open.
  - _Nội dung:_ Env đọc thêm ma trận giá open $[T, N]$. Yêu cầu ghi quy ước này vào docs TRƯỚC khi viết `trading_env.py`. Nhắc lại ở RESEARCH-002 (Sprint 9) rằng việc này chỉ còn là kiểm toán, không phải lúc quyết định.

"""
text = text.replace("- [ ] **[GNN-002] Phối hợp chạy Integration Test cho GNN (P0 - Cùng B)**", env_009 + "- [ ] **[GNN-002] Phối hợp chạy Integration Test cho GNN (P0 - Cùng B)**")

# 1.7 LEAK-002
leak_002 = """- [ ] **[LEAK-002] Kiểm thử rò rỉ dữ liệu mở rộng (P0)** _(mới)_
  - _Nội dung:_ Test rò rỉ dữ liệu cho MACD, volume_norm, scaler (train-only) và split, bổ sung cho LEAK-001 (chỉ phủ returns, volatility, RSI).

"""
text = text.replace(
    "### 🏁 SPRINT 4: Xây dựng Môi trường Giao dịch (Trading Environment)",
    leak_002 + "---\n\n### 🏁 SPRINT 4: Xây dựng Môi trường Giao dịch (Trading Environment)"
)

# 1.8 BASE-004
old_base_004 = """- [ ] **[BASE-004] Cùng Thành viên B lập trình và huấn luyện Single-Agent PPO (P0 - Chung A+B)**
  - _Tệp cùng thực hiện:_ `src/training/single_agent_ppo.py`, `scripts/train_single_agent.py`
  - _Phần việc của A:_
    - Kết nối Trading Environment với thuật toán PPO.
    - Thiết kế hàm Reward động (thưởng Return, phạt Transaction Cost và Volatility).
    - Theo dõi đồ thị học tập (learning curve) và phân tích hành vi đặt trọng số của Agent.
    - Cùng B tinh chỉnh (tune) siêu tham số: `learning_rate`, `clip_range`, `gamma`, `entropy_coef`."""
new_base_004 = """- [ ] **[BASE-004] Cùng Thành viên B lập trình và huấn luyện Single-Agent PPO (P0 - Chung A+B)** _(đã chỉnh)_
  - _Tệp cùng thực hiện:_ `src/training/single_agent_ppo.py`, `scripts/train_single_agent.py`
  - _Phần việc của A (Primary):_
    - Bổ sung vòng lặp huấn luyện PPO, tính GAE/advantage và clip loss.
    - Thiết kế hàm Reward động (thưởng Return, phạt Transaction Cost và Volatility).
    - LƯU Ý: Phần của B sẽ giảm về Adapter Obs/Action, Softmax, review. Nên tách lõi PPO dùng chung (vd. `src/training/ppo_core.py`) để hai người không cùng sửa `single_agent_ppo.py`.
    - Theo dõi đồ thị học tập (learning curve) và phân tích hành vi đặt trọng số của Agent."""
text = text.replace(old_base_004, new_base_004)

# 1.9 BASE-001
old_base_001 = """- [ ] **[BASE-001] Triển khai Baseline Giữ tiền mặt (Cash Baseline) (P0)**
  - _Tệp cần tạo:_ `src/evaluation/baselines/cash.py`
  - Giữ 100% Cash để làm mốc so sánh tối thiểu (Sanity check)."""
new_base_001 = """- [ ] **[BASE-001] Triển khai Baseline Giữ tiền mặt (Cash Baseline) (P0)** _(đã chỉnh)_
  - _Tệp cần tạo:_ `src/evaluation/baselines/cash.py`
  - Giữ 100% Cash để làm mốc so sánh, cho cash sinh lãi bằng risk-free rate thay vì 0%."""
text = text.replace(old_base_001, new_base_001)

# 1.9 Move EXP-001
exp_001 = """- [ ] **[EXP-001] Cài đặt đầy đủ các chỉ số tài chính chuẩn (P0)** _(đã chỉnh)_
  - _Tệp cần tạo:_ `src/evaluation/metrics.py`, `tests/test_metrics.py`
  - _Các chỉ số bắt buộc:_
    1. **Cumulative Return:** Tổng tỷ suất sinh lời toàn kỳ.
    2. **Annualized Return:** Lợi nhuận quy năm.
    3. **Annualized Volatility:** Biến động danh mục quy năm ($\sigma_{daily} \\times \sqrt{252}$).
    4. **Sharpe Ratio:** Tỷ suất sinh lời có điều chỉnh theo rủi ro (thêm risk-free rate vào Sharpe).
    5. **Maximum Drawdown (MDD):** Mức sụt giảm tài khoản sâu nhất từ đỉnh.
    6. **Calmar Ratio:** Tỷ số giữa Annualized Return và MDD.
    7. **Turnover & Total Costs:** Tỷ lệ đảo danh mục và tổng chi phí phát sinh.

"""
old_exp_001 = """- [ ] **[EXP-001] Cài đặt đầy đủ các chỉ số tài chính chuẩn (P0)**
  - _Tệp cần tạo:_ `src/evaluation/metrics.py`, `tests/test_metrics.py`
  - _Các chỉ số bắt buộc:_
    1. **Cumulative Return:** Tổng tỷ suất sinh lời toàn kỳ.
    2. **Annualized Return:** Lợi nhuận quy năm.
    3. **Annualized Volatility:** Biến động danh mục quy năm ($\sigma_{daily} \\times \sqrt{252}$).
    4. **Sharpe Ratio:** Tỷ suất sinh lời có điều chỉnh theo rủi ro.
    5. **Maximum Drawdown (MDD):** Mức sụt giảm tài khoản sâu nhất từ đỉnh.
    6. **Calmar Ratio:** Tỷ số giữa Annualized Return và MDD.
    7. **Turnover & Total Costs:** Tỷ lệ đảo danh mục và tổng chi phí phát sinh.

"""
text = text.replace(old_exp_001, "")
text = text.replace(
    "> 🏆 **Definition of Done (DoD) Sprint 5:**",
    exp_001 + "> 🏆 **Definition of Done (DoD) Sprint 5:**"
)

# 1.10 Sprint 6
marl_234 = """- [ ] **[MARL-002] Huấn luyện Sector Agent phân bổ tỷ trọng (P0)** _(mới)_
  - Huấn luyện Agent phân bổ nội bộ cho nhóm ngành được phân công.

- [ ] **[MARL-003] Huấn luyện cơ chế khen thưởng từng nhóm (P0)** _(mới)_
  - Thiết kế reward riêng cho mỗi Sector Agent.

- [ ] **[MARL-004] Tích hợp mạng MARL với Môi trường giao dịch (P0)** _(mới)_
  - Kết nối Multi-Agent với `portfolio_env.py`.

"""
text = text.replace(
    "- [ ] **[MARL-006] Đánh giá mô hình MARL sơ bộ (P0 - Chung)**",
    marl_234 + "- [ ] **[MARL-006] Đánh giá mô hình MARL sơ bộ (P0 - Chung)**"
)

# 1.11 Sprint 7 HMARL-000
hmarl_000 = """- [ ] **[HMARL-000] Tài liệu thiết kế huấn luyện phân cấp (P0)** _(mới)_
  - Chốt 3 điều:
    - (a) Reward của low-level agent là reward danh mục chung hay reward riêng theo sector?
    - (b) Tần suất ra quyết định của high-level (hàng ngày hay hàng tuần)?
    - (c) Lịch cập nhật hai tầng (xen kẽ hoặc đóng băng một tầng khi tầng kia học) để giảm non-stationarity.

"""
text = text.replace(
    "- [ ] **[HMARL-001] Xây dựng Trạng thái Quan sát cho High-Level Agent (P0)**",
    hmarl_000 + "- [ ] **[HMARL-001] Xây dựng Trạng thái Quan sát cho High-Level Agent (P0)**"
)

# 1.12 Sprint 7 HMARL-002
text = text.replace(
    "$$\\text{Budget}_{Tech}, \\text{Budget}_{Finance}, \\text{Budget}_{Healthcare}, \\text{Budget}_{Energy}, \\text{Ratio}_{Cash}$$",
    "$$\\text{Budget}_{Tech}, \\text{Budget}_{Finance}, \\text{Budget}_{Healthcare}, \\text{Budget}_{Energy\_Consumer}, \\text{Ratio}_{Cash}$$ _(đã chỉnh)_"
)

with open('manual/NHIEM_VU_THANH_VIEN_A.md', 'w') as f:
    f.write(text)

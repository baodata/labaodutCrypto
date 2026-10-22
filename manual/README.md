# HỆ THỐNG TÀI LIỆU DỰ ÁN (PROJECT DOCUMENTATION)

> Thư mục `manual/` lưu trữ toàn bộ tài liệu nghiên cứu, thiết kế kiến trúc, kế hoạch phân công công việc và nhật ký phát triển của đề tài **H-MARL-GNN: Hierarchical Multi-Agent Reinforcement Learning with Heterogeneous Dynamic Graph Neural Networks for Algorithmic Trading**.

---

## 📑 BẢN ĐỒ TRA CỨU TÀI LIỆU (TABLE OF CONTENTS)

```
manual/
├── 1. planning/       # Kế hoạch tổng thể, ticket & phân công 2 thành viên
├── 2. architecture/   # Kiến trúc hệ thống, quy ước môi trường & sổ lệnh chạy
├── 3. research/       # Nghiên cứu định lượng, mô hình dự báo & bộ phân bổ
├── 4. sprints/        # Kế hoạch triển khai & nghiệm thu các Sprint (Sprint 5, 6, GP)
└── 5. audits/         # Rà soát logic, chẩn đoán allocator & nhật ký sửa lỗi
```

---

### 1. KẾ HOẠCH & PHÂN CÔNG CÔNG VIỆC ([`planning/`](planning/))

Chứa các tài liệu phân công nhiệm vụ, lộ trình học và quản trị dự án giữa 2 thành viên:

| Tài liệu | Mô tả tóm tắt |
| :--- | :--- |
| [**BANG_DIEU_PHOI_2_NGUOI.md**](planning/BANG_DIEU_PHOI_2_NGUOI.md) | Bảng điều phối phân công công việc chi tiết giữa Thành viên A và Thành viên B qua từng Sprint. |
| [**NHIEM_VU_THANH_VIEN_A.md**](planning/NHIEM_VU_THANH_VIEN_A.md) | Danh sách Ticket và trách nhiệm của Thành viên A: Tải nến, Feature Pipeline, Gym Environment. |
| [**NHIEM_VU_THANH_VIEN_B.md**](planning/NHIEM_VU_THANH_VIEN_B.md) | Danh sách Ticket và trách nhiệm của Thành viên B: Dynamic Graph GNN, Mô hình PPO, Backtest Engine. |
| [**QUY_CHE_DONG_BO_VA_HOC_AI_70_30.md**](planning/QUY_CHE_DONG_BO_VA_HOC_AI_70_30.md) | Quy chế làm việc 70/30 và lộ trình 5 Level tự học làm chủ AI từ cơ bản đến nâng cao. |
| [**TICKET_CAI_THIEN_DINH_LUONG.md**](planning/TICKET_CAI_THIEN_DINH_LUONG.md) | Kế hoạch các ticket nâng cấp định lượng, chống rò rỉ dữ liệu (lookahead bias) và bảo toàn tính nhân quả. |
| [**H-MARL-GNN_2_Person_Master_Ticket_Plan.txt**](planning/H-MARL-GNN_2_Person_Master_Ticket_Plan.txt) | Bản thảo Master Ticket Plan gốc của nhóm 2 người. |
| [**H-MARL-GNN_Chi_Tiet_Do_An.txt**](planning/H-MARL-GNN_Chi_Tiet_Do_An.txt) | Bản đặc tả chi tiết đề tài đồ án tốt nghiệp / nghiên cứu. |
| [**H-MARL-GNN_Ticket_Plan.txt**](planning/H-MARL-GNN_Ticket_Plan.txt) | Danh sách tổng hợp các ticket công việc ban đầu. |
| [**EPIC_0.txt**](planning/EPIC_0.txt) | Kế hoạch Epic 0: Khởi tạo repository, virtualenv và cấu trúc khung dự án. |

---

### 2. KIẾN TRÚC HỆ THỐNG & QUY ƯỚC KỸ THUẬT ([`architecture/`](architecture/))

Chứa tài liệu kỹ thuật cốt lõi, quy ước mô phỏng sàn giao dịch và hướng dẫn chạy lệnh:

| Tài liệu | Mô tả tóm tắt |
| :--- | :--- |
| [**LUONG_VA_CO_CHE_HE_THONG.md**](architecture/LUONG_VA_CO_CHE_HE_THONG.md) | **Tài liệu kiến trúc trung tâm** (800+ dòng): Mô tả chi tiết toàn bộ luồng chạy của hệ thống từ lúc đọc dữ liệu thô, tính 20 features, xây đồ thị động, Actor-Critic đến khớp lệnh và tính thưởng. |
| [**KIEN_TRUC_MODULE_PPO.md**](architecture/KIEN_TRUC_MODULE_PPO.md) | Bản đồ phân tách các module PPO (data_preparation, env_factory, ppo_trainer, logger...) và ranh giới trách nhiệm A/B. |
| [**ENV_CONVENTIONS.md**](architecture/ENV_CONVENTIONS.md) | Quy ước chuẩn của môi trường giao dịch `TradingEnv`: Quyết định lúc `Close[t]`, khớp lệnh tại `Open[t+1]`, hạch toán phí và NAV tại `Close[t+1]`. |
| [**GP_CONVENTIONS.md**](architecture/GP_CONVENTIONS.md) | Quy ước thiết kế, không gian hàm toán học và ràng buộc cho nhánh Genetic Programming (GP). |
| [**LENH_CHAY.md**](architecture/LENH_CHAY.md) | Sổ tay lệnh thực thi (CLI cheat-sheet) cho tất cả các kịch bản nghiên cứu, train, test và ablation. |

---

### 3. NGHIÊN CỨU ĐỊNH LƯỢNG & MÔ HÌNH DỰ BÁO ([`research/`](research/))

Tài liệu thiết kế thuật toán phân bổ danh mục, mô hình dự báo và nhân tố alpha:

| Tài liệu | Mô tả tóm tắt |
| :--- | :--- |
| [**PREDICTION_BANK_VA_NHAN_GIAO_DICH.md**](research/PREDICTION_BANK_VA_NHAN_GIAO_DICH.md) | Thiết kế Prediction Bank: Tách biệt dự báo không nhãn (label-free) và nhãn lợi nhuận tương lai (forward returns) trên từng fold. |
| [**LOCK_PREDICTOR_VA_SO_SANH_ALLOCATOR.md**](research/LOCK_PREDICTOR_VA_SO_SANH_ALLOCATOR.md) | Cơ chế khóa (Predictor Lock) mô hình có Inner-IC cao nhất với chữ ký SHA-256 bất biến, chống data snooping khi phân bổ. |
| [**RIDGE_MOMENTUM_VA_ALLOCATOR_CHI_PHI.md**](research/RIDGE_MOMENTUM_VA_ALLOCATOR_CHI_PHI.md) | Nghiên cứu tín hiệu Ridge Regression, Momentum, Mean Reversion kết hợp bộ phân bổ danh mục nhận biết chi phí (cost-aware). |
| [**BAT_CHUOC_CONVEX_TARGETS.md**](research/BAT_CHUOC_CONVEX_TARGETS.md) | Phương pháp học bắt chước (Imitation Learning / Behavioral Cloning) giúp mạng PPO Actor hội tụ nhanh theo danh mục Convex tối ưu. |
| [**EXECUTION_ALPHA_THICH_NGHI.md**](research/EXECUTION_ALPHA_THICH_NGHI.md) | Thiết kế bộ điều khiển thực thi lệnh thích ứng theo biến động và trượt giá (Adaptive Execution Controller). |
| [**NANG_CAP_ALPHA_VIX_BLACKOUT_FF3.md**](research/NANG_CAP_ALPHA_VIX_BLACKOUT_FF3.md) | Tích hợp các nguồn dữ liệu vĩ mô và nhân tố bên ngoài: Biến động Cboe VIX, FED Blackout periods, Fama-French 3-Factor (FF3). |
| [**GRU_GNN_GRU_PREDICTOR_ABLATION.md**](research/GRU_GNN_GRU_PREDICTOR_ABLATION.md) | Nghiên cứu đối chứng (ablation) giữa bộ dự báo chuỗi thời gian GRU và bộ dự báo đồ thị kết hợp GNN-GRU. |

---

### 4. KẾ HOẠCH & NGHIỆM THU SPRINTS ([`sprints/`](sprints/))

Báo cáo nghiệm thu các giai đoạn phát triển chính của đề tài:

| Tài liệu | Mô tả tóm tắt |
| :--- | :--- |
| [**NGHIEM_THU_SPRINT_5.md**](sprints/NGHIEM_THU_SPRINT_5.md) | Báo cáo nghiệm thu Sprint 5: Giao thức nghiên cứu định lượng, chia tập Train/Val causal và Learning Gate. |
| [**TRIEN_KHAI_SPRINT_6.md**](sprints/TRIEN_KHAI_SPRINT_6.md) | Kế hoạch triển khai Sprint 6: Sector-based Multi-Agent RL (MARL) và Ma trận hiệp phương sai (Covariance Banks: Ledoit-Wolf, EWMA). |
| [**KE_HOACH_TRIEN_KHAI_GP.md**](sprints/KE_HOACH_TRIEN_KHAI_GP.md) | Kế hoạch chi tiết 9 bước (D1 -> D9) triển khai nhánh thuật toán Lập trình Di truyền (Genetic Programming). |
| [**TRIEN_KHAI_GP.md**](sprints/TRIEN_KHAI_GP.md) | Nhật ký triển khai thực tế và kết quả pilot của nhánh Genetic Programming Allocator. |

---

### 5. RÀ SOÁT, CHẨN ĐOÁN & SỬA LỖI ([`audits/`](audits/))

Ghi chép các đợt kiểm tra chất lượng mã nguồn, phòng chống rò rỉ dữ liệu và vá lỗi:

| Tài liệu | Mô tả tóm tắt |
| :--- | :--- |
| [**RA_SOAT_LOGIC_2026_10_14.md**](audits/RA_SOAT_LOGIC_2026_10_14.md) | Đợt rà soát toàn diện logic hệ thống ngày 14/10/2026: timeline khớp lệnh, chuẩn hóa scaling, kiểm toán phí. |
| [**CHAN_DOAN_NHAT_KY_ALLOCATOR_2026_10_19.md**](audits/CHAN_DOAN_NHAT_KY_ALLOCATOR_2026_10_19.md) | Chẩn đoán các hiện tượng bất thường trong nhật ký bộ phân bổ ngày 19/10/2026. |
| [**KIEM_CHUNG_NHAN_XET_2026_10_19.md**](audits/KIEM_CHUNG_NHAN_XET_2026_10_19.md) | Kiểm chứng các nhận xét, phản biện và câu hỏi chất vấn định lượng ngày 19/10/2026. |
| [**SUA_LOI_KIEM_CHUNG_2026_10_19.md**](audits/SUA_LOI_KIEM_CHUNG_2026_10_19.md) | Nhật ký chi tiết các bản vá lỗi sau đợt kiểm chứng (sửa test evaluation scope, accounting float64). |
| [**SUA_PATIENCE_VA_TRAIN_PROBE.md**](audits/SUA_PATIENCE_VA_TRAIN_PROBE.md) | Ghi chú kỹ thuật về việc điều chỉnh early stopping patience và probe train/val. |

---

## 🧭 LỘ TRÌNH ĐỌC KHUYẾN NGHỊ (RECOMMENDED READING PATH)

1. **Người mới bắt đầu tìm hiểu đồ án**:
   `planning/H-MARL-GNN_Chi_Tiet_Do_An.txt` $\longrightarrow$ `architecture/LUONG_VA_CO_CHE_HE_THONG.md` $\longrightarrow$ `architecture/LENH_CHAY.md`
2. **Thành viên code tính năng (Developer)**:
   `planning/BANG_DIEU_PHOI_2_NGUOI.md` $\longrightarrow$ `architecture/KIEN_TRUC_MODULE_PPO.md` $\longrightarrow$ `architecture/ENV_CONVENTIONS.md`
3. **Nghiên cứu thuật toán & Viết báo cáo/luận văn (Researcher)**:
   `research/PREDICTION_BANK_VA_NHAN_GIAO_DICH.md` $\longrightarrow$ `research/LOCK_PREDICTOR_VA_SO_SANH_ALLOCATOR.md` $\longrightarrow$ `sprints/TRIEN_KHAI_SPRINT_6.md`

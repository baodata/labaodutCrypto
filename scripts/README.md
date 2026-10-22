# Thư Mục Scripts

Thư mục `scripts/` chứa toàn bộ 21 scripts thực thi độc lập cho pipeline dữ liệu, phân tích đặc trưng, chiến lược cơ sở (baselines), nghiên cứu mô hình (research), và huấn luyện/backtest (training).

Các script đã được tổ chức thành 5 nhóm chuyên biệt:

```
scripts/
├── data/           # 1. Thu thập và tiền xử lý dữ liệu thô / alpha sources
├── features/       # 2. Phân tích tín hiệu & kiểm định đặc trưng (Rank-IC, Alpha)
├── baselines/      # 3. Chiến lược cơ sở định chuẩn (Equal-Weight Tilt, Robustness)
├── research/       # 4. Nghiên cứu, tối ưu hóa phân bổ, GP, và benchmark latency
└── training/       # 5. Huấn luyện mô hình (PPO, MARL, Convex Imitation) & Sprint Backtest
```

---

## 1. `scripts/data/` — Quản Lý & Tải Dữ Liệu

| Script | Mô tả | Lệnh mẫu |
|---|---|---|
| `download_data.py` | Tải nến OHLCV từ Binance REST API cho danh mục tài sản theo cấu hình `configs/core/assets.yaml`. Hỗ trợ resume khi ngắt kết nối. | `python scripts/data/download_data.py --assets BTC/USDT ETH/USDT --interval 1h` |
| `download_alpha_sources.py` | Tải dữ liệu vĩ mô và alpha phụ trợ bên ngoài (VIX từ CBOE, Fama-French từ Tuck School, FOMC calendar). | `python scripts/data/download_alpha_sources.py` |
| `import_alpha_sources.py` | Phân tích cú pháp và import dữ liệu alpha thô vào Parquet định dạng chuẩn hóa, lưu vào `data/external/`. | `python scripts/data/import_alpha_sources.py` |

---

## 2. `scripts/features/` — Phân Tích & Nghiên Cứu Đặc Trưng

| Script | Mô tả | Lệnh mẫu |
|---|---|---|
| `analyze_feature_signal.py` | Tính toán Rank-IC, IC t-stat và hồi quy Ridge cắt ngang cho các features kỹ thuật để kiểm tra độ dự báo và loại bỏ rò rỉ dữ liệu. | `python scripts/features/analyze_feature_signal.py` |
| `research_alpha_features.py` | Nghiên cứu và đo lường đóng góp alpha gia tăng của các nhóm đặc trưng vĩ mô (VIX, Fama-French, Blackout window). | `python scripts/features/research_alpha_features.py --groups vix,fama_french --run-ablation` |

---

## 3. `scripts/baselines/` — Chiến Lược Cơ Sở (Baselines)

| Script | Mô tả | Lệnh mẫu |
|---|---|---|
| `evaluate_equal_weight_tilt.py` | Chạy sweep và đánh giá chiến lược 1/N Tilt (Equal-Weight lõi kết hợp tilt theo tín hiệu momentum/RSI với vùng no-trade và chu kỳ tái cân bằng). | `python scripts/baselines/evaluate_equal_weight_tilt.py` |
| `evaluate_tilt_robustness.py` | Kiểm tra độ bền vững (robustness/stress-testing) của cấu hình Tilt qua nhiều chu kỳ thị trường lịch sử và kiểm định qua cổng gate. | `python scripts/baselines/evaluate_tilt_robustness.py` |

---

## 4. `scripts/research/` — Nghiên Cứu & Benchmark Thuật Toán

| Script | Mô tả | Lệnh mẫu |
|---|---|---|
| `build_prediction_bank.py` | Huấn luyện và xuất kho dự báo (prediction bank) từ nhiều kiến trúc (GRU, LightGBM, Random Forest...). | `python scripts/research/build_prediction_bank.py --config configs/core/quant_sprint6.yaml` |
| `lock_predictor.py` | Khóa và đóng băng checkpoint mô hình cùng metadata và SHA-256 hash để bảo đảm tính tái lập. | `python scripts/research/lock_predictor.py --run-dir models/runs/...` |
| `compare_signal_allocators.py` | So sánh hiệu suất giữa các bộ phân bổ tín hiệu (Equal-Weight, Risk Parity, Mean-Variance, Heuristic Tilt). | `python scripts/research/compare_signal_allocators.py` |
| `compare_gp_allocators.py` | Tối ưu hóa biểu thức phân bổ danh mục tự động bằng Genetic Programming (GP) và đối sánh với baseline. | `python scripts/research/compare_gp_allocators.py` |
| `audit_gp_run.py` | Kiểm toán độc lập kết quả chạy GP, xác minh không có lookahead bias và bảo đảm tính toàn vẹn source hashes. | `python scripts/research/audit_gp_run.py --run-dir models/runs/...` |
| `compare_portfolio_decisions.py` | Đối chiếu quyết định danh mục và phân tích đánh đổi Pareto giữa lợi nhuận - rủi ro - turnover. | `python scripts/research/compare_portfolio_decisions.py` |
| `summarize_portfolio_decisions.py` | Tổng hợp bảng tóm tắt và trực quan hóa các chỉ số portfolio decisions từ các lượt chạy thử nghiệm. | `python scripts/research/summarize_portfolio_decisions.py` |
| `compare_execution_speeds.py` | Benchmark tốc độ suy luận, memory profile và latency giữa các bộ giải và mô hình. | `python scripts/research/compare_execution_speeds.py` |

---

## 5. `scripts/training/` — Huấn Luyện & Backtest Chiến Lược

| Script | Mô tả | Lệnh mẫu |
|---|---|---|
| `train_ppo.py` | Huấn luyện thuật toán học tăng cường PPO (Proximal Policy Optimization) đơn tác tử trên môi trường giao dịch. | `python scripts/training/train_ppo.py --encoder mlp --seed 42` |
| `train_marl.py` | Huấn luyện học tăng cường đa tác tử (Multi-Agent RL) kết hợp Dynamic Correlation Graph và phân cấp tài sản. | `python scripts/training/train_marl.py --episodes 100` |
| `train_convex_imitation.py` | Huấn luyện mạng nơ-ron học bắt chước (Imitation Learning) hành vi từ bộ giải tối ưu lồi (Convex Oracle). | `python scripts/training/train_convex_imitation.py` |
| `run_sprint5_backtest.py` | Thực thi kiểm thử backtest hoàn chỉnh theo giao thức định lượng Sprint 5 với chi phí trượt giá thực tế. | `python scripts/training/run_sprint5_backtest.py` |
| `run_sprint6_ablation.py` | Chạy thực nghiệm bóc tách thành phần (ablation study) đánh giá ảnh hưởng của các module trong Sprint 6. | `python scripts/training/run_sprint6_ablation.py --variants gru_control,control` |
| `evaluate_sprint6_banks.py` | Đánh giá hiệu suất out-of-sample của các prediction bank trong giao thức Sprint 6. | `python scripts/training/evaluate_sprint6_banks.py --evaluation-scope validation` |

---

## Lưu Ý Khi Sử Dụng
- Tất cả các script tự động giải quyết `PROJECT_ROOT` từ vị trí tương đối của file (`Path(__file__).resolve().parents[2]`), do đó có thể thực thi từ bất kỳ thư mục làm việc nào.
- Để import giữa các module trong `scripts/`, hãy sử dụng package path đầy đủ, ví dụ:
  ```python
  from scripts.baselines.evaluate_equal_weight_tilt import make_tilt_policy
  from scripts.data.import_alpha_sources import cboe_close
  ```

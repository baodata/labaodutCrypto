# H-MARL-GNN: Hierarchical Multi-Agent Reinforcement Learning with Heterogeneous Dynamic Graph Neural Networks for Algorithmic Trading

Đề tài nghiên cứu xây dựng hệ thống phân bổ danh mục đầu tư và giao dịch tự động kết hợp mạng nơ-ron đồ thị động (Dynamic GNN) và học tăng cường đa tác tử phân cấp (Hierarchical Multi-Agent RL).

---

## 1. Cấu trúc dự án

```text
.
├── configs/                # Cấu hình YAML (assets, mô hình, siêu tham số)
├── data/
│   ├── raw/                # Dữ liệu thị trường thô (OHLCV từ Yahoo Finance)
│   ├── interim/            # Dữ liệu sau khi đồng bộ ngày và làm sạch
│   └── processed/          # Dữ liệu cuối cùng kèm các chỉ báo kỹ thuật
├── src/                    # Mã nguồn chính
│   ├── data/               # Tải, làm sạch và kiểm tra dữ liệu
│   ├── features/           # Tính toán các chỉ báo kỹ thuật (Return, RSI, MACD, Volatility...)
│   ├── graph/              # Xây dựng ma trận tương quan động và đồ thị tài chính
│   ├── env/                # Môi trường giao dịch (Gymnasium), phí giao dịch, rủi ro
│   ├── models/             # GNN / GAT Encoders
│   ├── agents/             # High-Level Agent & Low-Level Sector Agents
│   ├── training/           # Huấn luyện mô hình RL và GNN
│   ├── evaluation/         # Backtest engine và các chỉ số (Sharpe, MDD, Calmar...)
│   └── utils/              # Tiện ích bổ trợ (logging, visualization, seed...)
├── tests/                  # Bộ kiểm thử tự động (pytest)
├── notebooks/              # Jupyter Notebooks phục vụ EDA và thử nghiệm
├── experiments/            # Lưu trữ log huấn luyện và checkpoints
├── scripts/                # Các script chạy pipeline độc lập
├── manual/                 # Tài liệu kế hoạch và phân công ticket
├── requirements.txt        # Danh sách thư viện phụ thuộc
├── pyproject.toml          # Cấu hình package Python
└── README.md
```

---

## 2. Hướng dẫn cài đặt & Khởi chạy

### Bước 1: Kích hoạt môi trường ảo

```bash
source .venv/bin/activate
```

### Bước 2: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Bước 3: Cài đặt package ở chế độ Editable

Giúp import `src` từ bất kỳ đâu trong dự án mà không cần chỉnh `sys.path`:

```bash
pip install -e .
```

### Bước 4: Chạy kiểm thử tự động

```bash
pytest
```

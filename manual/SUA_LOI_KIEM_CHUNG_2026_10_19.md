# Bản sửa lỗi sau kiểm chứng — 19/10/2026

Đối chiếu với [báo cáo kiểm chứng trước sửa](KIEM_CHUNG_NHAN_XET_2026_10_19.md). Bản sửa xử lý các lỗi có bằng chứng trong code và phép tính; không coi các hạn chế nghiên cứu là lỗi có thể giải quyết chỉ bằng chỉnh source.

## 1. Entropy attention GAT

**Lỗi:** helper cũ cộng entropy theo head rồi lấy trung bình theo cạnh. Scale vì thế phụ thuộc số head và mật độ cạnh, không phải entropy trung bình của từng node/head.

**Sửa:** `src/models/gnn/gat_encoder.py` aggregate `-alpha * log(alpha)` theo node đích, rồi lấy trung bình trên node có cạnh vào và head. API yêu cầu `edge_index` đi kèm attention thực tế, gồm self-loop do GAT thêm:

```python
loss = compute_attention_entropy_loss(
    alpha, returned_edge_index, num_nodes=n_nodes,
    normalize_by_degree=False,
)
```

- Kiểm tra shape, device, kiểu dữ liệu và tổng probability mỗi node/head.
- Node có một hàng xóm có entropy 0; node không có cạnh vào được loại khỏi trung bình.
- Có tùy chọn chia `log(degree)`; attention uniform có kết quả 1 khi degree > 1.
- Graph rỗng và probability 0 không gây NaN; giữ gradient.
- Attention sau dropout có thể không còn normalize: helper báo lỗi thay vì âm thầm dùng sai phân phối.

**Phạm vi:** helper vẫn chưa được thêm vào PPO objective. Entropy PPO hiện tại là entropy Dirichlet của hành động. Tài liệu GNN-005 đã sửa để không mô tả attention regularizer như tính năng đang bật hoặc khẳng định residual bảo đảm không over-smoothing.

## 2. Precision accounting và PnL credit

**Lỗi:** danh mục và kết quả projection bị ép về float32, gây sai số khi đối chiếu tổng PnL theo tài sản/cash với delta NAV. Kiểm chứng trước sửa đo sai số đến khoảng `3.08e-5` đơn vị tiền trên mẫu NAV 100,000.

**Sửa:** internal weights, projection và execution weights trong diagnostics dùng float64. Tính gross return cũng ép các đầu vào về float64. Observation và tensor đầu vào policy vẫn float32. Manifest/checkpoint ghi `accounting_precision: float64`.

Regression dùng mẫu có gap qua đêm, biến động trong ngày và lệnh hold cho bốn chế độ: legacy, cash-funded, impact, adaptive execution. Kiểm tra:

- `sum(asset_net_pnl) + cash_pnl == NAV_next - NAV_previous`, sai số tuyệt đối tối đa cho phép `1e-8`.
- NAV khớp hai phân kỳ overnight/intraday sau khi trừ phí.
- Commission + slippage + impact khớp tổng phí.
- Hold giữ số cổ phiếu và có phí 0.
- Commission của nhánh cash-funded khớp actual stock notional.
- Dtype nội bộ float64, observation float32.

Đây là coverage trên mẫu kiểm soát, không phải chứng nhận mọi tổ hợp dữ liệu/tham số hoặc mọi quy mô NAV.

## 3. Evaluation và ngày split thực tế

**Lỗi:** cảnh báo test cũ chỉ nhìn CLI `test_start`, có thể bỏ sót ngày lấy từ assets config hoặc một khoảng bắt đầu 2023 nhưng bao gồm 2024–2025 đã được khảo sát.

**Sửa:** `src/evaluation/periods.py` mô tả các phiên thực sự trong test tensor. Nếu có phiên thuộc 2024–2025, checkpoint/metrics/manifest ghi `reused_exploratory_period_not_independent_holdout` và khi backtest phát cảnh báo. Khoảng khác được ghi `custom_period_freshness_unverified`, không tự nhận là holdout độc lập.

`resolved_split_dates` lưu cutoff sau khi resolve CLI/config; `test_period.start/end` ghi phiên thực tế. Hai loại ngày được lưu riêng để tránh nhầm cutoff cuối tuần với ngày giao dịch.

**Thay đổi mặc định có chủ ý:** `configs/training.yaml` và fallback config dùng `evaluation.scope: validation`. Muốn đánh giá test phải truyền tường minh:

```bash
python -m src.training.train_ppo --encoder gnn --evaluation-scope test
```

Research protocol vẫn không cho dùng test trong vòng phát triển. Đây không phải cơ chế khóa final holdout một lần; dữ liệu đã được xem không trở lại độc lập chỉ nhờ đổi config.

**Lỗi cutoff liên quan:** validation-only trước đây có thể từ chối `val_end` rơi vào cuối tuần vì tensor kết thúc ở phiên trước đó. Hiện chấp nhận khoảng cách tối đa 7 ngày lịch, tương tự đường outer validation; giữ cutoff đã cấu hình, không tự đổi sang split tỷ lệ. Kiểm thử riêng trường hợp cuối tuần và ngày lấy hoàn toàn từ assets config.

## 4. Kiểm thử sau sửa

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  --disable-warnings \
  --basetemp /tmp/ai-project1-full-fixes-final-20261019
```

**Kết quả: 234 passed, 3 warnings, 23.29 giây.** `git diff --check` đạt.

Các tệp regression bổ sung/cập nhật:

- `tests/models/test_attention_entropy.py`
- `tests/env/test_credit_accounting.py`
- `tests/integration/test_evaluation_periods.py`
- `tests/integration/test_training_artifacts.py`
- `tests/models/test_seed_summary.py`: kiểm tra điều phối nhiều seed cho cả validation và test.
- Các assertion dtype trong test portfolio/env cập nhật theo hợp đồng float64 nội bộ.

Không chạy lại thí nghiệm thật hoặc backtest thêm test lịch sử để nghiệm thu bản sửa này.

## 5. So sánh thí nghiệm và phần còn mở

Source/config và precision đã đổi: dùng thư mục run mới cho thí nghiệm sau sửa. Không gom seed chạy trước và sau sửa thành một cohort cùng phiên bản. Checkpoint cũ có source hash khác không đủ điều kiện resume chính xác với source hiện tại; không bỏ kiểm tra hash để ép resume. Artifact của các run đang/cũ được giữ nguyên.

Các vấn đề chưa thể tuyên bố đã giải quyết: alpha ngoài mẫu, survivorship bias do thiếu point-in-time universe/delisting, số ngày thị trường hạn chế, đóng góp residual graph và chi phí nhiều agent. H-MARL phân cấp chưa hoàn chỉnh không được mô tả như đã nghiệm thu. Taxonomy 5 sector → 4 agent groups đã nhất quán trong config nên không đổi nhóm một cách tùy ý.

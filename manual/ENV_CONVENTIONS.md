# QUY ƯỚC MÔI TRƯỜNG GIAO DỊCH (TRADING ENVIRONMENT CONVENTIONS)
**Ticket:** ENV-009 (Thành viên A)

Tài liệu này xác định các quy ước cốt lõi về thời gian thực thi và khớp lệnh của Môi trường Giao dịch (Trading Environment). Phải tuân thủ nghiêm ngặt quy ước này khi code module `Gymnasium Env` (ENV-008).

## 1. Thời điểm ra quyết định và Khớp lệnh
- **Thời điểm ra quyết định (Decision Time):** Cuối ngày $t$ (Lúc thị trường vừa đóng cửa). Agent sẽ quan sát toàn bộ dữ liệu Close, Volume, MACD, RSI... tính đến hết ngày $t$ để đưa ra quyết định mua/bán (tỷ trọng mong muốn).
- **Thời điểm khớp lệnh (Execution Time):** Lệnh sẽ được đẩy vào sàn giao dịch vào sáng ngày hôm sau ($t+1$) và khớp ở **Giá Mở cửa (Open Price)** của ngày $t+1$.

## 2. Cách tính Lợi nhuận (Returns)
- Observation và NAV cùng được đánh dấu tại **Close[t]**. Danh mục đang nắm giữ nhận lợi nhuận **Close[t] → Open[t+1]** trước khi khớp lệnh.
- Nếu có lệnh rebalance, phí được ghi trên NAV tại Open[t+1], sau đó danh mục mới nhận lợi nhuận **Open[t+1] → Close[t+1]**. Observation và NAV tiếp theo đều thuộc Close[t+1].
- Lệnh hold giữ nguyên số cổ phiếu qua hai phân kỳ, không tái cân bằng và không phát sinh turnover/phí; tiền mặt vẫn nhận lãi. Mỗi episode có **T−1** bước.
- Khi có Adj Close, pipeline dùng cùng hệ số Adj Close/Close cho Open và Close để mô phỏng lợi nhuận total return.

## 3. Dữ liệu cần cấp cho Environment
- Backtest thực tế cần hai ma trận Open và Close `[T,N]`, căn cùng ngày và thứ tự ticker với `MarketDataTensor`.

## 4. Lệnh giao dịch và khởi tạo baseline

```python
from src.env.orders import PortfolioOrder

env.step(PortfolioOrder.hold())
env.step(PortfolioOrder.rebalance(target_weights, action_alpha=1.0))
```

- Vector trọng số `[N+1]` dùng alpha mặc định của env và là một lệnh rebalance. Trả lại trọng số của Close[t] không phải lệnh hold.
- Baseline Buy & Hold và baseline theo lịch khớp đầy đủ target ở lần đầu tại Open[t+1], với phí bình thường; không nạp một danh mục miễn phí tại reset.
- Sau lần đầu, tilt/1/N theo lịch dùng alpha cấu hình ở ngày rebalance, trả `PortfolioOrder.hold()` ở ngày ngoài lịch hoặc trong vùng no-trade.
- Alpha riêng của một lệnh không đổi alpha mặc định của env. Trần vị thế áp dụng khi rebalance; một lệnh hold không tự bán tài sản vì trọng số trôi vượt trần.

## 5. Metrics và tổng hợp seeds

- `annualized_return` là CAGR. `sharpe_ratio` là `sqrt(252) * mean(r_t - rf_annual/252) / std(r_t - rf_annual/252, ddof=1)`.
- Cash và Sharpe cùng dùng quy ước lãi hàng ngày `rf_annual/252`. Khi chưa đủ hai mẫu hoặc độ biến động bằng 0, Sharpe trả 0.
- CSV/checkpoint mới ghi `sharpe_convention=daily_arithmetic_excess_sample_std`. Checkpoint đã chọn bằng công thức CAGR cũ cần được chọn lại sau khi huấn luyện/đánh giá theo công thức mới.
- `--seeds` xuất bảng mean/std của metric số vào `*_seeds_summary.csv`; epoch, status và các trường metadata nằm trong `*_seeds_metadata.csv`.

## 6. History mask

- `market_history` có shape `[L,N,F]`; `history_mask` `[L]` đánh dấu ngày có thật bằng `True`, left-padding bằng `False`.
- Mask dựa trên lịch sử có sẵn, không dựa trên việc feature có bằng 0 hay không. Padding được thêm sau chuẩn hóa; inference adapter cũng đưa phần bị mask về 0 sau chuẩn hóa.
- Mask đi riêng qua buffer tới cả rollout và reevaluate; không tăng kích thước vector observation hay đổi shape tham số checkpoint.
- GRU/LSTM/GNN chuyển timestep hợp lệ lên đầu trước khi pack; history rỗng có hidden state 0. API legacy thiếu mask giả định history đầy đủ.

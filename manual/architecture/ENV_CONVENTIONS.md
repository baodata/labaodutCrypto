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


## 7. Opt-in Sprint 6

`--quant-config configs/quant_sprint6.yaml` thêm causal risk/liquidity banks. Normalization, graph source, impact, adaptive execution và DSR bật độc lập; xem [contract đầy đủ và lệnh chạy](TRIEN_KHAI_SPRINT_6.md).

- ADV/volatility/covariance để quyết định tại Close[t] không dùng dữ liệu t+1. Impact dùng actual stock dollar legs tại Open[t+1], không tính cash leg.
- Khi bật impact/controller, execution trả phí từ cash và giảm buy nếu thiếu tiền; log commission/slippage/impact riêng. Controller chọn tốc độ và dollar cap từ Close, Open chỉ thực thi; no-trade giữ lượng cổ phiếu. Đường controller off/eta=0/accounting=legacy giữ accounting cũ. Có cash-funded control để tách ảnh hưởng accounting khỏi impact/controller.
- DSR thêm `reward_state=[A,B,count]` trước cập nhật vào observation; XS normalization giữ market context. Các context làm tăng observation width theo model metadata; history mask vẫn đi riêng.
- MARL thêm local `agent_state=[observed_budget,target_budget]`, mặc định equal sector budgets và shared reward. Sector-active reward là attribution, cash vẫn pooled.
- Các contract mới chưa được nghiệm thu runtime trong lượt triển khai; không trình bày như bằng chứng cải thiện alpha.


## 8. Bản sửa precision/evaluation 19/10/2026

- `PortfolioState.weights`, projection và `info.execution_weights` là float64; `market_history`, observation portfolio/cash và PPO buffer vẫn float32. PnL credit được đối chiếu với delta NAV ở precision kế toán.
- CLI/training YAML mặc định validation. Test cần chọn `--evaluation-scope test`; protocol phát triển tiếp tục chặn test. Cutoff validation có thể ở cuối tuần/ngày nghỉ (tolerance coverage 7 ngày).
- Lưu cả resolved cutoff dates và actual test session dates. Test tensor có phiên thuộc 2024–2025 được gắn reused exploratory dù không truyền CLI date hoặc bắt đầu trước 2024. Không tự xác nhận period chưa biết là holdout độc lập.

## 9. Execution alpha nghiên cứu

`execution.speed_mode` mặc định `cost_aware` giữ đường blend-then-brake cũ. Khi bật `fixed`, `efficiency_ratio` hoặc `asymmetric`, tốc độ từng mã thay thế action_alpha, không blend hai lần. Close proposal funding buy từ cash + sell; Open chỉ áp drift, cap/funding và phí. Per-order alpha override thay requested speed của lệnh, không bỏ giới hạn thực thi. ER không phải confidence/risk-off và không bảo đảm alpha. Xem [contract đầy đủ](EXECUTION_ALPHA_THICH_NGHI.md).

## 10. Final Close proposal cho GP

`PortfolioOrder.execution(weights, hold_mask=..., notional_caps=...)` đánh dấu proposal đã hoạch định tốc độ tại Close. Chỉ dùng cùng cash-funded engine, adaptive/scalar execution controller off; env không áp alpha lần hai. Bool hold mask và dollar caps chốt từ dữ liệu Close; Open chỉ drift/fill/funding/fees. Vector action và rebalance/hold cũ giữ nhánh API cũ. GP budget/risk/cost assumptions và kết quả NO-GO được ghi tại [GP_CONVENTIONS](GP_CONVENTIONS.md) và [triển khai GP](TRIEN_KHAI_GP.md).

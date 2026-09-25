# QUY ƯỚC MÔI TRƯỜNG GIAO DỊCH (TRADING ENVIRONMENT CONVENTIONS)
**Ticket:** ENV-009 (Thành viên A)

Tài liệu này xác định các quy ước cốt lõi về thời gian thực thi và khớp lệnh của Môi trường Giao dịch (Trading Environment). Phải tuân thủ nghiêm ngặt quy ước này khi code module `Gymnasium Env` (ENV-008).

## 1. Thời điểm ra quyết định và Khớp lệnh
- **Thời điểm ra quyết định (Decision Time):** Cuối ngày $t$ (Lúc thị trường vừa đóng cửa). Agent sẽ quan sát toàn bộ dữ liệu Close, Volume, MACD, RSI... tính đến hết ngày $t$ để đưa ra quyết định mua/bán (tỷ trọng mong muốn).
- **Thời điểm khớp lệnh (Execution Time):** Lệnh sẽ được đẩy vào sàn giao dịch vào sáng ngày hôm sau ($t+1$) và khớp ở **Giá Mở cửa (Open Price)** của ngày $t+1$.

## 2. Cách tính Lợi nhuận (Returns)
- Do khớp lệnh ở giá Open ngày $t+1$, lợi nhuận của lệnh đó sẽ được tính từ mức giá **Open ngày $t+1$ đến Open ngày $t+2$**.
- Công thức Gross Return của từng tài sản:
  $$r_i(t+1) = \frac{\text{Open}_{i, t+2} - \text{Open}_{i, t+1}}{\text{Open}_{i, t+1}}$$

## 3. Dữ liệu cần cấp cho Environment
- Môi trường cần được nạp ma trận giá Open (kích thước $[T, N]$) độc lập với `MarketDataTensor` (vốn dĩ chứa các feature học được dựa trên giá Close) để phục vụ cho việc tính toán Return Khớp lệnh một cách chân thực nhất.

## 4. Ghi chú cho Kiểm toán (RESEARCH-002)
- Khi đến khâu kiểm toán ở Sprint 9, chúng ta chỉ cần xác nhận lại rằng mã nguồn của `trading_env.py` đã tuân thủ đúng tài liệu này. Mọi quyết định thiết kế đã được chốt cứng ở đây, Sprint 9 không phải là lúc để tranh cãi hay thay đổi lại quy ước.

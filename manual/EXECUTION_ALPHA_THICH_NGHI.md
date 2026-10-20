# Tốc độ execution: ER, bất đối xứng và chi phí

Ngày: 19/10/2026. Mở rộng QNT-013; các cơ chế mới **tắt mặc định**.

**Cơ chế 4, cập nhật 20/10/2026:** [kế hoạch QNT-013D](KE_HOACH_TRIEN_KHAI_GP.md) đã được triển khai thành nhánh GP riêng; xem [code, lệnh và trạng thái nghiệm thu](TRIEN_KHAI_GP.md). Bảng lịch sử bên dưới mô tả lượt ER/asymmetric ngày 19/10, không phải trạng thái hiện tại của GP.

## 1. Bốn đề xuất đã có gì trong dự án?

| Cơ chế | Trước thay đổi | Quyết định triển khai |
|---|---|---|
| Kaufman ER | Chưa có ER bank hoặc ER alpha; QNT-013 đã có phạm vi execution thích nghi | Thêm `efficiency_ratio` để ablation riêng, ER10 causal từng mã |
| Mua chậm, bán nhanh | Chưa có tốc độ mua/bán riêng; QNT-028 đề cập circuit breaker nhưng là cơ chế khác | Thêm `asymmetric`, buy=0.25, sell=0.75 trong profile nghiên cứu |
| Conviction từ z-score | State allocator đã có forecast/fit-scale, chưa có calibrated confidence trigger cho execution | Chưa bật. Ghi backlog QNT-013C: đo calibration và net edge trước |
| Gârleanu–Pedersen | QNT-013 đã dẫn bài báo và cảnh báo heuristic không phải nghiệm tối ưu | Giữ controller chi phí/rủi ro hiện tại; QNT-013D nghiên cứu mô hình đúng, chưa gắn nhãn “GP optimal” |

Không có bằng chứng để gọi một trong các tốc độ mới là tối ưu. Giải pháp có cơ sở để tiếp tục làm đối chứng vẫn là **allocator tính chi phí + no-trade band + participation cap**, sau đó đo ER và bất đối xứng riêng.

## 2. Những khẳng định cần điều chỉnh

### ER đo tính thẳng của đường giá, không đo chiều hay alpha

`ER[t] = abs(C[t]-C[t-window]) / sum(abs(diff(C)), window)`.

Giá giảm đều cũng có ER=1. Nó có thể tăng tốc một target sai. Công thức `alpha_min + (alpha_max-alpha_min)*ER²` là heuristic cho execution của dự án. KAMA nguyên bản nội suy hai smoothing constants **rồi mới bình phương**, khác cách đặt biên alpha này; xem [công thức trong tài liệu TradingView](https://www.tradingview.com/support/solutions/43000773012-kaufman-s-adaptive-moving-average-kama/).

Bank dùng adjusted Close cùng basis với return engine, đúng `window+1` giá đến Close[t]. Cửa sổ phẳng hợp lệ có ER=0, tức alpha_min. Thiếu lịch sử/giá không hợp lệ khiến riêng mã đó unready và giữ cổ phiếu; không ffill/bfill. Không reset lịch sử theo đầu mỗi episode. Profile đặt biên 0.05–0.80, không tự mở lên 100%.

### Bán nhanh không đồng nghĩa phát hiện sập thị trường

Điều kiện `target_i < current_i` chỉ là giảm tỷ trọng, có thể do rebalance bình thường. Chế độ `asymmetric` áp dụng cho mọi giảm tỷ trọng; nó **không phải stop-loss/circuit breaker**. Nếu muốn chỉ tăng tốc khi risk-off, cần tín hiệu giảm rủi ro được định nghĩa và kiểm chứng riêng trong QNT-028.

Close[t] quyết định, Open[t+1] mới khớp. Dù alpha_sell=1, danh mục vẫn chịu gap qua đêm và chịu giới hạn ADV/cash/phí. Không thể hứa thoát nguyên vẹn khỏi COVID hoặc hoàn thành mọi lệnh trong một ngày. Có thể cấu hình alpha_sell=1 như một scenario riêng, không được bỏ các giới hạn thực thi.

### Z-score cực trị không phải xác suất dự báo đúng

Z-score cắt ngang chỉ cho biết forecast của một mã xa trung bình rổ bao nhiêu độ lệch chuẩn. Nó không cung cấp sai số dự báo hay xác suất sinh lời. Forecast không nhất thiết Gaussian, các mã phụ thuộc nhau và N=24 nhỏ. Vì thế `|z|>=2.5` không chứng minh “99% có sóng lớn”.

QNT-013C cần OOF calibration theo horizon/regime, sai số/độ bất định, độ bền tín hiệu và expected net benefit sau chi phí. Giữ raw decimal forecast, xử lý std=0/ties, không lấy outer/test để chọn threshold.

### Công thức GP được đề xuất không phải nghiệm của bài báo

Bài gốc dùng chi phí bậc hai và giao dịch dần về **aim portfolio**, có xét decay của predictor. Tốc độ tổng quát dạng ma trận `Lambda^{-1} A_xx`; `gamma * variance / cost` không thay thế được nghiệm này. Gamma aversion của mô hình tài chính cũng không phải gamma discount của PPO. Xem [bản tác giả Gârleanu–Pedersen (2013), mục II–III](https://research-api.cbs.dk/ws/portalfiles/portal/60191441/lasse_heje_pedersen_et_al_dynamic_trading_acceptedmanuscript.pdf).

Controller hiện tại là heuristic brake theo covariance/ADV/volatility/chi phí, chưa ước lượng Lambda, decay và aim portfolio. Commission + slippage cấu hình vẫn là **0.15%**, kể cả AAPL/MSFT; không giả định chi phí bằng 0.

## 3. Contract code

| `execution.speed_mode` | Target đầu vào controller | Alpha trước giới hạn thực thi |
|---|---|---|
| `cost_aware` (mặc định tương thích) | Target đã blend bởi action_alpha | Brake hiện tại, giữ đường tính cũ |
| `fixed` | Policy target chưa blend | action_alpha × brake nếu bật brake |
| `efficiency_ratio` | Policy target chưa blend | (alpha_min + range × ER²) × brake nếu bật |
| `asymmetric` | Policy target chưa blend | alpha_sell nếu giảm, alpha_buy nếu tăng; nhân brake nếu bật |

Ba mode mới **thay thế scalar smoothing**, không nhân thêm action_alpha lần nữa. `PortfolioOrder.rebalance(..., action_alpha=...)` vẫn có thể override alpha yêu cầu cho một lệnh; brake/giới hạn/readiness tiếp tục áp dụng. `PortfolioOrder.hold()` vẫn giữ số cổ phiếu và không tính phí.

- Brake hiện tại chỉ là heuristic cost/risk, không phải GP. Bật brake có thể làm alpha thực nhỏ hơn alpha_min; alpha_min là biên của ER trước brake, không phải cam kết tốc độ khớp.
- No-trade band trên chênh lệch policy target với Close weights ở ba mode mới. Mode cũ giữ band trên chênh lệch đã blend để không đổi hành vi các run cũ.
- Cả ER, readiness, tốc độ và dollar caps đều chốt tại Close. Open chỉ xử lý drift/fill, cap mỗi mã, cash funding và phí; không refit tốc độ từ giá tương lai.
- Mỗi mã có alpha riêng nên tổng stock proposal có thể vượt ngân sách: giảm buy theo tiền khả dụng sau sell, phần dư giữ ở cash. Không chuẩn hóa lại stock legs để vô tình tái đầu tư tiền vừa bán.
- Vùng hold giữ cổ phiếu qua gap; drift vượt position cap không tự tạo lệnh bán trên một leg hold. Position cap tiếp tục chặn mua mới.
- `info.execution_controller` lưu requested/scheduled/funded Close alpha, hold mask và Close proposal. `info.execution_weights` là trọng số thực sau Open/cost. Không gọi scheduled alpha là realized fill ratio; gap có thể đổi cả chiều giao dịch.
- ER bank có metadata/readiness, kiểm tra ticker/date order và window. Qua `prepare_quant_context`, bank được lưu cùng artifact. ER/các mode mới không tự trở thành input mới của actor.

Tệp chính: `src/features/trend_efficiency.py`, `src/env/rebalance.py`, `src/env/trading_env.py`, `src/training/quant_context.py`. Các runner predictor/decision/imitation cũ vẫn giữ controller off.

## 4. Đối chứng không cần train lại

```bash
python scripts/compare_execution_speeds.py \
  --predictor-lock models/runs/locked_predictor_h5_20261019/predictor_lock.json \
  --execution-config configs/execution_speed_research.yaml
```

Profile cố định fold `development_2021`, chỉ train OOF và inner, không outer/test. Khóa predictor hiện tại được chọn offline bằng inner IC nên inner đã là dữ liệu phát triển; không coi đây là đánh giá độc lập.

Runner tạo **một convex target tape causal** trên shadow portfolio với fixed action_alpha theo protocol, sau đó phát lại y nguyên tape ở mọi scenario. Convex solver không được sửa objective hay âm thầm refit mục tiêu theo từng controller. Đối chứng này đo execution của cùng các target, chưa đo một allocator được tối ưu lại đồng thời cho alpha động.

Mỗi scenario chạy cả convex tape và matched 1/N, cùng controller/chi phí/cash start, cùng các ngày có forecast và hold ở horizon-purged tail. Các scenario khai báo trước:

1. `fixed`: alpha=protocol (0.25), không brake, vẫn cùng band/cap/accounting.
2. `fixed_brake`: chỉ thêm brake.
3. `er_brake`: thay fixed alpha bằng ER, còn lại như fixed_brake.
4. `asymmetric_brake`: thay fixed alpha bằng buy/sell alpha, còn lại như fixed_brake.

`fixed` có band và participation cap nên không phải bản sao metrics của run cũ controller off. Cả `fixed_brake`, ER và asymmetric đều phải so với cùng control phù hợp, không quy mọi thay đổi cho một cơ chế.

Artifact: `execution_comparison.csv`, `daily_returns.csv`, target tapes/hash, per-asset execution plans và ER bank. Bảng có delta Sharpe so matched cùng controller và matched fixed; ghi mean cash, hold, turnover, chi phí và reconciliation. Mean cash tính trên Close holdings trước từng quyết định, gồm ramp đầu kỳ. Cash khác nhau cần được giải thích trước khi gán cải thiện Sharpe cho chọn tài sản.

Không chọn thắng cuộc tự động hay chạy promotion/bootstrap gate từ một fold. Chỉ tiếp tục nếu cải thiện ròng bền trên nhiều development folds, mean cash/risk được đối chiếu, chi phí/impact stress hợp lý và không phá constraints. Tham số đã nhìn inner rồi thay đổi cần ghi là lượt nghiên cứu mới.

## 5. Trạng thái nghiệm thu

Đã triển khai code/config/runner và kiểm tra cú pháp. Không thêm/chạy unit tests trong lượt này. Kết quả chạy scenario thực tế (nếu có) được ghi bên dưới; chưa có nghiệm thu toàn bộ nhiều fold/seed hoặc chứng minh alpha.

## 6. Pilot thực tế: chưa có lý do bật ER/bán nhanh mặc định

Run: `models/runs/execution_speeds_20261019T110836Z_50500782/`.

Predictor GRU đã khóa, fold `development_2021`; **inner của fold này là năm 2020**, không phải năm outer 2021. 16 trường hợp = 2 phase × 4 tốc độ × 2 chiến lược. Không train predictor/actor, không đánh giá outer/test. Tham số scenario khai báo trước trong `configs/execution_speed_research.yaml`.

Kết quả **convex tape trên inner**, cùng target tape:

| Controller | Sharpe | ΔSharpe vs matched cùng controller | Cash Close trung bình | Chi phí tiền |
|---|---:|---:|---:|---:|
| Fixed 0.25 | 1.024475 | +0.147098 | 6.40% | 254.98 |
| Fixed + brake | 1.020385 | +0.147142 | 6.67% | 251.84 |
| ER + brake | 1.009666 | +0.143828 | 7.72% | 267.09 |
| Buy 0.25 / sell 0.75 + brake | 1.024708 | +0.145923 | 7.23% | 270.17 |

Matched dùng cùng **target cash 4%**, không ép cash thực tế bằng nhau sau ramp/drift/execution. Ví dụ matched inner của ER có cash trung bình 7.51%, của fixed 6.04%; cần đọc cùng cột cash/volatility/drawdown.

- ER không cải thiện ở pilot này: so fixed+brake, Sharpe và delta vs matched đều giảm, phí tăng. Độ thẳng của giá chưa đem lại lợi ích execution trên tape này.
- Bất đối xứng có Sharpe tuyệt đối nhỉnh hơn fixed rất nhỏ (~0.00023), nhưng delta vs matched thấp hơn cả hai fixed controls và phí cao hơn. Chưa đủ lý do trả thêm turnover/chi phí cho bán nhanh mọi target reduction.
- Train OOF convex tape có delta Sharpe so matched âm ở cả bốn controller (khoảng −0.0069 đến −0.0169); inner dương không chứng minh hiệu quả bền nhiều giai đoạn.
- Đây là kiểm tra cơ chế trên một fold đã dùng phát triển, không chọn alpha hay công bố alpha từ bảng này. Chưa có bằng chứng rằng chuyển execution sẽ giải quyết actor ít nhạy với forecast.

Khuyến nghị hiện tại: giữ fixed alpha làm control, giữ allocator tính chi phí, tiếp tục sửa/đo fidelity actor và tín hiệu predictor; chỉ nghiên cứu thêm ER/asymmetric qua các development folds đã định trước nếu cần. Không đổi controller trong predictor/allocator benchmark đã khóa.

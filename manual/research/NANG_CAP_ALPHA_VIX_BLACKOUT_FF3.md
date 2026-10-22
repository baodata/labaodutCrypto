# Nâng cấp predictor: VIX, Blackout và Fama–French

Ngày triển khai: 20/10/2026. Ticket QNT-026, QNT-027, phần feature của QNT-029.

## 1. Trạng thái thực tế

Đã thêm bộ nạp, tính feature và runner đối chứng. Người dùng xác nhận **chưa có dữ liệu ngoài OHLCV**, yêu cầu tạo bộ nạp và hướng dẫn. Vì vậy chưa có kết quả alpha mới, chưa train lại predictor và chưa thay predictor lock cũ.

| Nhóm | Đã nối vào predictor | Còn thiếu để nghiệm thu ticket đầy đủ |
|---|---|---|
| VIX | Mức VIX, VIX/VIX3M, cờ đảo ngược, tương tác momentum/beta | Dữ liệu lịch sử với availability/vintage; đánh giá; macro risk controller/cash target riêng |
| Blackout | Cờ earnings/FOMC, tương tác momentum; bộ nạp revision/coverage; API giới hạn target | Lịch có thời điểm công bố; chạy đối chứng; tích hợp overlay vào execution và kiểm toán cùng baseline |
| FF3 | Beta Mkt-RF/SMB/HML; tổng và volatility residual | Daily vintages có thời điểm phát hành; residual graph/covariance bank và báo cáo exposure riêng |

Các ticket vẫn để mở. QNT-028 circuit breaker và QNT-030 broker không thuộc bước nâng cấp predictor này.

## 2. Luồng đã thêm

```text
CSV + metadata/vintage + availability evidence
       → bộ nạp chỉ nhận thông tin được biết trước cutoff
       → feature bank + readiness + input hashes
       → cùng khoảng warm-up cho control và các nhánh
       → Ridge / GRU / GNN-GRU hiện có
       → prediction bank train OOF và inner
       → cùng matched 1/N, bounded tilt, cost-aware allocator
       → rank-IC và đóng góp ròng của nhóm feature
```

`FeaturePipeline` OHLCV và feature mặc định của PPO giữ giao diện hiện tại. Feature ngoài chỉ được bật qua `external_alpha` trong config prediction hoặc runner mới. Không thêm VIX vào action space. Config mặc định bắt đầu bằng Ridge cố định để giảm chi phí nghiên cứu.

### Thời điểm dữ liệu

- Decision của hệ thống vẫn Close[t], lệnh tại Open[t+1]. Các nguồn ngoài dùng **cutoff bảo thủ 13:00 New York** trên ngày t để xử lý cả phiên đóng sớm khi chưa có exchange calendar đầy đủ. Đây là giới hạn thông tin ngoài sớm hơn Close, không đổi accounting của môi trường. Dữ liệu công bố 14:00–16:00 sẽ đến predictor ở phiên sau; đây là độ trễ được chấp nhận và ghi trong metadata.
- Daily VIX/VIX3M đóng cửa chỉ được nhận sau **16:30 New York** theo giả định bảo thủ. Do đó dùng phiên đã biết trước đó. VIX cùng ngày không tự coi là biết tại Close cổ phiếu.
- CSV hỗ trợ nhiều revision của cùng ngày. Mỗi quyết định lấy phiên bản mới nhất có `available_at <= cutoff`; không lấy bản sửa tương lai.
- Thiếu, quá cũ hoặc hổng dữ liệu trong khoảng dùng sẽ chặn ablation. Không backfill tương lai và không thay feature thiếu bằng 0. Chỉ phần đầu trước warm-up dùng padding 0 và bị loại khỏi fit.
- Timestamp bắt buộc có múi giờ; ngày quan sát và ngày giao dịch không có múi giờ. Đơn vị được kiểm tra rõ ràng.

## 3. VIX term structure

File `src/features/macro_volatility.py`, nguồn riêng ngoài `data/raw`:

```csv
date,available_at,source,vintage_id,vix,vix3m
```

Metadata cùng tên `.json`: `availability`, `availability_evidence`, `unit: index_points`.

Feature:

- `vix_level`, `vix_term_ratio`, `vix_inverted`.
- `vix_term_x_mom_12_1 = (ratio-1) × mom_12_1`.
- `vix_term_x_beta_spy_60d = (ratio-1) × beta_spy_60d`.

VIX là context chung. Khi Ridge dự báo excess return cắt ngang rồi center theo ngày, một hệ số nhân với biến chung đơn thuần bị triệt tiêu. Tương tác với momentum và beta cho phép phản ứng khác nhau giữa các mã. Đây là giả thuyết cần đo, không phải bằng chứng cải thiện.

Ratio dùng implied volatility ở hai kỳ hạn, **không phải** giá futures VIX. Ratio >1 không đảm bảo thị trường sẽ sập và không tự động chuyển 50% sang cash trong ablation predictor. Cash/risk overlay phải là thí nghiệm riêng với baseline cùng overlay.

Nguồn: [Cboe historical data](https://www.cboe.com/tradable_products/vix/vix_historical_data), [Cboe term indices](https://www.cboe.com/tradable_products/vix/term_structure), [Cboe data timing](https://datashop.cboe.com/faqs).

## 4. Fama–French 3 factors

File `src/features/factor_risk.py`:

```csv
date,available_at,source,vintage_id,mkt_rf,smb,hml,rf
```

Metadata khai báo `unit: percent_return` hoặc `decimal_return`. Giá trị `1.0` ở chế độ percent nghĩa là 1%, được chia 100 đúng một lần. Không thay sentinel thiếu thành lợi suất.

Hồi quy dùng:

```text
stock_total_return - RF = intercept + beta_mkt*Mkt_RF + beta_smb*SMB + beta_hml*HML + residual
```

- Beta tại decision dùng 60 quan sát đã phát hành, có ngày trước decision. Nếu lịch phát hành chậm, cửa sổ này cũng chậm tương ứng, không tự nhận là 60 phiên gần nhất trên thị trường.
- Residual ngày u dùng hệ số fit trên 60 quan sát trước u, tuyệt đối không fit hệ số bằng chính lợi suất u. Tại decision t, toàn bộ factor rows/revisions để tính residual phải đã available ở t.
- Feature mặc định: ba beta, `ff_residual_mom_20d` (tổng residual cộng tính), `ff_residual_vol_20d` (std daily, ddof=1). Tổng residual không được gọi là lợi nhuận danh mục có thể giao dịch.
- Regression có intercept và ridge nhỏ để ổn định; condition number, ngày factor gần nhất, tuổi dữ liệu, readiness được ghi riêng.
- FF3 **không có momentum factor**. Residual FF3 không chứng minh alpha thuần hoặc triệt tiêu mọi rủi ro phong cách.

**Không đổi ngày quan sát thành ngày phát hành.** French daily factors tải hôm nay có thể chứa lịch sử sửa đổi. Gán trễ 1 hay 60 ngày không khôi phục vintage cũ. Strict mode yêu cầu archived daily vintage cùng bằng chứng ngày công bố. Các archive monthly được công bố trên thư viện không tự thay thế daily vintages.

Nguồn: [French factor definitions](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html), [French library và thay đổi lịch sử CRSP](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

## 5. Earnings/FOMC và Blackout

File `src/data/calendar_events.py`. Cần hai bảng và metadata:

```csv
# events.csv
event_id,event_type,ticker,scheduled_at,known_at,status,source,vintage_id

# event_coverage.csv
ticker,event_type,covered_from,covered_through,known_at,source,vintage_id
```

Các dòng `# ...` phía trên chỉ chú thích trong tài liệu, không nằm trong CSV thật.

- `event_type`: `earnings` hoặc `fomc`; ticker earnings thuộc universe, ticker FOMC là `*`.
- `scheduled_at`: giờ sự kiện dự kiến có timezone; `known_at`: thời điểm lịch/revision thực sự được công bố. **Không được gán `known_at = scheduled_at - 2 ngày` bằng suy đoán.**
- `event_id` cố định qua các bản sửa; `status` là `scheduled`/`cancelled`. Hủy/sửa chỉ tác động từ `known_at` của revision.
- Coverage là cam kết lịch của nguồn đã được kiểm tra cho ticker/type và khoảng ngày, ở thời điểm `known_at`. Bảng lịch rỗng không chứng minh ngày đó không có sự kiện. Coverage không dự đoán các thông báo bất ngờ chưa được công bố.
- Cần coverage earnings của mọi mã và FOMC toàn thị trường cho cửa sổ trước/sau. Không đủ coverage sẽ không phát feature “không blackout”.
- Blackout mặc định từ 2 phiên trước đến hết 1 phiên sau ngày sự kiện, tính theo ngày New York. Sự kiện cuối tuần map sang phiên equity kế tiếp; không suy đoán reaction intraday bằng lịch ngày.
- Feature: `earnings_blackout`, `fomc_blackout`, `blackout_x_mom_12_1`. Không dùng earnings surprise hoặc kết quả họp trong feature báo trước.

Hàm `blackout_target(target, current, blocked, asset_cap=.02)` có sẵn: với mã bị chặn, target không vượt current và 2%; phần giảm chuyển cash. Đây là **target tại Close**, không đảm bảo bán được ở mức giá đó hoặc tránh gap trước Open. Hàm chưa được tự bật trong TradingEnv hoặc runner predictor. Khi nghiên cứu overlay, cần nối cùng lớp execution cho strategy lẫn matched baseline và đo riêng tác động giảm exposure.

Nguồn FOMC: [Fed meeting calendars](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm). Lịch hiện tại chỉ cho biết ngày họp; cần bản lưu/announcement cũ để chứng minh lịch đã biết vào từng decision. “Blackout” ở đây là quy tắc giao dịch của đồ án, khác blackout communications/trading áp dụng cho thành viên Fed.

## 6. Các bước chuẩn bị dữ liệu

### Bước A — Lưu bản gốc và bằng chứng

1. Tải VIX và VIX3M daily history từ Cboe, lưu hai CSV gốc và ngày nhận.
2. Tải **U.S. Fama/French 3 Factors [Daily]**, không chọn monthly/FF5. Lưu ZIP gốc, source URL và timestamp nhận. Muốn backtest strict, thu thập thêm daily vintages đã lưu từ từng thời điểm công bố.
3. Lịch earnings: lấy lịch đã công bố từ vendor có PIT snapshots hoặc announcement của doanh nghiệp được lưu tại thời điểm đó. Lưu cả lịch sửa/hủy. Tự tải lịch hiện tại rồi tra ngược ngày là chưa đủ.
4. FOMC: lưu từng bản lịch/announcement được công bố, gồm những lần đổi lịch. Không đưa cuộc họp bất thường vào feature trước khi được công bố.

Thư mục mẫu chỉ có header/metadata rỗng trong `configs/templates/alpha/`. Sao chép sang `data/external/pit/`, điền dữ liệu thật, đơn vị và `availability_evidence`. Không chỉ sửa nhãn thành `point_in_time` để vượt bộ nạp.

### Bước B — Import file công khai hiện tại để khảo sát schema

Sau khi đã tải hai CSV Cboe và ZIP French về máy:

```bash
python scripts/import_alpha_sources.py \
  --vix-csv /duong/dan/VIX_History.csv \
  --vix3m-csv /duong/dan/VIX3M_History.csv \
  --french-daily /duong/dan/F-F_Research_Data_Factors_daily_CSV.zip
```

Script tự tạo thư mục mới `data/external/alpha_import_<timestamp>_<id>/`. Không ghi đè dữ liệu cũ. Nó chuẩn hóa schema, giữ hash nguồn; mặc định `available_at` là thời điểm import hiện tại. Dữ liệu này **không đủ cho backtest lịch sử strict** và manifest ghi `current_vintage_research_only`.

`--research-lag-days N` chỉ dành cho khảo sát độ nhạy theo một độ trễ giả định; dữ liệu vẫn bị gắn current-vintage/research-only. Không có độ trễ mặc định nào được coi là đúng cho nguồn French. Muốn khảo sát dạng này phải tạo bản config riêng với `strict_pit: false` và ghi rõ giả định; Blackout vẫn luôn yêu cầu PIT. Không dùng kết quả đó để chứng minh không có leakage.

### Bước C — Kiểm tra readiness của nguồn

Điền đường dẫn vào `configs/alpha_features.yaml`, sau đó chạy:

```bash
python scripts/research_alpha_features.py
```

Mặc định chỉ chuẩn bị feature banks và báo readiness; không train. Chưa có dữ liệu sẽ tạo manifest `needs_data` và `input_banks/data_status.csv`, không tạo kết quả giả.

Có dữ liệu VIX trước thì nghiên cứu riêng VIX:

```bash
python scripts/research_alpha_features.py --groups vix
```

### Bước D — Chạy ablation predictor khi dữ liệu đạt yêu cầu

```bash
python scripts/research_alpha_features.py \
  --groups vix,fama_french,blackout \
  --run-ablation
```

Runner tạo `control`, từng nhóm riêng và `combined`. Control và các nhánh dùng cùng warm-up của toàn nhóm được chọn; không so các lịch khác nhau. Nếu một nhóm thiếu, lệnh có `--run-ablation` dừng mã 2 và giữ manifest `needs_data`. Không tự bỏ nhóm lỗi để chỉ báo nhánh thuận lợi.

Sau đối chứng Ridge có thể dùng nguyên khung neural đã có, vẫn cùng allocator:

```bash
python scripts/research_alpha_features.py \
  --groups vix \
  --prediction-config configs/prediction_bank_neural.yaml \
  --allocation-config configs/signal_allocation_neural.yaml \
  --run-ablation
```

Lệnh này có ngân sách lớn hơn và không được chạy trong lần triển khai này.

## 7. Đọc kết quả và quyết định

Artifacts trong thư mục mới `models/runs/alpha_features_.../`:

- `input_banks/data_status.csv`: nguồn nào thiếu/chưa usable.
- `<group>_readiness.csv`, `<group>_features.csv`, `<group>_metadata.json`: ngày quan sát/available, tuổi dữ liệu, feature values và hash nguồn.
- `rank_ic_by_fold.csv`: IC predictor trên train OOF và inner riêng, số ngày hợp lệ, MSE so với forecast 0 và độ phân tán forecast. Forecast hằng không có IC hợp lệ, không tự gán IC=0.
- `inner_metrics.csv`: Sharpe, turnover, costs, drawdown so với matched baseline; không phải metrics outer.
- `feature_contribution_by_fold.csv`, `feature_contribution_summary.csv`: feature arm so với **control cùng model/allocator/horizon**, paired block bootstrap theo fold; block tối thiểu horizon. Holm áp dụng trên các contrast feature được runner xuất. CI là pointwise, không phải simultaneous CI.
- Mỗi nhánh có prediction bank và weights neural nếu bật; không dùng lại khóa predictor cũ.

OOF dùng để học allocator và chẩn đoán predictor, inner là vùng phát triển đã được xem nhiều lần. Các cửa sổ train giữa fold chồng lấn, không được gộp như các thị trường độc lập. Runner không forecast/backtest outer, không test 2024–2025, không tự promotion hoặc khóa predictor mới. Dữ liệu giá ngoài inner có thể được tải để giữ calendar, nhưng không dùng kết quả outer để chọn nhánh.

Ưu tiên feature khi IC và lợi ích ròng nhất quán theo fold, tốt hơn control dưới cùng allocator/cost, không chỉ do đổi cash/turnover. Blackout có thể giúp tail risk mà không nâng IC; cần báo riêng. Chỉ sau khi chốt cấu hình ở inner mới thiết kế vòng đánh giá tiếp theo, ghi rõ vùng thời gian đã khảo sát.

## 8. Giới hạn nghiệm thu lần này

Đã kiểm tra cú pháp các module/script thay đổi. Không thêm/chạy unit tests. Chưa chạy fit/ablation có dữ liệu ngoài; chưa có bằng chứng chất lượng alpha tăng. Bộ nạp kiểm tra schema và provenance được khai báo, không thể tự chứng minh file gắn `point_in_time` là dữ liệu lưu thật từ quá khứ. Cần kiểm toán bằng chứng nguồn trước khi nghiệm thu thực nghiệm.

Đã chạy lệnh chuẩn bị `python scripts/research_alpha_features.py` trên dữ liệu thật của repo. Run `models/runs/alpha_features_20261019T185828Z_bca0dd4b/` (UTC; ngày địa phương 20/10/2026) ghi cả ba nhóm `needs_data`, 0/2014 phiên có dữ liệu ngoài. Các file còn thiếu: `macro_volatility.csv/.json`, `ff3_daily.csv/.json`, `events.csv/.json`, `event_coverage.csv` theo config. Không có predictor fit, outer hoặc test evaluation trong bước này.

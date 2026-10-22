**Rà soát logic dự án — 14/10/2026**

## Cập nhật sau khi xử lý toàn bộ báo cáo

Đã sửa **12/12 mục** và triển khai ba cải thiện hiệu năng được nêu ở cuối báo cáo. Các phát hiện chi tiết bên dưới được giữ làm lịch sử của phiên bản trước sửa; số dòng trong phần lịch sử có thể đã thay đổi.

### Bổ sung sau phản hồi GAT trên CPU chạy lâu

- Run người dùng ngắt chỉ có checkpoint epoch 0. Profiling model GAT đúng kích thước cấu hình cho thấy backward 16 mẫu dùng khoảng 1 GiB peak RSS; minibatch mặc định 256 cần giới hạn activation.
- Thêm `--microbatch-size`: mặc định GNN 16 trên CPU / 32 trên CUDA. Tích lũy gradient theo số mẫu, giữ một Adam step và KL guard cho mỗi effective minibatch; log ghi số microbatch riêng.
- GNN gom ngày trùng nhau trong microbatch trước encode, giữ autograd. GRU bỏ pack/sort/copy khi history đầy đủ; nhánh có padding được giữ. Thêm cờ thread/cache/model dimensions và log tiến độ mỗi giai đoạn.
- Profiling trên dữ liệu thật với model GAT lớn, 8 cửa sổ, 1 lượt PPO: cache **1,54 s**, rollout **0,92 s**, update **28,00 s**, eval **3,43 s**. Cache không phải phần chậm nhất.
- GAT nhỏ (hidden 16, heads 2, out 16, GRU 64, lookback 10), 8 cửa sổ, **3 lượt PPO**: cache **0,28 s**, rollout **0,89 s**, update **9,00 s**, eval **1,59 s**; tổng khoảng **12 s/epoch**, chưa tính khởi động. Đây là cấu hình thăm dò khác kiến trúc/rollout gốc, không phải so tốc độ trên cùng model.
- So đường graph dense với đường gom ngày/tích lũy gradient trên dữ liệu tổng hợp: log-prob/value sai khác tối đa **0**; gradient sai khác tối đa **1,2e-7**; sau một Adam step trọng số sai khác tối đa **8,4e-6** do thứ tự phép toán float32. Effective batch và số Adam step giữ nguyên; không cam kết bằng nhau từng bit.
- Hai run profiling chỉ train một update và đánh giá train/validation, không chạy test 2024–2025. Hướng dẫn lệnh thăm dò nằm trong README. Bộ 204 test dưới đây thuộc đợt audit trước bổ sung hiệu năng này; đợt bổ sung đã kiểm tra compile, đường train thật và các số đo trên.

| Mục | Trạng thái và cách sửa |
|---|---|
| 1. Hold | `PortfolioOrder.hold()` bỏ rebalance/phí, giữ số cổ phiếu và vẫn đánh dấu NAV qua đêm/trong ngày. Buy & Hold, baseline theo lịch và vùng no-trade dùng lệnh này. |
| 2. Exposure ban đầu | Baseline theo lịch khớp toàn bộ target ở lần đầu, có phí; các lần sau mới dùng alpha. CSV ghi cash thực tế và turnover/phí ngoài lịch. |
| 3. Sharpe | Dùng trung bình daily excess / sample std, năm hóa; CAGR giữ riêng. Kết quả/checkpoint có nhãn quy ước. |
| 4. Seeds | Chỉ tổng hợp metric số; epoch/status/metadata ra CSV riêng; từ chối seed trùng. Đã kiểm tra cả CLI hai seed từ train đến tổng hợp. |
| 5. Null | API `PolicyDiagnostics.diagnostics()` chung cho MLP/GRU/LSTM/GNN; không gọi trực tiếp chữ ký private riêng của GNN. |
| 6. Baseline/CBB | Selection excess, Sharpe delta và bootstrap dùng chung cash/alpha. Có tên baseline trong CSV; mặc định cash `1/(N+1)`. P-value dùng phân phối bootstrap null đã center; tỷ lệ bootstrap excess ≤ 0 có tên riêng. |
| 7. Gate | YAML thực sự điều khiển orientation, score weights, lịch, cash, alpha và ngưỡng Sharpe; validate cấu hình; manifest lưu cấu hình resolve, hash source/data/config, quyết định tổng thể cần qua mọi baseline. |
| 8. Dropout | GRU/LSTM/GNN từ chối dropout khác 0 trong policy PPO để rollout và reevaluate có cùng log-prob trước update. |
| 9. Alignment/data | Kiểm tra ngày/ticker/feature duy nhất, thứ tự và equality; history phải khớp ngày market; graph thiếu ticker/ngày báo lỗi; validator chạy trước align/feature. |
| 10. Artifact/resume | Run mặc định có thư mục riêng; checkpoint nhúng scaler/feature/config/split/hash, scheduler/RNG. Ghi CSV/last checkpoint mỗi epoch, ghi trạng thái khi ngắt/lỗi. Resume kiểm tra source/data/config/runtime và tham số train. |
| 11. Warm-up/risk/IC | Đánh dấu 252 phiên warm-up, loại khỏi fit scaler/train; PPO còn yêu cầu lookback đã warm-up. Downside là RMS của phần return âm trên mọi ngày. Thêm nhãn executable Open[t+1]→Close[t+h], purge nhãn vượt split và CI block cho IC. |
| 12. API/cài đặt/test | Baseline `.run(close_prices=...)` và backtester dùng chung env T−1; có `CheckpointPolicy` dùng scaler nhúng, history và 3 relation channels. MARL validate đủ action, budget rõ, local weights chuẩn hóa, có history/graph. Khai báo PyG trong requirements/package; sửa test theo timeline/phí/ràng buộc hiện tại. |

### Bằng chứng kiểm tra

- `.venv/bin/python -m pytest -q --tb=short`: **204 passed**, khoảng **20,83 giây**. Bốn warning gồm thông báo deprecation từ PyG/PyTorch và cảnh báo epoch 0 chủ ý ở smoke test. Log: `/tmp/ai_roject1_full_logic_fix_pytest.log`.
- Các test tích hợp chạy train một update với GRU, GCN+GRU, GAT+GRU; null đủ bốn encoder; CLI nhiều seed; so baseline standalone với TradingEnv; so cache với đường GNN trực tiếp cả action/value/log-prob và gradient.
- Test ngắt ở update thứ hai rồi resume so với chạy liên tục: **model tensors bằng nhau tuyệt đối**, scheduler/RNG khớp. Resume với dữ liệu đã đổi bị từ chối trước khi ghi đè manifest/scaler cũ.
- Với raw bắt đầu 2015 và lookback 20, train PPO bắt đầu **2016-02-01**, còn **1.492 phiên** sau warm-up. Muốn giữ train từ 2015 cần cung cấp dữ liệu trước đó; không dùng feature chưa sẵn sàng như mẫu train hợp lệ.
- Đã tính lại tilt, robustness và rank-IC/linear trên validation/các giai đoạn đã khai báo; không chạy lại test lịch sử 2024–2025 hoặc train PPO đầy đủ trên dữ liệu thật trong đợt sửa này. Bản trước đợt sửa toàn bộ nằm ở `experiments/archive_before_full_logic_fix_2026_10_14/`.
- Linear fit trên train sau warm-up: validation Sharpe **0.4354**, matched baseline **0.4120**, chênh lệch **+0.0235**; turnover **41.28**, phí **$6.271,72**. Đây là khảo sát, chưa chứng minh edge.
- Null GRU 10 seed trên validation: Sharpe **0.41231 ± 0.00037**, delta so matched baseline **+0.00035 ± 0.00037**. Baseline cash 4%, alpha 0.25. Không đánh giá test.
  Artifact: [CSV null](../models/runs/audit_null_gru_2026_10_14_final/untrained_null_validation.csv), [manifest null](../models/runs/audit_null_gru_2026_10_14_final/manifest.json).
- IC executable 20 phiên của `mom_12_1` trên validation **0.0447**, CI block 95% **[-0.0609, +0.1465]**; không đọc naive t-stat như bằng chứng chắc chắn khi nhãn chồng lấn.

### Hiệu năng đã triển khai

- Buffer GNN lưu chỉ số ngày và graph bank chung `[T,3,N,N]`; chỉ gather history cho minibatch. Với ví dụ 6.016 bước, lookback 20, 24 mã, 2.765 ngày: phần lưu graph giảm từ **1,55 GiB xuống khoảng 36,5 MiB**, gồm bank và chỉ số, chưa tính obs/activation/optimizer.
- Cache chỉ dùng khi trọng số cố định trong rollout/evaluation; có kiểm tra version, bị từ chối sau optimizer update. PPO update tính lại GNN nên gradient vẫn đi tới graph encoder.
- Benchmark GAT+GRU tổng hợp 99 bước, 24 mã, 20 feature, lookback 20, hidden 16, CPU một thread: trực tiếp **0,417 s**, cache + inference **0,063 s**, nhanh hơn **6,62 lần**, sai khác action tối đa **0**. Đây là benchmark inference nhỏ, không phải mức tăng tốc đã đo cho toàn bộ train. Artifact: `experiments/audit_cache_benchmark.json`.
- Train evaluation mặc định mỗi 5 epoch, cùng epoch đầu/cuối; validation vẫn mỗi epoch. Ngày không eval train ghi NaN và `train_evaluated=0`, không tái sử dụng metric cũ. Log có thời gian rollout/cache/update/eval và dung lượng graph bank.

### Kết quả giữ nguyên sau sửa engine

Các giá trị validation tilt và cổng bên dưới đã được chạy lại. Manifest mới nằm cùng CSV kết quả. Cổng tổng thể vẫn **FAIL**, không thay ngưỡng để thích nghi kết quả. Survivorship bias và việc test lịch sử đã được dùng để điều chỉnh là giới hạn dữ liệu/nghiên cứu, không thể loại bỏ bằng sửa code.

## Lịch sử sửa bốn mục đầu

Các mục 1–4 bên dưới là phát hiện tại thời điểm audit và đã được sửa. Env hỗ trợ `PortfolioOrder.hold()`; Buy & Hold và baseline theo lịch khớp đầy đủ target ở lần đầu với phí, rồi dùng hold hoặc alpha cấu hình cho các ngày sau. Sharpe dùng trung bình excess return hàng ngày và sample std; tổng hợp seeds chỉ lấy metric số, xuất metadata ra CSV riêng. Bộ kiểm thử sau sửa: **169 passed**.

Đã chạy lại validation tilt, baseline tuyến tính và robustness. Bảng cũ được giữ ở `experiments/*_before_execution_sharpe_fix.csv`. Turnover/phí ngoài lịch bằng 0 ở tất cả các giai đoạn. Baseline tháng trên validation có cash trung bình **3.96%**, return **17.96%**, Sharpe **0.4362**; baseline tuần có cash **3.99%**, return **17.10%**, Sharpe **0.4122**. Tilt RSI/momentum hàng tuần tốt nhất trong sweep có return **18.51%**, Sharpe **0.4581**; đó vẫn là kết quả khảo sát validation.

Giữ nguyên các ngưỡng cổng. Tilt đóng băng so với 1/N tuần thắng **3/4** giai đoạn, excess ròng **0.9469%/năm**, CI bootstrap 95% **[-1.5116%, +3.3881%]**; so với 1/N tháng thắng **2/4**, excess **0.3680%/năm**, CI **[-1.6247%, +2.3617%]**. Cả hai vẫn không đạt. Không huấn luyện PPO hoặc chạy lại test 2024–2025 trong bước sửa này; các checkpoint đã chọn bằng Sharpe cũ cần được chọn lại theo quy ước mới.

Phạm vi audit ban đầu: luồng data → feature/scaler → dynamic graph → actor/critic → PPO → TradingEnv → backtest/metrics; các script rank-IC, tilt, robustness; cấu hình, adapter/MARL và bộ test hiện có. Các phát hiện và phép tái hiện dưới đây ghi lại phiên bản trước sửa; kết quả hiện tại nằm trong phần cập nhật phía trên.

`./.venv/bin/python -m pytest -q --tb=short`: **147 passed, 6 failed**, khoảng 6.8 giây. Các phép tái hiện bổ sung dùng dữ liệu tổng hợp hoặc chỉ backtest validation 2022–2023, không huấn luyện PPO. Log pytest: `/tmp/ai_roject1_audit_pytest.txt`.

**1. [P1] Hold/no-trade vẫn tạo giao dịch qua đêm — đã tái hiện**

- [TradingEnv.step](/home/tuan/AI/AI_roject1/src/env/trading_env.py:328) đóng băng target từ trọng số ở Close[t], sau đó so với trọng số đã trôi đến Open[t+1] tại dòng 352.
- [BuyAndHoldBaseline.__call__](/home/tuan/AI/AI_roject1/src/evaluation/baselines/buy_and_hold.py:43) trả lại trọng số ở Close[t]. [make_tilt_policy](/home/tuan/AI/AI_roject1/scripts/evaluate_equal_weight_tilt.py:105) cũng trả lại trọng số đó vào ngày ngoài lịch hoặc khi nằm trong vùng no-trade.
- Hai trường hợp trên được hiểu thành lệnh tái cân bằng về trọng số của tối hôm trước. Không tồn tại lệnh giữ nguyên số cổ phiếu.
- Tái hiện với hai tài sản 50/50, giá qua đêm +10% và −10%, không giới hạn vị thế: sau lần mua đầu, ngày được gọi là hold vẫn có turnover **0.10**, phí **$14.9775**.
- Trên validation, baseline hàng tuần có **400/400** ngày ngoài lịch phát sinh turnover, tổng phí ở các ngày này **$351.09**. Hàng tháng: **476/476** ngày, phí **$387.52**.

Hướng sửa: biểu diễn riêng lệnh hold/skip-rebalance. Env vẫn đánh dấu NAV và tính lãi khi giữ nguyên số lượng cổ phiếu, nhưng bỏ bước đặt lại target và phí. Chính sách được quyết định ở Close[t]; việc thực hiện hold tại Open[t+1] không cần đưa giá Open tương lai vào observation.

**2. [P1] Các baseline theo lịch có mức cash thực tế khác hẳn nhau — đã đo**

[Script tilt](/home/tuan/AI/AI_roject1/scripts/evaluate_equal_weight_tilt.py:165) dùng alpha 0.25 và khởi tạo 100% cash, chỉ tiến vào target ở ngày theo lịch. Mục tiêu cash 4% không tương đương phơi nhiễm cash 4% trong backtest.

| Lịch | Cash trung bình sau mỗi bước, validation | Cash sau 21 phiên |
|---|---:|---:|
| Hàng ngày | 4.57% | 4.16% |
| Mỗi 5 phiên | 6.88% | 26.83% |
| Mỗi 21 phiên | 16.03% | 76.30% |

Chênh lệch này tái xuất hiện đầu mỗi giai đoạn vì [robustness script](/home/tuan/AI/AI_roject1/scripts/evaluate_tilt_robustness.py:94) tạo env mới và reset về cash. Do đó Sharpe 0.599 của baseline tháng chưa chứng minh riêng tác dụng của lịch tái cân bằng. Phải thống nhất cách vào danh mục: ví dụ lần đầu khớp đầy đủ target và ghi phí, sau đó mới dùng alpha cho các thay đổi. Báo cáo thêm cash/exposure thực tế và so baseline cùng lịch, cùng khởi tạo.

Kết luận trước về việc bác bỏ tilt phải coi là kết quả của phiên bản mô phỏng hiện tại. Sau khi sửa mục 1–2 cần đánh giá lại với ngưỡng đã đóng băng, không đổi ngưỡng để thích nghi kết quả.

**3. [P1] `sharpe_ratio` đang dùng CAGR làm tử số — đã tái hiện**

[FinancialMetrics](/home/tuan/AI/AI_roject1/src/evaluation/metrics.py:63) tính `(CAGR - annual_rf) / annualized_volatility`. Sharpe thông thường dùng trung bình excess return cùng kỳ chia cho độ lệch chuẩn excess return. Với rf cố định, cách tính hàng ngày là `sqrt(252) * mean(r_t - rf_daily) / std(r_t - rf_daily)`; CAGR giữ ở một metric riêng. Đây là định nghĩa trong [bài viết gốc của William Sharpe](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm).

Chuỗi tổng hợp lặp `[+10%, -8%, +4%, -2%]` cho metric hiện tại khoảng **5.64**, trong khi tỷ số từ trung bình excess return khoảng **2.35**. Sai khác ảnh hưởng cả bảng kết quả, ngưỡng Sharpe và việc chọn checkpoint. Sau khi thống nhất định nghĩa, cần tính lại metrics; checkpoint được chọn theo Sharpe cũ có thể không còn tối ưu theo định nghĩa mới.

**4. [P1] Tổng hợp `--seeds` sẽ lỗi vì cột chuỗi — đã tái hiện**

[evaluate_test](/home/tuan/AI/AI_roject1/src/training/train_ppo.py:2059) thêm `checkpoint_status` là chuỗi. [main](/home/tuan/AI/AI_roject1/src/training/train_ppo.py:2142) lấy `.agg(["mean", "std"])` trên toàn bảng, bao gồm cột này. Pandas hiện tại báo `TypeError: dtype 'str' does not support operation 'mean'`. Lỗi xảy ra sau khi các run train/evaluate đã hoàn thành, nên rất tốn thời gian nếu mới phát hiện lúc cuối.

Hướng sửa: chỉ tổng hợp các metric số; giữ epoch/status và các thông tin run trong bảng riêng. Kiểm tra seed không trùng trước khi chạy, vì dictionary hiện tại cũng ghi đè khi seed trùng.

**5. [P1] Null evaluation chỉ hoạt động với GNN — đã tái hiện**

[evaluate_untrained_null](/home/tuan/AI/AI_roject1/src/training/train_ppo.py:772) luôn gọi `_forward_distribution(state, graph)` và unpack ba kết quả. MLP/GRU/LSTM nhận một tham số observation và trả hai kết quả. Cả ba encoder đều báo TypeError khi tái hiện lời gọi này. Parser mặc định `gru`, nên đường null mặc định cũng lỗi.

Hướng sửa: cung cấp API diagnostics chung hoặc phân nhánh như `action_for_model`; không dựa vào chữ ký private riêng của GNN.

**6. [P1] Sharpe delta và bootstrap đang so với hai baseline khác nhau**

[sharpe_delta_vs_matched_baseline](/home/tuan/AI/AI_roject1/src/training/train_ppo.py:2067) dùng 1/N + cash 4%, cùng alpha. Nhưng [CBB](/home/tuan/AI/AI_roject1/src/training/train_ppo.py:1999) dùng 1/N + cash 4.4%, alpha 1.0. Selection metric excess cũng dùng reference 4.4% ở [dòng 1342](/home/tuan/AI/AI_roject1/src/training/train_ppo.py:1342).

Vì vậy CI/p được xuất hiện không kiểm định lợi thế so với baseline matched đang báo ở cột Sharpe delta. Nên có tên baseline rõ ràng cho mỗi phép so và dùng chung matched baseline khi đánh giá đóng góp của việc chọn tài sản. CBB hiện trả tỷ lệ mẫu bootstrap có excess ≤ 0; không nên gọi đó là một kiểm định dưới giả thuyết H0 nếu chưa triển khai phân phối null phù hợp.

**7. [P2] Manifest robustness chưa thực sự quyết định toàn bộ chiến lược**

[YAML](/home/tuan/AI/AI_roject1/configs/tilt_robustness_gate.yaml:5) có `rsi_orientation` và `score_weights`, nhưng [robustness script](/home/tuan/AI/AI_roject1/scripts/evaluate_tilt_robustness.py:109) hard-code orientation `1`, còn [signal](/home/tuan/AI/AI_roject1/scripts/evaluate_equal_weight_tilt.py:69) hard-code 50/50. Sửa YAML ở hai mục này không đổi chiến lược chạy. Cổng còn yêu cầu `mean_sharpe_delta > 0` trong code nhưng không khai báo ngưỡng này trong YAML.

Giá trị mặc định hiện khớp code nên không gây sai số riêng ở lần vừa chạy. Tuy nhiên khả năng tái lập và ý nghĩa “đóng băng cấu hình” chưa đầy đủ. Cần đọc toàn bộ tham số, validate chúng, lưu bản cấu hình đã resolve và hash source/data cùng kết quả. Nên xuất quyết định tổng thể yêu cầu qua tất cả baseline, bên cạnh quyết định riêng từng baseline.

**8. [P2] Dropout trong GRU/LSTM phá tính nhất quán của log-prob**

[Recurrent actor-critic](/home/tuan/AI/AI_roject1/src/models/actor_critic.py:167) cho phép dropout khi nhiều lớp; rollout và evaluate đều ở train mode nhưng mỗi lần dùng mask khác. Lời gọi evaluate trước khi đổi tham số vẫn cho log-prob khác rollout. GNN đã chặn trường hợp này ở constructor, GRU/LSTM chưa chặn.

Đã tái hiện với GRU hai lớp, dropout 0.5: chênh lệch log-prob tối đa khoảng **0.00336** dù chưa bước optimizer. Mức sai lệch phụ thuộc trọng số và trạng thái; mặc định một lớp, dropout 0 không mắc lỗi. Cần giữ encoder policy deterministic khi thu rollout và reevaluate, hoặc quản lý mask một cách nhất quán.

**9. [P2] Kiểm tra căn ngày/ticker và chất lượng OHLCV chưa đủ**

- [build_market_data_tensor](/home/tuan/AI/AI_roject1/src/features/pipeline.py:204) chỉ so số dòng rồi gán theo vị trí; hai mã cùng số dòng nhưng đảo ngày vẫn được chấp nhận. Đã tái hiện với một DataFrame đảo thứ tự ngày. Đường `run_from_raw` có aligner nên được bảo vệ phần này, nhưng API public build tensor chưa bảo vệ.
- [TradingEnv](/home/tuan/AI/AI_roject1/src/env/trading_env.py:116) kiểm tra shape/tickers/features và phạm vi history index, nhưng chưa so dates của market với history slice. Caller dùng sai history index vẫn có thể nhận feature thuộc ngày khác, kể cả ngày tương lai.
- [Graph sequence](/home/tuan/AI/AI_roject1/src/graph/dynamic_graph.py:80) reindex ticker rồi chỉ so shape. Thiếu ticker được biến thành cột NaN và đồ thị rỗng thay vì báo lỗi. Đã tái hiện với ticker không tồn tại.
- `DataValidator` không được gọi trong đường pipeline chính. Một nến có Open = 3 × Close và High = Low bị validator từ chối nhưng `FeaturePipeline.process_asset` vẫn chấp nhận trong tái hiện.

Hướng sửa: assert index/date/ticker uniqueness và equality ở các ranh giới API; gọi validator trước khi tính feature; phân biệt warm-up hợp lệ với dữ liệu bị hỏng để không che lỗi bằng fillna(0).

**10. [P2] Artifact chưa đủ tự chứa để tái lập, và dễ mất log khi ngắt chạy**

[prepare_data](/home/tuan/AI/AI_roject1/src/training/train_ppo.py:524) ghi đè một scaler chung; [checkpoint](/home/tuan/AI/AI_roject1/src/training/train_ppo.py:1168) chỉ lưu đường dẫn scaler. Một run với dữ liệu/split khác có thể thay scaler mà checkpoint cũ vẫn trỏ cùng đường dẫn.

[CSV training log](/home/tuan/AI/AI_roject1/src/training/train_ppo.py:1834) chỉ được ghi sau khi vòng train hoàn tất. Ctrl+C đi qua finally để đóng logger rồi thoát trước dòng ghi CSV, nên CSV có thể không có các epoch mới hoặc vẫn là file của lần trước.

Nên có thư mục riêng cho từng run, nhúng scaler params vào checkpoint, lưu cấu hình đã resolve, split dates, feature params, source/data hashes. Flush log mỗi epoch và lưu thông tin dừng chạy. Nếu muốn resume đúng quỹ đạo, cần scheduler state và RNG state ngoài optimizer/model state.

**11. [P2] Warm-up và các chỉ báo rủi ro cần làm rõ**

[Momentum](/home/tuan/AI/AI_roject1/src/features/momentum.py:23) cần 252 phiên nhưng dữ liệu bắt đầu ngay đầu train 2015. Khoảng 252 phiên đầu `mom_12_1` là 0 do fillna; `dist_52w_high` dùng đỉnh từ đầu dữ liệu khi chưa đủ một năm. Nên tải một năm trước ngày train hoặc có availability mask và ngày bắt đầu train sau warm-up.

[Downside volatility](/home/tuan/AI/AI_roject1/src/features/risk.py:47) hiện là standard deviation có điều kiện trên những ngày âm. Nó không phải downside deviation quanh 0 và không phản ánh tần suất ngày âm. Không nhất thiết sai nếu đó là định nghĩa chủ ý; tên và báo cáo phải nói rõ hoặc đổi sang downside semideviation.

Các nhãn rank-IC 5/20 phiên trong [script phân tích](/home/tuan/AI/AI_roject1/scripts/analyze_feature_signal.py:56) là Close→Close trong khi lệnh đầu tiên khớp tại Open ngày kế tiếp. Đó là nhãn khảo sát hợp lệ nhưng chưa tương đương lợi nhuận mà một lệnh mới có thể nhận; nên thêm nhãn từ thời điểm khớp lệnh, purging và CI theo block cho các horizon chồng lấn.

**12. [P2] Đường legacy và cài đặt chưa đồng bộ**

- [Baseline.run](/home/tuan/AI/AI_roject1/src/evaluation/baselines/equal_weight.py:59) và Buy & Hold `.run()` còn timeline Open→Open, T−2; env hiện dùng Close→Open→Close, T−1. API `run_strategy` đang dùng env, nhưng hai đường public có thể cho kết quả khác nhau. Nên giữ một engine mô phỏng duy nhất và cập nhật [ENV_CONVENTIONS](/home/tuan/AI/AI_roject1/manual/ENV_CONVENTIONS.md).
- [ObservationAdapter](/home/tuan/AI/AI_roject1/src/models/adapters.py:71) dùng graph gộp và snapshot, khác policy PPO mới có ba quan hệ và GRU lịch sử. Nó chưa phải inference adapter tương thích checkpoint hiện tại.
- [MARL wrapper](/home/tuan/AI/AI_roject1/src/env/marl_env.py:69) thiếu validate từng action. Thiếu một agent sẽ làm tổng budget dưới 1 rồi env normalize lại, tự đổi budget các agent còn lại. Local obs thiếu history/graph và dùng weights toàn danh mục mà chưa định nghĩa local normalization.
- [requirements.txt](/home/tuan/AI/AI_roject1/requirements.txt:7) thiếu `torch-geometric` dù GNN import thư viện này. Môi trường hiện tại có sẵn nên test chạy được; cài mới theo README chưa đảm bảo chạy GNN.
- Sáu test fail vẫn kỳ vọng T−2, phí cũ hoặc full allocation vào một mã dù config giới hạn 20%. Cần fixture cấu hình riêng và assertion theo timeline hiện tại; thêm trường hợp overnight gap, hold, lịch tái cân bằng và aggregation đa seed.

**Các phần đã được kiểm tra là nối đúng**

- SPY được giữ ngoài tensor/action trong đường pipeline chính.
- Phí cổ phiếu không cộng thêm chân cash; commission và slippage được resolve và áp dụng; trần vị thế có thực thi.
- Observation và NAV trong `step` đều được đánh dấu Close[t+1]. Lỗi lệch timeline được phản ánh ở bản cũ đã được sửa trong đường PPO chính.
- GNN hiện có lịch sử thời gian, ba relation channels, fusion và GRU; GAT truyền trọng số cạnh bằng `edge_attr`.
- Thử tính feature trên prefix 70 phiên so với 100 phiên cho sai khác **0** ở phần chung. Ma trận correlation tại cuối prefix cũng sai khác **0**. Các phép này kiểm tra nhân quả của phép tính; không thay thế kiểm tra alignment của từng caller.

**Cải thiện hiệu năng có giá trị**

- Hai graph buffers dense `[steps,L,R,N,N]` với khoảng 6,016 bước, L=20, R=3, N=24 tiêu thụ khoảng **1.55 GiB**, chưa tính obs, activation và optimizer. Lưu chỉ số ngày cho từng transition và dùng graph sequence chung sẽ tránh lặp dữ liệu.
- GNN đang encode lại các ngày trùng nhau trong các lookback. Khi policy cố định trong rollout/evaluation và dropout=0, có thể cache embedding theo ngày cho đúng phiên bản model. Cache phải bị invalidated sau mỗi thay đổi tham số; không dùng embedding cũ trong cập nhật gradient.
- Đánh giá train toàn kỳ mỗi 5 epoch là đủ cho diagnostics; giữ lịch validation/checkpoint đã quy định. Đo thời gian rollout/update/eval và bộ nhớ trước khi chọn tối ưu.

Thứ tự sửa đề nghị: (1) hold và khởi tạo exposure; (2) Sharpe và baseline của bootstrap; (3) các lỗi `--seeds`/null và cập nhật test; (4) alignment/manifest/artifact; sau đó chạy lại baseline và cổng giữ nguyên. Các tiêu chí và kết quả nghiên cứu trước đây cần được gắn với phiên bản engine tương ứng.

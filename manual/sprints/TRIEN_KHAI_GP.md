# Triển khai GP temporary-cost — 20/10/2026

## 1. Đã đưa vào dự án

Đã triển khai pipeline chạy độc lập từ predictor lock → calibration/dynamics → conditional risk/cost → GP reference → constrained allocator/MPC → execution/accounting → báo cáo.

Đặc tả thực thi, giả định và các lệnh: [GP_CONVENTIONS.md](GP_CONVENTIONS.md). Kế hoạch ban đầu: [KE_HOACH_TRIEN_KHAI_GP.md](KE_HOACH_TRIEN_KHAI_GP.md).

| Module | Trách nhiệm |
|---|---|
| `src/models/signal_dynamics.py` | H5 forecast thành one-step calibrated alpha; centered AR diagonal + intercept; conditional total-return residual risk; save/load |
| `src/env/quadratic_cost.py` | Diagonal quadratic surrogate từ OHLCV square-root scenario và phép đổi shares/weights |
| `src/evaluation/gp_allocator.py` | Bellman reference, constrained step, MPC có terminal value, primal/objective certificate, numerical oracle |
| `src/models/gp_adapter.py` | Cùng aim/speed controls, equity budget chung, final execution orders và snapshot inference |
| `src/evaluation/gp_research.py` | Expanding OOF/refits, inner, optional full development outer, paired bootstrap/Holm, artifacts và case resume |
| `scripts/compare_gp_allocators.py` | CLI; giới hạn CPU BLAS thread, không có tùy chọn test |
| `configs/gp_allocator.yaml` | Search-free profile đầu tiên, giả định chi phí và solver tolerances |

Không sửa predictor lock, không tự train lại PPO/Sector MARL/H-MARL, không bật GP trong các run cũ. GP phải được đo như decision model riêng trước khi dùng làm teacher/residual cho RL.

## 2. Tích hợp môi trường

Thêm `PortfolioOrder.execution(...)` để truyền **final Close proposal** với bool hold mask và dollar caps. Order này yêu cầu cash-funded engine và scalar/adaptive controller off. Môi trường giữ Open drift, quantity holds, mechanical ADV/cap/cash funding và phí, bỏ toàn bộ smoothing bổ sung.

`PortfolioOrder.hold()` và vector action/rebalance cũ tiếp tục dùng nhánh cũ. GP không yêu cầu thay observation width, history mask hoặc weights của actor hiện tại.

Chi phí quadratic chỉ nằm trong planner. Realized NAV thu commission/slippage/impact thật theo scenario, không thêm phí Lambda lần hai. Cả bảy method dùng cùng planned cash ramp và cùng fill constraints. Không gán chênh lệch metrics giữa run này với run cũ cho GP nếu execution protocol khác.

## 3. Hai tầng tối ưu được phân biệt

- **Reference:** nghiệm không constraints của temporary-cost linear/quadratic Bellman contract. Scalar oracle, finite-horizon limit, FOC, aim identity và đơn vị tiền được xuất thành artifact mỗi run.
- **Portfolio:** active NAV-fraction planning approximation; hard budget/position/participation constraints, L1 commission và unconstrained GP continuation. MPC H=3 dùng expected signals từ F, không dùng future forecasts thực nhận.

Không gọi portfolio version là exact constrained GP optimum. Điều kiện nhận candidate là primal residual ≤1e-7 và convex objective gap ≤1e-9 NAV fraction. LP certificate dùng cùng objective/polytope, nên không coi “solver trả một vector khả thi” là đủ.

Nếu SLSQP có sai số feasibility nhỏ, runner có thể L1-repair candidate vào polytope rồi **chứng nhận lại objective gốc**. Repair được log, không biến thành một chiến lược clip/renormalize không kiểm soát. Nếu không đạt gap/feasibility thì dừng; không silently chuyển sang 1/N.

## 4. Lệnh

```bash
python scripts/compare_gp_allocators.py \
  --predictor-lock models/runs/locked_predictor_h5_20261019/predictor_lock.json \
  --gp-config configs/gp_allocator.yaml \
  --fold development_2021
```

Bỏ `--fold` để chạy OOF/inner tất cả bốn development folds. `--evaluate-development-outer` là lựa chọn rõ ràng cho full exploratory walk-forward/gates, không cho kết hợp chọn một fold riêng. Không có test 2024–2025 trong runner này.

Resume giữ cùng các tham số lệnh và thêm `--resume <thư_mục_run>`. Nạp case đã hoàn tất sau khi kiểm tra hashes; chạy lại case dở từ cash, không khôi phục/ghép NAV giữa chừng. Sửa code/config cần run mới.

## 5. Artifacts cần đọc

- `reference_numerical_acceptance.json`: numerical reference checks.
- `decision_protocol.json`: khác biệt execution so với source prediction protocol, có hash mới.
- `fits/<fold>/<fit_id>/`: calibration, dynamics, risk, Lambda, reference matrices và cutoff/provenance.
- `<fold>_refits.csv`: tiến độ refit và residuals.
- `cases/<fold>/<phase>/<method>/decisions.jsonl`: factor/aim/proposal/fill/cost/NAV diagnostics từng ngày.
- `metrics_by_fold.csv`, `daily_returns.csv`: kết quả portfolio theo method/phase.
- `paired_comparison.csv`: chỉ có khi full outer development được chạy; không tạo một gate giả từ inner.
- `gate_report.json`, `manifest.json`: phạm vi, trạng thái, dữ liệu/source/config/model hashes.

Audit độc lập trên run đã hoàn tất, không refit hoặc mở thêm portfolio:

```bash
python scripts/audit_gp_run.py --run-dir models/runs/THU_MUC_GP_DA_HOAN_TAT
```

Audit tạo thư mục mới trong `audits/`, kiểm tra artifact hashes, cutoff causal, bounds/budget, NAV/phí/QP certificates và xuất raw-h5/calibrated-step rank-IC/MSE theo phase. Outcomes chỉ dùng trong báo cáo offline, không đưa ngược vào allocator.

Deterministic GP không được nhân bản năm lần rồi coi là năm seed độc lập. `aim_gp` và `gp_constrained` là hai cách biểu diễn cùng một one-step problem nên không tính thành hai hypotheses alpha độc lập.

## 6. Trạng thái nghiệm thu

- D1–D4: đã có contract, code và scenario metadata. Lambda chưa được hiệu chỉnh bằng fills, chỉ là declared OHLCV scenario.
- D5: đã triển khai reference solver và numerical acceptance artifacts.
- D6: đã nối final proposal vào env; có accounting/feasibility/objective-gap diagnostics.
- D7: pilot OOF/inner một fold để kiểm tra runtime, ghi kết quả bên dưới sau khi hoàn tất.
- D8: runner/gates đã có và đã hoàn tất full development OOF/inner/outer theo profile cố định; **không qua performance gate**. Xem mục 8.
- D9: chưa warm-start RL; cần bằng chứng/benchmark trước khi mở nhánh này.

Không thêm/chạy pytest/unit tests trong lượt này. Numerical diagnostics và real-data pilot không thay thế full unit/integration regression suite. Checklist ticket vẫn mở cho nghiệm thu nhiều fold và đối chiếu ngoài mẫu đúng protocol.

Survivorship bias của universe cố định chưa được giải quyết. Forecast family đã chọn bằng inner; các giai đoạn development đã được đọc, không có độc lập alpha confirmation từ pilot.

## 7. Pilot đã hoàn tất

Run: `models/runs/gp_research_20261019T182556Z_a6d47deb/` (tên UTC 19/10, giờ Việt Nam 20/10). Fold `development_2021`, predictor GRU h5 đã khóa; 14 case = 7 methods × train OOF/inner, không outer/test.

Inner của fold là **năm 2020**. Kết quả trên protocol cash ramp chung mới:

| Method | Sharpe inner | ΔSharpe vs matched | Chi phí tiền | Cash Close trung bình |
|---|---:|---:|---:|---:|
| Matched 1/N | 0.886245 | 0 | 718.48 | 5.520% |
| Convex | 1.023838 | +0.137593 | 604.58 | 5.520% |
| Aim + fixed speed | 0.897363 | +0.011118 | 477.17 | 5.520% |
| GP constrained | 0.881702 | −0.004543 | 523.20 | 5.520% |
| GP MPC | 0.886789 | +0.000544 | 673.60 | 5.520% |
| GP zero calibrated alpha | 0.890777 | +0.004532 | 458.98 | 5.521% |

GP/MPC chưa hơn convex trong pilot này. Bỏ calibrated forecast còn có Sharpe nhỉnh hơn GP constrained, nên chưa có bằng chứng rằng mapping/dynamics mới khai thác alpha tốt. Một fold không đủ quy kết nguyên nhân hoặc chứng minh mô hình GP không hiệu quả nói chung; không tune profile theo bảng này.

Các mục numerical/cost/NAV được ghi trong artifact, không phải full unit test pass. Pilot dùng đúng cùng cấu hình kinh tế sẽ đem đi development walk-forward; các chỉnh sửa sau pilot chỉ mở rộng numerical diagnostics/runtime/resume, không đổi risk aversion, discount, forecast/calibration/cost/constraints hay budget.

## 8. Full development walk-forward đã hoàn tất

Run: `models/runs/gp_research_20261019T182840Z_45322a22/`.

84 case = 4 folds × train OOF/inner/outer × 7 methods. Outer gồm 2019–2022, vẫn là dữ liệu development đã được khảo sát; **không dùng test 2024–2025**. Tất cả tham số kinh tế/fit/scenario được giữ nguyên từ pilot. Resume trên run hoàn tất đã kiểm tra hashes và tái tạo báo cáo từ 84 case có sẵn.

Kết quả temporal paired comparison trên outer:

| Candidate / reference | Mean ΔSharpe theo fold | Net active return năm hóa | CI 95% | Fold ΔSharpe dương | Gate |
|---|---:|---:|---|---:|---|
| GP constrained / matched 1/N | −0.010454 | −1.167% | [−2.545%, +0.366%] | 2/4 | NO-GO |
| GP constrained / convex | −0.002153 | −2.514% | [−4.375%, −0.685%] | 2/4 | NO-GO |
| GP MPC / matched 1/N | −0.037180 | −1.884% | [−3.585%, −0.189%] | 1/4 | NO-GO |
| GP MPC / convex | −0.028879 | −3.231% | [−5.516%, −0.953%] | 2/4 | NO-GO |

Net active return là `252 * mean(daily_candidate - daily_reference)`, không phải hiệu hai CAGR. Holm-adjusted p-value của cả bốn giả thuyết **positive excess** đều là 1.0; không diễn giải nó là xác suất GP đúng/sai. Mean ΔSharpe là trung bình Sharpe delta từng fold; khác đại lượng daily net excess nên hai cột không nhất thiết cùng mức độ chênh lệch.

Ở mức runtime:

- Reference oracle errors đạt tolerance 1e-8; scalar closed-form rate error 0, low-cost limit error khoảng 8.76e-10.
- NAV/PnL reconciliation error lớn nhất khoảng **6.95e-11 đơn vị tiền**.
- QP primal residual lớn nhất khoảng **1.89e-10**, convex objective gap lớn nhất khoảng **4.55e-10 NAV fraction**, dưới các ngưỡng đã khai báo.
- Source files của run không đổi trong lúc chạy; fits/cases có hashes, source/config/library được kiểm tra khi resume.

Đây là kết quả **GP profile đã cài chạy được nhưng chưa tạo lợi thế kinh tế**. Không bật GP làm mặc định và chưa dùng làm teacher/residual cho RL. So GP-vs-convex hiện là so cả pipeline (h5 raw vs calibrated step dynamics), chưa cô lập thuật toán tối ưu với cùng mu một bước. Nếu mở nghiên cứu tiếp, cần đăng ký ablation/calibration/cost assumptions trước; không tune theo outer trong bảng này.

## 9. Audit độc lập đã hoàn tất

Thư mục: `models/runs/gp_research_20261019T182840Z_45322a22/audits/readout_20261019T183801Z_7bc57a9e/`.

- Đã đối chiếu **27.734 quyết định**, **40 snapshot fits** và case hashes; source/config/data/fit hashes đều khớp.
- **0 cutoff violations**: label end không sau fit cutoff, fit cutoff trước decision date.
- Max target/budget violation ≈9.997e-11; cost reconciliation error ≈6.89e-15 tiền; NAV error ≈6.95e-11 tiền.
- Max QP convex gap ≈4.55e-10 NAV fraction, primal residual ≈1.89e-10. Numerical feasibility repair lớn nhất ≈2.14e-7 tổng độ lệch weights; mọi candidate sau repair vẫn được chứng nhận bằng objective gốc.
- `aim_gp` và `gp_constrained` outer Sharpe chỉ khác cỡ 3.4e-8 đến 3.3e-7 theo fold, phù hợp hai cách biểu diễn cùng one-step problem với stopping tolerance khác trajectory rất nhỏ.

`signal_validation_summary.csv` đo trên **nhãn intraday một bước**, không phải nhãn h5 gốc. Trung bình đơn giản của bốn fold outer: raw-h5 forecast có step rank-IC khoảng **+0.01775**, calibrated-step alpha khoảng **+0.00080**. Riêng outer 2022, raw step rank-IC +0.03880 còn calibrated −0.05209; mapping MSE phần lớn không tốt hơn zero-alpha baseline. Đây là chẩn đoán offline của các giai đoạn đã xem, không phải giấy phép tune lại B/F bằng outer.

Điểm nghiên cứu cần ưu tiên nếu tiếp tục là **độ bền ngoài train của mapping/dynamics và cost calibration**, không mặc định tăng độ phức tạp GP hoặc warm-start RL bằng target này. Muốn cô lập allocator phải đăng ký convex-step control cùng calibrated mu trong lượt mới. Audit và numerical checks không thay thế full unit/integration regression suite.

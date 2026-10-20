# Contract GP temporary-cost — 20/10/2026

## 1. Sản phẩm đã triển khai

| Thành phần | Tệp |
|---|---|
| Forecast → step calibration, AR diagonal, residual risk | `src/models/signal_dynamics.py` |
| Quadratic impact scenario và phép đổi đơn vị | `src/env/quadratic_cost.py` |
| GP Bellman reference, constrained QP, MPC, numerical oracle | `src/evaluation/gp_allocator.py` |
| Forecast-only adapter và snapshot/load | `src/models/gp_adapter.py` |
| Causal refits, reporting, gates, resume | `src/evaluation/gp_research.py` |
| Entry point | `scripts/compare_gp_allocators.py` |
| Tham số scenario cố định | `configs/gp_allocator.yaml` |

Reference là mô hình tuyến tính và temporary quadratic cost không constraints. Portfolio dùng **active NAV fractions với planning approximation**, có constraints và unconstrained terminal value; không tuyên bố exact constrained dynamic optimum. Persistent impact và RL warm-start không thuộc phiên bản này.

## 2. Timeline và đơn vị

- Decision và thông tin tại Close[t], giao dịch tại Open[t+1], mark NAV/observation tại Close[t+1]. Vị thế cũ vẫn chịu overnight gap.
- Forecast bank giữ horizon h5 theo lock. Không lấy forecast/5; fit mapping sang cross-sectional intraday return `Close[t+1]/Open[t+1]-1` chỉ trên historical OOF đã có outcome.
- Expected cross-sectional overnight alpha đặt bằng 0. Conditional planning risk dùng residual của **total Close-to-Close returns** trừ intraday expected active return. Đây là giả định có model mismatch; overnight risk/PnL không bị loại khỏi backtest.
- Covariance, expected payoff, cost và discount theo phiên. Risk aversion=10, discount=0.995 theo profile đầu tiên; không dùng PPO gamma hay annualized covariance.
- Planner weights là NAV fractions. `0.5 delta_w'Lambda_w delta_w` là **cost/NAV**; tiền dự kiến nhân NAV. Lambda khác ma trận trong đơn vị shares.
- Với giá p, NAV V cố định trong bước planning: `delta_x = V diag(1/p) delta_w`, `Lambda_x = diag(p) Lambda_w diag(p)/V`. Holdings risk aversion tương ứng `gamma_x=gamma_w/V`, covariance price-change `diag(p) Sigma_return diag(p)` và loading `diag(p) B_return`.
- Giá total return là chỉ số điều chỉnh; số đơn vị của nó không tự là shares thật cho ADV. ADV vẫn lấy raw split-adjusted Close × Volume từ liquidity bank hiện có.

## 3. Calibration và causality

Factors gồm `(forecast-train_mean)/train_std` từng mã và constant=1. Mapping B là diagonal loading có ridge shrink về pooled slope, cộng train intercept; center expected payoff theo từng rổ. Không fit full N×N return regression trên vài trăm ngày.

Stochastic F là AR diagonal, fit chỉ các cặp calendar-liền-kề cùng `fit_id`; không nối qua predictor refit. Persistence được clip vào ±0.995 theo quy tắc khai báo trước, ghi raw/clipped count và half-life. Constant factor có hệ số 1 là deterministic intercept, không phải stochastic unit root. Forecast persistence và lag predictive efficacy được báo riêng.

Covariance là Ledoit–Wolf của residual total returns trong tập fit, có SPD floor 1e-10 được log. Không gọi trailing unconditional risk bank là conditional residual covariance. Model này cố định giữa các refit; đổi tham số rồi solve lại là receding-horizon approximation.

| Phase | Dữ liệu để fit | Cutoff |
|---|---|---|
| train_oof | Chỉ prefix OOF đã mature | Trước decision; refit mỗi 63 phiên |
| inner | Train OOF | train_end, đóng băng trong inner |
| outer development (opt-in) | Train OOF + inner forecasts đã có outcome | inner_end, không dùng outer labels |

Tất cả mapping labels kết thúc không muộn hơn cutoff. Giá Close[t+1] không bao giờ đi vào fit cho decision tại t. Mọi nhánh chia sẻ ngày bắt đầu sau tối thiểu 126 calibration outcomes và cùng purged tail hold.

Predictor family hiện chọn offline bằng inner IC. Do đó OOF backtest là causal conditional trên family đã chọn, không phải mô phỏng live family-selection được biết trước ở quá khứ; inner cũng đã được dùng chọn predictor. Không gọi chúng là holdout mới.

## 4. Cost scenario

- Planning NAV=100,000, eta=0.1, order-weight grid `[0.001,0.005,0.01,0.02]`.
- Fit diagonal nonnegative quadratic surrogate lên square-root cost từ trailing ADV và volatility của calibration prefix; Lambda floor=1e-8.
- Đây là **OHLCV scenario**, chưa có fills để ước lượng impact thật. Lambda được fit ở planning NAV cố định; khi NAV thực thay đổi, mismatch được giữ minh bạch, cần stress theo NAV/multiplier.
- Linear commission+slippage thực lấy từ environment config (hiện tổng 0.15%). Planner constrained/MPC tính thêm L1 này; reference vẫn pure quadratic.
- Realized accounting cash-funded dùng commission/slippage + eta=0.1 scenario cho mọi nhánh. Slippage baseline và impact bổ sung là hai giả định được khai báo; cần hiệu chỉnh nếu có fills để kiểm tra overlap. Không thu thêm Lambda surrogate vào NAV.

## 5. Common budget, constraints và initialization

Không dùng alpha động như một bộ market-timing thiếu forecast tuyệt đối. Cả bảy nhánh khởi tạo 100% cash, có **cùng planned equity budget**:

`b[k] = 0.96 * (1 - 0.75**(k+1))`, với k đếm từ đầu từng episode.

Mỗi bước planner có `sum(stock_weights)=b[k]`, lower/upper quanh `b[k]/N` ±25%, per-asset cap từ env. Active tracking reference là full core `0.96/N`, cố định trong Bellman planning. Early cash ramp được áp như equality constraint, không coi là nghiệm unconstrained GP.

Participation cap=1% ADV, chốt bằng Close ADV và NAV. GP QP/MPC có cap trong planning; mọi nhánh đều chịu cùng cap cơ học tại Open. Convex/1N controls có thể shortfall khi cap/gap bind; record thực fill và cash, không giả vờ đạt hết target. Nếu GP planning constraints infeasible, dừng với lỗi, không âm thầm nới band/budget.

Budget chung là lịch **độc lập holdings/returns**, khác scalar blend quanh weights trôi của benchmark cũ. Vì vậy có decision protocol/hash dẫn xuất; không ghi đè prediction protocol đóng băng hoặc so trực tiếp metrics với run cũ như cùng execution.

## 6. Solver và controls

- `matched_1n`: chia đều planned equity budget.
- `convex`: solver hiện có với action_alpha=1 và cash target theo budget; cùng conditional covariance và realized execution, không sửa objective/solver gốc.
- `aim_fixed`: cùng GP aim, tốc độ 0.25 rồi QP theo metric M và L1 cost.
- `aim_gp`: cùng aim, GP matrix rate, cùng projection/QP metric và linear cost.
- `gp_constrained`: một Bellman step với constraints + L1 cost.
- `gp_mpc`: ba bước hoạch định theo F, covariance/ADV/NAV frozen, budget tương lai đã biết và unconstrained GP terminal value; chỉ lấy action đầu.
- `gp_zero_forecast`: bằng `gp_constrained` nhưng zero toàn bộ calibrated factors, kể cả intercept; vẫn giữ risk/execution.

`aim_gp` và `gp_constrained` phải gần trùng vì là hai biểu diễn cùng one-step quadratic trước L1. Đây là đối chứng số học, không phải hai chứng cứ alpha độc lập. Khác biệt nhỏ có thể đến từ stopping tolerance và trajectory, phải đo.

Convex control giữ raw h5 payoff và horizon h5 của benchmark gốc; GP dùng calibrated step payoff và dynamics. Vì vậy so GP-vs-convex là so **hai pipeline decision đầy đủ từ cùng forecast source**, chưa cô lập riêng thuật toán tối ưu với cùng vector mu một bước. Nếu cần đo riêng contribution của dynamic optimizer, phải khai báo thêm convex-step control dùng cùng Bf trước thí nghiệm mới; không bổ sung/chọn control theo outer vừa xem.

SLSQP chỉ cung cấp candidate. Mỗi QP/MPC được chứng nhận bằng LP tối thiểu hóa linearization của **cùng feasible polytope**; convex objective gap ≤1e-9 NAV fraction và primal residual ≤1e-7 mới nhận. Code 8 không tự được nhận chỉ vì target khả thi. Nếu gap chưa đạt, có một lần restart với objective scale khác đã định trước; thất bại thì dừng. Log status gốc, attempts và certificate.

Sai số feasibility số học có thể được L1-repair bằng LP; sau đó recompute epigraph và chứng nhận lại objective gốc. Không dùng repair thay chứng nhận tối ưu, không đổi constraints. Numerical repair distance được lưu cho audit.

No-trade band mặc định 0 để tránh thay budget bằng overlay tùy ý. `PortfolioOrder.execution()` chuyển final Close proposal vào env: không scalar blend, không ER/asymmetric/brake tiếp. Hold mask và dollar caps chốt tại Close; Open chỉ drift, fill/cash funding và phí. Residual buy cash có thể làm filled weights khác planned, và fees làm cash thực khác nhau rất nhỏ.

## 7. Reference nghiệm thu số học

Mỗi run xuất `reference_numerical_acceptance.json`: scalar closed-form rate, Bellman residual, FOC, aim/proposal identity, long finite-horizon oracle và tiền khi đổi shares/weights. Tolerance 1e-8 cho oracle; core Bellman convergence theo config 1e-10. Đây là numerical diagnostics, không phải performance gate hay chứng minh model thực tế đúng.

Backtest xuất NAV reconciliation từ asset/cash PnL, primal/objective gaps, cost decomposition, participation, planned/fill shortfall, refit cutoffs, factor/model provenance. Chưa thay thế bộ unit/integration tests đầy đủ.

## 8. Lệnh đang chạy được

Pilot một fold, chỉ train OOF + inner:

```bash
python scripts/compare_gp_allocators.py \
  --predictor-lock models/runs/locked_predictor_h5_20261019/predictor_lock.json \
  --gp-config configs/gp_allocator.yaml \
  --fold development_2021
```

Toàn bộ bốn fold, vẫn **không outer/test**:

```bash
python scripts/compare_gp_allocators.py \
  --predictor-lock models/runs/locked_predictor_h5_20261019/predictor_lock.json
```

Full development walk-forward, có outer exploratory và gates, cần chọn rõ:

```bash
python scripts/compare_gp_allocators.py \
  --predictor-lock models/runs/locked_predictor_h5_20261019/predictor_lock.json \
  --evaluate-development-outer
```

Không hỗ trợ `--fold` cùng outer để tạo báo cáo gate từ fold chọn tay. Không có tùy chọn test. Hai GP chính so với 1/N và convex tạo bốn hypotheses định trước, Holm adjustment và temporal block bootstrap trong từng fold; không coi GP deterministic là năm seed độc lập.

Resume cùng source/data/config/library/lock/scope:

```bash
python scripts/compare_gp_allocators.py \
  --predictor-lock models/runs/locked_predictor_h5_20261019/predictor_lock.json \
  --fold development_2021 \
  --resume models/runs/THU_MUC_GP_CAN_TIEP_TUC
```

Resume ở ranh giới case: verify artifact hashes, skip case đã hoàn tất, case dở chạy lại từ cash với cùng fits. **Không hỗ trợ nối tiếp NAV giữa một case đang dở**. Không ghép hai đoạn portfolio giả; sửa code/config cần run mới.

## 9. Phạm vi chưa nghiệm thu

Chi phí từ fills, persistent cross-impact, exact constrained continuation value, market-timing alpha và RL teacher/residual chưa nằm trong bản này. Full development gates đã chạy trong run `gp_research_20261019T182840Z_45322a22`, cả GP constrained/MPC đều NO-GO; xem báo cáo triển khai. 2024–2025 và các development folds đã được xem trước vẫn exploratory; không có cơ chế nào tự phục hồi tính độc lập của holdout.

Xem [kế hoạch gốc D1–D9](KE_HOACH_TRIEN_KHAI_GP.md) và kết quả pilot trong `manual/TRIEN_KHAI_GP.md`.

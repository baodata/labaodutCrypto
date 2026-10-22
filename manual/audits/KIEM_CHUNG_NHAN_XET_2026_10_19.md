# Kiểm chứng các nhận xét kỹ thuật — 19/10/2026

Phạm vi: đọc source/config/tài liệu, đối chiếu artifact hiện có, chạy một nhóm test và kiểm tra số học có kiểm soát. Không sửa source/config của thí nghiệm đang chạy và không backtest thêm giai đoạn test lịch sử.

**Sau kiểm chứng:** đã triển khai bản sửa entropy helper, float64 credit và evaluation dates/default; xem [bản sửa và regression coverage](SUA_LOI_KIEM_CHUNG_2026_10_19.md). Phần mô tả source/precision bên dưới ghi nhận trạng thái trước bản sửa, không phải kết quả kiểm thử sau sửa.

## 1. Kết luận từng nhận xét

| Nhận xét | Kết luận đối với hiện trạng | Bằng chứng và giới hạn |
|---|---|---|
| Chưa chứng minh alpha ngoài mẫu | **Đúng** | 5 seed GRU của run Sprint 6 hiện có đều chọn epoch 0; ΔSharpe outer trung bình +0.00002335, sample std 0.00053219. Đây là policy khởi tạo. GNN chưa có đủ kết quả để kết luận. |
| Survivorship bias | **Đúng về nguy cơ chưa được xử lý** | Universe cố định 24 cổ phiếu, không có historical constituent membership/delisted data. Cảnh báo và metadata hiện chỉ công khai hạn chế, không sửa được selection bias. Chưa định lượng mức thiên lệch. |
| Dữ liệu train hạn chế | **Đúng về nguy cơ, có dấu hiệu overfit thực nghiệm** | Run hiện tại train 2017-01-31–2020-12-31 chỉ 988 ngày; các cửa sổ/seed tái sử dụng cùng thị trường. GRU tăng train metrics trong khi inner 2021 giảm. Với cấu hình h128/F20/N24, GRU 277,018 tham số; GNN 262,789. GNN không nhiều tham số hơn GRU trong cấu hình này, dù update đắt hơn. |
| Credit assignment H-MARL | **Đúng cho phần phân cấp chưa triển khai; sector MARL đã có định nghĩa reward** | `marl_env.py` có `shared` và `sector_active` với pooled-cash attribution; budget sector hiện cố định bằng nhau. Chưa có manager/high-level trainer trong source để nghiệm thu reward liên cấp hoặc manager credit. Không nên mô tả hệ hiện tại như H-MARL hoàn chỉnh. |
| Portfolio accounting cần kiểm thử | **Đúng về yêu cầu mở rộng; không đúng nếu hiểu là chưa có kiểm thử nào** | Nhóm test hiện có pass; stress Sprint 6 có artifact thật. Các kiểm tra mới bên dưới xác nhận một số invariants, nhưng chưa đủ bao phủ mọi tổ hợp cash/cap/split/impact/participation và đặc biệt cần chuẩn hóa precision của PnL credit. |
| GNN học lại market correlation | **Là giả thuyết hợp lý, chưa được chứng minh cho model này** | Residual graph đã có code causal, bank riêng và nhánh ablation. Config mẫu vẫn chọn `graph.covariance_source: legacy`. Chưa có bảng residual vs raw và GRU hoàn chỉnh để xác nhận đóng góp hoặc sự dư thừa của graph. |
| Chi phí nhiều agent | **Là rủi ro cần đo** | Chi phí update GNN trên CPU đã thể hiện trong log. Sector MARL chưa có run hoàn tất; chưa có benchmark latency inference/RAM/throughput và lợi ích ròng để chứng minh hoặc bác bỏ chi phí triển khai nhiều agent. Training latency không phải phép đo deployment latency. |
| Tái lập end-to-end | **Đã có bằng chứng nhỏ, chưa đủ nghiệm thu nghiên cứu đầy đủ** | Test synthetic đi qua train GRU/GCN/GAT, checkpoint/scaler/hash, resume model/optimizer/RNG và CLI multi-seed đều có trong nhóm test pass. Chưa có full real-data multi-seed/multi-fold hoàn tất; resume/state các nhánh Sprint 6 mới cần coverage riêng. |

## 2. Entropy GAT: nhận xét cần phân biệt ba cơ chế

### Hệ số attention trong GATv2

GATv2Conv đã softmax theo tập cạnh đi vào mỗi node, cho từng head, với self-loop theo cấu hình. Không tìm thấy lỗi thay softmax theo hàng xóm thành softmax trên toàn bộ graph trong encoder hiện tại.

Nguồn: [PyG GATv2Conv](https://pytorch-geometric.readthedocs.io/en/2.5.2/generated/torch_geometric.nn.conv.GATv2Conv.html).

### Helper entropy trong đồ án

`src/models/gnn/gat_encoder.py:169` nhận alpha `[E,H]`, tính tổng theo head rồi mean theo cạnh:

`L_helper = sum_(edge,head)(-alpha * log(alpha + epsilon)) / E`.

Đây không phải entropy trung bình mỗi node/head. Nếu alpha đã normalize theo node thì, bỏ qua epsilon, `L_helper = (N_active * H / E) * mean_node_head_entropy`. Vì vậy scale thay đổi với mật độ cạnh và số head. Hàm cũng không nhận edge_index nên không thể tự kiểm tra normalization hoặc degree của từng node.

Kiểm tra số học với 8 node, một head, attention uniform:

| Degree | Helper hiện tại | Entropy mỗi node |
|---|---:|---:|
| 2 | 0.34657359 | 0.69314718 |
| 8 | 0.25993016 | 2.07944154 |

Nếu dùng regularizer, cần aggregate `-alpha log alpha` theo node đích, trung bình node/head, quy định rõ self-loop/degree=1/isolated nodes. Có thể dùng entropy/log(degree) khi mục tiêu là so độ tập trung tương đối giữa graph khác degree. Lấy attention trước dropout hoặc đảm bảo dropout=0; các alpha sau attention dropout không nhất thiết còn tổng bằng 1.

### Loss đang được train

`src/training/ppo_core.py:319` dùng `loss_pi + coef_v * loss_v - entropy_coef * distribution_entropy`. Distribution ở đây là **Dirichlet action**, không phải graph attention. Không tìm thấy lời gọi `compute_attention_entropy_loss` từ trainer/updater hiện tại; việc module có helper không có nghĩa helper đã tham gia học.

Vì vậy đoạn tài liệu GNN-005 nói entropy attention được đưa vào tổng RL loss chưa phản ánh pipeline PPO đang chạy.

Residual và attention tập trung không cung cấp bằng chứng thực nghiệm rằng embeddings chưa over-smooth. Cần đo cosine similarity giữa node, variance cắt ngang và effective rank trước/sau encoder, theo ngày/relation/layer, với ablation cùng dữ liệu. Chưa thể kết luận model này đã over-smooth chỉ từ lý thuyết hoặc portfolio gần 1/N.

Nguồn lý thuyết: [Wu et al., Demystifying Oversmoothing in Attention-Based Graph Neural Networks](https://arxiv.org/abs/2305.16102). Kết quả của bài không thay thế việc đo encoder hai lớp có residual/LayerNorm của đồ án.

## 3. Taxonomy: 5 sector và 4 agent groups nhất quán trong config

Đếm trực tiếp `configs/assets.yaml`:

| Sector | Cổ phiếu | Agent group |
|---|---:|---|
| Technology | 6 | technology |
| Financials | 5 | financials |
| Healthcare | 5 | healthcare |
| Energy | 3 | energy_consumer |
| Consumer | 5 | energy_consumer |

Tổng 24 cổ phiếu, 5 sector, 4 low-level agent groups với số tài sản 6/5/5/8. SPY chỉ là benchmark. Quy tắc ghép Energy+Consumer đã khai báo trong config và wrapper đọc `agent_group` từ đó.

Như vậy **không có mâu thuẫn logic giữa số sector và số agent**. Tài liệu nào viết một agent cho mỗi sector hoặc 5 sector agent cần sửa thành phân nhóm hiện tại. Đây là taxonomy riêng của đồ án; lời giải thích ghép nhóm vẫn cần được nêu trong báo cáo và ablation.

## 4. Holdout: nhận xét đúng nguyên tắc, mốc thời gian cần cập nhật

Hiện có ba tầng khác nhau:

- `configs/assets.yaml`: train đến 2019, val 2020–2022, test bắt đầu **2023-01-01**.
- `src/data/split.py`: fallback mặc định test bắt đầu **2024-01-01**.
- Protocol Sprint 5/Sprint 6 đang chạy: fold development_2022 train đến 2020, **inner 2021** chọn checkpoint, **outer 2022** báo cáo, không đánh giá test.

`--evaluation-scope validation` cắt raw input theo cutoff trước alignment/feature và không tạo test tensor. Đây là bảo vệ thực sự cho các run protocol hiện tại.

Tuy nhiên `configs/training.yaml` vẫn mặc định `evaluation.scope: test`; CLI không dùng protocol có thể tiếp tục backtest test. Chưa thấy cơ chế one-shot final evaluation hoặc ledger khóa quyền đọc final holdout. Cảnh báo lịch sử trong `ppo_evaluation.py:102` chỉ so `args.test_start` với fallback 2024; trong khi mốc từ assets được resolve ở biến local của data preparation. Vì vậy cảnh báo có thể bỏ sót trường hợp không truyền CLI date hoặc test bắt đầu 2023.

2024–2025 đã xuất hiện nhiều lần trong lịch sử nghiên cứu nên không còn là final holdout độc lập. Đổi tên hoặc mở rộng thành 2023–2025 không khôi phục tính độc lập của phần đã xem. Phần dữ liệu final xác nhận cần thật sự chưa được dùng, khóa protocol/model/data/cost trước khi mở; nếu chưa có thì báo cáo kết quả hiện tại là exploratory.

## 5. Kiểm chứng đã chạy trong lượt này

### Pytest hiện có

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/env \
  tests/integration/test_data_leakage.py \
  tests/integration/test_training_artifacts.py \
  tests/data/test_assets_config.py \
  -q --disable-warnings --basetemp /tmp/ai-project1-audit-20261019
```

**76 passed, 3 warnings, 18.36 giây.** Đây là nhóm test đã chọn, không phải tuyên bố toàn bộ test suite pass.

### Artifact thật được đối chiếu

- Run ablation: `models/runs/sprint6_ablation_20261018T171523Z_800f6bda`. 5 GRU đã completed/NO-GO/epoch 0. GNN seed 7 chưa có báo cáo outer hoàn tất lúc đối chiếu.
- Bank/stress: `models/runs/sprint6_banks_gru_seed7_20261018T171514Z_04bd7819`: completed, 12 case eta/NAV/controller. Max `cost_reconciliation_error` = 2.728484e-12 tiền tài khoản. Phép đo này đối chiếu tổng commission+slippage+impact với total costs, không tự chứng minh mọi NAV invariant.

### Kiểm tra bổ sung trong bộ nhớ

- Sample/EWMA/Ledoit–Wolf: sửa returns sau cutoff 25, covariance đến cutoff không đổi; eigenvalues không âm ngoài tolerance 1e-12 trong mẫu 45 ngày/3 mã/window 10.
- Bốn chế độ legacy, cash-funded, impact, adaptive: mẫu 5 ngày/2 mã có gap qua đêm, intraday và hold. Công thức NAV theo hai phân kỳ khớp; tổng phí khớp; hold giữ shares; commission trên actual stock notional khớp trong tolerance serialization.
- Kiểm tra **PnL credit** ở tolerance USD 1e-7 ban đầu thất bại. Đo lại sai số tuyệt đối lớn nhất giữa `sum(asset_net_pnl)+cash_pnl` và delta NAV: legacy 2.03e-5, cash-funded 7.03e-6, impact 1.63e-5, adaptive 3.08e-5 USD trên NAV 100,000 USD. NAV reconstruction của cả bốn mẫu có sai số 0 trong phép tính đã dùng. Các số này gợi ý cần thống nhất float64 cho PnL credit/định nghĩa tolerance; chưa phải bằng chứng NAV sai lớn. Không đánh dấu kiểm tra PnL strict này là pass.

## 6. Những việc thật sự còn mở

1. Chứng minh tín hiệu/lợi thế ròng bằng protocol đóng băng qua nhiều fold; không dùng kết quả fallback làm bằng chứng học được alpha.
2. Universe point-in-time/delisting audit hoặc giới hạn rõ mọi kết luận vào rổ cố định đã chọn.
3. Sửa mô tả entropy attention đang active; nếu triển khai thì định nghĩa node/head aggregation và đo embeddings trực tiếp.
4. Final holdout policy/ledger, ngày split resolved trong metadata/cảnh báo và default development scope.
5. Accounting/covariance/normalization/resume coverage cho các nhánh Sprint 6 mới, bao gồm precision credit.
6. Nghiệm thu sector MARL và thiết kế reward/high-level credit trước khi trình bày hệ thống như H-MARL hoàn chỉnh.

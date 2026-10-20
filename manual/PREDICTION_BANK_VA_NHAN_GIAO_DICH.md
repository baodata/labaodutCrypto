# Prediction bank và nhãn có thể giao dịch

Triển khai: 19/10/2026. Schema: `prediction_bank_v1`.

## 1. Timeline và đơn vị

```text
Close[t]                 Open[t+1]                 Close[t+h]
feature → dự báo          bắt đầu vị thế            đánh dấu kết thúc nhãn
```

`h` đếm phiên giao dịch đã căn, không đếm ngày lịch:

```text
realized_return[i,t,h] = adjusted_Close[i,t+h] / adjusted_Open[i,t+1] - 1
realized_cross_sectional_excess_return
    = realized_return - mean(realized_return trên các mã cùng ngày/horizon)
```

- `h=1`: Open đến Close phiên kế tiếp, không chứa gap Close[t] → Open[t+1].
- `h=5`: vào Open[t+1], đánh dấu cuối tại Close[t+5]; gồm các đêm nắm giữ bên trong khoảng đó.
- Đơn vị là gross simple return thập phân: `0.01` là 1%; chưa annualize hoặc trừ phí.
- Target Ridge vượt trung bình cắt ngang của rổ cổ phiếu, không phải vượt SPY/matched portfolio có cash và phí.
- Nhãn không thay thế PnL danh mục hiện hữu: cổ phiếu đang giữ vẫn chịu gap qua đêm trong TradingEnv.
- Open/Close dùng chung hệ số Adj Close / Close như simulator. Runner mới yêu cầu Adj Close; không tự fallback price-only rồi ghi metadata total return.
- Universe vẫn là rổ cố định từ assets config, chưa có point-in-time membership. SPY không nằm trong action universe.

Công thức dùng chung: `src/data/labels.py`. Các đường IC/Ridge trong `feature_signal.py`, `signal_research.py` và `analyze_feature_signal.py` đã gọi chung hàm này. Close-to-close được giữ cho chẩn đoán với `executable=False`; không trộn vào IC nhãn giao dịch. Cột cũ `horizon_days` còn để tương thích, cột chuẩn bổ sung là `horizon_sessions`.

## 2. Dự báo tách khỏi outcome

| Tệp | Nội dung |
|---|---|
| `predictions.csv` | Dự báo và provenance, không có lợi suất thực tế |
| `prediction_fits.json` | Scaler, coefficients/intercept, cutoff và phạm vi dự báo của từng lần fit |
| `prediction_bank_manifest.json` | Schema, ticker order, lịch phiên, config/protocol/source/data hash và hash artifact |
| `labels.csv` | Lợi suất thực tế, chỉ dành cho predictor fit trên quá khứ hoặc đánh giá offline |
| `labels_manifest.json` | Công thức/đơn vị, data/protocol hash và hash của nhãn |
| `manifest.json` | Trạng thái run, cấu hình nguồn và số forecast/fit |

Khóa duy nhất: `fold_id, split, model_id, horizon_sessions, decision_date, ticker`.

Các trường bắt buộc khác gồm `execution_date`, `label_end_date`, `prediction`, `fit_id`, `training_start`, `training_cutoff`, `last_training_label_date`, `selection_cutoff`, `prediction_kind`, `target`, `label_definition`, `return_basis`.

Cutoff là ngày Close bao gồm trong phạm vi fit. Mọi nhãn fit và scaler phải sẵn có trước ngày dự báo. `selection_cutoff` là giới hạn thông tin lựa chọn; runner fixed-config không tìm hyperparameter.

Validator chặn khóa trùng, panel thiếu mã, dữ liệu không hữu hạn, ticker/calendar lệch, sai horizon, cutoff không causal và endpoint vượt biên phase. Không ghi đè bank đã có. Hash phát hiện artifact bị sửa.

Validator kiểm tra provenance được khai báo, không chứng minh một predictor bên ngoài thực sự tuân thủ causal preprocessing. Đơn vị expected return cũng chưa chứng minh calibration ngoài mẫu.

## 3. Ba phase và Ridge causal

Config mới `configs/prediction_bank.yaml` tách khỏi protocol đã đóng băng:

| Phase | Fit predictor | Dùng cho |
|---|---|---|
| `train_oof` | Expanding window; fit kết thúc trước ngày đầu mỗi block | Huấn luyện allocator |
| `inner` | Freeze predictor/scaler fit đến train_end | Chọn cấu hình allocator |
| `outer` | Refit predictor/scaler đến inner_end, hyperparameter cố định | Báo cáo fold phát triển |

Mặc định: Ridge alpha=1, feature `xs_rank_ret_20d`, `mom_12_1`, `excess_ret_20d`, horizon 1/5/20 phiên. OOF bắt đầu sau 252 phiên feature-ready, refit mỗi 63 phiên. Không lấp lịch sử chưa đủ bằng dự báo in-sample. Config bổ sung hai baseline calibrate trên quá khứ: momentum 12–1 và mean-reversion theo return 5 phiên. Xem [đối chiếu ba model và allocator chi phí](RIDGE_MOMENTUM_VA_ALLOCATOR_CHI_PHI.md).

Scaler fit riêng trên quá khứ của mỗi lần fit. Nhãn cuối mỗi phase phải kết thúc trong phase, nên `h` phiên quyết định cuối phase không có forecast trong bank đánh giá này. N=1 có target tương đối và dự báo center bằng 0.

Không chọn alpha bằng inner tương lai rồi dùng ngược để tạo lại `train_oof` như thể alpha đó được biết trong quá khứ. Đổi config sau khi xem kết quả là một thử nghiệm phát triển khác. Runner không random K-fold, không chọn theo outer.

Protocol cung cấp folds/development cutoff. Grid tìm tilt và performance gates của signal-research cũ không được runner bank thực thi. Đây là bước tạo dữ liệu dự báo, chưa phải thí nghiệm chứng minh alpha.

## 4. Lệnh tạo bank

Từ repository root trong virtualenv:

```bash
python scripts/build_prediction_bank.py \
  --prediction-config configs/prediction_bank.yaml \
  --research-protocol configs/quant_research_protocol.yaml
```

Mặc định tạo thư mục mới `models/runs/prediction_bank_<timestamp>_<id>/`. Có thể chỉ định `--output-dir` tới thư mục mới. Feature/label chỉ tính đến `development_end`; không train PPO/MARL hoặc backtest test 2024–2025.

## 5. API đọc cho allocator

```python
from src.evaluation.prediction_bank import PredictionBank

run_dir = "models/runs/prediction_bank_<timestamp>_<id>"
bank = PredictionBank.load(run_dir)  # không đọc labels.csv
model_id = bank.metadata["model_ids"][0]
train_predictions = bank.training_panel(
    model_id=model_id, fold_id="development_2022", horizon=5,
)
outer_predictions = bank.panel(
    model_id=model_id, fold_id="development_2022", split="outer", horizon=5,
)
# Date x Asset, columns theo bank.metadata['tickers']; chọn đúng ngày,
# không forward-fill forecast thiếu qua biên phase.
```

Join outcome chỉ khi gọi tường minh cho đánh giá offline:

```python
evaluated = bank.evaluation_frame_from_directory(run_dir)
# Dùng đo IC/calibration; không đưa cột realized_* vào allocator/observation.
```

Không concat các fold có lịch sử train chồng lấn rồi xem như ngày độc lập. Báo cáo OOS dùng outer disjoint; horizon chồng lấn cần xét serial dependence. Dùng cùng dự báo/risk/constraint/execution khi so allocator.

## 6. Phạm vi và trạng thái

Đã có hợp đồng nhãn, builder Ridge/rule causal và API bank; allocator convex có chi phí được bổ sung trong [luồng đối chiếu](RIDGE_MOMENTUM_VA_ALLOCATOR_CHI_PHI.md). Profile riêng đã thêm [supervised GRU/GNN-GRU](GRU_GNN_GRU_PREDICTOR_ABLATION.md) và checkpoint weights có hash. PPO/MARL chưa tự chuyển sang nhận forecast bank.

Lượt tạo hợp đồng ban đầu chỉ kiểm tra cú pháp Python và diff. Kết quả thực nghiệm bổ sung được ghi riêng ở tài liệu đối chiếu; kết quả test của phiên bản trước không nghiệm thu các module mới.

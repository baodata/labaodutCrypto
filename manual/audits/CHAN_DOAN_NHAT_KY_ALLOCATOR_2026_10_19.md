# Chẩn đoán: chất lượng forecast và hiệu quả học allocator

Nguồn: `models/runs/portfolio_decision_locked_h5_20261019/`, 60 training logs và 60 checkpoint `last.pth`.

Artifacts chẩn đoán: `models/runs/allocator_learning_diagnosis_20261019T045002Z_9681ad7b/`.

## Kết luận

Có bằng chứng cho **cả hai vấn đề**: predictor chưa có tín hiệu ổn định theo thời gian, và learner hiện chưa khai thác forecast hiệu quả. Chưa thể quy kết bao nhiêu phần trăm thất bại cho từng nguyên nhân hoặc khẳng định tăng số epoch sẽ giải quyết được.

Đặc biệt, profile tối đa 20 epoch thực tế chỉ chạy **9 epoch cho cả 60 lượt**, do điều kiện patience dùng chung với gate chấp nhận checkpoint. Đây là một yếu tố gây nhiễu cần sửa trước khi kết luận về ngân sách train.

## 1. Forecast không hoàn toàn vô dụng, nhưng chưa ổn định

Mean Spearman rank-IC trên nhãn executable 5 phiên, với family được khóa riêng từng fold:

| Fold | Family | Train OOF | Inner | Outer |
|---|---|---:|---:|---:|
| 2019 | GRU | −0.0057 | +0.0524 | +0.0103 |
| 2020 | GNN-GRU | +0.0165 | +0.0159 | +0.0952 |
| 2021 | GRU | +0.0173 | +0.1057 | −0.0514 |
| 2022 | GNN-GRU | +0.0344 | −0.0427 | +0.0431 |

Nguồn train OOF: join forecasts đã lưu với labels đã lưu, không fit lại predictor. Các số inner/outer khớp bảng IC đã có từ thí nghiệm predictor.

CI95% pointwise của outer IC:

| Fold | Mean IC | CI95% |
|---|---:|---|
| 2019 | +0.0103 | [−0.0635; +0.0796] |
| 2020 | +0.0952 | [+0.0138; +0.1763] |
| 2021 | −0.0514 | [−0.1375; +0.0277] |
| 2022 | +0.0431 | [−0.0191; +0.1023] |

Chỉ một CI pointwise outer nằm hoàn toàn trên 0; các CI này chưa điều chỉnh cho nhiều candidate/lựa chọn family. Inner tốt nhất ở 2021 đảo dấu khi ra outer. Vì vậy không thể nói “không có chút tín hiệu nào”, cũng chưa thể nói đã có alpha ổn định.

Outer chỉ được đọc lại cho chẩn đoán mô tả trên giai đoạn đã khảo sát, không dùng chọn lại family, horizon, checkpoint hoặc cấu hình.

## 2. Vì sao cả 60 lượt đều dừng epoch 9?

Trong `src/training/decision_trainer.py:155`, biến `eligible` đòi hỏi đồng thời:

1. Epoch ít nhất 5.
2. Smoothed inner Sharpe vượt epoch 0 ít nhất 0.03.
3. Sharpe hiện tại vượt best score được chấp nhận.

`stale` chỉ reset khi `eligible=True`. Chưa vượt ngưỡng 0.03 thì dù tăng từ +0.001 lên +0.002 vẫn tăng `stale`. Các epoch 5, 6, 7, 8, 9 tạo đủ 5 lần rồi break.

Như vậy điều kiện dừng đang đo **chưa đạt gate promotion**, không đơn thuần đo **không tiến bộ**. Run hiện tại chưa cho thấy điều gì sẽ xảy ra sau đủ 20 epoch. Đây là hạn chế logic điều phối; không phải bằng chứng rằng chạy lâu hơn chắc chắn có lợi.

Hướng sửa: tách best score dùng cho patience khỏi gate promotion +0.03. Lưu `best_trained.pth` để chẩn đoán độc lập với checkpoint được phép thay baseline. Giữ gate chấp nhận mô hình như cũ.

## 3. Optimizer hoạt động, KL không chặn gần hết cập nhật

| Nhánh | KL trung bình trong log | Adam steps/lượt, trung bình | Tỷ lệ minibatch thực hiện so với lịch dự kiến trong 9 epoch |
|---|---:|---:|---:|
| PPO | 0.0178 | 145.65 | 96.46% |
| Sector MARL | 0.0195 | 143.20 | 94.83% |
| H-MARL | 0.0214 | 134.75 | 89.24% |

Adam steps đọc từ optimizer state trong checkpoint cuối. Lịch minibatch tái dựng từ RNG episode/seed và batch size 256, 5 PPO iterations, 8 episode/update; trung bình 151 steps dự kiến trong 9 epoch.

Có cập nhật tham số thực tế; không có dấu hiệu toàn bộ learner bị đứng hoặc KL guard chỉ cho đi một vài bước. Vẫn có KL early stopping ở một phần minibatch, nhưng log không ủng hộ chẩn đoán “target_kl chặn hầu hết train”.

## 4. Critic cải thiện loss nhưng giải thích return còn yếu

| Nhánh | Value loss trung bình epoch 1 → 9 | Median EV ở epoch 9 | Tỷ lệ lượt có EV cuối âm |
|---|---|---:|---:|
| PPO | 0.014307 → 0.000124 | −0.772 | 95% |
| Sector MARL | 0.014891 → 0.000118 | −1.152 | 95% |
| H-MARL | 0.014998 → 0.000258 | −0.511 | 90% |

Loss nhỏ không đồng nghĩa critic đã học tốt phần biến thiên return. EV âm nghĩa là phương sai sai số dự báo lớn hơn phương sai target theo định nghĩa đang dùng.

Lưu ý: EV được tính từ value thu thập **trước update** trên rollout đang thay đổi; value loss được tổng hợp trong quá trình update. Chúng không phải một bộ đo convergence trên tập probe cố định, cũng không phải metric ngoài mẫu inner của critic. Vì vậy chỉ kết luận hiện có dấu hiệu value learning yếu, chưa kết luận lỗi toán học của critic.

Reward episode trung bình có chiều cải thiện nhưng còn âm. Các episode được random lại và dài 63–126 bước nên không thể dùng riêng đường reward này để kết luận learner hội tụ hoặc học giảm turnover. Log chưa có train deterministic evaluation trên cùng tập cửa sổ cố định.

## 5. Policy cuối còn ít nhạy với forecast

Để tránh nhầm với checkpoint epoch 0, đọc **`last.pth` epoch 9**. Kiểm tra forward trên các ngày inner cách nhau 10 phiên, dùng cùng holdings của matched reference cho mọi policy. So sánh output khi giữ nguyên forecast và khi đặt hai nhóm input forecast (raw và scaled) bằng 0; covariance, holdings và thông tin chi phí giữ nguyên.

Đây là kiểm tra độ nhạy trên state tham chiếu chung, **không phải backtest policy cuối**, không có optimizer update và không dùng kết quả để chọn checkpoint. Zeroing forecast có thể tạo state khác phân phối huấn luyện, vì vậy không xem đây là phép quy kết nhân quả hoặc tỷ lệ feature importance.

| Nhánh | Mean absolute stock target tilt khỏi core | Mean absolute target change khi zero forecast |
|---|---:|---:|
| PPO | 1.214 bp | 0.0416 bp |
| Sector MARL | 1.087 bp | 0.0326 bp |
| H-MARL | 1.565 bp | 0.0502 bp |

1 basis point của weight = 0.0001 = 0.01 điểm phần trăm. Các số trên tính ở **policy target**, trước execution alpha.

Cho riêng PPO, tuyến tính hóa gần core cho độ lệch chuẩn sampling khoảng `0.04 × 0.25 × 0.10 = 0.001`, tức khoảng **10 bp**, trước hiệu ứng projection. Con số này xấp xỉ 8 lần tilt deterministic trung bình 1.214 bp. Đây là ước lượng từ parameterization, không phải phương sai action đo trực tiếp; manager/worker cần đo riêng.

Điều đã quan sát được: policy cuối có thay đổi nhưng còn gần core; việc thay input forecast trong probe làm target đổi rất ít. Chưa có bằng chứng learner đã khai thác mạnh thông tin dự báo. Không nên dùng kết quả epoch 0 trong bảng paired để suy ra đặc điểm này; bằng chứng đến từ checkpoint đã train.

## 6. Ngân sách dữ liệu khác ngân sách optimizer

| Fold | Ngày forecast OOF khả dụng |
|---|---:|
| 2019 | 246 |
| 2020 | 497 |
| 2021 | 749 |
| 2022 | 1002 |

Mỗi lượt có 6,439–6,997 transition draws trong 9 epoch nhưng tái sử dụng các ngày trên. Tái dựng RNG cho thấy đã gặp khoảng 98.2–100% ngày OOF khả dụng. Do đó không có dấu hiệu chỉ vì sampling mà bỏ qua một phần lớn lịch sử.

Tăng epoch tạo thêm cập nhật và tái sử dụng dữ liệu, không tạo thêm quan sát thị trường độc lập. Đặc biệt fold đầu chỉ có 246 ngày OOF để học allocator. Năm seed cũng không tạo thêm năm lần lịch sử thị trường.

## 7. Chưa đủ thông tin để tách nguyên nhân hoàn toàn

Nhật ký hiện thiếu:

- Train deterministic performance trên cùng probe windows qua các epoch, để phân biệt underfit với train tốt/inner xấu.
- Advantage/return variance, gradient norms, policy score scale và actual sampled-weight std.
- KL first/max/stop và số minibatch theo epoch (hiện chỉ khôi phục được tổng Adam steps).
- Best trained weights độc lập promotion gate; log có epoch trung gian nhưng không lưu riêng weights của mỗi epoch.

Vì thế không thể khẳng định “thiếu alpha là nguyên nhân duy nhất” hoặc “chỉ cần train thêm”.

## 8. Thứ tự bước tiếp theo

1. **Sửa patience độc lập promotion gate, bổ sung đo learner trên tập probe train cố định.** Giữ predictor/cost/constraints và gate +0.03.
2. Khai báo trước một budget phát triển cố định; chỉ dùng train/inner để đánh giá learning curves. Không tăng budget/hạ gate theo outer vừa xem.
3. Một đối chứng hữu ích tiếp theo là học bắt chước convex targets trên train OOF: xem actor/action mapping có biểu diễn và fit được hành vi đối chứng hay không. Đây là đề xuất thí nghiệm, chưa chạy.
4. Nếu fit đối chứng được nhưng PPO không khai thác forecast, tập trung reward/advantage/credit assignment và exploration. Nếu allocator học tốt trên train mà suy giảm trên inner, tập trung generalization và signal stability.

**Ưu tiên hiện tại:** gỡ yếu tố gây nhiễu trong việc dừng train và đo hiệu quả learner, trước khi đầu tư thêm độ phức tạp H-MARL. Đồng thời giữ kết luận predictor chưa có tín hiệu bền vững. Chẩn đoán này không thay cấu hình, không train lại và không đánh giá test 2024–2025.

Các CSV và script tái tạo chẩn đoán nằm trong thư mục artifacts nêu ở đầu tài liệu. Không thêm/chạy unit tests trong bước đọc nhật ký này.

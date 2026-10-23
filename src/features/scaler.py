# bộ chuẩn hóa đặc trưng (scaler) dùng cho các mô hình định lượng và RL
# nguyên tắc sống còn: chỉ học tham số (fit) trên tập Train, sau đó áp dụng (transform) cho Val và Test

from __future__ import annotations

from typing import Optional
import numpy as np


class FeatureScaler:
    def __init__(self, clip_value: float = 5.0, eps: float = 1e-8) -> None:
        self.clip_value = clip_value
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.is_fitted = False

    # học giá trị trung bình và độ lệch chuẩn từ tập Train
    # đầu vào train_data có dạng [T_train, N, K]
    def fit(self, train_data: np.ndarray) -> FeatureScaler:
        if train_data.ndim != 3:
            raise ValueError(f"Dữ liệu huấn luyện phải có dạng 3D [T, N, K], nhận được {train_data.shape}")

        # tính mean và std theo từng đặc trưng K (gộp chung cả trục thời gian T và cổ phiếu N)
        # làm như vậy để giữ nguyên tương quan so sánh giữa các cổ phiếu (ví dụ mã nào biến động mạnh hơn)
        self.mean = np.nanmean(train_data, axis=(0, 1), keepdims=True).astype(np.float32)
        self.std = np.nanstd(train_data, axis=(0, 1), keepdims=True).astype(np.float32)

        # nếu std quá nhỏ hoặc bằng 0 thì gán bằng 1 để tránh chia cho 0
        self.std[self.std < self.eps] = 1.0

        self.is_fitted = True
        return self

    # chuẩn hóa dữ liệu dựa trên mean và std đã học từ tập Train
    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.mean is None or self.std is None:
            raise RuntimeError("Scaler chưa được fit trên tập Train! Hãy gọi fit() trước.")

        if data.ndim != 3:
            raise ValueError(f"Dữ liệu phải có dạng 3D [T, N, K], nhận được {data.shape}")

        # công thức z-score: (x - mean) / (std + eps)
        scaled = (data - self.mean) / (self.std + self.eps)

        # cắt ngưỡng ngoại lai (outliers) để tránh làm vỡ gradient khi huấn luyện mạng nơ-ron
        if self.clip_value is not None and self.clip_value > 0:
            scaled = np.clip(scaled, -self.clip_value, self.clip_value)

        return scaled.astype(np.float32)

    # vừa fit vừa transform luôn trên tập Train
    def fit_transform(self, train_data: np.ndarray) -> np.ndarray:
        return self.fit(train_data).transform(train_data)

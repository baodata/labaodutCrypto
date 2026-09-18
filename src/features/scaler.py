"""
src/features/scaler.py
Mô-đun chuẩn hóa đặc trưng độc quyền trên tập Train (Train-Only Market Feature Scaler).

Ticket: SPLIT-002 (P0 - Thành viên A)
Sprint: 3

Mục đích:
1. Thực thi nguyên tắc vàng chống rò rỉ dữ liệu (No Data Leakage / Anti-Lookahead Bias):
   - Mọi thông số thống kê (Mean mu, Std sigma) BẮT BUỘC chỉ được học từ tập Train.
   - Khi chuẩn hóa tập Validation và Test, hệ thống "đóng băng" mu_train và sigma_train
     để biến đổi, tuyệt đối không tính toán lại thống kê trên tập Val/Test.
2. Bảo toàn tính tương đồng chéo giữa các tài sản (Cross-asset comparability):
   - Tính toán mu và sigma trên toàn bộ các mã cổ phiếu cho từng đặc trưng F (kích thước [1, 1, F]),
     giúp mạng GNN và RL bảo toàn được sự so sánh lợi suất/rủi ro tương đối giữa các công ty.
3. Hỗ trợ đa dạng cấu trúc dữ liệu:
   - MarketDataTensor [T, N, F] (đầu vào cho GNN / H-MARL)
   - pd.DataFrame (bảng đặc trưng phẳng)
   - np.ndarray
4. Hỗ trợ lưu trữ và khôi phục tham số chuẩn hóa (save/load) ra tệp JSON
   tại data/processed/scaler_params.json nhằm đảm bảo tính tái lập (reproducibility).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.utils.data_types import MarketDataTensor


class MarketFeatureScaler:
    """Bộ chuẩn hóa đặc trưng thị trường tuân thủ nguyên tắc Train-Only."""

    def __init__(
        self,
        eps: float = 1e-8,
        clip_range: Optional[tuple[float, float]] = (-10.0, 10.0),
    ) -> None:
        """Khởi tạo MarketFeatureScaler.

        Args:
            eps: Epsilon chống chia cho 0.
            clip_range: Khoảng cắt tỉa giá trị ngoại lai (outliers) sau khi chuẩn hóa Z-score
                        (mặc định [-10.0, 10.0] để bảo vệ mạng nơ-ron khỏi gradient nổ).
        """
        self.eps = eps
        self.clip_range = clip_range
        self.is_fitted = False
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.feature_names: List[str] = []

    def fit(
        self,
        train_data: Union[MarketDataTensor, pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> "MarketFeatureScaler":
        """Chỉ học thống kê Mean và Std trên tập Train.

        Args:
            train_data: MarketDataTensor, DataFrame, hoặc numpy array.
            feature_names: Danh sách tên đặc trưng (nếu truyền numpy array).

        Returns:
            self
        """
        if isinstance(train_data, MarketDataTensor):
            # Tensor shape: [T_train, N, F]
            f_names = list(train_data.feature_names)
            t, n, f = train_data.tensor.shape
            # Tính toán thống kê gộp qua cả trục thời gian T và tài sản N cho từng feature
            # shape: [T * N, F]
            reshaped = train_data.tensor.reshape(-1, f)
            mu = np.mean(reshaped, axis=0)
            sigma = np.std(reshaped, axis=0)

        elif isinstance(train_data, pd.DataFrame):
            # Chọn các cột số (loại bỏ cột non-numeric nếu có)
            numeric_df = train_data.select_dtypes(include=[np.number])
            f_names = list(numeric_df.columns)
            mu = numeric_df.mean().to_numpy()
            sigma = numeric_df.std().to_numpy()

        elif isinstance(train_data, np.ndarray):
            if train_data.ndim == 3:
                t, n, f = train_data.shape
                reshaped = train_data.reshape(-1, f)
                mu = np.mean(reshaped, axis=0)
                sigma = np.std(reshaped, axis=0)
            elif train_data.ndim == 2:
                mu = np.mean(train_data, axis=0)
                sigma = np.std(train_data, axis=0)
            else:
                raise ValueError("numpy array phải có 2 hoặc 3 chiều.")
            f_names = feature_names or [f"feature_{i}" for i in range(len(mu))]

        else:
            raise TypeError(f"Không hỗ trợ kiểu dữ liệu: {type(train_data).__name__}")

        # Lưu lại thông số
        self.feature_names = f_names
        self.means = {name: float(m) for name, m in zip(f_names, mu)}
        self.stds = {name: float(s) for name, s in zip(f_names, sigma)}
        self.is_fitted = True
        return self

    def transform(
        self,
        data: Union[MarketDataTensor, pd.DataFrame, np.ndarray],
    ) -> Union[MarketDataTensor, pd.DataFrame, np.ndarray]:
        """Áp dụng thông số chuẩn hóa từ tập Train lên dữ liệu (Train/Val/Test)."""
        if not self.is_fitted:
            raise RuntimeError("Scaler chưa được fit trên tập Train. Hãy gọi .fit() trước.")

        if isinstance(data, MarketDataTensor):
            new_tensor = data.tensor.copy()
            for idx, feat in enumerate(data.feature_names):
                if feat in self.means and feat in self.stds:
                    m = self.means[feat]
                    s = self.stds[feat]
                    # Z-score: (X - mu) / (sigma + eps)
                    new_tensor[:, :, idx] = (new_tensor[:, :, idx] - m) / (s + self.eps)

            if self.clip_range is not None:
                new_tensor = np.clip(new_tensor, self.clip_range[0], self.clip_range[1])

            out_obj = MarketDataTensor(
                tensor=new_tensor.astype(np.float32),
                tickers=list(data.tickers),
                feature_names=list(data.feature_names),
                dates=list(data.dates),
            )
            out_obj.validate()
            return out_obj

        elif isinstance(data, pd.DataFrame):
            new_df = data.copy()
            for feat in self.feature_names:
                if feat in new_df.columns:
                    m = self.means[feat]
                    s = self.stds[feat]
                    new_df[feat] = (new_df[feat] - m) / (s + self.eps)
                    if self.clip_range is not None:
                        new_df[feat] = new_df[feat].clip(self.clip_range[0], self.clip_range[1])
            return new_df

        elif isinstance(data, np.ndarray):
            new_arr = data.copy()
            mu_arr = np.array([self.means[f] for f in self.feature_names])
            std_arr = np.array([self.stds[f] for f in self.feature_names])
            if new_arr.ndim == 3:
                new_arr = (new_arr - mu_arr) / (std_arr + self.eps)
            else:
                new_arr = (new_arr - mu_arr) / (std_arr + self.eps)

            if self.clip_range is not None:
                new_arr = np.clip(new_arr, self.clip_range[0], self.clip_range[1])
            return new_arr

        raise TypeError(f"Không hỗ trợ kiểu dữ liệu: {type(data).__name__}")

    def fit_transform(
        self,
        train_data: Union[MarketDataTensor, pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> Union[MarketDataTensor, pd.DataFrame, np.ndarray]:
        """Vừa học thông số trên Train vừa biến đổi dữ liệu Train."""
        self.fit(train_data, feature_names=feature_names)
        return self.transform(train_data)

    def save(self, file_path: Union[Path, str] = "data/processed/scaler_params.json") -> Path:
        """Lưu trữ thông số chuẩn hóa ra tệp JSON."""
        if not self.is_fitted:
            raise RuntimeError("Chưa có thông số để lưu. Hãy fit scaler trước.")

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": "1.0",
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
            "means": self.means,
            "stds": self.stds,
            "eps": self.eps,
            "clip_range": list(self.clip_range) if self.clip_range else None,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return path

    @classmethod
    def load(cls, file_path: Union[Path, str]) -> "MarketFeatureScaler":
        """Nạp lại thông số scaler từ tệp JSON đã lưu."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file tham số: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        scaler = cls(
            eps=data.get("eps", 1e-8),
            clip_range=tuple(data["clip_range"]) if data.get("clip_range") else None,
        )
        scaler.feature_names = data["feature_names"]
        scaler.means = data["means"]
        scaler.stds = data["stds"]
        scaler.is_fitted = data["is_fitted"]
        return scaler


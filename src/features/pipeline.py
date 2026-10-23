# quy trình tích hợp toàn bộ các bước tính toán và chuẩn hóa đặc trưng
# biến đổi dữ liệu nến thô [T, N, F] thành tensor đặc trưng đã scale [T, N, K]

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from src.features.scaler import FeatureScaler
from src.features.technical import FEATURE_NAMES, compute_technical_features
from src.utils.data_types import MarketDataTensor


# chạy toàn bộ quy trình tính đặc trưng và chuẩn hóa cho train / val / test
def build_feature_pipeline(
    full_data: MarketDataTensor,
    splits: Dict[str, MarketDataTensor],
    warmup_days: int = 30,
    clip_value: float = 5.0,
) -> Tuple[Dict[str, MarketDataTensor], FeatureScaler]:
    # 1. Trích xuất giá đóng cửa và khối lượng từ dữ liệu toàn bộ chuỗi
    close = full_data.get_feature("close")
    volume = full_data.get_feature("volume")

    # 2. Tính toán các chỉ báo kỹ thuật trên toàn bộ chuỗi thời gian liên tục
    # việc tính trên toàn bộ chuỗi đảm bảo ngày đầu tiên của tập val và test
    # đã có đầy đủ dữ liệu quá khứ nhìn lại (lookback window), không bị dính NaN
    raw_features, feat_names = compute_technical_features(close, volume)

    dt_series = pd.to_datetime(full_data.dates)

    # 3. Phân chia đặc trưng theo từng tập train, val, test
    split_raw_features: Dict[str, np.ndarray] = {}
    split_dates: Dict[str, np.ndarray] = {}

    for split_name, sp in splits.items():
        start_dt = pd.to_datetime(sp.dates[0])
        end_dt = pd.to_datetime(sp.dates[-1])
        mask = (dt_series >= start_dt) & (dt_series <= end_dt)

        # Tập train phải bỏ qua warmup_days đầu tiên để loại bỏ các giá trị NaN khởi động
        if split_name == "train":
            mask[:warmup_days] = False

        indices = np.where(mask)[0]
        split_dates[split_name] = full_data.dates[indices]
        split_raw_features[split_name] = raw_features[indices]

    # Kiểm tra xem có NaN nào lọt vào không
    for name, arr in split_raw_features.items():
        if np.isnan(arr).any():
            raise ValueError(f"Phát hiện giá trị NaN trong đặc trưng của tập {name}!")
        if np.isinf(arr).any():
            raise ValueError(f"Phát hiện giá trị Inf trong đặc trưng của tập {name}!")

    # 4. Khởi tạo và Fit Scaler CHỈ TRÊN TẬP TRAIN (ngăn chặn rò rỉ dữ liệu)
    scaler = FeatureScaler(clip_value=clip_value)
    scaler.fit(split_raw_features["train"])

    # 5. Dùng mean và std của tập Train để Transform cho cả Train, Val, Test
    scaled_splits: Dict[str, MarketDataTensor] = {}
    for split_name, raw_arr in split_raw_features.items():
        scaled_arr = scaler.transform(raw_arr)

        # Đóng gói thành MarketDataTensor chuẩn hóa
        scaled_splits[split_name] = MarketDataTensor(
            dates=split_dates[split_name],
            tickers=list(full_data.tickers),
            feature_names=list(feat_names),
            values=scaled_arr,
            benchmark_ticker=full_data.benchmark_ticker,
            # Benchmark values nếu có thì cũng cắt theo ngày tương ứng
            benchmark_values=None,
        )

    return scaled_splits, scaler

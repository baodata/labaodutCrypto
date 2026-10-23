# module này dùng để tính toán các chỉ báo kỹ thuật cơ bản cho từng cổ phiếu
# đầu vào là giá đóng cửa và khối lượng, đầu ra là mảng đặc trưng [T, N, K]

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd


# Danh sách tên các đặc trưng kỹ thuật được tính toán
FEATURE_NAMES = [
    "return_1d",          # lợi suất 1 ngày
    "return_5d",          # lợi suất 5 ngày (1 tuần)
    "volatility_20d",     # độ biến động 20 ngày (độ lệch chuẩn lợi suất)
    "rsi_14",             # chỉ báo sức mạnh tương đối RSI 14 ngày (chuẩn hóa 0-1)
    "macd",               # đường MACD chuẩn hóa theo giá đóng cửa
    "macd_signal",        # đường tín hiệu Signal của MACD
    "volume_ratio_20d",   # tỷ lệ khối lượng so với trung bình 20 ngày
]


# tính toàn bộ các chỉ báo kỹ thuật từ giá close và volume của N cổ phiếu
def compute_technical_features(
    close: np.ndarray,
    volume: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    # close và volume có kích thước [T, N]
    df_close = pd.DataFrame(close)
    df_volume = pd.DataFrame(volume)

    # 1. Tính lợi suất 1 ngày và 5 ngày
    ret_1d = df_close.pct_change(1).values
    ret_5d = df_close.pct_change(5).values

    # 2. Tính biến động 20 ngày (độ lệch chuẩn rolling 20 phiên của lợi suất 1 ngày)
    vol_20d = pd.DataFrame(ret_1d).rolling(window=20).std().values

    # 3. Tính RSI-14 theo công thức Wilder chuẩn hóa về đoạn [0, 1]
    delta = df_close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi_14 = (1.0 - (1.0 / (1.0 + rs))).values

    # 4. Tính MACD (12, 26, 9) chia cho giá close để scale-free giữa các mã đắt/rẻ
    ema_12 = df_close.ewm(span=12, adjust=False).mean()
    ema_26 = df_close.ewm(span=26, adjust=False).mean()
    macd_line = ((ema_12 - ema_26) / (df_close + 1e-8)).values
    macd_signal = pd.DataFrame(macd_line).ewm(span=9, adjust=False).mean().values

    # 5. Tính tỷ lệ khối lượng hôm nay so với trung bình 20 ngày gần nhất
    vol_sma20 = df_volume.rolling(window=20).mean().values
    volume_ratio = (df_volume.values / (vol_sma20 + 1e-8))

    # Ghép tất cả thành khối tensor [T, N, K]
    feature_list = [
        ret_1d,
        ret_5d,
        vol_20d,
        rsi_14,
        macd_line,
        macd_signal,
        volume_ratio,
    ]
    features_3d = np.stack(feature_list, axis=-1).astype(np.float32)

    return features_3d, list(FEATURE_NAMES)

"""
src/features/__init__.py
Gói tính toán đặc trưng tài chính định lượng (Quantitative Feature Engineering).

Mô-đun bao gồm:
- returns: Tính toán Simple Return và Log Return (FEAT-001).
- volatility: Tính toán Rolling Volatility (20-day sample std) và Annualized Volatility (FEAT-002).
- rsi: Tính toán chỉ báo Wilder Relative Strength Index (RSI-14) (FEAT-003).
- macd: Tính toán chỉ báo Moving Average Convergence Divergence (FEAT-004).
- volume: Chuẩn hóa khối lượng giao dịch qua Log1p và Rolling Z-score (FEAT-005).
- pipeline: Hợp nhất toàn bộ 6 đặc trưng và đúc MarketDataTensor [T, N, F] (FEAT-006).
- scaler: Chuẩn hóa Z-score độc quyền chỉ học trên tập Train (SPLIT-002).
"""

from src.features.macd import (
    MACDEngine,
    compute_macd,
)
from src.features.pipeline import (
    FeaturePipeline,
)
from src.features.returns import (
    ReturnsEngine,
    compute_log_returns,
    compute_simple_returns,
)
from src.features.rsi import (
    RSIEngine,
    compute_rsi,
)
from src.features.scaler import (
    MarketFeatureScaler,
)
from src.features.volatility import (
    VolatilityEngine,
    compute_rolling_volatility,
)
from src.features.volume import (
    VolumeEngine,
    compute_log_volume,
    compute_normalized_volume,
)

__all__ = [
    "ReturnsEngine",
    "compute_simple_returns",
    "compute_log_returns",
    "VolatilityEngine",
    "compute_rolling_volatility",
    "RSIEngine",
    "compute_rsi",
    "MACDEngine",
    "compute_macd",
    "VolumeEngine",
    "compute_log_volume",
    "compute_normalized_volume",
    "FeaturePipeline",
    "MarketFeatureScaler",
]

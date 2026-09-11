"""
src/features/__init__.py
Gói tính toán đặc trưng tài chính định lượng (Quantitative Feature Engineering).

Mô-đun bao gồm:
- returns: Tính toán Simple Return và Log Return.
- volatility: Tính toán Rolling Volatility (20-day sample std) và Annualized Volatility.
- rsi: Tính toán chỉ báo Wilder Relative Strength Index (RSI-14).
"""

from src.features.returns import (
    ReturnsEngine,
    compute_log_returns,
    compute_simple_returns,
)
from src.features.rsi import (
    RSIEngine,
    compute_rsi,
)
from src.features.volatility import (
    VolatilityEngine,
    compute_rolling_volatility,
)

__all__ = [
    "ReturnsEngine",
    "compute_simple_returns",
    "compute_log_returns",
    "VolatilityEngine",
    "compute_rolling_volatility",
    "RSIEngine",
    "compute_rsi",
]

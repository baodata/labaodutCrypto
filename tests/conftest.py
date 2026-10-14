"""
cái này tạo data giả lập để test
"""

import pytest
import pandas as pd
import numpy as np
import yaml


@pytest.fixture
def trading_env_config_path(tmp_path):
    """Explicit test costs/limits, independent of mutable research configuration."""
    path = tmp_path / "env.yaml"
    path.write_text(yaml.safe_dump({"environment": {
        "initial_balance": 100000.0,
        "transaction_fee_pct": 0.001,
        "slippage_pct": 0.0,
        "max_asset_weight": 1.0,
        "risk_free_rate": 0.02,
        "reward_beta": 1.0,
        "reward_scaling": 100.0,
    }}), encoding="utf-8")
    return path

@pytest.fixture(scope="session")
def synthetic_returns_df():
    """Tạo DataFrame Returns giả lập cho 3 cổ phiếu trong 50 ngày (Phục vụ GRAPH-001)"""
    np.random.seed(42)
    # 50 ngày làm việc (Business days)
    dates = pd.date_range(start="2024-01-01", periods=50, freq="B")
    tickers = ["AAPL", "MSFT", "NVDA"]
    
    # Tạo lợi nhuận giả ngẫu nhiên có phân phối chuẩn
    returns_data = np.random.normal(loc=0.001, scale=0.02, size=(50, 3))
    
    # Ép MSFT và AAPL có tương quan cao (bằng cách cho MSFT = AAPL + nhiễu nhỏ)
    returns_data[:, 1] = returns_data[:, 0] + np.random.normal(0, 0.005, size=50)
    
    df = pd.DataFrame(returns_data, index=dates, columns=tickers)
    return df

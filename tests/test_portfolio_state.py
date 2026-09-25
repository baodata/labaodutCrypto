import pytest
import numpy as np
from src.env.portfolio import PortfolioState

def test_portfolio_state_initialization():
    state = PortfolioState(initial_capital=100000.0, num_assets=24)
    
    assert state.portfolio_value == 100000.0
    assert state.peak_value == 100000.0
    assert len(state.weights) == 25
    assert state.cash_weight == 1.0
    assert np.all(state.asset_weights == 0.0)
    assert state.cash_value == 100000.0
    assert state.assets_value == 0.0

def test_portfolio_state_update():
    state = PortfolioState(initial_capital=100000.0, num_assets=2)
    
    # Giả sử sau giao dịch, vốn còn 99000, 50% Asset 1, 30% Asset 2, 20% Cash
    new_weights = np.array([0.5, 0.3, 0.2])
    state.update(99000.0, new_weights)
    
    assert state.portfolio_value == 99000.0
    assert state.peak_value == 100000.0 # Peak không đổi vì 99k < 100k
    assert state.cash_weight == 0.2
    assert state.cash_value == 19800.0
    assert state.assets_value == 79200.0
    
    # Cập nhật lần 2, giá trị tăng vọt lên 110000
    new_weights_2 = np.array([0.4, 0.4, 0.2])
    state.update(110000.0, new_weights_2)
    
    assert state.portfolio_value == 110000.0
    assert state.peak_value == 110000.0 # Peak tăng lên

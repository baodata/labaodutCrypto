# kiểm thử cho môi trường giao dịch TradingEnv theo chuẩn Gymnasium

import numpy as np
import pytest

from src.env.trading_env import TradingEnv, make_trading_env


def test_env_initialization():
    # kiểm tra khởi tạo môi trường trên tập train
    env = make_trading_env(split_name="train")

    assert env.N == 24
    assert env.K == 7
    # 24 * 7 = 168 đặc trưng thị trường + 25 tỷ trọng hiện tại = 193
    assert env.observation_space.shape == (193,)
    assert env.action_space.shape == (25,)

    obs, info = env.reset(seed=42)
    assert obs.shape == (193,)
    assert info["portfolio_value"] == 100_000.0
    assert info["weights"][0] == 1.0  # Ban đầu 100% tiền mặt


def test_action_to_weights_simplex():
    # kiểm tra cơ chế softmax đảm bảo tỷ trọng luôn là phân phối hợp lệ
    env = make_trading_env(split_name="train")

    test_actions = [
        np.zeros(25),
        np.ones(25) * 10.0,
        np.ones(25) * -5.0,
        np.array([100.0] + [0.0] * 24),
        np.random.normal(size=25),
    ]

    for action in test_actions:
        weights = env._action_to_weights(action)
        assert np.isclose(np.sum(weights), 1.0, atol=1e-5)
        assert np.all(weights >= 0.0)


def test_cash_only_behavior():
    # nếu agent chỉ giữ 100% tiền mặt thì không tốn phí và giá trị danh mục không đổi
    env = make_trading_env(split_name="train")
    env.reset(seed=42)

    # action chọn 100% tiền mặt (cho phần tử 0 giá trị rất lớn)
    cash_action = np.zeros(25)
    cash_action[0] = 50.0

    obs, reward, terminated, truncated, info = env.step(cash_action)

    assert np.isclose(info["turnover"], 0.0, atol=1e-5)
    assert np.isclose(info["fee"], 0.0, atol=1e-5)
    assert np.isclose(info["net_return"], 0.0, atol=1e-5)
    assert np.isclose(info["portfolio_value"], 100_000.0, atol=1e-2)


def test_full_episode_equal_weight_rollout():
    # chạy thử trọn vẹn 1 episode với chiến lược Equal-Weight trên tập Test (501 phiên)
    env = make_trading_env(split_name="test")
    obs, info = env.reset(seed=42)

    # phân bổ đều 24 cổ phiếu (vị trí 0 tiền mặt = 0, 1..24 đều nhau)
    equal_weight_action = np.zeros(25)
    equal_weight_action[1:] = 10.0

    terminated = False
    step_count = 0

    while not terminated:
        obs, reward, terminated, truncated, info = env.step(equal_weight_action)
        step_count += 1

    # tập test có 501 ngày nên số bước giao dịch là 500
    assert step_count == 500
    assert terminated is True

    # kiểm tra báo cáo tổng kết
    summary = env.get_summary()
    assert summary["total_steps"] == 500
    assert "cumulative_return" in summary
    assert "sharpe_ratio" in summary
    assert "max_drawdown" in summary
    assert not np.isnan(summary["sharpe_ratio"])
    assert not np.isnan(summary["cumulative_return"])

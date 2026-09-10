"""
tests/test_contracts.py
Unit tests cho hợp đồng dữ liệu kỹ thuật CONTRACT-001.
"""

import numpy as np
import pytest

from src.utils.contracts import (
    DynamicGraphData,
    MarketDataTensor,
    MarketObservation,
    PortfolioAction,
)


class TestContracts:
    def test_market_data_tensor_valid(self):
        """Kiểm tra MarketDataTensor hợp lệ."""
        tensor = np.ones((10, 4, 5))
        mdt = MarketDataTensor(
            tensor=tensor,
            tickers=["AAPL", "MSFT", "NVDA", "AMD"],
            feature_names=["ret", "vol", "rsi", "macd", "volume"],
            dates=[f"2023-01-{i+1:02d}" for i in range(10)],
        )
        mdt.validate()  # Không ném lỗi

    def test_market_data_tensor_invalid_shape(self):
        """Kiểm tra ném lỗi khi tensor không phải 3D hoặc lệch kích thước."""
        tensor = np.ones((10, 4))
        mdt = MarketDataTensor(
            tensor=tensor,
            tickers=["AAPL", "MSFT", "NVDA", "AMD"],
            feature_names=["ret", "vol"],
            dates=[f"2023-01-{i+1:02d}" for i in range(10)],
        )
        with pytest.raises(ValueError, match="3 chiều"):
            mdt.validate()

    def test_market_data_tensor_nan_detection(self):
        """Kiểm tra phát hiện NaN trong tensor."""
        tensor = np.ones((5, 2, 2))
        tensor[0, 0, 0] = np.nan
        mdt = MarketDataTensor(
            tensor=tensor,
            tickers=["AAPL", "MSFT"],
            feature_names=["ret", "vol"],
            dates=[f"2023-01-{i+1:02d}" for i in range(5)],
        )
        with pytest.raises(ValueError, match="NaN"):
            mdt.validate()

    def test_dynamic_graph_valid(self):
        """Kiểm tra cấu trúc đồ thị động hợp lệ."""
        node_features = np.ones((3, 16))
        edge_index = np.array([[0, 1, 2], [1, 2, 0]])
        edge_weight = np.array([0.8, 0.7, 0.9])

        graph = DynamicGraphData(
            node_features=node_features,
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=3,
        )
        graph.validate()

    def test_dynamic_graph_out_of_bounds_edge(self):
        """Kiểm tra ném lỗi khi chỉ số node vượt quá phạm vi."""
        node_features = np.ones((3, 16))
        edge_index = np.array([[0, 1], [1, 99]])  # 99 vượt quá 3 nodes
        edge_weight = np.array([0.5, 0.5])

        graph = DynamicGraphData(
            node_features=node_features,
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=3,
        )
        with pytest.raises(ValueError, match="phạm vi"):
            graph.validate()

    def test_market_observation_raw_mode(self):
        """Kiểm tra observation ở chế độ raw_features (Single-Agent PPO)."""
        obs = MarketObservation(
            mode="raw_features",
            market_state=np.zeros((4, 8)),
            portfolio_weights=np.array([0.2, 0.2, 0.2, 0.2]),
            cash_ratio=0.2,
            current_step=0,
            portfolio_value=10000.0,
        )
        obs.validate()

    def test_market_observation_graph_mode(self):
        """Kiểm tra observation ở chế độ graph_embeddings (GNN/H-MARL)."""
        obs = MarketObservation(
            mode="graph_embeddings",
            market_state=np.random.randn(4, 32),
            portfolio_weights=np.array([0.25, 0.25, 0.25, 0.25]),
            cash_ratio=0.0,
            current_step=10,
            portfolio_value=10500.0,
        )
        obs.validate()

    def test_market_observation_invalid_weights_sum(self):
        """Kiểm tra ném lỗi khi tổng trọng số khác 1.0."""
        obs = MarketObservation(
            mode="raw_features",
            market_state=np.zeros((2, 4)),
            portfolio_weights=np.array([0.5, 0.3]),
            cash_ratio=0.1,  # Tổng = 0.9 != 1.0
            current_step=0,
            portfolio_value=10000.0,
        )
        with pytest.raises(ValueError, match="bằng 1.0"):
            obs.validate()

    def test_portfolio_action_valid(self):
        """Kiểm tra action phân bổ hợp lệ."""
        action = PortfolioAction(
            target_weights=np.array([0.25, 0.25, 0.25]),
            target_cash=0.25,
        )
        action.validate()

    def test_portfolio_action_negative_weight(self):
        """Kiểm tra ném lỗi khi có trọng số âm (Short selling)."""
        action = PortfolioAction(
            target_weights=np.array([-0.1, 0.6, 0.3]),
            target_cash=0.2,
        )
        with pytest.raises(ValueError, match="trọng số âm"):
            action.validate()

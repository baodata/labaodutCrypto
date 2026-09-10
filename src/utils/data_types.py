"""
src/utils/data_types.py
Giao diện và quy chuẩn kiểu dữ liệu kỹ thuật giữa Thành viên A và Thành viên B.

Ticket: CONTRACT-001 (P0 - Chung A + B)
Sprint: 1

Mục đích:
1. Đảm bảo giao diện dữ liệu giữa Data/Env Pipeline (A) và Graph/Model Pipeline (B) hoàn toàn thống nhất.
2. Quy định cấu trúc Tensor đầu vào: [T, N, F] (Time, Assets, Features).
3. Quy định cấu trúc Đồ thị tài chính: node_features [N, F], edge_index [2, E], edge_weight [E].
4. Quy định Observation hỗ trợ linh hoạt 2 chế độ:
   - 'raw_features' [N, F] (cho Single-Agent PPO baseline chạy độc lập không cần GNN).
   - 'graph_embeddings' [N, D] (cho mô hình GNN/GAT và H-MARL).
5. Quy định Action phân bổ danh mục: w_i >= 0, w_cash >= 0, sum(w_i) + w_cash = 1.0 (Long-only, no short-selling).
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np


@dataclass
class MarketDataTensor:
    """
    Khuôn đúc Feature Tensor [T, N, F] do Thành viên A bàn giao cho Thành viên B.
    - T: Số bước thời gian (trading days)
    - N: Số lượng tài sản (assets)
    - F: Số lượng đặc trưng kỹ thuật (features: return, volatility, rsi, macd, volume...)
    """
    tensor: np.ndarray          # Kích thước [T, N, F]
    tickers: List[str]          # Danh sách N mã tài sản theo đúng thứ tự trục 1
    feature_names: List[str]    # Danh sách F tên đặc trưng theo đúng thứ tự trục 2
    dates: List[str]            # Danh sách T mốc thời gian theo đúng thứ tự trục 0

    def validate(self) -> None:
        """Kiểm tra tính toàn vẹn của tensor."""
        if self.tensor.ndim != 3:
            raise ValueError(f"Tensor phải có đúng 3 chiều [T, N, F], hiện tại: {self.tensor.shape}")

        t, n, f = self.tensor.shape
        if len(self.dates) != t:
            raise ValueError(f"Số lượng dates ({len(self.dates)}) không khớp với trục thời gian T={t}")
        if len(self.tickers) != n:
            raise ValueError(f"Số lượng tickers ({len(self.tickers)}) không khớp với trục tài sản N={n}")
        if len(self.feature_names) != f:
            raise ValueError(f"Số lượng feature_names ({len(self.feature_names)}) không khớp với trục đặc trưng F={f}")

        if np.isnan(self.tensor).any():
            raise ValueError("Phát hiện NaN trong Feature Tensor!")
        if np.isinf(self.tensor).any():
            raise ValueError("Phát hiện Inf trong Feature Tensor!")


@dataclass
class DynamicGraphData:
    """
    Khuôn đúc Đồ thị tài chính do Thành viên B xây dựng tại mỗi timestep t.
    - node_features: Ma trận đặc trưng các node [N, F]
    - edge_index: Ma trận cạnh liên kết [2, E] (kiểu int)
    - edge_weight: Trọng số các cạnh [E] (kiểu float)
    - num_nodes: N
    """
    node_features: np.ndarray   # [N, F]
    edge_index: np.ndarray      # [2, E]
    edge_weight: np.ndarray     # [E]
    num_nodes: int              # N

    def validate(self) -> None:
        """Kiểm tra hợp lệ cấu trúc đồ thị."""
        if self.node_features.ndim != 2:
            raise ValueError(f"node_features phải có kích thước [N, F], hiện tại: {self.node_features.shape}")
        n, _ = self.node_features.shape
        if n != self.num_nodes:
            raise ValueError(f"num_nodes ({self.num_nodes}) không khớp với node_features ({n})")

        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError(f"edge_index phải có dạng [2, E], hiện tại: {self.edge_index.shape}")

        num_edges = self.edge_index.shape[1]
        if self.edge_weight.shape[0] != num_edges:
            raise ValueError(f"edge_weight [E] phải có độ dài bằng số cạnh E={num_edges}")

        if num_edges > 0:
            if self.edge_index.min() < 0 or self.edge_index.max() >= self.num_nodes:
                raise ValueError(f"Chỉ số cạnh trong edge_index nằm ngoài phạm vi [0, {self.num_nodes - 1}]")


@dataclass
class MarketObservation:
    """
    Khuôn đúc Observation linh hoạt của Gymnasium Environment (A) hỗ trợ cả 2 chế độ:
    1. mode='raw_features': state tensor [N, F] nạp thẳng vào Actor-Critic (Single-Agent PPO không GNN).
    2. mode='graph_embeddings': state tensor [N, D] đã qua GNN/GAT encoder.
    """
    mode: Literal["raw_features", "graph_embeddings"]
    market_state: np.ndarray        # [N, F] hoặc [N, D]
    portfolio_weights: np.ndarray   # Tỷ trọng hiện tại của N tài sản [N]
    cash_ratio: float               # Tỷ trọng tiền mặt hiện tại (0.0 đến 1.0)
    current_step: int               # Timestep hiện tại
    portfolio_value: float          # Tổng giá trị danh mục hiện tại ($)

    def validate(self) -> None:
        """Kiểm tra ràng buộc tài chính và dữ liệu observation."""
        if self.market_state.ndim != 2:
            raise ValueError(f"market_state phải là tensor 2 chiều, hiện tại: {self.market_state.shape}")

        num_assets = self.market_state.shape[0]
        if self.portfolio_weights.shape[0] != num_assets:
            raise ValueError(
                f"Kích thước portfolio_weights ({self.portfolio_weights.shape[0]}) không khớp với N={num_assets}"
            )

        total_weight = float(np.sum(self.portfolio_weights) + self.cash_ratio)
        if not np.isclose(total_weight, 1.0, atol=1e-4):
            raise ValueError(f"Tổng trọng số danh mục + tiền mặt phải bằng 1.0, hiện tại: {total_weight:.6f}")

        if np.any(self.portfolio_weights < -1e-6) or self.cash_ratio < -1e-6:
            raise ValueError("Tỷ trọng không được âm (No short-selling)!")


@dataclass
class PortfolioAction:
    """
    Khuôn đúc Action của Agent (đầu ra của Actor nạp vào Gym Env).
    - target_weights: Tỷ trọng phân bổ mong muốn cho N tài sản [N]
    - target_cash: Tỷ trọng tiền mặt mong muốn
    """
    target_weights: np.ndarray  # [N]
    target_cash: float          # float

    def validate(self) -> None:
        """Kiểm tra action có thỏa mãn điều kiện phân bổ danh mục hay không."""
        if np.any(self.target_weights < -1e-5) or self.target_cash < -1e-5:
            raise ValueError("Action chứa trọng số âm! Phải thỏa mãn w >= 0.")

        total = float(np.sum(self.target_weights) + self.target_cash)
        if not np.isclose(total, 1.0, atol=1e-3):
            raise ValueError(f"Tổng trọng số Action phải bằng 1.0, hiện tại: {total:.6f}")

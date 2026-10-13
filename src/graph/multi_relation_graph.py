"""
src/graph/multi_relation_graph.py
Đồ thị Đa quan hệ (Multi-relation Graph V1).
Ticket: GRAPH-005

Mục đích: Tích hợp Đồ thị Động (Tương quan) và Đồ thị Tĩnh (Ngành) thành một.
Sử dụng chéo các file sector_graph.py và dynamic_graph.py.
"""
import numpy as np
import pandas as pd
import torch
from typing import Tuple

# Sử dụng chéo các module từ GRAPH-003 và GRAPH-004
from src.graph.sector_graph import generate_sector_edges
from src.graph.dynamic_graph import DynamicGraphGenerator

class MultiRelationGraphBuilder:
    def __init__(
        self,
        tickers: list[str],
        rolling_corr_df: pd.DataFrame | None = None,
        threshold: float = 0.5,
        config_path: str = "configs/assets.yaml",
    ):
        self.tickers = tickers
        self.threshold = threshold
        
        # Khởi tạo GRAPH-004 nếu có rolling_corr_df
        self.dynamic_gen = (
            DynamicGraphGenerator(rolling_corr_df, threshold)
            if rolling_corr_df is not None
            else None
        )
        
        # Khởi tạo GRAPH-003
        self.sec_edge_index, self.sec_edge_weight = generate_sector_edges(tickers, config_path=config_path)
        
    def build_at_time(self, timestamp: pd.Timestamp) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gộp cả Tương quan và Ngành vào một hệ thống (dành cho Backtest)."""
        if self.dynamic_gen is None:
            raise ValueError("Cần cung cấp rolling_corr_df để dùng build_at_time.")
        corr_edge_index, corr_edge_weight = self.dynamic_gen.get_graph_at_time(timestamp)
        combined_index = torch.cat([corr_edge_index, self.sec_edge_index], dim=1)
        combined_weight = torch.cat([corr_edge_weight, self.sec_edge_weight], dim=0)
        return combined_index, combined_weight

    def build_from_matrix(
        self,
        corr_matrix: np.ndarray,
        threshold: float = 0.5,
        mode: str = "knn",
        top_k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gộp cả Tương quan và Ngành từ ma trận trực tiếp (Dành cho Adapter và Live Trading)."""
        from src.graph.builder import generate_correlation_edges
        corr_edge_index, corr_edge_weight = generate_correlation_edges(
            corr_matrix, threshold=threshold, mode=mode, top_k=top_k
        )
        # Luôn đảm bảo self-loops (i -> i) với trọng số 1.0 để mọi node có in-degree >= 1
        N = corr_matrix.shape[0]
        self_loop_index = torch.arange(N, dtype=torch.long).repeat(2, 1)
        self_loop_weight = torch.ones(N, dtype=torch.float32)

        combined_index = torch.cat([corr_edge_index, self.sec_edge_index, self_loop_index], dim=1)
        combined_weight = torch.cat([corr_edge_weight, self.sec_edge_weight, self_loop_weight], dim=0)
        return combined_index, combined_weight

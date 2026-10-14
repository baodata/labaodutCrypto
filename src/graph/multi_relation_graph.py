"""
src/graph/multi_relation_graph.py
Đồ thị Đa quan hệ (Multi-relation Graph V1).
Ticket: GRAPH-005

Mục đích: Giữ riêng quan hệ tương quan dương, tương quan âm và quan hệ ngành.
Sử dụng chéo các file sector_graph.py và dynamic_graph.py.
"""
import numpy as np
import pandas as pd
import torch
from typing import Literal, Tuple

# Sử dụng chéo các module từ GRAPH-003 và GRAPH-004
from src.graph.sector_graph import generate_sector_edges
from src.graph.dynamic_graph import DynamicGraphGenerator


GRAPH_RELATIONS = (
    "correlation_positive",
    "correlation_negative",
    "sector",
)


class MultiRelationGraphBuilder:
    def __init__(
        self,
        tickers: list[str],
        rolling_corr_df: pd.DataFrame,
        threshold: float = 0.5,
        sector_config_path: str = "configs/assets.yaml",
        graph_mode: Literal["threshold", "knn", "dynamic_threshold"] = "threshold",
        top_k: int = 5,
    ):
        self.tickers = tickers
        self.threshold = threshold
        self.graph_mode = graph_mode
        self.top_k = top_k
        
        # Khởi tạo GRAPH-004
        self.dynamic_gen = DynamicGraphGenerator(
            rolling_corr_df, threshold, mode=graph_mode, top_k=top_k, tickers=tickers
        )
        
        # Khởi tạo GRAPH-003
        self.sec_edge_index, self.sec_edge_weight = generate_sector_edges(
            tickers, config_path=sector_config_path
        )
        
    def build_at_time(self, timestamp: pd.Timestamp) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy flattened edge-list API; PPO uses build_relations_at_time."""
        corr_edge_index, corr_edge_weight = self.dynamic_gen.get_graph_at_time(timestamp)
        combined_index = torch.cat([corr_edge_index, self.sec_edge_index], dim=1)
        combined_weight = torch.cat([corr_edge_weight, self.sec_edge_weight], dim=0)
        return combined_index, combined_weight

    @staticmethod
    def _split_signed_correlation(
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        positive_mask = edge_weight > 0
        negative_mask = edge_weight < 0
        positive = (edge_index[:, positive_mask], edge_weight[positive_mask])
        negative = (
            edge_index[:, negative_mask],
            edge_weight[negative_mask].abs(),
        )
        return positive, negative

    def build_relations_at_time(
        self, timestamp: pd.Timestamp
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Return positive corr, negative corr magnitude, and sector separately.

        The relation axis is explicit so overlapping sector/correlation edges
        remain distinct. Negative correlation is represented by its own channel,
        allowing GCN layers to use non-negative edge magnitudes without erasing sign.
        """
        corr_edge_index, corr_edge_weight = self.dynamic_gen.get_graph_at_time(timestamp)
        positive, negative = self._split_signed_correlation(
            corr_edge_index, corr_edge_weight
        )
        return positive, negative, (self.sec_edge_index, self.sec_edge_weight)

    def build_relations_from_matrix(self, corr_matrix: np.ndarray) -> tuple:
        """Live relation API, identical to the relation layout used for training."""
        from src.graph.builder import generate_correlation_edges
        if np.asarray(corr_matrix).shape != (len(self.tickers), len(self.tickers)):
            raise ValueError("Correlation shape không khớp thứ tự tickers của builder.")
        edges, weights = generate_correlation_edges(
            corr_matrix, threshold=self.threshold, mode=self.graph_mode, top_k=self.top_k
        )
        positive, negative = self._split_signed_correlation(edges, weights)
        return positive, negative, (self.sec_edge_index, self.sec_edge_weight)

    def build_from_matrix(
        self,
        corr_matrix: np.ndarray,
        threshold: float | None = None,
        mode: Literal["threshold", "knn", "dynamic_threshold"] | None = None,
        top_k: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy flattened API; checkpoint inference uses build_relations_from_matrix."""
        from src.graph.dynamic_graph import generate_correlation_edges
        corr_edge_index, corr_edge_weight = generate_correlation_edges(
            corr_matrix,
            threshold=self.threshold if threshold is None else threshold,
            mode=self.graph_mode if mode is None else mode,
            top_k=self.top_k if top_k is None else top_k,
        )
        combined_index = torch.cat([corr_edge_index, self.sec_edge_index], dim=1)
        combined_weight = torch.cat([corr_edge_weight, self.sec_edge_weight], dim=0)
        return combined_index, combined_weight

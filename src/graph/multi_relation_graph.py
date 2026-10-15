"""
src/graph/multi_relation_graph.py
Đồ thị Đa quan hệ (Multi-relation Graph V1).
Ticket: GRAPH-005

Mục đích: Giữ riêng quan hệ tương quan dương, tương quan âm và quan hệ ngành.
Sử dụng chéo các file sector_graph.py và dynamic_graph.py.
"""
from typing import Literal, Tuple
import numpy as np
import pandas as pd
import torch

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
        rolling_corr_df: pd.DataFrame | None = None,
        threshold: float = 0.5,
        sector_config_path: str = "configs/assets.yaml",
        config_path: str | None = None,
        graph_mode: Literal["threshold", "knn", "dynamic_threshold"] = "threshold",
        top_k: int = 5,
    ):
        self.tickers = tickers
        self.threshold = threshold
        self.graph_mode = graph_mode
        self.top_k = top_k
        
        # Khởi tạo GRAPH-004 nếu có rolling_corr_df
        self.dynamic_gen = (
            DynamicGraphGenerator(
                rolling_corr_df, threshold, mode=graph_mode, top_k=top_k, tickers=tickers
            )
            if rolling_corr_df is not None
            else None
        )
        
        # Khởi tạo GRAPH-003
        actual_config_path = config_path or sector_config_path
        self.sec_edge_index, self.sec_edge_weight = generate_sector_edges(
            tickers, config_path=actual_config_path
        )
        
    def build_at_time(self, timestamp: pd.Timestamp) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gộp cả Tương quan và Ngành vào một hệ thống (dành cho Backtest)."""
        if self.dynamic_gen is None:
            raise ValueError("Cần cung cấp rolling_corr_df để dùng build_at_time.")
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
        if self.dynamic_gen is None:
            raise ValueError("Cần cung cấp rolling_corr_df để dùng build_relations_at_time.")
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
        mode: Literal["threshold", "knn", "dynamic_threshold"] | str | None = None,
        top_k: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gộp cả Tương quan và Ngành từ ma trận trực tiếp (Dành cho Adapter và Live Trading)."""
        from src.graph.builder import generate_correlation_edges
        actual_threshold = self.threshold if threshold is None else threshold
        actual_mode = self.graph_mode if mode is None else mode
        actual_top_k = self.top_k if top_k is None else top_k
        corr_edge_index, corr_edge_weight = generate_correlation_edges(
            corr_matrix,
            threshold=actual_threshold,
            mode=actual_mode,
            top_k=actual_top_k,
        )
        # Luôn đảm bảo self-loops (i -> i) với trọng số 1.0 để mọi node có in-degree >= 1
        N = corr_matrix.shape[0]
        self_loop_index = torch.arange(N, dtype=torch.long).repeat(2, 1)
        self_loop_weight = torch.ones(N, dtype=torch.float32)

        combined_index = torch.cat([corr_edge_index, self.sec_edge_index, self_loop_index], dim=1)
        combined_weight = torch.cat([corr_edge_weight, self.sec_edge_weight, self_loop_weight], dim=0)
        return combined_index, combined_weight

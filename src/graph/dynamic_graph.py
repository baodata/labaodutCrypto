"""
src/graph/dynamic_graph.py
Đồ thị Động (Dynamic Graph).
Ticket: GRAPH-004

Mục đích: Lấy ma trận tương quan tại một thời điểm `t` cụ thể và gọi builder để ra đồ thị.
"""
from typing import Literal, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

# Sử dụng chéo file builder.py (GRAPH-002)
from src.graph.builder import generate_correlation_edges

class DynamicGraphGenerator:
    """Sinh ra Đồ thị tương quan động tại bất kỳ ngày t nào."""
    
    def __init__(
        self,
        rolling_corr_df: pd.DataFrame,
        threshold: float = 0.5,
        mode: Literal["threshold", "knn", "dynamic_threshold"] = "threshold",
        top_k: int = 5,
        tickers: Sequence[str] | None = None,
    ):
        self.rolling_corr_df = rolling_corr_df
        self.threshold = threshold
        self.mode = mode
        self.top_k = top_k
        self.tickers = list(tickers) if tickers is not None else (list(rolling_corr_df.columns) if rolling_corr_df is not None else None)
        if rolling_corr_df is not None:
            if not rolling_corr_df.columns.is_unique or not rolling_corr_df.index.is_unique:
                raise ValueError("Rolling correlation chứa index/ticker trùng lặp.")
            if list(rolling_corr_df.columns) != self.tickers:
                raise ValueError("Thứ tự ticker trong rolling_corr_df không khớp graph.")
            for _, frame in rolling_corr_df.groupby(level=0, sort=False):
                if list(frame.index.get_level_values(-1)) != self.tickers:
                    raise ValueError("Thứ tự node của từng correlation matrix không khớp ticker.")
        
    def get_graph_at_time(self, timestamp: pd.Timestamp) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Với mỗi bước thời gian t, module trả về đồ thị tài chính tương ứng.
        """
        if timestamp not in self.rolling_corr_df.index:
            raise ValueError(f"Không có dữ liệu tương quan tại ngày {timestamp}")
            
        corr_matrix = self.rolling_corr_df.loc[timestamp].values
        
        # Gọi builder để tạo cạnh
        edge_index, edge_weight = generate_correlation_edges(
            corr_matrix,
            threshold=self.threshold,
            mode=self.mode,
            top_k=self.top_k,
        )
        return edge_index, edge_weight


def build_dynamic_graph_sequence(
    returns_df: pd.DataFrame,
    dates: Sequence,
    tickers: Sequence[str],
    *,
    window_size: int = 20,
    threshold: float = 0.5,
    graph_mode: Literal["threshold", "knn", "dynamic_threshold"] = "threshold",
    top_k: int = 5,
    sector_config_path: str = "configs/assets.yaml",
) -> tuple[np.ndarray, np.ndarray]:
    """Build causal, date-aligned multi-relation graph tensors.

    The correlation for date ``t`` uses returns up to and including ``t``.
    This is safe for the trading environment because the observation at ``t``
    places orders at Open[t+1], with the next observation marked at Close[t+1]. Static sector edges
    are added to every daily correlation graph.

    Returns:
        ``adjacency`` and ``edge_mask`` arrays with shape ``[T, R, N, N]``;
        R is positive correlation, negative-correlation magnitude, and sector.
        ``edge_mask`` keeps zero-valued edges distinct from absent edges.
    """
    if window_size <= 1:
        raise ValueError("window_size phải lớn hơn 1.")
    if len(tickers) == 0:
        raise ValueError("tickers không được rỗng.")
    if len(set(tickers)) != len(tickers) or not returns_df.columns.is_unique or not returns_df.index.is_unique:
        raise ValueError("Ticker/date không được trùng lặp.")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold phải là số hữu hạn trong [0, 1].")
    if graph_mode not in {"threshold", "knn", "dynamic_threshold"}:
        raise ValueError("graph_mode phải là threshold, knn hoặc dynamic_threshold.")
    if not isinstance(top_k, (int, np.integer)) or top_k <= 0:
        raise ValueError("top_k phải là số nguyên dương.")

    date_index = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    if date_index.has_duplicates or not date_index.is_monotonic_increasing:
        raise ValueError("dates phải tăng dần và không trùng lặp.")
    if set(tickers) - set(returns_df.columns) or not date_index.isin(returns_df.index).all():
        raise ValueError("returns_df thiếu ticker hoặc ngày được yêu cầu.")
    if not returns_df.index.is_monotonic_increasing:
        raise ValueError("returns_df phải tăng dần theo ngày.")
    aligned_returns = returns_df.reindex(index=date_index, columns=list(tickers))
    if aligned_returns.shape != (len(date_index), len(tickers)):
        raise ValueError("returns_df không thể căn theo dates và tickers.")
    # The first return is naturally undefined. Treat non-finite values as
    # missing so rolling correlation emits no dynamic edge until warm-up.
    if np.isinf(aligned_returns.to_numpy()).any():
        raise ValueError("Returns chứa Inf.")

    from src.graph.correlation import compute_rolling_correlation
    from src.graph.multi_relation_graph import GRAPH_RELATIONS, MultiRelationGraphBuilder

    rolling_corr = compute_rolling_correlation(aligned_returns, window_size=window_size)
    builder = MultiRelationGraphBuilder(
        tickers=list(tickers),
        rolling_corr_df=rolling_corr,
        threshold=threshold,
        sector_config_path=sector_config_path,
        graph_mode=graph_mode,
        top_k=top_k,
    )

    time_count = len(date_index)
    asset_count = len(tickers)
    relation_count = len(GRAPH_RELATIONS)
    adjacency = np.zeros(
        (time_count, relation_count, asset_count, asset_count), dtype=np.float32
    )
    edge_mask = np.zeros_like(adjacency)
    for time_index, timestamp in enumerate(date_index):
        for relation_index, (edge_index, edge_weight) in enumerate(
            builder.build_relations_at_time(timestamp)
        ):
            if edge_index.numel() == 0:
                continue
            source = edge_index[0].cpu().numpy()
            target = edge_index[1].cpu().numpy()
            weights = edge_weight.cpu().numpy()
            adjacency[time_index, relation_index, source, target] = weights
            edge_mask[time_index, relation_index, source, target] = 1.0

    return adjacency, edge_mask

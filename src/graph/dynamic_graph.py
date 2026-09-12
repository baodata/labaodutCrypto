"""
src/graph/dynamic_graph.py
Đồ thị Động (Dynamic Graph).
Ticket: GRAPH-004

Mục đích: Lấy ma trận tương quan tại một thời điểm `t` cụ thể và gọi builder để ra đồ thị.
"""
import numpy as np
import pandas as pd
import torch
from typing import Tuple

# Sử dụng chéo file builder.py (GRAPH-002)
from src.graph.builder import generate_correlation_edges

class DynamicGraphGenerator:
    """Sinh ra Đồ thị tương quan động tại bất kỳ ngày t nào."""
    
    def __init__(self, rolling_corr_df: pd.DataFrame, threshold: float = 0.5):
        self.rolling_corr_df = rolling_corr_df
        self.threshold = threshold
        
    def get_graph_at_time(self, timestamp: pd.Timestamp) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Với mỗi bước thời gian t, module trả về đồ thị tài chính tương ứng.
        """
        if timestamp not in self.rolling_corr_df.index:
            raise ValueError(f"Không có dữ liệu tương quan tại ngày {timestamp}")
            
        corr_matrix = self.rolling_corr_df.loc[timestamp].values
        
        # Gọi builder để tạo cạnh
        edge_index, edge_weight = generate_correlation_edges(corr_matrix, threshold=self.threshold)
        return edge_index, edge_weight

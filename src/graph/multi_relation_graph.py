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
    def __init__(self, tickers: list[str], rolling_corr_df: pd.DataFrame, threshold: float = 0.5):
        self.tickers = tickers
        
        # Khởi tạo GRAPH-004
        self.dynamic_gen = DynamicGraphGenerator(rolling_corr_df, threshold)
        
        # Khởi tạo GRAPH-003
        self.sec_edge_index, self.sec_edge_weight = generate_sector_edges(tickers)
        
    def build_at_time(self, timestamp: pd.Timestamp) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gộp cả Tương quan và Ngành vào một hệ thống."""
        # 1. Lấy đồ thị động tại ngày t (Correlation)
        corr_edge_index, corr_edge_weight = self.dynamic_gen.get_graph_at_time(timestamp)
        
        # 2. Hợp nhất với đồ thị tĩnh (Sector)
        combined_index = torch.cat([corr_edge_index, self.sec_edge_index], dim=1)
        combined_weight = torch.cat([corr_edge_weight, self.sec_edge_weight], dim=0)
        
        return combined_index, combined_weight

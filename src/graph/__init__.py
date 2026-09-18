"""
src/graph/__init__.py
Gói xây dựng đồ thị tài chính động (Financial Graph Construction).

Mô-đun bao gồm:
- correlation: Tính toán ma trận tương quan trượt Rolling Correlation (GRAPH-001).
- builder: Xây dựng cạnh đồ thị edge_index và edge_weight từ ma trận tương quan (GRAPH-002).
- sector_graph: Đồ thị tĩnh dựa trên phân loại nhóm ngành cổ phiếu (GRAPH-003).
- dynamic_graph: Đồ thị tài chính động theo từng bước thời gian t (GRAPH-004).
- multi_relation_graph: Đồ thị đa quan hệ Multi-relation Graph V1 (GRAPH-005).
"""

from src.graph.builder import generate_correlation_edges
from src.graph.correlation import compute_rolling_correlation
from src.graph.dynamic_graph import DynamicGraphGenerator
from src.graph.multi_relation_graph import MultiRelationGraphBuilder
from src.graph.sector_graph import generate_sector_edges

__all__ = [
    "compute_rolling_correlation",
    "generate_correlation_edges",
    "generate_sector_edges",
    "DynamicGraphGenerator",
    "MultiRelationGraphBuilder",
]


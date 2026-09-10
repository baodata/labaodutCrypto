"""
src/graph/builder.py
Module tạo cấu trúc Graph (Node, Edge) từ ma trận tương quan.

- Ticket: GRAPH-002 (Sprint 2 - Member B)

FILE NÀY ĐỂ LÀM GÌ?
- Input: Nhận ma trận tương quan, lấy ở correlation.py (graph-001).
- Lọc các cặp cổ phiếu có mức độ tương quan >= or > một threshold nhất định.
- Output: Chuyển đổi dữ liệu thành cấu trúc chuẩn của PyTorch Geometric bao gồm:
  + `edge_index`: Mảng 2D lưu các cặp đỉnh nối với nhau.
  + `edge_weight`: Mảng 1D lưu trọng số của các cạnh (chính là hệ số tương quan).
"""

import numpy as np
import torch
from typing import Tuple

def generate_correlation_edges(corr_matrix: np.ndarray, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tạo edge_index và edge_weight từ một ma trận tương quan 2D tại một thời điểm t.
    
    Args:
        corr_matrix: Numpy array (N x N) biểu diễn ma trận tương quan.
        threshold: Ngưỡng tương quan tối thiểu để tạo cạnh (mặc định 0.5).
        
    Returns:
        edge_index: Tensor kích thước [2, E], kiểu torch.long
        edge_weight: Tensor kích thước [E], kiểu torch.float32
    """
    # Thay thế các giá trị NaN bằng 0 để an toàn
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    N = corr_matrix.shape[0]
    
    sources = []
    targets = []
    weights = []
    
    for i in range(N):
        for j in range(N):
            if i != j:  # Không lấy đường chéo (self-loops)
                weight = corr_matrix[i, j]
                # Chỉ giữ các cạnh có độ lớn tương quan >= threshold 
                # (Dùng abs() nếu bạn cho rằng tương quan ngược chiều cũng là 1 mối liên hệ mạnh)
                if abs(weight) >= threshold:
                    sources.append(i)
                    targets.append(j)
                    weights.append(weight)
                    
    # Chuyển đổi sang Torch Tensor theo chuẩn của thư viện PyTorch Geometric
    if len(sources) > 0:
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float32)
    else:
        # Xử lý an toàn cho trường hợp không có cạnh nào thỏa mãn ngưỡng
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)
        
    return edge_index, edge_weight

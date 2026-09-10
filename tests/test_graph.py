"""
tests/test_graph.py
unit tests cho module Graph Builder (GRAPH-001, GRAPH-002).

- Ticket: GRAPH-001, GRAPH-002 (Sprint 2 - Member B)

FILE NÀY ĐỂ LÀM GÌ?
- Kiểm thử các hàm tính toán ma trận tương quan trượt (Rolling Correlation) và tạo cấu trúc Graph từ ma trận tương quan.
- Sử dụng pytest để mô phỏng dữ liệu tỷ suất lợi nhuận (Return) và kiểm tra tính đúng đắn của các hàm.
"""

import pytest
import pandas as pd
import numpy as np
import torch

from src.graph.correlation import compute_rolling_correlation
from src.graph.builder import generate_correlation_edges

def test_compute_rolling_correlation(synthetic_returns_df):
    """Test GRAPH-001: Tính ma trận tương quan trượt."""
    window = 20
    corr_df = compute_rolling_correlation(synthetic_returns_df, window_size=window)
    
    # 1. Kiểm tra kích thước DataFrame trả về
    assert isinstance(corr_df, pd.DataFrame)
    assert len(corr_df.columns) == 3  # AAPL, MSFT, NVDA
    
    # 2. Lấy ma trận tương quan của ngày cuối cùng
    last_date = synthetic_returns_df.index[-1]
    last_matrix = corr_df.loc[last_date].values
    
    # 3. Kiểm tra tính chất toán học của ma trận tương quan
    # Đường chéo chính (tương quan của cổ phiếu với chính nó) phải bằng 1
    assert np.allclose(np.diag(last_matrix), 1.0, atol=1e-5)
    
    # MSFT (cột 1) và AAPL (cột 0) phải có tương quan cao vì ta đã setup cố tình ở conftest.py
    assert last_matrix[0, 1] > 0.5


def test_generate_correlation_edges():
    """Test GRAPH-002: Tạo danh sách cạnh (edge_index) từ ma trận tương quan."""
    # Ma trận giả lập 3x3
    # Nút 0 và 1 có corr = 0.8
    # Nút 0 và 2 có corr = 0.2
    # Nút 1 và 2 có corr = -0.6
    corr_matrix = np.array([
        [1.0,  0.8,  0.2],
        [0.8,  1.0, -0.6],
        [0.2, -0.6,  1.0]
    ])
    
    # Chạy hàm với ngưỡng (threshold) 0.5
    edge_index, edge_weight = generate_correlation_edges(corr_matrix, threshold=0.5)
    
    # 1. Kiểm tra kiểu dữ liệu chuẩn của PyTorch Geometric
    assert isinstance(edge_index, torch.Tensor)
    assert isinstance(edge_weight, torch.Tensor)
    assert edge_index.dtype == torch.long
    assert edge_weight.dtype == torch.float32
    
    # 2. Kiểm tra kích thước
    # Các cạnh thỏa mãn abs(corr) >= 0.5 và không tính đường chéo:
    # (0,1): 0.8
    # (1,0): 0.8
    # (1,2): -0.6
    # (2,1): -0.6
    # Tổng cộng có 4 cạnh (E = 4)
    E = 4
    assert edge_index.shape == (2, E)
    assert edge_weight.shape == (E,)
    
    # 3. Kiểm tra tính đúng đắn của cạnh
    # Cạnh nối nút 0 với nút 1 phải tồn tại với trọng số 0.8
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    assert (0, 1) in edges
    idx = edges.index((0, 1))
    assert pytest.approx(edge_weight[idx].item(), 0.01) == 0.8
    
    # Cạnh nối nút 0 với nút 2 có corr = 0.2 < 0.5 nên phải bị loại bỏ, không được tồn tại
    assert (0, 2) not in edges

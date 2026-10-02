"""
tests/test_gnn.py
Tích hợp và kiểm định Mạng Nơ-ron Đồ thị.
Ticket: GNN-002 (Sprint 4)
Mục tiêu: Đảm bảo GCN và GAT chạy Forward Pass thành công mà không bị crash kích thước ma trận.
"""
import torch
import pytest
from src.models.gnn.gcn_encoder import GCNEncoder
from src.models.gnn.gat_encoder import GATEncoder

class TestGNNEncoders:
    @pytest.fixture
    def dummy_graph(self):
        """Tạo một đồ thị giả lập 24 cổ phiếu, 6 tính năng (Giống output của Pipeline)"""
        N, F = 24, 6
        # X: Node features
        x = torch.rand((N, F), dtype=torch.float32)
        
        # Edge Index: Nối ngẫu nhiên vài đỉnh
        edge_index = torch.tensor([
            [0, 1, 1, 2, 5, 23, 10],
            [1, 0, 2, 1, 10, 5, 23]
        ], dtype=torch.long)
        
        # Edge weight
        edge_weight = torch.rand((edge_index.shape[1],), dtype=torch.float32)
        
        return x, edge_index, edge_weight

    def test_gcn_forward_pass(self, dummy_graph):
        """GNN-001: Đảm bảo GCN nhận [N, F] và trả về đúng [N, D]"""
        x, edge_index, edge_weight = dummy_graph
        N = x.shape[0]
        D = 16 # out_channels
        
        model = GCNEncoder(in_channels=6, hidden_channels=32, out_channels=D)
        
        model.eval()
        out = model(x, edge_index, edge_weight)
        
        assert out.shape == (N, D), f"Kỳ vọng ({N}, {D}), nhưng nhận được {out.shape}"
        assert not torch.isnan(out).any(), "Mạng GCN xuất hiện NaN!"

    def test_gat_forward_pass(self, dummy_graph):
        """GNN-003: Đảm bảo GAT nhận [N, F] và trả về đúng [N, D] bằng Attention"""
        x, edge_index, edge_weight = dummy_graph
        N = x.shape[0]
        D = 16
        
        model = GATEncoder(in_channels=6, hidden_channels=32, out_channels=D, heads=2)
        
        model.eval()
        out = model(x, edge_index, edge_weight)
        
        assert out.shape == (N, D), f"Kỳ vọng ({N}, {D}), nhưng nhận được {out.shape}"
        assert not torch.isnan(out).any(), "Mạng GAT xuất hiện NaN!"

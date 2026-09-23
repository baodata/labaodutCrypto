"""
src/models/gnn/gcn_encoder.py
Kiến trúc Mạng nơ-ron Tích chập Đồ thị (Graph Convolutional Network).

Ticket: GNN-001 (P0 - Thành viên B)
Sprint: 4

Mục đích:
1. Biến đổi Node Features [N, F] thành Node Embeddings [N, D] thông qua Message Passing.
2. Tích hợp cấu trúc ma trận kề từ edge_index và sức mạnh liên kết từ edge_weight.
3. Đóng vai trò làm Baseline Encoder chuẩn mực trước khi nâng cấp lên GAT.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNEncoder(nn.Module):
    """
    Bộ mã hóa GCN 2 lớp cơ bản cho bài toán Portfolio Optimization.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.2):
        super(GCNEncoder, self).__init__()
        self.dropout = dropout
        
        # Lớp Convolution 1: Nhận [N, F] chuyển thành [N, hidden]
        self.conv1 = GCNConv(in_channels, hidden_channels)
        
        # Lớp Convolution 2: Nhận [N, hidden] chuyển thành [N, D]
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
        # Layer Normalization giúp ổn định gradient trong RL
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Tensor đặc trưng của các Node, shape [N, F]
            edge_index: Tensor chỉ số cạnh, shape [2, E]
            edge_weight: Tensor trọng số cạnh, shape [E] (Tương quan hoặc Tĩnh)
            
        Returns:
            Biểu diễn nhúng của các Node, shape [N, D]
        """
        # Message Passing Layer 1
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Message Passing Layer 2
        x = self.conv2(x, edge_index, edge_weight)
        
        # Chuẩn hóa đầu ra để nạp vào Actor-Critic mượt mà hơn
        x = self.layer_norm(x)
        
        return x

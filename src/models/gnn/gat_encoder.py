"""
src/models/gnn/gat_encoder.py
Kiến trúc Mạng nơ-ron Chú ý Đồ thị (Graph Attention Network v2).

Ticket: GNN-003 (Mốc B - Thành viên B)
Sprint: 4

Mục đích:
1. Nâng cấp từ GCN tĩnh sang GAT động. 
2. Thay vì gộp thông tin từ hàng xóm một cách cào bằng, GAT tự động học ra
   hệ số Attention: Cổ phiếu nào đáng để quan tâm hơn trong cùng 1 cụm.
3. Sử dụng GATv2Conv (Bản nâng cấp mạnh hơn bản GAT gốc năm 2018).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GATEncoder(nn.Module):
    """
    Bộ mã hóa GAT với cơ chế Multi-head Attention.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.2):
        super(GATEncoder, self).__init__()
        self.dropout = dropout
        
        # Lớp GAT 1: Multi-head Attention
        # Kết quả đầu ra sẽ bị nhân lên theo số lượng heads (hidden_channels * heads)
        self.gat1 = GATv2Conv(
            in_channels, 
            hidden_channels, 
            heads=heads, 
            dropout=dropout,
            concat=True,
            edge_dim=1
        )
        
        # Lớp GAT 2: Lớp gộp (concat=False để trả về đúng kích thước out_channels)
        self.gat2 = GATv2Conv(
            hidden_channels * heads, 
            out_channels, 
            heads=1, 
            concat=False, 
            dropout=dropout,
            edge_dim=1
        )
        
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        Lưu ý: GAT tự học ra Attention Weights, nên edge_weight (từ correlation) 
        có thể được dùng như edge_attr hoặc chỉ dùng edge_index làm cấu trúc (Topology).
        Ở đây ta truyền edge_weight vào edge_attr để trợ lực cho mạng.
        """
        # Nếu edge_weight là mảng 1D, định hình lại thành 2D [E, 1] cho GAT
        edge_attr = edge_weight.view(-1, 1) if edge_weight is not None else None
        
        # Attention Layer 1
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x) # ELU thường được dùng kết hợp với GAT
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Attention Layer 2
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        
        x = self.layer_norm(x)
        
        return x

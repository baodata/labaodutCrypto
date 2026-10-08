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

Nâng cấp: GNN-005 (Thành viên B)
4. Thêm Residual / Skip Connections giữa các lớp GATv2 để chống Over-smoothing.
   Khi đồ thị tiệm cận Fully Connected (do Market Correlation Spike), residual
   đảm bảo mỗi cổ phiếu vẫn giữ lại "bản sắc riêng" (idiosyncratic features)
   sau khi tổng hợp thông tin từ hàng xóm.
5. Thêm Entropy Regularization Loss để ép Attention weights phân bố "sắc nét"
   (sharp attention), tránh hiện tượng GATv2 chia đều trọng số 1/N cho tất cả
   hàng xóm khi chúng quá giống nhau trong giai đoạn khủng hoảng.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATEncoder(nn.Module):
    """
    Bộ mã hóa GAT với cơ chế Multi-head Attention, Residual Connections
    và hỗ trợ tính Entropy Regularization Loss.

    Kiến trúc:
        Input [N, in_channels]
            │
            ▼
        GATv2Conv Layer 1 (Multi-head, concat=True)
            │   ↑  Residual projection (Linear nếu chiều khác nhau)
            ├───┘
            ▼
        ELU + Dropout
            │
            ▼
        GATv2Conv Layer 2 (Single-head, concat=False)
            │   ↑  Residual connection (trực tiếp từ đầu vào layer 2)
            ├───┘
            ▼
        LayerNorm
            │
            ▼
        Output [N, out_channels]
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.2,
        residual_alpha: float = 0.5,
    ):
        """
        Args:
            in_channels:      Số chiều đặc trưng đầu vào của mỗi node.
            hidden_channels:  Số chiều ẩn mỗi attention head ở lớp 1.
            out_channels:     Số chiều embedding đầu ra cuối cùng.
            heads:            Số đầu Attention ở lớp GATv2 đầu tiên (mặc định 4).
            dropout:          Tỷ lệ Dropout (mặc định 0.2).
            residual_alpha:   Tỷ trọng của skip connection trong residual.
                              Output = (1 - α)*GATv2(x) + α*x_projected
                              α = 0.0 → không dùng residual (giống code gốc).
                              α = 0.5 → cân bằng giữa thông tin mới và cũ (mặc định).
                              α = 1.0 → chỉ giữ nguyên đặc trưng gốc (không học được).
        """
        super(GATEncoder, self).__init__()
        self.dropout = dropout
        self.residual_alpha = residual_alpha

        # ── Lớp GAT 1: Multi-head Attention ──────────────────────────────────
        # Kết quả đầu ra sẽ bị nhân lên theo số lượng heads (hidden_channels * heads)
        self.gat1 = GATv2Conv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True,
            edge_dim=1,
        )

        # Projection cho Residual lớp 1:
        # Đầu vào là [N, in_channels], đầu ra gat1 là [N, hidden_channels * heads].
        # Chiều không khớp → cần Linear để chiếu về cùng chiều trước khi cộng.
        self.residual_proj1 = nn.Linear(in_channels, hidden_channels * heads, bias=False)

        # ── Lớp GAT 2: Lớp gộp (concat=False để trả về đúng kích thước out_channels) ──
        self.gat2 = GATv2Conv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=1,
        )

        # Projection cho Residual lớp 2:
        # Đầu vào là [N, hidden_channels * heads], đầu ra gat2 là [N, out_channels].
        self.residual_proj2 = nn.Linear(hidden_channels * heads, out_channels, bias=False)

        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
        return_attention_weights: bool = False,
    ):
        """
        Forward pass.

        Args:
            x:                      Node features [N, in_channels].
            edge_index:             Tensor [2, E] (cấu trúc cạnh).
            edge_weight:            Tensor [E] trọng số cạnh từ tương quan (Pearson).
                                    Được dùng như edge_attr để trợ lực cho Attention.
            return_attention_weights:
                                    Nếu True, trả về thêm tuple (edge_index, alpha)
                                    của lớp 2 để dùng cho Entropy Regularization Loss.

        Returns:
            Khi return_attention_weights=False: Tensor [N, out_channels].
            Khi return_attention_weights=True:  Tuple(Tensor [N, out_channels],
                                                      Tuple(edge_index, alpha_weights)).
        """
        # Nếu edge_weight là mảng 1D, định hình lại thành 2D [E, 1] cho GAT
        edge_attr = edge_weight.view(-1, 1) if edge_weight is not None else None

        # ── Attention Layer 1 với Residual ────────────────────────────────────
        x_res1 = self.residual_proj1(x)  # Chiếu về cùng chiều [N, hidden * heads]
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)  # ELU thường được dùng kết hợp với GAT
        # Residual: output = (1 - α) * GATv2(x) + α * x_projected
        x = (1.0 - self.residual_alpha) * x + self.residual_alpha * x_res1
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ── Attention Layer 2 với Residual ────────────────────────────────────
        x_res2 = self.residual_proj2(x)  # Chiếu về [N, out_channels]

        if return_attention_weights:
            # Trả về attention weights để tính Entropy Regularization Loss bên ngoài
            x, (att_edge_index, att_alpha) = self.gat2(
                x, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
        else:
            x = self.gat2(x, edge_index, edge_attr=edge_attr)

        # Residual lớp 2
        x = (1.0 - self.residual_alpha) * x + self.residual_alpha * x_res2

        x = self.layer_norm(x)

        if return_attention_weights:
            return x, (att_edge_index, att_alpha)
        return x


def compute_attention_entropy_loss(alpha: torch.Tensor) -> torch.Tensor:
    """
    Tính Entropy Regularization Loss từ Attention weights của GATv2.

    Mục đích: Ép buộc GATv2 phân bổ Attention "sắc nét" (sharp / peaked),
    tránh hiện tượng Attention đồng đều (uniform) khi đồ thị gần Fully Connected —
    đặc biệt trong giai đoạn Market Correlation Spike (khủng hoảng).

    Hàm mất mát Entropy:
        L_ent = (1/|E|) * Σ_{(i,j) in E} [ -α_{ij} * log(α_{ij} + ε) ]

    Đưa vào tổng hàm mất mát như:
        L_total = L_RL + λ_ent * L_ent    (λ_ent nhỏ, ví dụ: 0.01)

    Minimize L_ent → GATv2 bị buộc phải chọn ra một số ít cạnh trọng yếu,
    thay vì chia đều 1/N cho tất cả hàng xóm.

    Args:
        alpha: Attention weight tensor [E, heads] hoặc [E, 1] từ GATv2.

    Returns:
        Scalar tensor: giá trị entropy loss trung bình trên tất cả các cạnh.
    """
    eps = 1e-8  # Tránh log(0)
    # Entropy = -sum(alpha * log(alpha)) — tính trung bình trên toàn bộ cạnh
    entropy = -torch.sum(alpha * torch.log(alpha + eps), dim=-1)
    return entropy.mean()

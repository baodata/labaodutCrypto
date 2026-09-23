"""
src/models/actor_critic.py
Kiến trúc Đầu não ra Quyết định (Actor-Critic) dùng cho Thuật toán PPO.

Ticket: BASE-004 (Sprint 5)

Mục đích:
Lắp ráp Mạng GNN (Hiểu thị trường) với 2 cái "Đầu" (Heads):
1. Actor Head: Nhìn vào Đồ thị để ra quyết định Mua/Bán tỷ trọng bao nhiêu.
2. Critic Head: Nhìn vào Đồ thị để phán đoán xem trạng thái thị trường hiện tại 
   sẽ đem lại bao nhiêu điểm lợi nhuận (Dùng để PPO tự kiểm điểm và sửa sai).
"""
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

class GNN_ActorCritic(nn.Module):
    def __init__(self, gnn_encoder: nn.Module, hidden_dim: int):
        """
        Khởi tạo kiến trúc.
        Args:
            gnn_encoder: Bộ mã hóa GCN hoặc GAT (Từ Sprint 4)
            hidden_dim: Kích thước Vector nhúng D (Từ out_channels của GNN)
        """
        super(GNN_ActorCritic, self).__init__()
        
        # 1. Bộ Não Lõi (Dùng chung cho cả Actor và Critic)
        self.gnn = gnn_encoder
        
        # 2. Đầu Diễn viên (Actor Head - Chính sách giao dịch)
        # Ép Vector [D] của mỗi cổ phiếu thành 1 con số duy nhất (Điểm tiềm năng)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 3. Đầu Phê bình (Critic Head - Đánh giá Thị trường)
        # Đánh giá ĐIỂM SỐ CHUNG của toàn bộ thị trường
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
        """
        Chạy dữ liệu qua hệ thống.
        """
        # --- BƯỚC 1: ĐỌC THỊ TRƯỜNG ---
        # GNN nhúng 24 cổ phiếu thành ma trận thông thái [N, D]
        node_embeddings = self.gnn(x, edge_index, edge_weight)
        
        # --- BƯỚC 2: ACTOR HÀNH ĐỘNG ---
        # Tính điểm thô (Logits) cho từng cổ phiếu: [N, 1]
        action_logits = self.actor_head(node_embeddings)
        # Ép phẳng về [N] để gửi qua ActionAdapter (Softmax)
        action_logits = action_logits.squeeze(-1) 
        
        # --- BƯỚC 3: CRITIC PHÊ BÌNH ---
        # Gom nhóm toàn bộ 24 cổ phiếu lại thành 1 Vector đại diện cho cả Thị trường (Market State)
        # Bằng cách lấy trung bình cộng (Global Mean Pool). 
        # Cần biến batch = zeros(N) để báo cho PyG biết tất cả N node này thuộc chung 1 đồ thị.
        batch_index = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        market_state = global_mean_pool(node_embeddings, batch_index) # Shape: [1, D]
        
        # Chấm điểm toàn thị trường (Value): [1, 1]
        state_value = self.critic_head(market_state)
        # Ép phẳng về [1]
        state_value = state_value.squeeze(-1)
        
        return action_logits, state_value

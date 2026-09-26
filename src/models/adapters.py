"""
src/models/adapters.py
Bộ mã nguồn Gộp (Theo yêu cầu): Chứa cả Adapter Dịch thuật và Kiến trúc Não bộ Actor-Critic.
Hỗ trợ xuất ra N cổ phiếu + 1 Tiền mặt.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import global_mean_pool

# ==========================================
# 1. KIẾN TRÚC NÃO BỘ (ACTOR - CRITIC)
# ==========================================
class GNN_ActorCritic(nn.Module):
    def __init__(self, gnn_encoder: nn.Module, hidden_dim: int):
        super(GNN_ActorCritic, self).__init__()
        
        self.gnn = gnn_encoder
        
        # 1. Chấm điểm từng cổ phiếu (24 mã)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 2. Chấm điểm Tiền mặt (Cash)
        self.cash_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 3. Đầu Phê bình (Critic Head)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
        # 1. Nhúng Đồ thị
        node_embeddings = self.gnn(x, edge_index, edge_weight)
        
        # 2. Đánh giá trạng thái chung của Thị trường (Market State)
        batch_index = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        market_state = global_mean_pool(node_embeddings, batch_index) # Shape: [1, D]
        
        # 3. Tính Logit cho Cổ phiếu [N]
        stock_logits = self.actor_head(node_embeddings).squeeze(-1) 
        
        # 4. Tính Logit cho Tiền mặt [1]
        cash_logit = self.cash_head(market_state).squeeze(-1)
        
        # GHÉP LẠI: [N+1] Logits (Cổ phiếu + Tiền mặt)
        action_logits = torch.cat([stock_logits, cash_logit], dim=0)
        
        # 5. Critic đánh giá toàn bộ thị trường
        state_value = self.critic_head(market_state).squeeze(-1)
        
        return action_logits, state_value

# ==========================================
# 2. CÁP DỊCH THUẬT (ADAPTERS)
# ==========================================
class ObservationAdapter:
    def __init__(self, graph_builder):
        self.graph_builder = graph_builder

    def process(self, obs_features: np.ndarray, obs_corr: np.ndarray) -> tuple:
        x = torch.tensor(obs_features, dtype=torch.float32)
        edge_index, edge_weight = self.graph_builder.build_from_matrix(obs_corr)
        return x, edge_index, edge_weight

class ActionAdapter:
    @staticmethod
    def process(actor_logits: torch.Tensor) -> np.ndarray:
        # Nhận vào 25 Logits (N+1), xuất ra 25 tỷ trọng tổng bằng 100%
        weights = F.softmax(actor_logits, dim=0)
        return weights.detach().cpu().numpy()

"""
src/models/adapters.py
Bộ Chuyển Đổi Dữ Liệu (Adapter) giữa Môi trường Gym và Mạng Nơ-ron Đồ thị.

Ticket: MODEL-001 (P0 - Thành viên B)
Sprint: 5

Mục đích: Dịch thuật ngôn ngữ giữa Numpy (Của Env) và PyTorch Tensor (Của GNN).
"""
import torch
import numpy as np
import torch.nn.functional as F

class ObservationAdapter:
    """
    Biến đổi Trạng thái (Observation) từ Môi trường thành đầu vào cho GNN.
    """
    def __init__(self, graph_builder):
        # Bộ tạo đồ thị Đa quan hệ (Sprint 3)
        self.graph_builder = graph_builder

    def process(self, obs_features: np.ndarray, obs_corr: np.ndarray) -> tuple:
        """
        Dịch thuật Observation.
        Args:
            obs_features: Đặc trưng 24 cổ phiếu tại ngày t (Numpy [N, F])
            obs_corr: Ma trận tương quan tại ngày t (Numpy [N, N])
            
        Returns:
            x, edge_index, edge_weight (Tất cả là PyTorch Tensors)
        """
        # 1. Chuyển đặc trưng Node thành Tensor
        x = torch.tensor(obs_features, dtype=torch.float32)
        
        # 2. Xây dựng Đồ thị Đa quan hệ (Tĩnh + Động) từ ma trận tương quan
        edge_index, edge_weight = self.graph_builder.build_from_matrix(obs_corr)
        
        return x, edge_index, edge_weight


class ActionAdapter:
    """
    Biến đổi Quyết định (Action) từ Mạng Nơ-ron thành Lệnh giao dịch cho Môi trường.
    """
    @staticmethod
    def process(actor_logits: torch.Tensor) -> np.ndarray:
        """
        Dịch thuật Action.
        Args:
            actor_logits: Đầu ra thô của Mạng Actor, shape [N] (Ví dụ: [2.5, -1.0, 3.2...])
            
        Returns:
            portfolio_weights: Tỷ trọng danh mục hợp lệ, tổng = 1.0 (Numpy [N])
        """
        # 1. Dùng Softmax để ép tất cả các số về khoảng (0, 1) và có tổng = 1.0
        weights = F.softmax(actor_logits, dim=0)
        
        # 2. Chuyển từ PyTorch Tensor về Numpy Array để gửi cho Env
        portfolio_weights = weights.detach().cpu().numpy()
        
        return portfolio_weights

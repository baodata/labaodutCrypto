# module xây dựng đồ thị tài chính cho các cổ phiếu (Graph Layer)
# kết hợp mối quan hệ cùng ngành (Sector) và tương quan lợi suất (Correlation) để phục vụ mạng GNN

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import yaml

from src.data.loader import get_all_tickers, load_config, load_market_data


@dataclass
class MarketGraph:
    # cấu trúc:
    # tickers: cho ta biết số lượng cổ phiếu N và thứ tự của từng cổ phiếu
    # sector_matrix: cho ta biết mối quan hệ cùng ngành giữa các cổ phiếu [N, N]
    #       ví dụ: nếu cổ phiếu A và B cùng ngành thì sector_matrix[A, B] gần 1, ngược lại gần 0

    # correlation_matrix: ma trận tương quan lợi suất [N, N]
    #       ví dụ: nếu cổ phiếu A và B có lợi suất tương quan cao thì correlation_matrix[A, B] gần 1, ngược lại gần 0
    #       lợi suất là log return: r_t = log(p_t / p_{t-1}) = log(p_t) - log(p_{t-1}) cho ta biết mức độ biến động của cổ phiếu theo thời gian
    #           r_t: lợi suất tại thời điểm t
    #           p_t: giá đóng cửa tại thời điểm t

    # combined_matrix: ma trận kết hợp [N, N]
    #       ví dụ: combined_matrix = alpha * sector_matrix + (1 - alpha) * correlation_matrix
    #       alpha là trọng số cho mối quan hệ cùng ngành, 1 - alpha là trọng số cho tương quan lợi suất

    # normalized_matrix: ma trận kề chuẩn hóa theo GCN [N, N]
    #       ví dụ: normalized_matrix = D^(-1/2) * combined_matrix * D^(-1/2) 
    #       chuẩn hóa ma trận kề theo chuẩn GCN
    #       giá trị nằm trong khoảng [0, 1] và có self-loop (tự nối chính nó)

    # edge_index: torch.Tensor  danh sách cạnh [2, E]
    # edge_weight: torch.Tensor  trọng số cạnh [E]
    #       ví dụ: edge_index = [[0, 1, 2], [1, 2, 3]] nghĩa là có cạnh từ node 0 đến node 1, từ node 1 đến node 2, và từ node 2 đến node 3
    #       edge_weight = [0.5, 0.3, 0.8] là trọng số tương ứng cho từng cạnh

    tickers: List[str]
    sector_matrix: np.ndarray       # ma trận kề nhóm ngành [N, N]
    correlation_matrix: np.ndarray  # ma trận tương quan lợi suất [N, N]
    combined_matrix: np.ndarray     # ma trận kết hợp [N, N]
    normalized_matrix: np.ndarray   # ma trận kề chuẩn hóa theo GCN [N, N]
    edge_index: torch.Tensor        # danh sách cạnh [2, E]
    edge_weight: torch.Tensor       # trọng số cạnh [E]

    @property
    def N(self) -> int:
        return len(self.tickers)

    # chuyển tensor cạnh sang thiết bị tính toán (CPU hoặc GPU)
    def to(self, device: torch.device | str) -> MarketGraph:
        return MarketGraph(
            tickers=self.tickers,
            sector_matrix=self.sector_matrix,
            correlation_matrix=self.correlation_matrix,
            combined_matrix=self.combined_matrix,
            normalized_matrix=self.normalized_matrix,
            edge_index=self.edge_index.to(device),
            edge_weight=self.edge_weight.to(device),
        )


# xây dựng ma trận kề nhóm ngành (sector matrix)
def build_sector_adjacency(tickers: List[str], sectors: Dict[str, List[str]]) -> np.ndarray:
    n = len(tickers)
    adj = np.zeros((n, n), dtype=np.float32)

    # ánh xạ từng ticker thuộc ngành nào
    ticker_to_sector = {}
    for sector_name, ticker_list in sectors.items():
        for ticker in ticker_list:
            ticker_to_sector[ticker] = sector_name

    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            # nối cạnh nếu cùng ngành hoặc chính nó (self-loop)
            if i == j or (ticker_to_sector.get(t1) is not None and ticker_to_sector.get(t1) == ticker_to_sector.get(t2)):
                adj[i, j] = 1.0

    return adj


# tính ma trận tương quan lợi suất và lọc theo ngưỡng threshold
# QUAN TRỌNG: Chỉ tính trên tập Train để tránh rò rỉ dữ liệu tương lai
def build_correlation_adjacency(
    train_prices: np.ndarray,
    threshold: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray]:
    # tính lợi suất 1 ngày
    df_prices = pd.DataFrame(train_prices)
    returns = df_prices.pct_change(1).dropna().values

    # ma trận tương quan Pearson
    corr_matrix = np.corrcoef(returns, rowvar=False).astype(np.float32)
    # xử lý trường hợp bất thường
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # tạo ma trận kề tương quan: giữ cạnh nếu độ tương quan tuyệt đối >= threshold
    corr_adj = np.where(np.abs(corr_matrix) >= threshold, np.abs(corr_matrix), 0.0).astype(np.float32)
    np.fill_diagonal(corr_adj, 1.0)  # Luôn có self-loop

    return corr_matrix, corr_adj


# chuẩn hóa ma trận kề theo công thức GCN: D^(-1/2) * A * D^(-1/2)
# giúp việc truyền tin (message passing) trong GNN không bị nổ hoặc triệt tiêu biên độ
def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    # đảm bảo có self-loop
    adj_loop = adj.copy()
    np.fill_diagonal(adj_loop, 1.0)

    # tính bậc của từng nút
    degrees = np.sum(adj_loop, axis=1)
    # tính d^(-1/2)
    deg_inv_sqrt = np.zeros_like(degrees, dtype=np.float32)
    mask = degrees > 0
    deg_inv_sqrt[mask] = np.power(degrees[mask], -0.5)

    d_mat = np.diag(deg_inv_sqrt)
    normalized = d_mat @ adj_loop @ d_mat
    return normalized.astype(np.float32)


# chuyển ma trận kề thành định dạng cạnh edge_index và edge_weight cho PyTorch
def adjacency_to_edge_index(adj: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    src, dst = np.nonzero(adj)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_weight = torch.tensor(adj[src, dst], dtype=torch.float32)
    return edge_index, edge_weight


# hàm chính xây dựng toàn bộ đồ thị tài chính từ dữ liệu và file config
def build_market_graph(
    config_path: str = "configs/assets.yaml",
    correlation_threshold: float = 0.4,
    sector_weight: float = 0.5,
) -> MarketGraph:
    cfg = load_config(config_path)
    full_data, splits = load_market_data(config_path)

    tickers = full_data.tickers
    sectors = cfg.get("sectors", {})

    # 1. Đồ thị nhóm ngành (Sector Graph)
    sector_adj = build_sector_adjacency(tickers, sectors)

    # 2. Đồ thị tương quan lợi suất (Correlation Graph) - CHỈ TÍNH TRÊN TẬP TRAIN
    train_prices = splits["train"].get_feature("close")
    corr_matrix, corr_adj = build_correlation_adjacency(train_prices, threshold=correlation_threshold)

    # 3. Kết hợp 2 đồ thị theo trọng số alpha
    # combined = alpha * sector + (1 - alpha) * correlation
    alpha = float(sector_weight)
    combined = (alpha * sector_adj) + ((1.0 - alpha) * corr_adj)

    # 4. Chuẩn hóa ma trận kề theo chuẩn GCN
    norm_adj = normalize_adjacency(combined)

    # 5. Xuất ra định dạng cạnh cho PyTorch
    edge_index, edge_weight = adjacency_to_edge_index(norm_adj)

    return MarketGraph(
        tickers=tickers,
        sector_matrix=sector_adj,
        correlation_matrix=corr_matrix,
        combined_matrix=combined,
        normalized_matrix=norm_adj,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )

# kiểm thử cho module xây dựng đồ thị tài chính (Graph Layer)

import numpy as np
import pytest
import torch

from src.graph.builder import (
    adjacency_to_edge_index,
    build_correlation_adjacency,
    build_market_graph,
    build_sector_adjacency,
    normalize_adjacency,
)


def test_build_sector_adjacency():
    tickers = ["AAPL", "MSFT", "JPM", "BAC"]
    sectors = {
        "Tech": ["AAPL", "MSFT"],
        "Finance": ["JPM", "BAC"],
    }

    adj = build_sector_adjacency(tickers, sectors)

    assert adj.shape == (4, 4)
    # ma trận phải đối xứng
    assert np.array_equal(adj, adj.T)
    # đường chéo chính phải bằng 1 (self-loop)
    assert np.all(np.diag(adj) == 1.0)
    # AAPL (0) và MSFT (1) cùng ngành -> 1
    assert adj[0, 1] == 1.0
    assert adj[1, 0] == 1.0
    # AAPL (0) và JPM (2) khác ngành -> 0
    assert adj[0, 2] == 0.0
    assert adj[2, 0] == 0.0


def test_build_correlation_adjacency():
    # tạo dữ liệu giá giả lập cho 3 cổ phiếu
    np.random.seed(42)
    t = 100
    p1 = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, size=t)))
    # p2 có tương quan rất cao với p1
    p2 = 50.0 * np.exp(np.cumsum(0.9 * np.random.normal(0, 0.01, size=t) + 0.1 * np.random.normal(0, 0.01, size=t)))
    # p3 độc lập
    p3 = 20.0 * np.exp(np.cumsum(np.random.normal(0, 0.03, size=t)))

    prices = np.stack([p1, p2, p3], axis=-1)

    corr_mat, corr_adj = build_correlation_adjacency(prices, threshold=0.5)

    assert corr_mat.shape == (3, 3)
    assert np.all(corr_mat >= -1.0) and np.all(corr_mat <= 1.0)
    assert np.all(np.diag(corr_adj) == 1.0)
    # ma trận kề tương quan đối xứng
    assert np.array_equal(corr_adj, corr_adj.T)


def test_normalize_adjacency_gcn():
    adj = np.array([
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=np.float32)

    norm_adj = normalize_adjacency(adj)

    assert norm_adj.shape == (3, 3)
    assert np.allclose(norm_adj, norm_adj.T)

    # kiểm tra trị riêng lớn nhất (bán kính phổ) của ma trận GCN luôn <= 1.0
    eigenvals = np.linalg.eigvalsh(norm_adj)
    assert np.max(eigenvals) <= 1.0 + 1e-5


def test_build_market_graph_integration():
    # kiểm thử tích hợp trên toàn bộ 24 cổ phiếu thực tế
    graph = build_market_graph(
        config_path="configs/assets.yaml",
        correlation_threshold=0.4,
        sector_weight=0.5,
    )

    assert graph.N == 24
    assert graph.sector_matrix.shape == (24, 24)
    assert graph.correlation_matrix.shape == (24, 24)
    assert graph.combined_matrix.shape == (24, 24)
    assert graph.normalized_matrix.shape == (24, 24)

    # kiểm tra cấu trúc PyTorch
    assert isinstance(graph.edge_index, torch.Tensor)
    assert isinstance(graph.edge_weight, torch.Tensor)
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] == graph.edge_weight.shape[0]

    # kiểm tra chuyển đổi device
    cpu_graph = graph.to("cpu")
    assert cpu_graph.edge_index.device.type == "cpu"
    assert not torch.isnan(cpu_graph.edge_weight).any()

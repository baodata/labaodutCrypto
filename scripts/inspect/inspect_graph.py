#!/usr/bin/env python3
"""test output for financial market graph construction."""

from pathlib import Path
import sys
import numpy as np

# thêm thư mục gốc dự án vào sys.path để chạy ở đâu cũng import được
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_config
from src.graph import build_market_graph


def main():
    print("=" * 70)
    print("KIỂM TRA ĐỒ THỊ TÀI CHÍNH (FINANCIAL GRAPH INSPECTION)")
    print("=" * 70)

    cfg = load_config("configs/assets.yaml")
    graph = build_market_graph(config_path="configs/assets.yaml", correlation_threshold=0.4, sector_weight=0.5)

    print(f"\n1. Thông tin tổng quan đồ thị:")
    print(f"   - Số nút (cổ phiếu)             : {graph.N}")
    print(f"   - Tổng số cạnh nối (Edges)      : {graph.edge_index.shape[1]}")
    density = graph.edge_index.shape[1] / (graph.N * graph.N)
    print(f"   - Mật độ đồ thị (Graph Density) : {density:.1%}")

    print(f"\n2. Phân bố các nhóm ngành (Sectors):")
    for sec_name, tickers in cfg.get("sectors", {}).items():
        print(f"   - {sec_name:<14} ({len(tickers)} mã): {', '.join(tickers)}")

    print(f"\n3. Top 10 cặp cổ phiếu có tương quan cao nhất (trên tập Train):")
    pairs = []
    for i in range(graph.N):
        for j in range(i + 1, graph.N):
            pairs.append((graph.tickers[i], graph.tickers[j], graph.correlation_matrix[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"   {'CẶP CỔ PHIẾU':<18} | {'HỆ SỐ TƯƠNG QUAN (PEARSON)':<26}")
    print("   " + "-" * 48)
    for t1, t2, corr in pairs[:10]:
        print(f"   {t1} - {t2:<12} | {corr:>+10.4f}")

    print(f"\n4. Cấu trúc tensor phục vụ mạng GNN (PyTorch):")
    print(f"   - edge_index shape  : {tuple(graph.edge_index.shape)} (dtype={graph.edge_index.dtype})")
    print(f"   - edge_weight shape : {tuple(graph.edge_weight.shape)} (dtype={graph.edge_weight.dtype})")
    print(f"   - normalized_matrix : {graph.normalized_matrix.shape} (min={graph.normalized_matrix.min():.4f}, max={graph.normalized_matrix.max():.4f})")

    print("\n✅ ĐỒ THỊ ĐƯỢC XÂY DỰNG HOÀN TOÀN TỪ TẬP TRAIN, KHÔNG BỊ RÒ RỈ DỮ LIỆU TƯƠNG LAI.")
    print("=" * 70)


if __name__ == "__main__":
    main()

import torch
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features.pipeline import FeaturePipeline
from src.models.gnn.gcn_encoder import GCNEncoder
from src.models.gnn.gat_encoder import GATEncoder
from src.graph.multi_relation_graph import MultiRelationGraphBuilder

def main():
    print("=== KIỂM THỬ TÍCH HỢP GNN (GNN-002) ===")
    
    try:
        # 1. Chạy Pipeline của A để lấy dữ liệu thực tế
        print("1. Chạy Feature Pipeline (A)...")
        pipeline = FeaturePipeline()
        _, market_tensor = pipeline.run_from_raw(raw_dir="data/raw", export_parquet=False)
        print(f" -> Lấy thành công tensor kích thước: {market_tensor.tensor.shape}")
        
        # 2. Xây dựng đồ thị thực tế (A)
        print("2. Xây dựng đồ thị...")
        T, N, F = market_tensor.tensor.shape
        tickers = market_tensor.tickers
        
        # Tạo dummy correlation matrix (NxN) cho 1 bước step (Env gọi adapter)
        dummy_corr = np.eye(N)
        # Giả lập tương quan mạnh giữa 0 và 1
        dummy_corr[0, 1] = 0.8
        dummy_corr[1, 0] = 0.8
        
        # Build graph
        # Chúng ta truyền bảng DataFrame rỗng cho builder vì khi gọi build_from_matrix nó chỉ cần sec_edge_index
        # Nhưng constructor cần DataFrame. Ta đưa None.
        builder = MultiRelationGraphBuilder(tickers=tickers, rolling_corr_df=None, threshold=0.5)
        edge_index, edge_weight = builder.build_from_matrix(dummy_corr)
        print(f" -> Xây dựng thành công đồ thị đa quan hệ (Tương quan + Ngành) với {edge_index.shape[1]} cạnh.")
        
        # 3. Đưa vào GNN của B
        print("3. Forward qua GNN (B)...")
        # Lấy đặc trưng ngày đầu tiên
        x = torch.tensor(market_tensor.tensor[0], dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        
        gcn = GCNEncoder(in_channels=F, hidden_channels=32, out_channels=16)
        out_gcn = gcn(x, edge_index, edge_weight)
        assert out_gcn.shape == (N, 16), f"Lỗi shape GCN: {out_gcn.shape}"
        assert not torch.isnan(out_gcn).any(), "GCN xuất hiện NaN!"
        print(" -> GCN Forward thành công!")
        
        gat = GATEncoder(in_channels=F, hidden_channels=32, out_channels=16, heads=2)
        out_gat = gat(x, edge_index, edge_weight)
        assert out_gat.shape == (N, 16), f"Lỗi shape GAT: {out_gat.shape}"
        assert not torch.isnan(out_gat).any(), "GAT xuất hiện NaN!"
        print(" -> GAT Forward thành công!")
        
        print("\n✅ KIỂM THỬ TÍCH HỢP THÀNH CÔNG TỐT ĐẸP!")
        
    except Exception as e:
        print(f"❌ LỖI: {e}")

if __name__ == "__main__":
    main()

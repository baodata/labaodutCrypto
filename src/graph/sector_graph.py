"""
src/graph/sector_graph.py
Đồ thị tĩnh (Static Graph) dựa trên nhóm ngành.
Ticket: GRAPH-003
"""
import torch
from typing import Tuple
from src.utils.config import AssetsConfig

def generate_sector_edges(tickers: list[str], config_path: str = "configs/assets.yaml") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tạo mạng lưới liên kết tĩnh dựa trên ngành nghề (Sector).
    Hai cổ phiếu cùng ngành sẽ được nối với nhau bằng trọng số 1.0.
    """
    config = AssetsConfig(config_path)
    ticker_to_sector = config.get_ticker_to_sector()
    
    sources = []
    targets = []
    weights = []
    
    N = len(tickers)
    for i in range(N):
        for j in range(N):
            if i != j:
                sec_i = ticker_to_sector.get(tickers[i])
                sec_j = ticker_to_sector.get(tickers[j])
                
                if sec_i and sec_j and sec_i == sec_j:
                    sources.append(i)
                    targets.append(j)
                    weights.append(1.0)
                    
    if len(sources) > 0:
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)
        
    return edge_index, edge_weight

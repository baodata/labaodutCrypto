"""
experiments/dry_run_training.py
Kịch bản chạy thử (Dry Run) toàn bộ hệ thống Não bộ của Thành viên B,
KẾT HỢP với Môi trường Sàn giao dịch thực (TradingEnv MODULE MỚI) của Thành viên A!
"""
import sys
from pathlib import Path
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.gnn.gat_encoder import GATEncoder
# Import TẤT CẢ từ file adapters.py (Vì đã gộp theo yêu cầu của bạn)
from src.models.adapters import GNN_ActorCritic, ObservationAdapter, ActionAdapter
from src.graph.multi_relation_graph import MultiRelationGraphBuilder
from src.training.logger import TrainerLogger
from src.env.trading_env import TradingEnv
from src.utils.data_types import MarketDataTensor

class MockMarketTensor:
    def __init__(self, T, N, F):
        self.tensor = np.random.randn(T, N, F) * 0.01

def main():
    print("="*60)
    print(" 🛠️ ĐANG CHẠY THỬ NGHIỆM VỚI TRADING ENV MODULE HÓA (A)")
    print("="*60)
    
    T_days, N_stocks, F_features = 2700, 24, 6
    mock_tensor = MockMarketTensor(T_days, N_stocks, F_features)
    # TradingEnv mới yêu cầu open_prices để khớp lệnh
    mock_open_prices = np.ones((T_days, N_stocks)) * 100.0 
    
    print(f"[1] Khởi tạo Sàn giao dịch TradingEnv cho {T_days} ngày (Hơn 10 năm)...")
    env = TradingEnv(market_tensor=mock_tensor, open_prices=mock_open_prices, config_path="configs/env.yaml")
    
    print("[2] Khởi tạo Graph Builder & Não bộ Tiền mặt (B)...")
    graph_builder = MultiRelationGraphBuilder(tickers=[f"T{i}" for i in range(N_stocks)], rolling_corr_df=None, threshold=0.5)
    obs_adapter = ObservationAdapter(graph_builder)
    
    gat = GATEncoder(in_channels=F_features, hidden_channels=32, out_channels=16, heads=2)
    model = GNN_ActorCritic(gnn_encoder=gat, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    logger = TrainerLogger(log_dir="experiments/logs_real_env", checkpoint_dir="experiments/checkpoints_real_env")
    
    print("\n🚀 BẮT ĐẦU HUẤN LUYỆN 5 EPOCHS (Khoảng thời gian cực dài)...")
    for epoch in range(1, 6):
        obs, info = env.reset()
        epoch_loss = 0.0
        
        for step in range(1, T_days):
            # Tính tương quan của tối đa 60 ngày gần nhất (Rolling Window)
            if step > 2:
                start_idx = max(0, step - 60)
                obs_corr = np.corrcoef(mock_tensor.tensor[start_idx:step, :, 0].T)
                obs_corr = np.nan_to_num(obs_corr, nan=0.0)
            else:
                obs_corr = np.zeros((N_stocks, N_stocks))
                
            # Dictionary obs trả về từ TradingEnv, ta lấy 'market_state'
            x, edge_index, edge_weight = obs_adapter.process(obs['market_state'], obs_corr)
            
            optimizer.zero_grad()
            # Mạng xuất ra 25 Logits (24 cổ + 1 tiền)
            action_logits, state_value = model(x, edge_index, edge_weight)
            
            # Action Adapter tự động Softmax ra 25 tỷ trọng tổng = 1
            portfolio_weights = ActionAdapter.process(action_logits)
            
            # Gửi 25 Tỷ trọng lên sàn
            next_obs, reward, terminated, truncated, info = env.step(portfolio_weights)
            
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            critic_loss = torch.nn.functional.mse_loss(state_value, reward_tensor)
            advantage = reward - state_value.item()
            
            import torch.nn.functional as F
            weights_tensor = F.softmax(action_logits, dim=0)
            actor_loss = -torch.mean(torch.log(weights_tensor + 1e-8) * advantage) 
            
            total_loss = critic_loss + actor_loss
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            obs = next_obs
            
            if terminated:
                break
                
        avg_loss = epoch_loss / T_days
        logger.log_metrics(epoch, {"Loss/Epoch": avg_loss, "Portfolio/FinalValue": info['portfolio_value']})
        print(f"🔥 Epoch {epoch:02d}/5 | Tài khoản: ${info['portfolio_value']:,.2f} | Lỗ TB: {avg_loss:.4f} | Rủi ro sập: {info['max_drawdown']*100:.2f}%")
            
    logger.close()
    print("\n✅ THÀNH CÔNG: Sàn mới của A (Có Tiền Mặt) đã chạy mượt mà cùng Não bộ của B!")

if __name__ == "__main__":
    main()

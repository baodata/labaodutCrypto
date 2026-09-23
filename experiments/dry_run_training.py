"""
experiments/dry_run_training.py
Kịch bản chạy thử (Dry Run) toàn bộ hệ thống Não bộ của Thành viên B,
KẾT HỢP với Môi trường Sàn giao dịch thực (PortfolioEnv) của Thành viên A!
"""
import sys
from pathlib import Path
import torch
import numpy as np
import warnings

# Tắt cảnh báo Numpy khi tính toán tương quan bị chia 0 ở ngày đầu tiên
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.gnn.gat_encoder import GATEncoder
from src.models.actor_critic import GNN_ActorCritic
from src.models.adapters import ObservationAdapter, ActionAdapter
from src.graph.multi_relation_graph import MultiRelationGraphBuilder
from src.train.logger import TrainerLogger
from src.env.portfolio_env import PortfolioEnv

class MockMarketTensor:
    def __init__(self, T, N, F):
        # Giả lập dữ liệu lợi nhuận và đặc trưng (thay vì chạy Pipeline nặng)
        self.tensor = np.random.randn(T, N, F) * 0.01

def main():
    print("="*60)
    print(" 🛠️ ĐANG CHẠY THỬ NGHIỆM VÒNG LẶP HUẤN LUYỆN (REAL ENV)")
    print("="*60)
    
    # 1. TẠO DỮ LIỆU ĐỂ BƠM VÀO SÀN GIAO DỊCH
    T_days, N_stocks, F_features = 2000, 24, 6
    mock_tensor = MockMarketTensor(T_days, N_stocks, F_features)
    
    print("[1] Khởi tạo Sàn giao dịch PortfolioEnv (Sprint 4 - A)...")
    env = PortfolioEnv(market_tensor=mock_tensor, config_path="configs/env.yaml")
    
    print("[2] Khởi tạo Graph Builder & Adapters (Sprint 5 - B)...")
    graph_builder = MultiRelationGraphBuilder(tickers=[f"T{i}" for i in range(N_stocks)], rolling_corr_df=None, threshold=0.5)
    obs_adapter = ObservationAdapter(graph_builder)
    
    print("[3] Khởi tạo Não bộ GNN và Actor-Critic (Sprint 4&5 - B)...")
    gat = GATEncoder(in_channels=F_features, hidden_channels=32, out_channels=16, heads=2)
    model = GNN_ActorCritic(gnn_encoder=gat, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("[4] Khởi tạo Logger...")
    logger = TrainerLogger(log_dir="experiments/logs_real_env", checkpoint_dir="experiments/checkpoints_real_env")
    
    # 3. CHẠY HUẤN LUYỆN 50 EPOCHS
    print("\n🚀 BẮT ĐẦU HUẤN LUYỆN 50 EPOCHS (VÒNG ĐỜI)...")
    
    for epoch in range(1, 51):
        obs, info = env.reset()
        epoch_loss = 0.0
        
        for step in range(1, T_days):
            # A. Ma trận tương quan
            if step > 2:
                obs_corr = np.corrcoef(mock_tensor.tensor[:step, :, 0].T)
                obs_corr = np.nan_to_num(obs_corr, nan=0.0)
            else:
                obs_corr = np.zeros((N_stocks, N_stocks))
                
            # B. Dịch thuật dữ liệu
            x, edge_index, edge_weight = obs_adapter.process(obs, obs_corr)
            
            # C. Suy nghĩ
            optimizer.zero_grad()
            action_logits, state_value = model(x, edge_index, edge_weight)
            
            # Tính Softmax weights bằng PyTorch để giữ Gradient cho lúc Backprop
            import torch.nn.functional as F
            weights_tensor = F.softmax(action_logits, dim=0)
            portfolio_weights = weights_tensor.detach().cpu().numpy()
            
            # D. Khớp lệnh
            next_obs, reward, terminated, truncated, info = env.step(portfolio_weights)
            
            # E. THUẬT TOÁN HỌC (POLICY GRADIENT + CRITIC)
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            
            # Lỗi của Critic (Đoán sai)
            critic_loss = F.mse_loss(state_value, reward_tensor)
            
            # Lỗi của Actor (Dùng Advantage)
            advantage = reward - state_value.item()
            # Khuyến khích hành động sinh ra Lãi lớn (Advantage > 0)
            actor_loss = -torch.mean(torch.log(weights_tensor + 1e-8) * advantage) 
            
            total_loss = critic_loss + actor_loss
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            obs = next_obs
            
            if terminated:
                break
                
        # Ghi log mỗi Epoch
        avg_loss = epoch_loss / T_days
        logger.log_metrics(epoch, {"Loss/Epoch": avg_loss, "Portfolio/FinalValue": info['portfolio_value']})
        logger.save_checkpoint(epoch, model, optimizer, current_reward=info['portfolio_value'])
        
        # In ra màn hình xem AI tiến bộ thế nào
        if epoch % 5 == 0 or epoch == 1:
            print(f"🔥 Epoch {epoch:02d}/50 | Tài khoản cuối năm: ${info['portfolio_value']:,.2f} | Avg Loss: {avg_loss:.4f}")
            
    logger.close()
    print("\n✅ THÀNH CÔNG: AI ĐÃ TRẢI QUA 50 KIẾP LUÂN HỒI!")
    print("Mở TensorBoard để xem AI học cách sống sót như thế nào nhé.")

if __name__ == "__main__":
    main()

"""
experiments/dry_run_training.py
Kịch bản chạy thử (Dry Run) toàn bộ hệ thống Não bộ của Thành viên B,
KẾT HỢP với Môi trường Sàn giao dịch thực (TradingEnv) của Thành viên A!
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import warnings

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.gnn.gat_encoder import GATEncoder
from src.graph.multi_relation_graph import MultiRelationGraphBuilder
from src.training.logger import TrainerLogger
from src.env.trading_env import TradingEnv
from src.utils.data_types import MarketDataTensor

# Tạm thời Mock ActorCritic (Vì Sprint 5 Thành viên A chưa code file src.models.actor_critic)
class GNN_ActorCritic(nn.Module):
    def __init__(self, gnn_encoder, hidden_dim=16):
        super().__init__()
        self.gnn = gnn_encoder
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x, edge_index, edge_weight):
        node_embeds = self.gnn(x, edge_index, edge_weight) # [N, hidden_dim]
        action_logits = self.actor(node_embeds) # [N, 1]
        market_embed = torch.mean(node_embeds, dim=0) # [hidden_dim]
        state_value = self.critic(market_embed) # [1]
        return action_logits, state_value

class MockMarketTensor:
    def __init__(self, T, N, F):
        self.tensor = np.random.randn(T, N, F) * 0.01

def main():
    print("="*60)
    print(" 🛠️ ĐANG CHẠY THỬ NGHIỆM VÒNG LẶP HUẤN LUYỆN (REAL ENV)")
    print("="*60)
    
    T_days, N_stocks, F_features = 200, 24, 6
    mock_tensor_data = MockMarketTensor(T_days, N_stocks, F_features)
    
    tickers = [f"T{i}" for i in range(N_stocks)]
    features = [f"F{i}" for i in range(F_features)]
    dates = [f"2024-01-{min(i+1, 31):02d}" for i in range(T_days)]
    mock_tensor = MarketDataTensor(mock_tensor_data.tensor, tickers, features, dates)
    
    open_prices = np.zeros((T_days, N_stocks))
    open_prices[0] = 100.0
    for t in range(1, T_days):
        open_prices[t] = open_prices[t-1] * (1.0 + np.random.randn(N_stocks) * 0.01)
        
    print("[1] Khởi tạo Sàn giao dịch TradingEnv (Sprint 4 - A)...")
    env = TradingEnv(market_tensor=mock_tensor, open_prices=open_prices, config_path="configs/env.yaml")
    
    print("[2] Khởi tạo Graph Builder (Sprint 5 - B)...")
    graph_builder = MultiRelationGraphBuilder(tickers=tickers, rolling_corr_df=None, threshold=0.5)
    
    print("[3] Khởi tạo Não bộ GNN và Actor-Critic (Mock vì A chưa làm Sprint 5)...")
    gat = GATEncoder(in_channels=F_features, hidden_channels=32, out_channels=16, heads=2)
    model = GNN_ActorCritic(gnn_encoder=gat, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("[4] Khởi tạo Logger...")
    logger = TrainerLogger(log_dir="experiments/logs_real_env", checkpoint_dir="experiments/checkpoints_real_env")
    
    print("\n🚀 BẮT ĐẦU HUẤN LUYỆN 50 EPOCHS (VÒNG ĐỜI)...")
    
    for epoch in range(1, 51):
        obs, info = env.reset()
        epoch_loss = 0.0
        
        while True:
            step = info['step']
            
            if step > 2:
                obs_corr = np.corrcoef(mock_tensor.tensor[:step, :, 0].T)
                obs_corr = np.nan_to_num(obs_corr, nan=0.0)
            else:
                obs_corr = np.zeros((N_stocks, N_stocks))
                
            market_state = obs["market_state"]
            x = torch.tensor(market_state, dtype=torch.float32)
            edge_index, edge_weight = graph_builder.build_from_matrix(obs_corr)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
            
            optimizer.zero_grad()
            action_logits, state_value = model(x, edge_index, edge_weight)
            
            import torch.nn.functional as F
            cash_logit = torch.tensor([0.0], dtype=torch.float32)
            logits_with_cash = torch.cat([action_logits.squeeze(), cash_logit], dim=0)
            
            weights_tensor = F.softmax(logits_with_cash, dim=0)
            portfolio_weights = weights_tensor.detach().cpu().numpy()
            
            next_obs, reward, terminated, truncated, info = env.step(portfolio_weights)
            
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            critic_loss = F.mse_loss(state_value.squeeze(), reward_tensor)
            
            advantage = reward - state_value.item()
            actor_loss = -torch.mean(torch.log(weights_tensor + 1e-8) * advantage) 
            
            total_loss = critic_loss + actor_loss
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            obs = next_obs
            
            if terminated:
                break
                
        avg_loss = epoch_loss / (env.T - 2)
        logger.log_metrics(epoch, {"Loss/Epoch": avg_loss, "Portfolio/FinalValue": info['portfolio_value']})
        
        if epoch % 10 == 0:
            logger.save_checkpoint(epoch, model, optimizer, current_reward=info['portfolio_value'])
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"🔥 Epoch {epoch:02d}/50 | Tài khoản cuối năm: ${info['portfolio_value']:,.2f} | Avg Loss: {avg_loss:.4f}")
            
    logger.close()
    print("\n✅ THÀNH CÔNG: AI ĐÃ TRẢI QUA 50 KIẾP LUÂN HỒI BẰNG TRADING ENV CHUẨN!")

if __name__ == "__main__":
    main()

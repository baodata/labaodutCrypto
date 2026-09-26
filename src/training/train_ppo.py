"""
src/training/train_ppo.py
Kịch bản Huấn luyện Chính thức (Real Training Loop).
Ticket: TRAIN-001 (Kết hợp toàn bộ Hệ thống)

Sử dụng Data thật, Config thật và Môi trường thật.
"""
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import torch.nn.functional as F

warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.features.pipeline import FeaturePipeline
from src.models.gnn.gat_encoder import GATEncoder
from src.models.adapters import GNN_ActorCritic, ObservationAdapter, ActionAdapter
from src.graph.multi_relation_graph import MultiRelationGraphBuilder
from src.training.logger import TrainerLogger
from src.env.trading_env import TradingEnv

def load_configs():
    with open("configs/model.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    with open("configs/env.yaml", "r") as f:
        env_cfg = yaml.safe_load(f)
    return model_cfg, env_cfg

def main():
    print("="*60)
    print(" 🚀 H-MARL-GNN: BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN CHÍNH THỨC")
    print("="*60)
    
    # 1. ĐỌC CONFIG THẬT
    model_cfg, env_cfg = load_configs()
    lr = model_cfg['rl_agent']['learning_rate']
    hidden_dim = model_cfg['gnn']['hidden_channels']
    out_dim = model_cfg['gnn']['out_channels']
    heads = model_cfg['gnn']['heads']
    
    # 2. XỬ LÝ DỮ LIỆU THẬT
    print("[1] Đang nạp và xử lý Dữ liệu thị trường thật...")
    pipeline = FeaturePipeline()
    # Chạy pipeline lấy Data thật (Nếu chạy lần đầu sẽ hơi lâu để tính RSI/MACD)
    processed_data, market_tensor = pipeline.run_from_raw(raw_dir="data/raw", export_parquet=False)
    
    T_days, N_stocks, F_features = market_tensor.tensor.shape
    tickers = market_tensor.tickers
    print(f" -> Đã nạp thành công {N_stocks} cổ phiếu trong {T_days} ngày. Số đặc trưng: {F_features}")
    
    # Trích xuất giá Mở cửa (Open prices) để đưa vào Sàn giao dịch
    open_prices = np.zeros((T_days, N_stocks))
    for idx, ticker in enumerate(tickers):
        df = processed_data[ticker]
        open_col = 'open' if 'open' in df.columns else 'Open'
        if open_col in df.columns:
            open_prices[:, idx] = df[open_col].values
        else:
            open_prices[:, idx] = 100.0  # Fallback
            
    # 3. KHỞI TẠO CÔNG CỤ
    print("[2] Khởi tạo Sàn giao dịch (TradingEnv)...")
    env = TradingEnv(market_tensor=market_tensor, open_prices=open_prices, config_path="configs/env.yaml")
    
    print("[3] Khởi tạo GNN, Actor-Critic và Adapters...")
    graph_builder = MultiRelationGraphBuilder(tickers=tickers, rolling_corr_df=None, threshold=model_cfg['gnn']['threshold_corr'])
    obs_adapter = ObservationAdapter(graph_builder)
    
    gat = GATEncoder(in_channels=F_features, hidden_channels=hidden_dim, out_channels=out_dim, heads=heads)
    model = GNN_ActorCritic(gnn_encoder=gat, hidden_dim=out_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    logger = TrainerLogger(log_dir="logs/real_training", checkpoint_dir="models/checkpoints")
    
    # 4. VÒNG LẶP HUẤN LUYỆN
    EPOCHS = 50
    print(f"\n🚀 BẮT ĐẦU HUẤN LUYỆN {EPOCHS} EPOCHS...")
    
    for epoch in range(1, EPOCHS + 1):
        obs, info = env.reset()
        epoch_loss = 0.0
        
        for step in range(1, T_days):
            # Tính Tương quan 60 ngày gần nhất (Tăng tốc độ)
            if step > 2:
                start_idx = max(0, step - 60)
                obs_corr = np.corrcoef(market_tensor.tensor[start_idx:step, :, 0].T)
                obs_corr = np.nan_to_num(obs_corr, nan=0.0)
            else:
                obs_corr = np.zeros((N_stocks, N_stocks))
                
            x, edge_index, edge_weight = obs_adapter.process(obs['market_state'], obs_corr)
            
            optimizer.zero_grad()
            action_logits, state_value = model(x, edge_index, edge_weight)
            
            weights_tensor = F.softmax(action_logits, dim=0)
            portfolio_weights = weights_tensor.detach().cpu().numpy()
            
            next_obs, reward, terminated, truncated, info = env.step(portfolio_weights)
            
            # CÔNG THỨC PPO ĐƠN GIẢN
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            critic_loss = F.mse_loss(state_value, reward_tensor)
            advantage = reward - state_value.item()
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
        logger.save_checkpoint(epoch, model, optimizer, current_reward=info['portfolio_value'])
        
        print(f"🔥 Epoch {epoch:02d}/{EPOCHS} | Tài khoản: ${info['portfolio_value']:,.2f} | Lỗ TB: {avg_loss:.4f} | Drawdown: {info['max_drawdown']*100:.2f}%")
            
    logger.close()
    print("\n✅ HUẤN LUYỆN HOÀN TẤT. CHÚC MỪNG BẠN!")

if __name__ == "__main__":
    main()

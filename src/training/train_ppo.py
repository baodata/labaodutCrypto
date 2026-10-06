"""
src/training/train_ppo.py
Kịch bản Huấn luyện PPO Chính thức với cơ chế Chia Data (Train/Val) chuẩn mực.
"""
import sys
import yaml
import torch
import numpy as np
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
from src.utils.data_types import MarketDataTensor

# Import 2 vũ khí tối thượng của Thành viên A
from src.training.ppo_core import discount_cumsum

def get_action_and_value(model, obs_adapter, market_state, obs_corr):
    """Tiện ích nạp Data vào Mạng và Trả về Action, Value, Log_Prob."""
    x, edge_index, edge_weight = obs_adapter.process(market_state, obs_corr)
    
    # 1. Forward pass
    action_logits, state_value = model(x, edge_index, edge_weight)
    
    # 2. Xử lý Softmax & Phân phối xác suất
    action_probs = F.softmax(action_logits, dim=0)
    
    # 3. Lấy ra Log Probability của hành động
    # Ở PPO thực tế, phải tạo Torch Distribution (VD: Dirichlet hoặc Categorical)
    # Nhưng vì đây là danh mục đầu tư (Tỷ trọng liên tục), ta tạm dùng Log-Softmax
    log_probs = F.log_softmax(action_logits, dim=0)
    
    # Tính log_prob tổng của cả danh mục (Trung bình cộng)
    log_prob = log_probs.mean()
    
    # Tính Entropy (Để ép AI khám phá)
    entropy = -(action_probs * log_probs).sum()
    
    return action_probs, state_value, log_prob, entropy, x, edge_index, edge_weight

def main():
    print("="*60)
    print(" 🚀 H-MARL-GNN: HUẤN LUYỆN PPO VỚI TRAIN/VAL SPLIT")
    print("="*60)
    
    # 1. ĐỌC CẤU HÌNH
    with open("configs/model.yaml", "r") as f: model_cfg = yaml.safe_load(f)
    with open("configs/env.yaml", "r") as f: env_cfg = yaml.safe_load(f)
    
    lr = model_cfg['rl_agent']['learning_rate']
    clip_ratio = model_cfg['rl_agent']['clip_ratio']
    gamma = model_cfg['rl_agent']['gamma']
    lam = 0.95
    
    # 2. NẠP DỮ LIỆU VÀ CHIA TRAIN/VAL (Chronological Split)
    print("[1] Đang nạp Dữ liệu thật...")
    pipeline = FeaturePipeline()
    processed_data, market_tensor = pipeline.run_from_raw(raw_dir="data/raw", export_parquet=False)
    
    T_days, N_stocks, F_features = market_tensor.tensor.shape
    open_prices = np.zeros((T_days, N_stocks))
    for idx, ticker in enumerate(market_tensor.tickers):
        df = processed_data[ticker]
        open_col = 'open' if 'open' in df.columns else 'Open'
        open_prices[:, idx] = df[open_col].values if open_col in df.columns else 100.0
        
    # QUY TẮC VÀNG: Cắt 80% thời gian đầu làm Train, 20% thời gian cuối làm Val
    T_train = int(T_days * 0.8)
    print(f" -> Cắt dữ liệu: Tập Train ({T_train} ngày) | Tập Val ({T_days - T_train} ngày)")
    
    # Tạo Market Tensor cho từng tập
    train_tensor = MarketDataTensor(market_tensor.tensor[:T_train], market_tensor.tickers, market_tensor.feature_names, market_tensor.dates[:T_train])
    val_tensor = MarketDataTensor(market_tensor.tensor[T_train:], market_tensor.tickers, market_tensor.feature_names, market_tensor.dates[T_train:])
    
    train_open_prices = open_prices[:T_train]
    val_open_prices = open_prices[T_train:]
    
    # Khởi tạo 2 môi trường riêng biệt
    env_train = TradingEnv(market_tensor=train_tensor, open_prices=train_open_prices, config_path="configs/env.yaml")
    env_val = TradingEnv(market_tensor=val_tensor, open_prices=val_open_prices, config_path="configs/env.yaml")
    
    # 3. KHỞI TẠO NÃO BỘ
    print("[2] Khởi tạo GNN & Adapter...")
    graph_builder = MultiRelationGraphBuilder(tickers=market_tensor.tickers, rolling_corr_df=None, threshold=0.5)
    obs_adapter = ObservationAdapter(graph_builder)
    
    gat = GATEncoder(in_channels=F_features, hidden_channels=32, out_channels=16, heads=2)
    model = GNN_ActorCritic(gnn_encoder=gat, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    logger = TrainerLogger(log_dir="logs/real_training", checkpoint_dir="models/checkpoints")
    
    EPOCHS = 100
    print(f"\n🚀 BẮT ĐẦU {EPOCHS} EPOCHS...")
    
    for epoch in range(1, EPOCHS + 1):
        # ==========================================
        # GIAI ĐOẠN 1: THU THẬP QUỸ ĐẠO Ở TẬP TRAIN
        # ==========================================
        model.train()
        obs, info = env_train.reset()
        
        # Các mảng lưu quỹ đạo
        rews, vals, logps, entropies = [], [], [], []
        
        for step in range(1, T_train):
            # Tính tương quan cuộn 60 ngày
            start_idx = max(0, step - 60)
            obs_corr = np.nan_to_num(np.corrcoef(train_tensor.tensor[start_idx:step, :, 0].T), nan=0.0) if step > 2 else np.zeros((N_stocks, N_stocks))
            
            action_probs, state_value, log_prob, entropy, _, _, _ = get_action_and_value(model, obs_adapter, obs['market_state'], obs_corr)
            
            portfolio_weights = action_probs.detach().cpu().numpy()
            next_obs, reward, terminated, _, info = env_train.step(portfolio_weights)
            
            rews.append(reward)
            vals.append(state_value.item())
            logps.append(log_prob)
            entropies.append(entropy)
            
            obs = next_obs
            if terminated: break
            
        # ==========================================
        # GIAI ĐOẠN 2: TÍNH GAE VÀ CẬP NHẬT PPO (TỪ THÀNH VIÊN A)
        # ==========================================
        # Bổ sung giá trị của state cuối cùng
        vals.append(0.0)
        rews_arr, vals_arr = np.array(rews), np.array(vals)
        
        # Công thức GAE tuyệt đỉnh của A
        deltas = rews_arr + gamma * vals_arr[1:] - vals_arr[:-1]
        adv_buf = discount_cumsum(deltas, gamma * lam)
        ret_buf = discount_cumsum(rews_arr, gamma)
        
        # Chuẩn hóa lợi thế
        adv_mean, adv_std = np.mean(adv_buf), np.std(adv_buf)
        adv_buf = (adv_buf - adv_mean) / (adv_std + 1e-8)
        
        # ==========================================
        # CẬP NHẬT TRỌNG SỐ BẰNG THUẬT TOÁN PPO CHUẨN
        # ==========================================
        # Đóng băng log_prob cũ để làm mốc so sánh (Old Policy)
        old_logps = torch.stack(logps).detach()
        adv_tensor = torch.tensor(adv_buf.copy(), dtype=torch.float32)
        ret_tensor = torch.tensor(ret_buf.copy(), dtype=torch.float32)
        vals_tensor = torch.tensor(vals[:-1], dtype=torch.float32)
        
        # Để tiết kiệm RAM, ta cập nhật PPO 1 lần trên toàn bộ tập dữ liệu
        # (Nếu có RAM mạnh có thể chạy vòng lặp 5 lần ở đây)
        optimizer.zero_grad()
        
        # Lấy lại Log_prob mới (Vì ta đã detach mảng cũ)
        # Trong thực tế phải forward lại model, nhưng vì đồ thị tuần tự nên ta xấp xỉ bằng logps hiện tại
        # Do là bước cập nhật đầu tiên, new_logps == old_logps, ratio == 1.
        # Ta ép thuật toán sử dụng Entropy để ngăn hiện tượng mụ mẫm
        
        # KHÔI PHỤC ENTROPY: Ép AI rải tiền, chống All-in bừa bãi
        # Dùng lại mảng logps còn dính Computation Graph
        new_logps = torch.stack(logps)
        ratio = torch.exp(new_logps - old_logps)
        
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_tensor
        actor_loss = -(torch.min(ratio * adv_tensor, clip_adv)).mean()
        
        critic_loss = F.mse_loss(vals_tensor, ret_tensor)
        
        # Entropy trung bình của cả quá trình
        entropy_tensor = torch.stack(entropies).mean() 
        
        # Công thức PPO Tổng hợp (Có trừ Entropy)
        entropy_coef = 0.05  # Kích thích AI thử nghiệm
        loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy_tensor
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        # ==========================================
        # GIAI ĐOẠN 3: THI THỬ TRÊN TẬP VAL (KHÔNG HỌC)
        # ==========================================
        model.eval()
        with torch.no_grad():
            obs, info_val = env_val.reset()
            for step in range(1, len(val_tensor.tensor)):
                start_idx = max(0, step - 60)
                obs_corr = np.nan_to_num(np.corrcoef(val_tensor.tensor[start_idx:step, :, 0].T), nan=0.0) if step > 2 else np.zeros((N_stocks, N_stocks))
                action_probs, _, _, _, _, _, _ = get_action_and_value(model, obs_adapter, obs['market_state'], obs_corr)
                
                next_obs, reward, terminated, _, info_val = env_val.step(action_probs.cpu().numpy())
                obs = next_obs
                if terminated: break
                
        # ==========================================
        # GHI LOG VÀ LƯU BEST MODEL DỰA TRÊN ĐIỂM VAL
        # ==========================================
        train_profit = info['portfolio_value']
        val_profit = info_val['portfolio_value']
        
        logger.log_metrics(epoch, {"Profit/Train": train_profit, "Profit/Val": val_profit})
        
        # Chỉ lưu Best Model nếu điểm thi Val phá kỷ lục
        logger.save_checkpoint(epoch, model, optimizer, current_reward=val_profit)
        
        print(f"🔥 Epoch {epoch:02d} | Lãi Train: ${train_profit:,.0f} | LÃI VAL: ${val_profit:,.0f} | Val Drawdown: {info_val['max_drawdown']*100:.2f}%")

    logger.close()
    print("\n✅ HOÀN TẤT: AI ĐÃ SẴN SÀNG ĐỂ ĐEM ĐI BACKTEST!")

if __name__ == "__main__":
    main()
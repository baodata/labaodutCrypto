# môi trường giao dịch danh mục đa tài sản theo chuẩn Gymnasium
# mô phỏng quá trình ra quyết định phân bổ tỷ trọng, trừ phí giao dịch và tính toán lợi suất ròng

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yaml

from src.data.loader import load_config, load_market_data
from src.features.pipeline import build_feature_pipeline
from src.utils.data_types import MarketDataTensor


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        dates: np.ndarray,
        tickers: List[str],
        benchmark_prices: Optional[np.ndarray] = None,
        initial_cash: float = 100_000.0,
        transaction_fee: float = 0.001,
        reward_type: str = "net_return",
        risk_penalty: float = 0.5,
        bankruptcy_threshold: float = 0.3,
    ) -> None:
        super().__init__()

        # features có dạng [T, N, K]
        # prices có dạng [T, N] (giá đóng cửa thực tế)
        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float32)
        self.dates = dates
        self.tickers = list(tickers)
        self.benchmark_prices = benchmark_prices.astype(np.float32) if benchmark_prices is not None else None

        self.T, self.N, self.K = self.features.shape
        if self.prices.shape != (self.T, self.N):
            raise ValueError(f"Kích thước prices {self.prices.shape} không khớp với features {(self.T, self.N)}")

        # Các tham số môi trường
        self.initial_cash = float(initial_cash)
        self.transaction_fee = float(transaction_fee)
        self.reward_type = reward_type
        self.risk_penalty = float(risk_penalty)
        self.bankruptcy_threshold = float(bankruptcy_threshold)

        # Không gian hành động: vector N + 1 (vị trí 0 là Tiền mặt Cash, 1..N là các cổ phiếu)
        # Giá trị hành động chưa chuẩn hóa, môi trường sẽ tự động áp dụng Softmax để tạo tỷ trọng tổng bằng 1
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.N + 1,),
            dtype=np.float32,
        )

        # Không gian quan sát: vector 1D gồm đặc trưng thị trường [N * K] ghép với tỷ trọng hiện tại [N + 1]
        obs_dim = (self.N * self.K) + (self.N + 1)
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Trạng thái nội tại của danh mục
        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.current_weights = np.zeros(self.N + 1, dtype=np.float32)
        self.current_weights[0] = 1.0  # Ban đầu 100% tiền mặt
        self.history: List[Dict[str, Any]] = []

    # chuyển đổi hành động thô của agent thành tỷ trọng danh mục hợp lệ (tổng = 1, không âm)
    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        # sử dụng softmax ổn định số học
        exp_a = np.exp(action - np.max(action))
        sum_exp = np.sum(exp_a)
        if sum_exp <= 0:
            weights = np.zeros(self.N + 1, dtype=np.float32)
            weights[0] = 1.0
            return weights
        return (exp_a / sum_exp).astype(np.float32)

    # ghép đặc trưng phiên hiện tại và tỷ trọng danh mục thành vector quan sát
    def _get_obs(self) -> np.ndarray:
        step = min(self.current_step, self.T - 1)
        # đặc trưng thị trường tại phiên hiện tại [N, K]
        market_feats = self.features[step].flatten()
        # ghép với tỷ trọng hiện tại [N + 1]
        obs = np.concatenate([market_feats, self.current_weights]).astype(np.float32)
        return obs

    # lấy ma trận đặc trưng thị trường dạng 2D [N, K] nếu mô hình (như GNN) cần cấu trúc nút
    def get_market_state(self) -> np.ndarray:
        step = min(self.current_step, self.T - 1)
        return self.features[step].copy()

    # khởi tạo lại môi trường về phiên giao dịch đầu tiên
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.current_weights = np.zeros(self.N + 1, dtype=np.float32)
        self.current_weights[0] = 1.0  # 100% tiền mặt
        self.history = []

        obs = self._get_obs()
        info = {
            "step": 0,
            "date": str(self.dates[0])[:10],
            "portfolio_value": self.portfolio_value,
            "weights": self.current_weights.copy(),
        }
        return obs, info

    # thực hiện một bước giao dịch sang phiên tiếp theo
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.current_step >= self.T - 1:
            # đã hết dữ liệu chuỗi thời gian
            obs = self._get_obs()
            return obs, 0.0, True, False, {"step": self.current_step, "portfolio_value": self.portfolio_value}

        t = self.current_step

        # 1. Chuyển hành động thành tỷ trọng mục tiêu mong muốn
        target_weights = self._action_to_weights(action)

        # 2. Tính khối lượng tái cơ cấu (Turnover) và phí giao dịch
        # chỉ tính phí trên phần cổ phiếu được mua bán (bỏ qua tiền mặt)
        turnover = float(np.sum(np.abs(target_weights[1:] - self.current_weights[1:])))
        fee = turnover * self.transaction_fee

        # 3. Lấy lợi suất thực tế của cổ phiếu từ phiên t sang t+1
        price_today = self.prices[t]
        price_next = self.prices[t + 1]
        stock_rets = (price_next / (price_today + 1e-8)) - 1.0

        # 4. Tính lợi suất gộp và lợi suất ròng sau khi trừ phí
        gross_return = float(np.sum(target_weights[1:] * stock_rets))
        net_return = gross_return - fee

        # 5. Cập nhật giá trị danh mục
        prev_value = self.portfolio_value
        self.portfolio_value = max(0.0, prev_value * (1.0 + net_return))

        # 6. Tính tỷ trọng trôi dạt (Drifted Weights) cho phiên tiếp theo
        v_cash = max(0.0, target_weights[0] - fee)
        v_stocks = np.maximum(0.0, target_weights[1:] * (1.0 + stock_rets))
        v_total = v_cash + np.sum(v_stocks)

        if v_total > 0:
            self.current_weights = np.concatenate([[v_cash], v_stocks]) / v_total
        else:
            self.current_weights = np.zeros(self.N + 1, dtype=np.float32)
            self.current_weights[0] = 1.0

        # 7. Tính phần thưởng (Reward)
        if self.reward_type == "log_return":
            reward = float(np.log(max(1.0 + net_return, 1e-8)))
        elif self.reward_type == "risk_adjusted":
            reward = float(net_return - 0.5 * self.risk_penalty * (net_return ** 2))
        else:
            reward = float(net_return)

        # 8. Tăng bước thời gian
        self.current_step += 1

        # 9. Kiểm tra điều kiện kết thúc
        is_bankrupt = self.portfolio_value <= (self.bankruptcy_threshold * self.initial_cash)
        is_finished = self.current_step >= (self.T - 1)
        terminated = bool(is_bankrupt or is_finished)
        truncated = False

        # Lợi suất benchmark SPY nếu có
        bench_ret = 0.0
        if self.benchmark_prices is not None:
            bench_ret = float((self.benchmark_prices[t + 1] / self.benchmark_prices[t]) - 1.0)

        info = {
            "step": self.current_step,
            "date": str(self.dates[self.current_step])[:10],
            "portfolio_value": self.portfolio_value,
            "net_return": net_return,
            "gross_return": gross_return,
            "fee": fee,
            "turnover": turnover,
            "weights": target_weights.copy(),
            "benchmark_return": bench_ret,
        }
        self.history.append(info)

        next_obs = self._get_obs()
        return next_obs, reward, terminated, truncated, info

    # xuất báo cáo tổng kết hiệu suất đầu tư sau khi chạy hết episode
    def get_summary(self) -> Dict[str, float]:
        if not self.history:
            return {}

        df_hist = pd.DataFrame(self.history)
        total_steps = len(df_hist)
        cum_return = (self.portfolio_value - self.initial_cash) / self.initial_cash

        # Lợi suất hàng năm (CAGR, quy ước 252 phiên/năm)
        years = total_steps / 252.0
        cagr = ((1.0 + cum_return) ** (1.0 / max(years, 0.01))) - 1.0 if cum_return > -1.0 else -1.0

        # Biến động lợi suất hàng năm
        net_rets = df_hist["net_return"].values
        daily_std = float(np.std(net_rets))
        ann_vol = daily_std * np.sqrt(252.0)

        # Tỷ lệ Sharpe (giả sử lãi suất phi rủi ro = 0)
        daily_mean = float(np.mean(net_rets))
        sharpe = (daily_mean / (daily_std + 1e-8)) * np.sqrt(252.0)

        # Mức sụt giảm tối đa (Maximum Drawdown - MDD)
        values = df_hist["portfolio_value"].values
        peaks = np.maximum.accumulate(values)
        drawdowns = (values - peaks) / (peaks + 1e-8)
        max_drawdown = float(np.min(drawdowns))

        # Tổng phí giao dịch đã trả
        total_fees = float(df_hist["fee"].sum() * self.initial_cash)
        avg_turnover = float(df_hist["turnover"].mean())

        return {
            "initial_cash": self.initial_cash,
            "final_value": float(self.portfolio_value),
            "cumulative_return": float(cum_return),
            "annualized_return_cagr": float(cagr),
            "annualized_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "avg_turnover": float(avg_turnover),
            "total_steps": total_steps,
        }

    # hiển thị thông tin tóm tắt ra màn hình
    def render(self) -> None:
        if not self.history:
            print(f"Step {self.current_step} | Vốn: ${self.portfolio_value:,.2f} | 100% Tiền mặt")
            return
        last = self.history[-1]
        print(
            f"Phiên {last['step']:>4} ({last['date']}) | "
            f"Giá trị: ${last['portfolio_value']:>10,.2f} | "
            f"Lợi suất ròng: {last['net_return']:>+6.2%} | "
            f"Turnover: {last['turnover']:>5.2%}"
        )


# hàm tiện ích tạo môi trường giao dịch trực tiếp từ tên phân vùng (train / val / test)
def make_trading_env(
    split_name: str = "train",
    config_path: str = "configs/assets.yaml",
    env_config_path: str = "configs/env.yaml",
) -> TradingEnv:
    # 1. Đọc cấu hình
    env_cfg_path = Path(env_config_path)
    env_params: Dict[str, Any] = {}
    if env_cfg_path.is_file():
        with open(env_cfg_path, "r", encoding="utf-8") as f:
            env_params = yaml.safe_load(f) or {}

    # 2. Nạp dữ liệu thô và chạy pipeline tính đặc trưng
    full_data, splits = load_market_data(config_path)
    scaled_splits, scaler = build_feature_pipeline(full_data, splits, warmup_days=30)

    if split_name not in scaled_splits:
        raise KeyError(f"Phân vùng '{split_name}' không tồn tại. Các phân vùng có sẵn: {list(scaled_splits.keys())}")

    target_split = scaled_splits[split_name]

    # 3. Lấy giá đóng cửa thực tế khớp với các ngày của tập này
    date_mask = np.isin(full_data.dates, target_split.dates)
    matched_prices = full_data.get_feature("close")[date_mask]

    # Lấy giá benchmark SPY nếu có
    matched_bench = None
    if full_data.benchmark_values is not None:
        close_idx = full_data.get_feature_index("close")
        matched_bench = full_data.benchmark_values[date_mask, close_idx]

    return TradingEnv(
        features=target_split.values,
        prices=matched_prices,
        dates=target_split.dates,
        tickers=target_split.tickers,
        benchmark_prices=matched_bench,
        initial_cash=env_params.get("initial_cash", 100_000.0),
        transaction_fee=env_params.get("transaction_fee", 0.001),
        reward_type=env_params.get("reward_type", "net_return"),
        risk_penalty=env_params.get("risk_penalty", 0.5),
        bankruptcy_threshold=env_params.get("bankruptcy_threshold", 0.3),
    )

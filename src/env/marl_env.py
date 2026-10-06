import numpy as np
import yaml
from typing import Dict, List, Tuple, Any

from src.env.trading_env import TradingEnv

class MultiAgentTradingEnv:
    """
    Multi-Agent wrapper cho TradingEnv (MARL-004).
    """

    def __init__(self, global_env: TradingEnv, agent_groups_config: dict, tickers: list[str]):
        """
        Khởi tạo môi trường Multi-Agent.
        """
        self.global_env = global_env
        self.agent_groups_config = agent_groups_config
        self.tickers = tickers
        self.agents = list(self.agent_groups_config.keys())
        self.num_agents = len(self.agents)

        # Đọc cấu hình assets.yaml để lấy thông tin phân nhóm
        with open("configs/assets.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        self.ticker_to_agent = {}
        for asset in config.get("assets", []):
            if asset["ticker"] in self.tickers:
                self.ticker_to_agent[asset["ticker"]] = asset["agent_group"]

        # Map từng agent đến danh sách các index của tickers mà nó quản lý
        self.agent_to_indices = {agent_key: [] for agent_key in self.agents}
        for i, ticker in enumerate(self.tickers):
            agent_key = self.ticker_to_agent.get(ticker)
            if agent_key and agent_key in self.agent_to_indices:
                self.agent_to_indices[agent_key].append(i)

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, Dict[str, np.ndarray]], dict]:
        """Reset môi trường."""
        global_obs, info = self.global_env.reset(seed=seed, options=options)
        return self._split_observation(global_obs), info

    def _split_observation(self, global_obs: dict) -> dict[str, dict]:
        """
        MARL-002: Chia observation của global_env cho từng agent.
        """
        split_obs = {}
        for agent_name in self.agents:
            indices = self.agent_to_indices[agent_name]
            if not indices:
                # Nếu agent không quản lý mã nào
                continue
            
            market_state = global_obs["market_state"][indices]
            portfolio_weights = global_obs["portfolio_weights"][indices]
            cash_ratio = global_obs["cash_ratio"]
            
            split_obs[agent_name] = {
                "market_state": market_state,
                "portfolio_weights": portfolio_weights,
                "cash_ratio": cash_ratio
            }
        return split_obs

    def step(self, agent_actions: dict[str, np.ndarray]) -> Tuple[Dict[str, Dict], Dict[str, float], bool, bool, Dict]:
        """
        MARL-003: Step môi trường đa tác tử.
        """
        N = len(self.tickers)
        global_weights = np.zeros(N + 1, dtype=np.float32)
        budget = 1.0 / self.num_agents if self.num_agents > 0 else 1.0

        for agent_name, action in agent_actions.items():
            indices = self.agent_to_indices[agent_name]
            if not indices:
                continue

            # action: [N_agent + 1], tổng = 1.0
            agent_weight = budget * action

            # Phân bổ tỷ trọng cho từng tài sản (cổ phiếu)
            for i, idx in enumerate(indices):
                global_weights[idx] += agent_weight[i]

            # Phân bổ tỷ trọng cho tiền mặt (phần tử cuối cùng)
            global_weights[-1] += agent_weight[-1]

        next_global_obs, reward, terminated, truncated, info = self.global_env.step(global_weights)

        next_obs = self._split_observation(next_global_obs)
        rewards = {agent: reward for agent in self.agents}

        return next_obs, rewards, terminated, truncated, info

import numpy as np
from typing import Dict, List, Tuple, Any

from src.env.trading_env import TradingEnv
from src.utils.config import AssetsConfig

class MultiAgentTradingEnv:
    """
    Multi-Agent wrapper cho TradingEnv (MARL-004).
    """

    def __init__(self, global_env: TradingEnv, agent_groups_config: dict, tickers: list[str],
                 assets_config_path: str = "configs/core/assets.yaml", reward_mode: str = "shared"):
        """
        Khởi tạo môi trường Multi-Agent.
        """
        if reward_mode not in {"shared", "sector_active"}:
            raise ValueError("MARL reward must be shared or sector_active.")
        if reward_mode == "sector_active" and global_env.dsr is not None:
            raise ValueError("Sector active credit and DSR are separate ablations.")
        self.reward_mode = reward_mode
        self.global_env = global_env
        self.agent_groups_config = agent_groups_config
        self.tickers = tickers
        if tickers != global_env.market_tensor.tickers or len(set(tickers)) != len(tickers):
            raise ValueError("MARL tickers phải khớp thứ tự ticker của global_env.")
        self.agents = list(self.agent_groups_config.keys())
        self.num_agents = len(self.agents)

        # Đọc cấu hình assets.yaml để lấy thông tin phân nhóm
        config = AssetsConfig(assets_config_path).raw_config

        self.ticker_to_agent = {}
        for asset in config.get("assets", []):
            if asset["ticker"] in self.tickers:
                self.ticker_to_agent[asset["ticker"]] = asset["agent_group"]

        # Map từng agent đến danh sách các index của tickers mà nó quản lý
        self.agent_to_indices = {agent_key: [] for agent_key in self.agents}
        for i, ticker in enumerate(self.tickers):
            agent_key = self.ticker_to_agent.get(ticker)
            if agent_key not in self.agent_to_indices:
                raise ValueError(f"Ticker {ticker} không thuộc agent group được cấu hình.")
            self.agent_to_indices[agent_key].append(i)
        self.agents = [agent for agent in self.agents if self.agent_to_indices[agent]]
        self.num_agents = len(self.agents)
        if not self.agents:
            raise ValueError("MARL cần ít nhất một agent có tài sản.")

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, Dict[str, np.ndarray]], dict]:
        """Reset môi trường."""
        global_obs, info = self.global_env.reset(seed=seed, options=options)
        return self._split_observation(global_obs), info

    def _split_observation(self, global_obs: dict) -> dict[str, dict]:
        """
        MARL-002: Chia observation của global_env cho từng agent.
        """
        return split_sector_observation(global_obs,self.agent_to_indices,self.agents)

    def step(self, agent_actions: dict[str, np.ndarray]) -> Tuple[Dict[str, Dict], Dict[str, float], bool, bool, Dict]:
        """
        MARL-003: Step môi trường đa tác tử.
        """
        N = len(self.tickers)
        if set(agent_actions) != set(self.agents):
            raise ValueError("Cần action cho đúng tất cả active agents; không tự phân bổ lại budget khi thiếu agent.")
        global_weights = np.zeros(N + 1, dtype=np.float32)
        budget = 1.0 / self.num_agents if self.num_agents > 0 else 1.0

        for agent_name, action in agent_actions.items():
            indices = self.agent_to_indices[agent_name]
            if not indices:
                continue

            action = np.asarray(action, dtype=np.float64)
            if action.shape != (len(indices) + 1,) or not np.isfinite(action).all() or np.any(action < 0) or not np.isclose(action.sum(), 1.0, atol=1e-6):
                raise ValueError(f"Action của {agent_name} phải hữu hạn, không âm, shape [N_agent+1], tổng 1.")
            agent_weight = budget * action

            # Phân bổ tỷ trọng cho từng tài sản (cổ phiếu)
            for i, idx in enumerate(indices):
                global_weights[idx] += agent_weight[i]

            # Phân bổ tỷ trọng cho tiền mặt (phần tử cuối cùng)
            global_weights[-1] += agent_weight[-1]

        next_global_obs, reward, terminated, truncated, info = self.global_env.step(global_weights)

        next_obs = self._split_observation(next_global_obs)
        rewards = {agent: reward for agent in self.agents}
        local_returns = {}
        for agent, indices in self.agent_to_indices.items():
            if agent not in self.agents:
                continue
            pnl = float(np.sum(info['asset_net_pnl'][indices]) + info['cash_pnl']/self.num_agents)
            local_returns[agent] = pnl/info['previous_value']*self.num_agents
            if self.reward_mode == 'sector_active':
                t = self.global_env.current_step
                benchmark = float(np.mean(self.global_env.close_prices[t, indices]/self.global_env.close_prices[t-1, indices]-1))
                rewards[agent] = (local_returns[agent]-benchmark)*self.global_env.config['reward_scaling']
        info['sector_net_returns'] = local_returns
        info['sector_reward_mode'] = self.reward_mode
        info['sector_credit_convention'] = 'PnL divided by previous global NAV times number of sectors; cash attribution equal'

        return next_obs, rewards, terminated, truncated, info


def split_sector_observation(global_obs, agent_to_indices, agents=None):
    agents = list(agent_to_indices) if agents is None else agents
    num_agents = len(agents)
    split_obs = {}
    for agent_name in agents:
        indices = agent_to_indices[agent_name]
        if not indices:
            # Nếu agent không quản lý mã nào
            continue

        market_state = global_obs["market_state"][indices]
        global_weights = np.asarray(global_obs["portfolio_weights"][indices], dtype=np.float64)
        # Attribute pooled cash equally; normalize within the observed group NAV.
        cash_allocation = float(global_obs["cash_ratio"][0]) / num_agents
        local_budget = float(global_weights.sum() + cash_allocation)
        portfolio_weights = global_weights / local_budget if local_budget > 1e-12 else np.zeros(len(indices))
        cash_ratio = np.asarray([cash_allocation / local_budget if local_budget > 1e-12 else 1.0], dtype=np.float32)

        split_obs[agent_name] = {
            "market_state": market_state,
            "portfolio_weights": portfolio_weights,
            "cash_ratio": cash_ratio
        }
        split_obs[agent_name]["portfolio_weights"] = portfolio_weights.astype(np.float32)
        split_obs[agent_name]["agent_budget"] = np.asarray([local_budget], dtype=np.float32)
        split_obs[agent_name]["target_budget"] = np.asarray([1.0 / num_agents], dtype=np.float32)
        split_obs[agent_name]["agent_state"] = np.asarray([local_budget, 1.0/num_agents], dtype=np.float32)
        if "market_history" in global_obs:
            split_obs[agent_name]["market_history"] = global_obs["market_history"][:, indices].copy()
        if "market_context" in global_obs:
            split_obs[agent_name]["market_context"] = global_obs["market_context"].copy()
        if "reward_state" in global_obs:
            split_obs[agent_name]["reward_state"] = global_obs["reward_state"].copy()
        if "history_mask" in global_obs:
            split_obs[agent_name]["history_mask"] = global_obs["history_mask"].copy()
        for key in ("graph_adjacency_history", "graph_edge_mask_history"):
            if key in global_obs:
                split_obs[agent_name][key] = np.take(np.take(global_obs[key], indices, axis=-2), indices, axis=-1)
    return split_obs

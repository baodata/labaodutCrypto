"""Trading environment factory shared by training and evaluation. Owner: A."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.env.trading_env import TradingEnv
from src.utils.data_types import MarketDataTensor


def create_env(
    tensor,
    open_prices: np.ndarray,
    env_config: Path,
    reward_beta: float,
    reward_scaling: float | None = None,
    history_tensor: MarketDataTensor | None = None,
    history_start_index: int = 0,
    reward_baseline: str = "none",
    lookback_window: int = 1,
    action_alpha: float = 1.0,
    graph_adjacency_sequence: np.ndarray | None = None,
    graph_edge_mask_sequence: np.ndarray | None = None,
    graph_start_index: int = 0,
    close_prices: np.ndarray | None = None,
) -> TradingEnv:
    return TradingEnv(
        market_tensor=tensor,
        open_prices=open_prices,
        close_prices=close_prices,
        config_path=str(env_config),
        reward_beta=reward_beta,
        reward_scaling=reward_scaling,
        reward_baseline=reward_baseline,
        history_tensor=history_tensor,
        history_start_index=history_start_index,
        lookback_window=lookback_window,
        action_alpha=action_alpha,
        graph_adjacency_sequence=graph_adjacency_sequence,
        graph_edge_mask_sequence=graph_edge_mask_sequence,
        graph_start_index=graph_start_index,
        quant_context=getattr(history_tensor if history_tensor is not None else tensor, "quant_context", None),
    )

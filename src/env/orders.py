"""Explicit execution instructions for deterministic trading strategies."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PortfolioOrder:
    """Hold share quantities, or rebalance with an optional per-order alpha.

    PPO continues to send weight vectors using the environment's default alpha.
    Baselines can place a full initial allocation without changing that default.
    """

    target_weights: np.ndarray | None = None
    action_alpha: float | None = None

    def __post_init__(self) -> None:
        if self.target_weights is None and self.action_alpha is not None:
            raise ValueError("Lệnh hold không được có action_alpha.")
        if self.action_alpha is not None and (
            not np.isfinite(self.action_alpha) or not 0.0 < self.action_alpha <= 1.0
        ):
            raise ValueError("action_alpha của lệnh phải hữu hạn và thuộc (0, 1].")

    @classmethod
    def hold(cls) -> "PortfolioOrder":
        """Keep shares unchanged; accrue returns and cash interest without trading."""
        return cls()

    @classmethod
    def rebalance(
        cls, target_weights: np.ndarray, *, action_alpha: float | None = None
    ) -> "PortfolioOrder":
        return cls(target_weights=target_weights, action_alpha=action_alpha)

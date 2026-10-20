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
    final_execution: bool = False
    hold_mask: np.ndarray | None = None
    notional_caps: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.final_execution:
            target = np.asarray(self.target_weights, dtype=np.float64)
            if (self.action_alpha is not None or target.ndim != 1 or target.size < 2
                    or not np.isfinite(target).all() or np.any(target < 0)
                    or abs(target.sum()-1.) > 1e-8
                    or np.shape(self.hold_mask) != (target.size-1,)
                    or np.shape(self.notional_caps) != (target.size-1,)
                    or np.asarray(self.hold_mask).dtype != np.bool_
                    or not np.isfinite(self.notional_caps).all() or np.any(np.asarray(self.notional_caps) < 0)):
                raise ValueError('Final execution requires normalized target, bool hold mask, finite dollar caps and no alpha.')
        elif self.hold_mask is not None or self.notional_caps is not None:
            raise ValueError('Mechanical hold/caps require a final execution order.')
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

    @classmethod
    def execution(cls, weights, *, hold_mask, notional_caps):
        """Already planned at Close; skip alpha/controller, preserve Open fill limits."""
        return cls(target_weights=np.asarray(weights, dtype=np.float64).copy(),
            final_execution=True, hold_mask=np.asarray(hold_mask, dtype=bool).copy(),
            notional_caps=np.asarray(notional_caps, dtype=np.float64).copy())

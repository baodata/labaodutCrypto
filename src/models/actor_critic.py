"""Actor-Critic models với actions constrained to the portfolio simplex.

Có 3 lựa chọn encoder:
  - SimplexActorCritic  : MLP thuần trên vector quan sát đã flatten
  - LSTMActorCritic     : LSTM encoder, xử lý [L, N*F] tuần tự theo thời gian
  - GRUActorCritic      : GRU encoder, ít tham số hơn LSTM, hội tụ nhanh hơn

Với encoder tuần tự, vector quan sát chỉ là định dạng lưu trong PPO buffer.
_RecurrentActorCritic dựng lại [B,L,N,F] rồi chạy RNN dọc theo trục thời gian.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet


# ---------------------------------------------------------------------------
# 1. MLP Baseline (Không thay đổi — giữ tương thích backward)
# ---------------------------------------------------------------------------

class SimplexActorCritic(nn.Module):
    """PPO policy for raw market features and portfolio state.

    The actor uses a Dirichlet distribution, so every sampled action is
    non-negative and sums to one (including the cash allocation).
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        if observation_dim <= 0 or action_dim <= 1 or hidden_dim <= 0:
            raise ValueError("Dimensions must be positive and action_dim must include assets and cash.")

        self.trunk = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def _forward_distribution(self, obs: torch.Tensor) -> tuple[Dirichlet, torch.Tensor]:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        hidden = self.trunk(obs)
        # Keep concentrations >= 1 to avoid singular density at the simplex
        # boundary and zero-valued samples that make PPO log-probabilities infinite.
        concentration = (F.softplus(self.actor_head(hidden)) + 1.0).clamp(max=100.0)
        return Dirichlet(concentration), self.critic_head(hidden).squeeze(-1)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return action, log probability, and value for one observation or a batch."""
        single_observation = obs.ndim == 1
        distribution, value = self._forward_distribution(obs)
        action = distribution.mean if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action)
        if single_observation:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            value = value.squeeze(0)
        return action, log_prob, value

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPOUpdater interface: log probability, entropy, and state value."""
        distribution, value = self._forward_distribution(obs)
        return distribution.log_prob(actions), distribution.entropy(), value


# ---------------------------------------------------------------------------
# 2. Recurrent Base (LSTM / GRU)
# ---------------------------------------------------------------------------

class _RecurrentActorCritic(nn.Module):
    """Base class cho LSTM/GRU Actor-Critic.

    Kiến trúc 3 tầng:
        1. Recurrent Encoder: [L, N*F] → hidden state h_t  (nén thứ tự thời gian)
        2. FC Layer: concat(h_t, portfolio_state) → shared_hidden
        3. Actor/Critic Heads: shared_hidden → Dirichlet α / V(s)

    So với SimplexActorCritic (flatten [L,N,F] → 3026 dim → MLP):
        - MLP không phân biệt ngày t-1 và ngày t-20: cả hai có cùng "quyền" ảnh hưởng.
        - LSTM/GRU đọc dữ liệu theo đúng thứ tự thời gian, h_t phản ánh
          trạng thái "bộ nhớ" cập nhật đến ngày mới nhất.
        - Số tham số ít hơn đáng kể: input_size = N*F (thay vì L*N*F).

    Args:
        n_assets   : Số mã chứng khoán N.
        n_features : Số feature F trên mỗi mã.
        lookback   : Số ngày lịch sử L (chỉ dùng để validate shape, không hardcode).
        action_dim : N + 1 (bao gồm tiền mặt).
        hidden_dim : Chiều hidden state RNN và FC layer.
        rnn_layers : Số lớp RNN chồng nhau.
        rnn_cell   : 'lstm' hoặc 'gru'.
        dropout    : Dropout giữa các RNN layers (chỉ áp dụng khi rnn_layers > 1).
    """

    def __init__(
        self,
        n_assets: int,
        n_features: int,
        lookback: int,
        action_dim: int,
        hidden_dim: int = 128,
        rnn_layers: int = 1,
        rnn_cell: str = "lstm",
        dropout: float = 0.0,
    ):
        super().__init__()
        if n_assets <= 0 or n_features <= 0 or lookback <= 0:
            raise ValueError("n_assets, n_features, và lookback phải lớn hơn 0.")
        if action_dim <= 1:
            raise ValueError("action_dim phải lớn hơn 1 (bao gồm tiền mặt).")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim phải lớn hơn 0.")

        self.n_assets = n_assets
        self.n_features = n_features
        self.lookback = lookback
        self.hidden_dim = hidden_dim
        self.rnn_cell = rnn_cell.lower()

        # Input mỗi timestep: features của tất cả N tài sản tại ngày t → [N*F]
        rnn_input_dim = n_assets * n_features

        # Lớp Recurrent Encoder
        rnn_dropout = dropout if rnn_layers > 1 else 0.0
        if self.rnn_cell == "lstm":
            self.rnn = nn.LSTM(
                input_size=rnn_input_dim,
                hidden_size=hidden_dim,
                num_layers=rnn_layers,
                batch_first=True,          # input shape: [B, L, input_size]
                dropout=rnn_dropout,
            )
        elif self.rnn_cell == "gru":
            self.rnn = nn.GRU(
                input_size=rnn_input_dim,
                hidden_size=hidden_dim,
                num_layers=rnn_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        else:
            raise ValueError(f"rnn_cell phải là 'lstm' hoặc 'gru', nhận: {rnn_cell!r}")

        # Portfolio state: [N weights + 1 cash_ratio]
        portfolio_state_dim = n_assets + 1

        # Shared FC sau khi ghép h_t + portfolio_state
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim + portfolio_state_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unflatten_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Tách obs phẳng thành market_history [B, L, N, F] và portfolio_state [B, N+1]."""
        single = obs.ndim == 1
        if single:
            obs = obs.unsqueeze(0)

        B = obs.shape[0]
        hist_dim = self.lookback * self.n_assets * self.n_features
        port_dim = self.n_assets + 1
        expected_dim = hist_dim + port_dim
        if obs.shape[1] != expected_dim:
            raise ValueError(f"Expected obs dim {expected_dim}, got {obs.shape[1]}")

        market_history_flat = obs[:, :hist_dim]
        portfolio_state = obs[:, hist_dim:]

        market_history = market_history_flat.view(B, self.lookback, self.n_assets, self.n_features)

        if single:
            market_history = market_history.squeeze(0)
            portfolio_state = portfolio_state.squeeze(0)

        return market_history, portfolio_state

    def _encode(
        self,
        market_history: torch.Tensor,
        portfolio_state: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        """Encode thành shared hidden vector.

        Args:
            market_history  : [B, L, N, F] hoặc [L, N, F] (single obs)
            portfolio_state : [B, N+1]     hoặc [N+1]

        Returns:
            (shared [B, hidden_dim], was_single_obs)
        """
        single = market_history.ndim == 3
        if single:
            market_history = market_history.unsqueeze(0)    # [1, L, N, F]
            portfolio_state = portfolio_state.unsqueeze(0)  # [1, N+1]

        B, L, N, F = market_history.shape
        # Flatten không gian asset-feature: [B, L, N, F] → [B, L, N*F]
        rnn_input = market_history.reshape(B, L, N * F)

        # Forward qua RNN — lấy hidden state tại bước cuối (t = L-1)
        _, hidden = self.rnn(rnn_input)
        if self.rnn_cell == "lstm":
            h_t = hidden[0][-1]  # hidden[0]: h_n shape [num_layers, B, H] → lấy layer cuối
        else:
            h_t = hidden[-1]     # shape [num_layers, B, H] → lấy layer cuối

        # Concat h_t + portfolio_state → [B, H + N+1]
        combined = torch.cat([h_t, portfolio_state], dim=-1)
        shared = self.shared_fc(combined)  # [B, hidden_dim]
        return shared, single

    def _forward_distribution(
        self,
        obs: torch.Tensor,
    ) -> tuple[Dirichlet, torch.Tensor]:
        market_history, portfolio_state = self._unflatten_obs(obs)
        shared, _ = self._encode(market_history, portfolio_state)
        concentration = (F.softplus(self.actor_head(shared)) + 1.0).clamp(max=100.0)
        return Dirichlet(concentration), self.critic_head(shared).squeeze(-1)

    # ------------------------------------------------------------------
    # Public interface — khớp hoàn toàn với SimplexActorCritic
    # ------------------------------------------------------------------

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (action, log_prob, value)."""
        single = obs.ndim == 1
        distribution, value = self._forward_distribution(obs)
        action = distribution.mean if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action)
        if single:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            value = value.squeeze(0)
        return action, log_prob, value

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPO interface: (log_prob, entropy, value)."""
        distribution, value = self._forward_distribution(obs)
        return distribution.log_prob(actions), distribution.entropy(), value


# ---------------------------------------------------------------------------
# 3. Concrete subclasses: LSTM và GRU
# ---------------------------------------------------------------------------

class LSTMActorCritic(_RecurrentActorCritic):
    """LSTM-based Actor-Critic cho portfolio allocation.

    LSTM phù hợp khi cần nhớ các pattern dài hạn (volatility regimes,
    multi-week trends). Phù hợp nhất với lookback >= 20 ngày.
    """

    def __init__(
        self,
        n_assets: int,
        n_features: int,
        lookback: int,
        action_dim: int,
        hidden_dim: int = 128,
        rnn_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__(
            n_assets=n_assets,
            n_features=n_features,
            lookback=lookback,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            rnn_layers=rnn_layers,
            rnn_cell="lstm",
            dropout=dropout,
        )


class GRUActorCritic(_RecurrentActorCritic):
    """GRU-based Actor-Critic cho portfolio allocation.

    GRU ít tham số hơn LSTM (~25%), hội tụ nhanh hơn với tập Train nhỏ
    (T=1763 ngày). Khuyến nghị thử GRU trước khi chuyển sang LSTM.
    """

    def __init__(
        self,
        n_assets: int,
        n_features: int,
        lookback: int,
        action_dim: int,
        hidden_dim: int = 128,
        rnn_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__(
            n_assets=n_assets,
            n_features=n_features,
            lookback=lookback,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            rnn_layers=rnn_layers,
            rnn_cell="gru",
            dropout=dropout,
        )


def build_actor_critic(
    *,
    encoder: str,
    n_assets: int,
    n_features: int,
    lookback: int,
    action_dim: int,
    hidden_dim: int = 128,
    rnn_layers: int = 1,
    rnn_dropout: float = 0.0,
) -> nn.Module:
    """Create an actor-critic from serializable architecture metadata."""
    encoder = encoder.lower()
    if encoder == "mlp":
        observation_dim = lookback * n_assets * n_features + n_assets + 1
        return SimplexActorCritic(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )
    if encoder == "lstm":
        return LSTMActorCritic(
            n_assets=n_assets,
            n_features=n_features,
            lookback=lookback,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            rnn_layers=rnn_layers,
            dropout=rnn_dropout,
        )
    if encoder == "gru":
        return GRUActorCritic(
            n_assets=n_assets,
            n_features=n_features,
            lookback=lookback,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            rnn_layers=rnn_layers,
            dropout=rnn_dropout,
        )
    raise ValueError(f"encoder phải là 'mlp', 'lstm' hoặc 'gru', nhận: {encoder!r}")

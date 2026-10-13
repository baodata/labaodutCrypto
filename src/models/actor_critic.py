"""Actor-Critic models với actions constrained to the portfolio simplex.

Có 3 lựa chọn encoder:
  - SimplexActorCritic  : MLP thuần trên vector quan sát đã flatten
  - LSTMActorCritic     : LSTM encoder, xử lý [L, N*F] tuần tự theo thời gian
  - GRUActorCritic      : GRU encoder, ít tham số hơn LSTM, hội tụ nhanh hơn
  - DynamicGraphActorCritic: GAT/GCN theo ngày, sau đó GRU theo thời gian

Với encoder tuần tự, vector quan sát chỉ là định dạng lưu trong PPO buffer.
_RecurrentActorCritic dựng lại [B,L,N,F] rồi chạy RNN dọc theo trục thời gian.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch.nn.utils.rnn import pack_padded_sequence


# ---------------------------------------------------------------------------
# 1. MLP Baseline (Không thay đổi — giữ tương thích backward)
# ---------------------------------------------------------------------------

class SimplexActorCritic(nn.Module):
    """PPO policy for raw market features and portfolio state.

    The actor uses a Dirichlet distribution, so every sampled action is
    non-negative and sums to one (including the cash allocation).
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        dirichlet_kappa: float | None = None,
    ):
        super().__init__()
        if observation_dim <= 0 or action_dim <= 1 or hidden_dim <= 0:
            raise ValueError("Dimensions must be positive and action_dim must include assets and cash.")

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.dirichlet_kappa = float(dirichlet_kappa) if dirichlet_kappa is not None else None

        self.actor_trunk = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.critic_trunk = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

        if self.dirichlet_kappa is not None:
            if self.dirichlet_kappa <= action_dim:
                raise ValueError("dirichlet_kappa phải lớn hơn action_dim.")
            nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
            nn.init.zeros_(self.actor_head.bias)

    def _forward_distribution(self, obs: torch.Tensor) -> tuple[Dirichlet, torch.Tensor]:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        actor_hidden = self.actor_trunk(obs)
        critic_hidden = self.critic_trunk(obs)
        if self.dirichlet_kappa is not None:
            K = self.action_dim
            mean = torch.softmax(self.actor_head(actor_hidden), dim=-1)
            concentration = 1.0 + (self.dirichlet_kappa - K) * mean
        else:
            # Keep concentrations >= 1 to avoid singular density at the simplex
            # boundary and zero-valued samples that make PPO log-probabilities infinite.
            concentration = (F.softplus(self.actor_head(actor_hidden)) + 1.0).clamp(max=100.0)
        return Dirichlet(concentration), self.critic_head(critic_hidden).squeeze(-1)

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
        2. Shared recurrent representation: concat(h_t, portfolio_state)
        3. Separate Actor/Critic FC trunks and heads

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
        dirichlet_kappa: float | None = None,
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
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.rnn_cell = rnn_cell.lower()
        self.dirichlet_kappa = float(dirichlet_kappa) if dirichlet_kappa is not None else None

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

        # Keep the temporal representation shared, but separate actor/critic
        # trunks so the value objective cannot directly overwrite policy layers.
        self.actor_fc = nn.Sequential(
            nn.Linear(hidden_dim + portfolio_state_dim, hidden_dim),
            nn.Tanh(),
        )
        self.critic_fc = nn.Sequential(
            nn.Linear(hidden_dim + portfolio_state_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

        if self.dirichlet_kappa is not None:
            if self.dirichlet_kappa <= action_dim:
                raise ValueError("dirichlet_kappa phải lớn hơn action_dim.")
            nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
            nn.init.zeros_(self.actor_head.bias)

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
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Encode thành shared hidden vector.

        Args:
            market_history  : [B, L, N, F] hoặc [L, N, F] (single obs)
            portfolio_state : [B, N+1]     hoặc [N+1]

        Returns:
            (actor_hidden [B, H], critic_hidden [B, H], was_single_obs)
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
        actor_hidden = self.actor_fc(combined)
        critic_hidden = self.critic_fc(combined)
        return actor_hidden, critic_hidden, single

    def _forward_distribution(
        self,
        obs: torch.Tensor,
    ) -> tuple[Dirichlet, torch.Tensor]:
        market_history, portfolio_state = self._unflatten_obs(obs)
        actor_hidden, critic_hidden, _ = self._encode(market_history, portfolio_state)
        if self.dirichlet_kappa is not None:
            K = self.action_dim
            mean = torch.softmax(self.actor_head(actor_hidden), dim=-1)
            concentration = 1.0 + (self.dirichlet_kappa - K) * mean
        else:
            concentration = (F.softplus(self.actor_head(actor_hidden)) + 1.0).clamp(max=100.0)
        return Dirichlet(concentration), self.critic_head(critic_hidden).squeeze(-1)

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
        dirichlet_kappa: float | None = None,
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
            dirichlet_kappa=dirichlet_kappa,
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
        dirichlet_kappa: float | None = None,
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
            dirichlet_kappa=dirichlet_kappa,
        )


class DynamicGraphActorCritic(nn.Module):
    """Spatio-temporal PPO actor-critic: daily GNN embeddings followed by a GRU."""

    def __init__(
        self,
        n_assets: int,
        n_features: int,
        lookback: int,
        action_dim: int,
        hidden_dim: int = 128,
        gnn_architecture: str = "gat",
        gnn_hidden_channels: int = 64,
        gnn_out_channels: int = 32,
        gnn_heads: int = 4,
        temporal_encoder: str = "gru",
        temporal_layers: int = 1,
        temporal_dropout: float = 0.0,
        dirichlet_kappa: float | None = None,
    ):
        super().__init__()
        if n_assets <= 0 or n_features <= 0 or lookback <= 0:
            raise ValueError("n_assets, n_features và lookback phải lớn hơn 0.")
        if action_dim != n_assets + 1:
            raise ValueError("action_dim phải gồm mỗi asset và một chân tiền mặt.")
        self.n_assets = n_assets
        self.n_features = n_features
        self.lookback = lookback
        self.action_dim = action_dim
        architecture = gnn_architecture.lower()
        self.gnn_architecture = architecture
        self.graph_relation_names = (
            "correlation_positive",
            "correlation_negative",
            "sector",
        )
        self.graph_relation_count = len(self.graph_relation_names)
        self.dirichlet_kappa = float(dirichlet_kappa) if dirichlet_kappa is not None else None
        if temporal_encoder.lower() != "gru":
            raise ValueError("DynamicGraphActorCritic hiện hỗ trợ temporal_encoder='gru'.")
        if temporal_layers <= 0 or not 0.0 <= temporal_dropout < 1.0:
            raise ValueError("Cấu hình GRU thời gian không hợp lệ.")
        if temporal_dropout != 0.0:
            raise ValueError(
                "PPO GNN yêu cầu temporal_dropout=0 để log-prob rollout và update nhất quán."
            )
        self.temporal_encoder_name = "gru"

        # PPO re-evaluates stored actions on later minibatches. Dropout inside
        # the policy encoder would make log probabilities inconsistent between
        # rollout collection and update, so the policy GNN is deterministic.
        if architecture == "gcn":
            from src.models.gnn.gcn_encoder import GCNEncoder

            self.graph_encoders = nn.ModuleList(
                [
                    GCNEncoder(
                        in_channels=n_features,
                        hidden_channels=gnn_hidden_channels,
                        out_channels=gnn_out_channels,
                        dropout=0.0,
                    )
                    for _ in self.graph_relation_names
                ]
            )
        elif architecture == "gat":
            from src.models.gnn.gat_encoder import GATEncoder

            self.graph_encoders = nn.ModuleList(
                [
                    GATEncoder(
                        in_channels=n_features,
                        hidden_channels=gnn_hidden_channels,
                        out_channels=gnn_out_channels,
                        heads=gnn_heads,
                        dropout=0.0,
                    )
                    for _ in self.graph_relation_names
                ]
            )
        else:
            raise ValueError("gnn_architecture phải là 'gcn' hoặc 'gat'.")
        self.relation_fusion = nn.Sequential(
            nn.Linear(gnn_out_channels * self.graph_relation_count, gnn_out_channels),
            nn.ELU(),
            nn.LayerNorm(gnn_out_channels),
        )

        self.temporal_encoder = nn.GRU(
            input_size=gnn_out_channels,
            hidden_size=hidden_dim,
            num_layers=temporal_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.asset_pool_score = nn.Linear(hidden_dim, 1)
        self.critic_pool_score = nn.Linear(hidden_dim, 1)
        global_input_dim = hidden_dim + n_assets + 1
        self.actor_global_trunk = nn.Sequential(
            nn.Linear(global_input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.critic_trunk = nn.Sequential(
            nn.Linear(global_input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.asset_actor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.cash_actor = nn.Linear(hidden_dim, 1)
        self.critic_head = nn.Linear(hidden_dim, 1)

        if self.dirichlet_kappa is not None:
            if self.dirichlet_kappa <= action_dim:
                raise ValueError("dirichlet_kappa phải lớn hơn action_dim.")
            nn.init.orthogonal_(self.asset_actor[-1].weight, gain=0.01)
            nn.init.zeros_(self.asset_actor[-1].bias)
            nn.init.orthogonal_(self.cash_actor.weight, gain=0.01)
            nn.init.zeros_(self.cash_actor.bias)

    def _unflatten(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        single = obs.ndim == 1
        if single:
            obs = obs.unsqueeze(0)
        batch_size = obs.shape[0]
        history_dim = self.lookback * self.n_assets * self.n_features
        expected_dim = history_dim + self.n_assets + 1
        if obs.ndim != 2 or obs.shape[1] != expected_dim:
            raise ValueError(f"Expected obs shape [B, {expected_dim}], got {tuple(obs.shape)}")
        history = obs[:, :history_dim].reshape(
            batch_size, self.lookback, self.n_assets, self.n_features
        )
        # Environment left-pads early histories with zeros. The mask lets the
        # GRU skip those artificial frames instead of treating them as data.
        valid_history = history.abs().sum(dim=(2, 3)) > 1e-12
        portfolio_state = obs[:, history_dim:]
        return history, valid_history, portfolio_state, single

    def _graph_batch(
        self,
        graph: dict[str, torch.Tensor] | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if graph is None or "adjacency_history" not in graph or "edge_mask_history" not in graph:
            raise ValueError(
                "DynamicGraphActorCritic cần adjacency_history và edge_mask_history."
            )
        adjacency = torch.as_tensor(graph["adjacency_history"], device=device, dtype=dtype)
        edge_mask = torch.as_tensor(graph["edge_mask_history"], device=device)
        if adjacency.ndim == 3:
            # Legacy single-observation format [L, N, N].
            adjacency = adjacency.unsqueeze(0).unsqueeze(2)
            edge_mask = edge_mask.unsqueeze(0).unsqueeze(2)
        elif adjacency.ndim == 4:
            if (
                adjacency.shape[0] == batch_size
                and adjacency.shape[1] == self.lookback
                and adjacency.shape[2:] == (self.n_assets, self.n_assets)
            ):
                # Legacy batched format [B, L, N, N].
                adjacency = adjacency.unsqueeze(2)
                edge_mask = edge_mask.unsqueeze(2)
            else:
                # Relation-aware single-observation format [L, R, N, N].
                adjacency = adjacency.unsqueeze(0)
                edge_mask = edge_mask.unsqueeze(0)
        expected_shape = (
            batch_size,
            self.lookback,
            adjacency.shape[2] if adjacency.ndim == 5 else -1,
            self.n_assets,
            self.n_assets,
        )
        if (
            adjacency.ndim != 5
            or tuple(adjacency.shape) != expected_shape
            or tuple(edge_mask.shape) != expected_shape
        ):
            raise ValueError(
                "Graph batch phải có dạng [B, L, R, N, N]; nhận "
                f"{tuple(adjacency.shape)} và {tuple(edge_mask.shape)}."
            )
        relation_count = adjacency.shape[2]
        if relation_count == 1 and self.graph_relation_count == 3:
            # Backward compatibility for older signed, merged graphs: split
            # correlation sign into separate non-negative channels. Relation
            # overlap cannot be recovered from legacy data, so sector is empty.
            signed_weights = adjacency[:, :, 0]
            zero_relation = torch.zeros_like(signed_weights)
            adjacency = torch.stack(
                (
                    signed_weights.clamp_min(0.0),
                    (-signed_weights).clamp_min(0.0),
                    zero_relation,
                ),
                dim=2,
            )
            edge_mask = torch.stack(
                (
                    (edge_mask[:, :, 0] > 0.5) & (signed_weights >= 0.0),
                    (edge_mask[:, :, 0] > 0.5) & (signed_weights < 0.0),
                    torch.zeros_like(edge_mask[:, :, 0], dtype=torch.bool),
                ),
                dim=2,
            )
            relation_count = self.graph_relation_count
        if relation_count != self.graph_relation_count:
            raise ValueError(
                f"Expected {self.graph_relation_count} graph relations, received {relation_count}."
            )

        adjacency = adjacency.reshape(
            batch_size * self.lookback,
            self.graph_relation_count,
            self.n_assets,
            self.n_assets,
        )
        edge_mask = edge_mask.reshape(
            batch_size * self.lookback,
            self.graph_relation_count,
            self.n_assets,
            self.n_assets,
        )

        relation_edge_indices: list[torch.Tensor] = []
        relation_edge_weights: list[torch.Tensor] = []
        for relation_index in range(self.graph_relation_count):
            relation_mask = edge_mask[:, relation_index] > 0.5
            edge_positions = torch.nonzero(relation_mask, as_tuple=False)
            if edge_positions.numel() == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                edge_weight = torch.empty((0,), dtype=dtype, device=device)
            else:
                graph_ids, sources, targets = edge_positions.unbind(dim=1)
                offsets = graph_ids * self.n_assets
                edge_index = torch.stack((sources + offsets, targets + offsets), dim=0)
                edge_weight = adjacency[
                    graph_ids, relation_index, sources, targets
                ]
                if torch.any(edge_weight < 0.0):
                    raise ValueError(
                        "Relation weights must be non-negative magnitudes; encode sign in relation channel."
                    )
            relation_edge_indices.append(edge_index)
            relation_edge_weights.append(edge_weight)
        return relation_edge_indices, relation_edge_weights

    def _forward_distribution(
        self,
        obs: torch.Tensor,
        graph: dict[str, torch.Tensor] | None,
    ) -> tuple[Dirichlet, torch.Tensor, bool]:
        history, valid_history, portfolio_state, single = self._unflatten(obs)
        batch_size = history.shape[0]
        relation_edge_indices, relation_edge_weights = self._graph_batch(
            graph, batch_size, history.device, history.dtype
        )
        node_features = history.reshape(
            batch_size * self.lookback * self.n_assets, self.n_features
        )
        relation_embeddings = [
            encoder(node_features, edge_index, edge_weight)
            for encoder, edge_index, edge_weight in zip(
                self.graph_encoders,
                relation_edge_indices,
                relation_edge_weights,
            )
        ]
        node_embeddings = self.relation_fusion(
            torch.cat(relation_embeddings, dim=-1)
        )
        node_embeddings = node_embeddings.reshape(
            batch_size, self.lookback, self.n_assets, -1
        )
        node_embeddings = node_embeddings * valid_history[:, :, None, None].to(
            node_embeddings.dtype
        )

        # Move the left-padded valid sequence to the front, then use packed GRU
        # sequences so the recurrent state only advances over real observations.
        lengths = valid_history.sum(dim=1).clamp(min=1)
        time_offsets = (self.lookback - lengths).unsqueeze(1)
        compact_indices = (
            time_offsets
            + torch.arange(self.lookback, device=history.device).unsqueeze(0)
        ).clamp(max=self.lookback - 1)
        gather_indices = compact_indices[:, :, None, None].expand(
            -1, -1, self.n_assets, node_embeddings.shape[-1]
        )
        compact_embeddings = torch.gather(node_embeddings, dim=1, index=gather_indices)
        asset_sequences = compact_embeddings.permute(0, 2, 1, 3).reshape(
            batch_size * self.n_assets, self.lookback, -1
        )
        packed_sequences = pack_padded_sequence(
            asset_sequences,
            lengths.repeat_interleave(self.n_assets).detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, final_hidden = self.temporal_encoder(packed_sequences)
        asset_embeddings = final_hidden[-1].reshape(batch_size, self.n_assets, -1)

        # Learned attention retains asset-specific information while forming
        # the shared market context used by the cash actor and critic.
        pool_weights = torch.softmax(
            self.asset_pool_score(asset_embeddings).squeeze(-1), dim=1
        )
        actor_pooled = torch.sum(asset_embeddings * pool_weights.unsqueeze(-1), dim=1)
        critic_pool_weights = torch.softmax(
            self.critic_pool_score(asset_embeddings).squeeze(-1), dim=1
        )
        critic_pooled = torch.sum(
            asset_embeddings * critic_pool_weights.unsqueeze(-1), dim=1
        )
        actor_input = torch.cat((actor_pooled, portfolio_state), dim=-1)
        critic_input = torch.cat((critic_pooled, portfolio_state), dim=-1)
        actor_global = self.actor_global_trunk(actor_input)
        critic_hidden = self.critic_trunk(critic_input)
        asset_context = actor_global.unsqueeze(1).expand(-1, self.n_assets, -1)
        asset_inputs = torch.cat(
            (
                asset_embeddings,
                asset_context,
                portfolio_state[:, : self.n_assets].unsqueeze(-1),
            ),
            dim=-1,
        )
        if self.dirichlet_kappa is not None:
            K = self.action_dim
            asset_logits = self.asset_actor(asset_inputs).squeeze(-1)
            cash_logit = self.cash_actor(actor_global)
            all_logits = torch.cat((asset_logits, cash_logit), dim=-1)
            mean = torch.softmax(all_logits, dim=-1)
            concentrations = 1.0 + (self.dirichlet_kappa - K) * mean
        else:
            asset_concentrations = F.softplus(self.asset_actor(asset_inputs).squeeze(-1)) + 1.0
            cash_concentration = F.softplus(self.cash_actor(actor_global)) + 1.0
            concentrations = torch.cat((asset_concentrations, cash_concentration), dim=-1)
            concentrations = concentrations.clamp(max=100.0)
        distribution = Dirichlet(concentrations)
        value = self.critic_head(critic_hidden).squeeze(-1)
        return distribution, value, single

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        graph: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value, single = self._forward_distribution(obs, graph)
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
        graph: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value, _ = self._forward_distribution(obs, graph)
        return distribution.log_prob(actions), distribution.entropy(), value


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
    gnn_architecture: str = "gat",
    gnn_hidden_channels: int = 64,
    gnn_out_channels: int = 32,
    gnn_heads: int = 4,
    temporal_encoder: str = "gru",
    dirichlet_kappa: float | None = None,
) -> nn.Module:
    """Create an actor-critic from serializable architecture metadata."""
    encoder = encoder.lower()
    if encoder == "mlp":
        observation_dim = lookback * n_assets * n_features + n_assets + 1
        return SimplexActorCritic(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            dirichlet_kappa=dirichlet_kappa,
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
            dirichlet_kappa=dirichlet_kappa,
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
            dirichlet_kappa=dirichlet_kappa,
        )
    if encoder == "gnn":
        return DynamicGraphActorCritic(
            n_assets=n_assets,
            n_features=n_features,
            lookback=lookback,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            gnn_architecture=gnn_architecture,
            gnn_hidden_channels=gnn_hidden_channels,
            gnn_out_channels=gnn_out_channels,
            gnn_heads=gnn_heads,
            temporal_encoder=temporal_encoder,
            temporal_layers=rnn_layers,
            temporal_dropout=rnn_dropout,
            dirichlet_kappa=dirichlet_kappa,
        )
    raise ValueError(f"encoder phải là 'mlp', 'lstm', 'gru' hoặc 'gnn', nhận: {encoder!r}")

"""Actor-Critic models với actions constrained to the portfolio simplex.

Có 3 lựa chọn encoder:
  - SimplexActorCritic  : MLP thuần trên vector quan sát đã flatten
  - LSTMActorCritic     : LSTM encoder, xử lý [L, N*F] tuần tự theo thời gian
  - GRUActorCritic      : GRU encoder, ít tham số hơn LSTM, hội tụ nhanh hơn

Với encoder tuần tự, vector quan sát chỉ là định dạng lưu trong PPO buffer.
_RecurrentActorCritic dựng lại [B,L,N,F] rồi chạy RNN dọc theo trục thời gian.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet

from src.models.gnn.gat_encoder import GATEncoder
from src.models.gnn.gcn_encoder import GCNEncoder
from src.graph.multi_relation_graph import MultiRelationGraphBuilder


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


# ---------------------------------------------------------------------------
# 3. Graph Neural Network Actor-Critic (GAT & GCN) - Ticket GNN-004
# ---------------------------------------------------------------------------

class GNNActorCritic(nn.Module):
    """GNN-based Actor-Critic (GAT & GCN) tích hợp Dynamic Financial Graph (GNN-004).

    Kiến trúc:
        1. Graph Construction: Từ market_history [B, L, N, F], tính ma trận tương quan
           trượt (rolling Pearson) trên feature return (F_0) của L ngày gần nhất.
           Tạo đồ thị đa quan hệ (MultiRelationGraphBuilder) kết hợp cạnh tương quan (k-NN)
           và cạnh cùng ngành (same-sector).
        2. GNN Encoding: Đưa node features tại timestep hiện tại [B*N, F] cùng cấu trúc
           cạnh ghép batch vào GATv2/GCN encoder → node embeddings [B, N, D].
        3. Global Pooling: Lấy trung bình toàn bộ node embeddings → market_state [B, D].
        4. Simplex Dirichlet Policy:
           - Stock Actor Head: [B, N, D + 1] (kèm trọng số hiện tại của cổ phiếu) → N logits.
           - Cash Head: [B, D + 1] (kèm tỷ lệ tiền mặt hiện tại) → 1 logit tiền mặt.
           - Ghép thành N + 1 logits → softplus + 1.0 → Dirichlet distribution.
        5. Critic Head: [B, D + (N + 1)] → state value V(s).
    """

    def __init__(
        self,
        gnn_type: str,
        n_assets: int,
        n_features: int,
        lookback: int,
        action_dim: int,
        hidden_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.0,
        tickers: list[str] | None = None,
        knn_k: int = 5,
        corr_threshold: float = 0.5,
        config_path: str = "configs/assets.yaml",
    ):
        super().__init__()
        gnn_type = gnn_type.lower()
        if gnn_type not in {"gat", "gcn"}:
            raise ValueError(f"gnn_type phải là 'gat' hoặc 'gcn', nhận: {gnn_type!r}")
        if n_assets <= 0 or n_features <= 0 or lookback <= 0:
            raise ValueError("n_assets, n_features, và lookback phải lớn hơn 0.")
        if action_dim <= 1:
            raise ValueError("action_dim phải lớn hơn 1 (bao gồm tiền mặt).")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim phải lớn hơn 0.")

        self.gnn_type = gnn_type
        self.n_assets = n_assets
        self.n_features = n_features
        self.lookback = lookback
        self.hidden_dim = hidden_dim
        self.knn_k = min(knn_k, n_assets - 1) if n_assets > 1 else 1
        self.corr_threshold = corr_threshold

        if tickers is None:
            tickers = [f"ASSET_{i}" for i in range(n_assets)]
        self.tickers = list(tickers)

        # Graph Builder đa quan hệ (tương quan động + cùng ngành)
        self.graph_builder = MultiRelationGraphBuilder(
            tickers=self.tickers,
            rolling_corr_df=None,
            threshold=corr_threshold,
            config_path=config_path,
        )

        # GNN Encoder
        if self.gnn_type == "gat":
            self.gnn = GATEncoder(
                in_channels=n_features,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=heads,
                dropout=dropout,
            )
        else:
            self.gnn = GCNEncoder(
                in_channels=n_features,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim,
                dropout=dropout,
            )

        # Heads
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.cash_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim + (n_assets + 1), hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

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

    def _forward_distribution(self, obs: torch.Tensor) -> tuple[Dirichlet, torch.Tensor]:
        single = obs.ndim == 1
        market_history, portfolio_state = self._unflatten_obs(obs)
        if single:
            market_history = market_history.unsqueeze(0)
            portfolio_state = portfolio_state.unsqueeze(0)

        B = market_history.shape[0]
        N = self.n_assets
        F_dim = self.n_features
        device = obs.device

        # 1. Trích xuất Node Features tại ngày mới nhất (t)
        # Shape: [B, N, F_dim] -> Flatten theo batch: [B * N, F_dim]
        current_node_features = market_history[:, -1, :, :].reshape(B * N, F_dim)
        current_node_features = torch.nan_to_num(current_node_features, nan=0.0)

        # 2. Xây dựng đồ thị động cho từng sample trong batch
        # Tính tương quan từ chuỗi return trong cửa sổ lookback: market_history[:, :, :, 0]
        returns_history = market_history[:, :, :, 0].detach().cpu().numpy()  # [B, L, N]

        batch_edge_indices: list[torch.Tensor] = []
        batch_edge_weights: list[torch.Tensor] = []

        for b in range(B):
            sample_returns = returns_history[b]  # [L, N]
            if sample_returns.shape[0] >= 3:
                std = np.std(sample_returns, axis=0)
                if np.all(std > 1e-6):
                    corr = np.corrcoef(sample_returns.T)
                    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    corr = np.eye(N, dtype=np.float32)
            else:
                corr = np.eye(N, dtype=np.float32)

            e_idx, e_wt = self.graph_builder.build_from_matrix(
                corr,
                threshold=self.corr_threshold,
                mode="knn",
                top_k=self.knn_k,
            )
            # Offset node index cho batch
            batch_edge_indices.append(e_idx + (b * N))
            batch_edge_weights.append(e_wt)

        edge_index = torch.cat(batch_edge_indices, dim=1).to(device)
        edge_weight = torch.cat(batch_edge_weights, dim=0).to(device)

        # 3. GNN Encoding
        node_embeddings = self.gnn(current_node_features, edge_index, edge_weight)  # [B * N, D]
        node_embeddings = torch.nan_to_num(node_embeddings, nan=0.0)
        node_embeddings = node_embeddings.view(B, N, self.hidden_dim)  # [B, N, D]

        # 4. Global Market Pooling
        market_state = node_embeddings.mean(dim=1)  # [B, D]
        market_state = torch.nan_to_num(market_state, nan=0.0)

        # 5. Stock Head & Cash Head
        clean_portfolio_state = torch.nan_to_num(portfolio_state, nan=1.0 / (N + 1))
        current_asset_weights = clean_portfolio_state[:, :N].unsqueeze(-1)  # [B, N, 1]
        stock_input = torch.cat([node_embeddings, current_asset_weights], dim=-1)  # [B, N, D + 1]
        stock_logits = self.actor_head(stock_input).squeeze(-1)  # [B, N]

        current_cash_ratio = clean_portfolio_state[:, -1:]  # [B, 1]
        cash_input = torch.cat([market_state, current_cash_ratio], dim=-1)  # [B, D + 1]
        cash_logit = self.cash_head(cash_input)  # [B, 1]

        # Gom N + 1 logits
        action_logits = torch.cat([stock_logits, cash_logit], dim=-1)  # [B, N + 1]
        action_logits = torch.nan_to_num(action_logits, nan=0.0).clamp(min=-20.0, max=20.0)

        # Nén qua softplus để tạo concentration parameters cho Dirichlet
        concentration = (F.softplus(action_logits) + 1.0).clamp(min=1.001, max=100.0)
        concentration = torch.nan_to_num(concentration, nan=1.001)

        # 6. Critic Value
        critic_input = torch.cat([market_state, clean_portfolio_state], dim=-1)  # [B, D + (N + 1)]
        values = self.critic_head(critic_input).squeeze(-1)  # [B]
        values = torch.nan_to_num(values, nan=0.0).clamp(min=-1e6, max=1e6)

        if single:
            values = values.squeeze(0)

        return Dirichlet(concentration), values

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
    gnn_heads: int = 4,
    gnn_dropout: float = 0.0,
    gnn_knn: int = 5,
    tickers: list[str] | None = None,
    config_path: str = "configs/assets.yaml",
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
    if encoder in {"gat", "gcn"}:
        return GNNActorCritic(
            gnn_type=encoder,
            n_assets=n_assets,
            n_features=n_features,
            lookback=lookback,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            heads=gnn_heads,
            dropout=gnn_dropout,
            tickers=tickers,
            knn_k=gnn_knn,
            config_path=config_path,
        )
    raise ValueError(f"encoder phải là 'mlp', 'lstm', 'gru', 'gat' hoặc 'gcn', nhận: {encoder!r}")

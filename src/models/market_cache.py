"""Bounded, version-checked market embeddings for frozen PPO inference."""

from dataclasses import dataclass

import torch


@dataclass
class MarketEmbeddingCache:
    first_day: int
    embeddings: torch.Tensor
    parameter_versions: tuple

    def get(self, model, day: int):
        if tuple(p._version for p in model.parameters()) != self.parameter_versions:
            raise RuntimeError("Market cache đã cũ sau khi model parameters thay đổi.")
        if not self.first_day <= day < self.first_day + len(self.embeddings):
            raise ValueError("Ngày observation nằm ngoài market cache.")
        return self.embeddings[day - self.first_day]


@torch.no_grad()
def prepare_market_cache(model, env, batch_size=None, temporal_batch_size=None, progress=None):
    if not hasattr(model, "encode_days"):
        return None
    batch_size = getattr(model, "cache_batch_size", 64) if batch_size is None else batch_size
    temporal_batch_size = getattr(model, "cache_temporal_batch_size", batch_size) if temporal_batch_size is None else temporal_batch_size
    if batch_size <= 0 or temporal_batch_size <= 0:
        raise ValueError("Cache batch sizes phải lớn hơn 0.")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    first = max(0, env.history_start_index - model.lookback + 1)
    end = env.history_start_index + env.T
    positions = torch.arange(first, end, device=device)
    features = torch.as_tensor(env.history_tensor.tensor[first:end], dtype=dtype, device=device)
    graph_positions = positions + env.graph_start_index - env.history_start_index
    valid_graph = (graph_positions >= 0) & (graph_positions < len(env.graph_adjacency_sequence))
    graph_positions = graph_positions.clamp(0, len(env.graph_adjacency_sequence) - 1)
    adjacency = torch.as_tensor(env.graph_adjacency_sequence, dtype=dtype, device=device)[graph_positions]
    edge_mask = torch.as_tensor(env.graph_edge_mask_sequence, dtype=dtype, device=device)[graph_positions]
    adjacency = adjacency * valid_graph[:, None, None, None]
    edge_mask = edge_mask * valid_graph[:, None, None, None]
    daily_chunks = []
    for i in range(0, len(features), batch_size):
        daily_chunks.append(model.encode_days(features[i:i + batch_size], adjacency[i:i + batch_size], edge_mask[i:i + batch_size]))
        if progress is not None:
            progress("GNN", min(len(features), i + batch_size), len(features))
    daily = torch.cat(daily_chunks)
    encoded = []
    for i in range(env.history_start_index, end, temporal_batch_size):
        days = torch.arange(i, min(end, i + temporal_batch_size), device=device)
        histories = days[:, None] - torch.arange(model.lookback - 1, -1, -1, device=device)
        valid = histories >= first
        indices = (histories - first).clamp(min=0)
        windows = daily[indices] * valid[..., None, None]
        encoded.append(model.encode_temporal(windows, valid))
        if progress is not None:
            progress("GRU", min(end, i + temporal_batch_size) - env.history_start_index, env.T)
    return MarketEmbeddingCache(env.history_start_index, torch.cat(encoded), tuple(p._version for p in model.parameters()))

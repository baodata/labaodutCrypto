"""Shared observation serialization and checkpoint-backed PPO inference."""

import numpy as np
import torch

from src.features.scaler import MarketFeatureScaler
from src.models.actor_critic import build_actor_critic


def history_mask_array(obs: dict) -> np.ndarray:
    lookback = len(obs["market_history"])
    mask = np.asarray(obs.get("history_mask", np.ones(lookback, dtype=bool)))
    if mask.shape != (lookback,) or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("Observation history_mask phải có shape [L], chỉ chứa 0/1.")
    return mask.astype(bool, copy=False)


def history_mask_from_obs(obs: dict, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(history_mask_array(obs), dtype=torch.bool, device=device)


def observation_vector(obs: dict) -> np.ndarray:
    history = np.asarray(obs["market_history"], dtype=np.float32)
    history = np.where(history_mask_array(obs)[:, None, None], history, 0.0)
    return np.concatenate((
        history.reshape(-1),
        np.asarray(obs["portfolio_weights"], dtype=np.float32).reshape(-1),
        np.asarray(obs["cash_ratio"], dtype=np.float32).reshape(-1),
    ))


def graph_tensors_from_obs(obs: dict, device: torch.device) -> dict | None:
    keys = ("graph_adjacency_history", "graph_edge_mask_history")
    if any(key in obs for key in keys) and not all(key in obs for key in keys):
        raise ValueError("Observation cần cả graph adjacency và edge mask.")
    if not all(key in obs for key in keys):
        return None
    return {
        "adjacency_history": torch.as_tensor(obs[keys[0]], dtype=torch.float32, device=device),
        "edge_mask_history": torch.as_tensor(obs[keys[1]], dtype=torch.float32, device=device),
    }


def relation_history_from_correlations(builder, correlation_history, tickers):
    """Encode live rolling matrices into the same [L,R,N,N] channels as train."""
    if list(tickers) != builder.tickers:
        raise ValueError("Live ticker order không khớp graph builder.")
    matrices = np.asarray(correlation_history, dtype=np.float64)
    n = len(tickers)
    if matrices.ndim != 3 or matrices.shape[1:] != (n, n):
        raise ValueError("Live correlations phải có shape [L,N,N].")
    adjacency = np.zeros((len(matrices), 3, n, n), dtype=np.float32)
    mask = np.zeros_like(adjacency)
    for t, matrix in enumerate(matrices):
        for r, (edges, weights) in enumerate(builder.build_relations_from_matrix(matrix)):
            if edges.numel():
                source, target = edges.numpy()
                adjacency[t, r, source, target] = weights.numpy()
                mask[t, r, source, target] = 1
    return {"graph_adjacency_history": adjacency, "graph_edge_mask_history": mask}


class CheckpointPolicy:
    def __init__(self, checkpoint_path, *, tickers, feature_names, device="cpu", features_scaled=False,
                 feature_parameters=None):
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if list(tickers) != checkpoint["tickers"] or list(feature_names) != checkpoint["feature_names"]:
            raise ValueError("Inference ticker/feature order không khớp checkpoint.")
        if feature_parameters is not None and feature_parameters != checkpoint.get("feature_parameters"):
            raise ValueError("Định nghĩa feature inference khác checkpoint.")
        if "scaler_params" not in checkpoint:
            raise ValueError("Checkpoint cần scaler_params nhúng; không dùng scaler chung có thể bị ghi đè.")
        self.scaler = MarketFeatureScaler.from_dict(checkpoint["scaler_params"])
        self.model = build_actor_critic(**checkpoint["model_config"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.features_scaled = features_scaled
        self.requires_history_mask = checkpoint.get("history_mask_convention") is not None
        self.expected_shape = (checkpoint["lookback"], len(tickers), len(feature_names))

    @torch.no_grad()
    def __call__(self, obs, info=None):
        history = np.asarray(obs["market_history"], dtype=np.float32)
        if history.shape != self.expected_shape or not np.isfinite(history).all():
            raise ValueError("Inference market_history sai shape hoặc không hữu hạn.")
        if self.requires_history_mask and "history_mask" not in obs:
            raise ValueError("Checkpoint này yêu cầu history_mask tường minh khi inference.")
        mask = history_mask_array(obs)
        prepared = dict(obs)
        if not self.features_scaled:
            prepared["market_history"] = self.scaler.transform(history)
        prepared["market_history"] = np.where(mask[:, None, None], prepared["market_history"], 0.0)
        state = torch.as_tensor(observation_vector(prepared), device=self.device)
        graph = graph_tensors_from_obs(prepared, self.device)
        kwargs = {"graph": graph} if graph is not None else {}
        kwargs["history_mask"] = history_mask_from_obs(prepared, self.device)
        action = self.model.act(state, deterministic=True, **kwargs)[0]
        return action.cpu().numpy().astype(np.float32)

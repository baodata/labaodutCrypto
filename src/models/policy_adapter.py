"""Shared observation serialization and checkpoint-backed PPO inference."""

import pandas as pd

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
        np.asarray(obs.get("reward_state", []), dtype=np.float32).reshape(-1),
        np.asarray(obs.get("market_context", []), dtype=np.float32).reshape(-1),
        np.asarray(obs.get("agent_state", []), dtype=np.float32).reshape(-1),
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

    def set_session_calendar(self, dates):
        if not hasattr(self.scaler,'set_session_calendar'):
            raise ValueError('Static scaler has no rolling session calendar.')
        self.scaler.set_session_calendar(dates)

    def normalization_state_dict(self):
        return self.scaler.to_dict()

    def load_normalization_state_dict(self, state):
        restored = MarketFeatureScaler.from_dict(state)
        if restored.feature_names != self.scaler.feature_names or getattr(restored, 'tickers', None) != getattr(self.scaler, 'tickers', None):
            raise ValueError("Inference normalization state order mismatch.")
        self.scaler = restored
        if hasattr(self, '_normalization_history'):
            del self._normalization_history

    def prepare_observation(self, obs):
        history = np.asarray(obs["market_history"], dtype=np.float32)
        if history.shape != self.expected_shape or not np.isfinite(history).all():
            raise ValueError("Inference market_history sai shape hoặc không hữu hạn.")
        if self.requires_history_mask and "history_mask" not in obs:
            raise ValueError("Checkpoint này yêu cầu history_mask tường minh khi inference.")
        mask = history_mask_array(obs)
        prepared = dict(obs)
        if not self.features_scaled:
            if hasattr(self.scaler, 'transform_day'):
                if self.scaler.context_features:
                    prepared['market_context'] = self.scaler.market_context(history[-1])
                dates = obs.get('history_dates')
                if dates is None or len(dates) != len(history):
                    raise ValueError("Causal inference needs history_dates and sequential dates after stored scaler state.")
                cache = getattr(self, '_normalization_history', {str(pd.Timestamp(d)): np.asarray(v, dtype=np.float32) for d,v in self.scaler.normalized_history})
                for i, date in enumerate(dates):
                    if not mask[i]:
                        continue
                    key = str(pd.Timestamp(date))
                    if key not in cache:
                        cache[key] = self.scaler.transform_day(history[i], date, tickers=self.scaler.tickers)
                prepared['market_history'] = np.asarray([cache[str(pd.Timestamp(d))] if mask[i] else np.zeros_like(history[i]) for i,d in enumerate(dates)])
                self._normalization_history = {str(pd.Timestamp(d)): cache[str(pd.Timestamp(d))] for i,d in enumerate(dates) if mask[i]}
            else:
                prepared["market_history"] = self.scaler.transform(history)
        prepared["market_history"] = np.where(mask[:, None, None], prepared["market_history"], 0.0)
        return prepared

    @torch.no_grad()
    def __call__(self, obs, info=None):
        prepared = self.prepare_observation(obs)
        state = torch.as_tensor(observation_vector(prepared), device=self.device)
        graph = graph_tensors_from_obs(prepared, self.device)
        kwargs = {"graph": graph} if graph is not None else {}
        kwargs["history_mask"] = history_mask_from_obs(prepared, self.device)
        action = self.model.act(state, deterministic=True, **kwargs)[0]
        return action.cpu().numpy().astype(np.float32)


class SectorCheckpointPolicy(CheckpointPolicy):
    """Inference for train_marl's jointly executed, equal-budget sector policies.

    Normalize the GLOBAL universe first, then slice each sector in the exact
    training order. Rolling scaler/reward state contracts are unchanged.
    """
    def __init__(self, checkpoint_path, *, tickers, feature_names, device='cpu',features_scaled=False,
                 feature_parameters=None):
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path,map_location=self.device,weights_only=False)
        if list(tickers) != checkpoint['tickers'] or list(feature_names) != checkpoint['feature_names']:
            raise ValueError('Sector inference ticker/feature order mismatch.')
        if feature_parameters is not None and checkpoint.get('feature_parameters') != feature_parameters:
            raise ValueError('Sector feature parameters mismatch.')
        self.scaler = MarketFeatureScaler.from_dict(checkpoint['scaler_params'])
        self.models = {}
        self.mapping = checkpoint['sector_mapping']
        for agent,config in checkpoint['model_config'].items():
            model = build_actor_critic(**config).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'][agent])
            self.models[agent] = model.eval()
        self.features_scaled = features_scaled
        self.requires_history_mask = True
        lookback = next(iter(checkpoint['model_config'].values()))['lookback']
        self.expected_shape = (lookback,len(tickers),len(feature_names))

    @torch.no_grad()
    def __call__(self, obs, info=None):
        from src.env.marl_env import split_sector_observation
        prepared = self.prepare_observation(obs)
        local = split_sector_observation(prepared,self.mapping)
        global_weights = np.zeros(self.expected_shape[1]+1,dtype=np.float32)
        for agent,model in self.models.items():
            state = torch.as_tensor(observation_vector(local[agent]),device=self.device)
            action = model.act(state,deterministic=True,history_mask=history_mask_from_obs(local[agent],self.device))[0].cpu().numpy()
            global_weights[self.mapping[agent]] = action[:-1]/len(self.models)
            global_weights[-1] += action[-1]/len(self.models)
        return global_weights

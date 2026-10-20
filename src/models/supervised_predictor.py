"""Per-asset GRU/GNN-GRU return predictors, independent of portfolio allocation.

Both share feature projection, temporal GRU and a scalar head per stock. The
GNN adds separate dense message transforms for positive/negative/sector
relations before the GRU. No pooling across assets discards stock identities.
"""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


NEURAL_FAMILIES = {"gru", "gnn_gru", "gnn_gru_no_edges"}


class SupervisedReturnPredictor(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, family: str, relations: int = 3):
        super().__init__()
        if family not in NEURAL_FAMILIES or min(n_features, hidden_dim, relations) <= 0:
            raise ValueError("Invalid supervised predictor architecture.")
        self.model_config = dict(n_features=n_features, hidden_dim=hidden_dim, family=family, relations=relations)
        self.family, self.relations = family, relations
        # Common modules initialize first: same seed gives identical common
        # parameters for GRU and GNN-GRU, including the no-edge control.
        self.input_projection = nn.Linear(n_features, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.head.weight, std=.01)
        nn.init.zeros_(self.head.bias)
        self.relation_transforms = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(relations)]
            if family != "gru" else []
        )

    def forward(self, history, history_mask, adjacency=None):
        if history.ndim != 4 or history.shape[-1] != self.model_config["n_features"]:
            raise ValueError("Predictor features must be [B,L,N,F].")
        batch, length, assets, _ = history.shape
        if (history_mask.shape != (batch, length) or history_mask.dtype != torch.bool
                or not history_mask.any(dim=1).all()
                or torch.any(history_mask[:, :-1] & ~history_mask[:, 1:])):
            raise ValueError("A nonempty explicit left-padding mask is required for every history.")
        encoded = torch.tanh(self.input_projection(history.masked_fill(~history_mask[:, :, None, None], 0)))
        if self.family == "gnn_gru":
            if adjacency is None or adjacency.shape != (batch, length, self.relations, assets, assets):
                raise ValueError("GNN-GRU needs [B,L,R,N,N] aligned adjacency histories.")
            if not torch.isfinite(adjacency).all() or torch.any(adjacency < 0):
                raise ValueError("Relation strengths must be finite nonnegative magnitudes.")
            # Producer stores A[source,target]; aggregate incoming neighbors.
            incoming = adjacency.transpose(-2, -1)
            incoming = incoming / incoming.sum(-1, keepdim=True).clamp_min(1e-12)
            messages = torch.zeros_like(encoded)
            for relation, transform in enumerate(self.relation_transforms):
                messages = messages + torch.matmul(incoming[:, :, relation], transform(encoded))
            encoded = encoded + torch.tanh(messages / self.relations)
        # Compact real left-padded steps to the beginning, then pack them.
        lengths = history_mask.sum(dim=1)
        indices = (torch.arange(length, device=history.device)[None] + length - lengths[:, None]).clamp_max(length-1)
        encoded = encoded.gather(1, indices[:, :, None, None].expand_as(encoded))
        series = encoded.permute(0, 2, 1, 3).reshape(batch*assets, length, -1)
        packed = pack_padded_sequence(series, lengths.repeat_interleave(assets).cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        result = self.head(hidden[-1]).reshape(batch, assets)
        return result - result.mean(dim=1, keepdim=True)


def history_batch(features, positions, *, ready_start, lookback, device, graph=None):
    positions = np.asarray(positions, dtype=np.int64)
    if positions.ndim != 1 or not len(positions) or np.any(positions < ready_start) or np.any(positions >= len(features)):
        raise ValueError("History positions must refer to feature-ready observed sessions.")
    indices = positions[:, None] - lookback + 1 + np.arange(lookback)[None]
    mask = indices >= ready_start
    indices = np.maximum(indices, ready_start)
    history = np.asarray(features[indices], dtype=np.float32).copy()
    history[~mask] = 0
    adjacency = None
    if graph is not None:
        adjacency = np.asarray(graph[indices], dtype=np.float32).copy()
        adjacency[~mask] = 0
        adjacency = torch.as_tensor(adjacency, device=device)
    return (torch.as_tensor(history, device=device), torch.as_tensor(mask, device=device), adjacency)


def predict_neural(model, features, positions, *, ready_start, settings, graph=None, target_scale=1.):
    model.eval()
    output = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for begin in range(0, len(positions), settings["batch_size"]):
            batch = history_batch(features, positions[begin:begin+settings["batch_size"]], ready_start=ready_start,
                                  lookback=settings["lookback"], device=device, graph=graph if model.family == "gnn_gru" else None)
            output.append(model(*batch).cpu().numpy().astype(np.float64) * target_scale)
    if not output:
        raise ValueError("No positions requested for neural prediction.")
    predictions = np.concatenate(output)
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite supervised return predictions.")
    return predictions


def fit_neural(features, positions, targets, *, ready_start, settings, family, graph=None):
    """Fixed epoch budget, dates shuffled as whole asset panels, no val selection."""
    device = torch.device(settings["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested for supervised predictor but unavailable.")
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    target_scale = max(float(np.std(targets)), 1e-8)
    targets = np.asarray(targets, dtype=np.float32) / target_scale
    if targets.shape != (len(positions), features.shape[1]) or not np.isfinite(targets).all():
        raise ValueError("Training labels must align with the selected historical decision panels.")
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(settings["seed"])
        model = SupervisedReturnPredictor(features.shape[-1], settings["hidden_dim"], family).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
        rng = np.random.default_rng(settings["seed"])
        losses = []
        for _ in range(settings["epochs"]):
            model.train()
            total = 0.
            order = rng.permutation(len(positions))
            for begin in range(0, len(positions), settings["batch_size"]):
                indices = order[begin:begin+settings["batch_size"]]
                batch = history_batch(features, positions[indices], ready_start=ready_start,
                    lookback=settings["lookback"], device=device, graph=graph if family == "gnn_gru" else None)
                expected = torch.as_tensor(targets[indices], device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(*batch), expected)
                if not torch.isfinite(loss):
                    raise ValueError("Supervised training loss is not finite.")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total += float(loss.detach()) * len(indices)
            losses.append(total / len(positions))
    model.eval()
    fit_info = dict(model=family, model_config=model.model_config, training_settings=dict(settings),
        target_scale=target_scale, train_normalized_mse_by_epoch=losses, optimizer="Adam",
        parameter_count=sum(p.numel() for p in model.parameters()), torch_version=str(torch.__version__),
        selection="fixed_epochs_no_validation_labels", history_mask="explicit_left_padding_true_real",
        graph_mode="causal_multi_relation" if family == "gnn_gru" else "none")
    return model, fit_info


def save_neural_checkpoint(path, model, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("Neural fit checkpoint already exists; choose a new bank directory.")
    temporary = path.with_name(path.name + f".{uuid4().hex}.tmp")
    payload = dict(metadata, state_dict={name: value.detach().cpu() for name, value in model.state_dict().items()})
    torch.save(payload, temporary)
    temporary.replace(path)


def load_neural_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    model = SupervisedReturnPredictor(**payload["model_config"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload

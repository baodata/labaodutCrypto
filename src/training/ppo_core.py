import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from typing import Dict, Any

def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """
    Tính tổng tích lũy có chiết khấu (discounted cumulative sum) một cách hiệu quả.
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class PPOBuffer:
    """
    Bộ đệm lưu trữ quỹ đạo (trajectory) cho một epoch của huấn luyện PPO.
    """
    def __init__(
        self,
        max_size: int,
        obs_dim: int,
        act_dim: int,
        gamma: float = 0.99,
        lam: float = 0.95,
        graph_dim: int | None = None,
        graph_lookback: int = 1,
        graph_relations: int = 1,
        graph_adjacency_sequence: np.ndarray | None = None,
        graph_edge_mask_sequence: np.ndarray | None = None,
        history_lookback: int | None = None,
    ):
        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((max_size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(max_size, dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.ret_buf = np.zeros(max_size, dtype=np.float32)
        self.val_buf = np.zeros(max_size, dtype=np.float32)
        self.logp_buf = np.zeros(max_size, dtype=np.float32)
        if history_lookback is not None and history_lookback <= 0:
            raise ValueError("history_lookback phải lớn hơn 0.")
        self.history_mask_buf = np.ones((max_size, history_lookback), dtype=bool) if history_lookback is not None else None
        self.graph_lookback = graph_lookback
        if (graph_adjacency_sequence is None) != (graph_edge_mask_sequence is None):
            raise ValueError("Indexed buffer cần cả graph adjacency và mask sequence.")
        self.graph_sequence = graph_adjacency_sequence
        self.graph_mask_sequence = graph_edge_mask_sequence
        self.graph_day_index_buf = np.zeros(max_size, dtype=np.int64) if self.graph_sequence is not None else None
        if self.graph_sequence is not None:
            if graph_dim is None or self.graph_sequence.ndim != 4 or self.graph_sequence.shape[1:] != (graph_relations, graph_dim, graph_dim) or self.graph_mask_sequence.shape != self.graph_sequence.shape:
                raise ValueError("Indexed graph sequence phải có shape [T,R,N,N].")
        if graph_dim is not None and (
            graph_dim <= 0 or graph_lookback <= 0 or graph_relations <= 0
        ):
            raise ValueError("graph_dim, graph_lookback và graph_relations phải lớn hơn 0.")
        self.graph_adjacency_history_buf = (
            np.zeros(
                (max_size, graph_lookback, graph_relations, graph_dim, graph_dim),
                dtype=np.float32,
            )
            if graph_dim is not None and self.graph_sequence is None
            else None
        )
        self.graph_edge_mask_history_buf = (
            np.zeros(
                (max_size, graph_lookback, graph_relations, graph_dim, graph_dim),
                dtype=np.float32,
            )
            if graph_dim is not None and self.graph_sequence is None
            else None
        )
        
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, max_size

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        val: float,
        logp: float,
        graph_adjacency_history: np.ndarray | None = None,
        graph_edge_mask_history: np.ndarray | None = None,
        graph_day_index: int | None = None,
        history_mask: np.ndarray | None = None,
    ):
        """
        Lưu trữ một bước chuyển (transition).
        """
        assert self.ptr < self.max_size
        if self.history_mask_buf is not None:
            mask = np.asarray(history_mask)
            if mask.shape != self.history_mask_buf.shape[1:] or not np.all((mask == 0) | (mask == 1)):
                raise ValueError("Buffer cần history_mask shape [L], chỉ chứa 0/1 cho mọi bước.")
            self.history_mask_buf[self.ptr] = mask
        elif history_mask is not None:
            raise ValueError("Khởi tạo PPOBuffer với history_lookback để lưu mask.")
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        if self.graph_day_index_buf is not None:
            if graph_day_index is None or not 0 <= graph_day_index < len(self.graph_sequence):
                raise ValueError("Indexed graph buffer cần day index hợp lệ.")
            if graph_adjacency_history is not None or graph_edge_mask_history is not None:
                raise ValueError("Không trộn graph history dense với indexed graph buffer.")
            self.graph_day_index_buf[self.ptr] = graph_day_index
        elif self.graph_adjacency_history_buf is not None:
            if graph_adjacency_history is None or graph_edge_mask_history is None:
                raise ValueError("Graph buffer cần đủ graph history cho mọi bước.")
            if graph_adjacency_history.shape != self.graph_adjacency_history_buf.shape[1:]:
                raise ValueError("graph_adjacency_history sai shape trong PPOBuffer.")
            if graph_edge_mask_history.shape != self.graph_edge_mask_history_buf.shape[1:]:
                raise ValueError("graph_edge_mask_history sai shape trong PPOBuffer.")
            self.graph_adjacency_history_buf[self.ptr] = graph_adjacency_history
            self.graph_edge_mask_history_buf[self.ptr] = graph_edge_mask_history
        elif graph_adjacency_history is not None or graph_edge_mask_history is not None:
            raise ValueError("Khởi tạo PPOBuffer với graph_dim để lưu graph data.")
        self.ptr += 1

    def finish_path(self, last_val: float = 0.0):
        """
        Tính toán lợi thế GAE-Lambda và phần thưởng tích lũy khi kết thúc một quỹ đạo.
        """
        if not np.isfinite(last_val):
            raise ValueError("last_val phải hữu hạn.")
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # GAE: delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # Returns-to-go: discounted cumulative rewards
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self) -> Dict[str, torch.Tensor]:
        """
        Trả về dữ liệu dưới dạng torch tensors, và chuẩn hóa lợi thế (advantages).
        """
        assert self.ptr == self.max_size  # buffer must be full
        self.ptr, self.path_start_idx = 0, 0
        
        # Advantage normalization
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            val=self.val_buf,
            adv=self.adv_buf,
            logp=self.logp_buf
        )
        if self.graph_adjacency_history_buf is not None:
            data["graph_adjacency_history"] = self.graph_adjacency_history_buf
            data["graph_edge_mask_history"] = self.graph_edge_mask_history_buf
        tensors = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
        if self.history_mask_buf is not None:
            tensors["history_mask"] = torch.as_tensor(self.history_mask_buf, dtype=torch.bool)
        if self.graph_day_index_buf is not None:
            tensors.update(
                graph_sequence=torch.as_tensor(self.graph_sequence, dtype=torch.float32),
                graph_mask_sequence=torch.as_tensor(self.graph_mask_sequence, dtype=torch.float32),
                graph_day_index=torch.as_tensor(self.graph_day_index_buf, dtype=torch.long),
                graph_lookback=torch.tensor(self.graph_lookback, dtype=torch.long),
            )
        return tensors


def graph_minibatch(data: dict, indices=None) -> dict | None:
    """Gather only the requested histories from the immutable daily graph bank."""
    if "graph_sequence" in data:
        days = data["graph_day_index"] if indices is None else data["graph_day_index"][indices]
        lookback = int(data["graph_lookback"].item())
        history = days[:, None] - torch.arange(lookback - 1, -1, -1, device=days.device)
        valid = history >= 0
        gathered = history.clamp(min=0)
        mask = valid[..., None, None, None]
        return {
            "adjacency_history": data["graph_sequence"][gathered] * mask,
            "edge_mask_history": data["graph_mask_sequence"][gathered] * mask,
            "day_index_history": history,
        }
    if "graph_adjacency_history" in data:
        return {
            "adjacency_history": data["graph_adjacency_history"] if indices is None else data["graph_adjacency_history"][indices],
            "edge_mask_history": data["graph_edge_mask_history"] if indices is None else data["graph_edge_mask_history"][indices],
        }
    return None

class PPOUpdater:
    """
    Thực hiện cập nhật PPO theo hàm mục tiêu bị cắt xén (clipped objective).
    """
    def __init__(self, clip_ratio: float = 0.2, target_kl: float = 0.01, 
                 value_loss_coef: float = 0.5, entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 value_clip_range: float | None = None):
        if value_clip_range is not None and (
            not np.isfinite(value_clip_range) or value_clip_range <= 0
        ):
            raise ValueError("value_clip_range phải là None hoặc số hữu hạn lớn hơn 0.")
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.value_clip_range = (
            None if value_clip_range is None else float(value_clip_range)
        )

    def update(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        buffer_data: Dict[str, torch.Tensor],
        train_iters: int = 10,
        batch_size: int | None = None,
        microbatch_size: int | None = None,
        progress=None,
    ) -> Dict[str, Any]:
        """
        Thực hiện nhiều epoch cập nhật PPO.
        """
        if train_iters <= 0:
            raise ValueError("train_iters phải lớn hơn 0.")

        obs = buffer_data['obs']
        act = buffer_data['act']
        ret = buffer_data['ret']
        adv = buffer_data['adv']
        old_logp = buffer_data['logp']
        sample_count = len(obs)
        if sample_count == 0:
            raise ValueError("PPO buffer không được rỗng.")
        if batch_size is None:
            batch_size = sample_count
        if batch_size <= 0:
            raise ValueError("batch_size phải lớn hơn 0.")
        batch_size = min(batch_size, sample_count)
        if microbatch_size is not None and microbatch_size <= 0:
            raise ValueError("microbatch_size phải lớn hơn 0.")
        microbatch_size = min(microbatch_size or batch_size, batch_size)

        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "kl": [], "clip_frac": []}
        weights = []
        minibatches_completed = 0
        minibatches_attempted = 0
        epochs_completed = 0
        stopped_early = False
        first_kl = float("nan")
        stop_kl = float("nan")
        max_kl = float("nan")
        microbatches_completed = 0
        planned_minibatches = train_iters * int(np.ceil(sample_count / batch_size))

        # Explained variance uses the critic predictions collected alongside the
        # rollout, before this update changes the model. This makes the metric
        # comparable across PPO epochs and avoids an extra full-batch forward pass.
        old_values = buffer_data.get("val")
        if old_values is None:
            with torch.no_grad():
                values = []
                for first in range(0, sample_count, microbatch_size):
                    indices = torch.arange(first, min(sample_count, first + microbatch_size), device=obs.device)
                    graph = graph_minibatch(buffer_data, indices)
                    kwargs = {"graph": graph} if graph is not None else {}
                    if "history_mask" in buffer_data:
                        kwargs["history_mask"] = buffer_data["history_mask"][indices]
                    values.append(model.evaluate(obs[indices], act[indices], **kwargs)[2].reshape(-1))
                old_values = torch.cat(values)
        ret_var = torch.var(ret.reshape(-1), unbiased=False)
        if ret_var.item() > 1e-12:
            explained_variance = (
                1.0
                - torch.var(ret.reshape(-1) - old_values.reshape(-1), unbiased=False)
                / ret_var
            ).item()
        else:
            explained_variance = float("nan")

        for epoch in range(train_iters):
            permutation = torch.randperm(sample_count, device=obs.device)
            for indices in permutation.split(batch_size):
                # Accumulate one effective minibatch at fixed parameters. Only
                # the forward/backward activation size changes; Adam and the KL
                # guard still operate once on the full minibatch.
                optimizer.zero_grad(set_to_none=True)
                accumulated = {name: 0.0 for name in stats}
                pieces = list(indices.split(microbatch_size))
                for piece_index, piece in enumerate(pieces, 1):
                    piece_graph = graph_minibatch(buffer_data, piece)
                    kwargs = {"graph": piece_graph} if piece_graph is not None else {}
                    if "history_mask" in buffer_data:
                        kwargs["history_mask"] = buffer_data["history_mask"][piece]
                    log_prob, entropy, value = model.evaluate(obs[piece], act[piece], **kwargs)
                    value = value.reshape(-1)
                    log_ratio = log_prob - old_logp[piece]
                    ratio = torch.exp(log_ratio)
                    clipped_ratio = ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio)
                    loss_pi = -torch.minimum(ratio * adv[piece], clipped_ratio * adv[piece]).mean()
                    target = ret[piece].reshape(-1)
                    unclipped_v = (value - target).square()
                    if self.value_clip_range is None:
                        loss_v = unclipped_v.mean()
                    else:
                        previous = old_values[piece].reshape(-1)
                        clipped_value = previous + (value - previous).clamp(
                            -self.value_clip_range, self.value_clip_range
                        )
                        loss_v = torch.maximum(unclipped_v, (clipped_value - target).square()).mean()
                    ent = entropy.mean()
                    fraction = len(piece) / len(indices)
                    loss = loss_pi + self.value_loss_coef * loss_v - self.entropy_coef * ent
                    (loss * fraction).backward()
                    with torch.no_grad():
                        piece_kl = ((ratio - 1.0) - log_ratio).mean().item()
                        piece_clip = (torch.abs(ratio - 1.0) > self.clip_ratio).float().mean().item()
                    piece_stats = {"policy_loss": loss_pi.item(), "value_loss": loss_v.item(),
                                   "entropy": ent.item(), "kl": piece_kl, "clip_frac": piece_clip}
                    for key, value_stat in piece_stats.items():
                        accumulated[key] += fraction * value_stat
                    microbatches_completed += 1
                    if progress is not None:
                        progress(minibatches_completed, planned_minibatches, piece_index, len(pieces))
                approx_kl = accumulated["kl"]
                clip_frac = accumulated["clip_frac"]
                minibatches_attempted += 1
                if minibatches_attempted == 1:
                    first_kl = approx_kl
                max_kl = approx_kl if not np.isfinite(max_kl) else max(max_kl, approx_kl)

                # Do not apply a minibatch update that already exceeds the KL guard.
                if approx_kl > 1.5 * self.target_kl:
                    stopped_early = True
                    stop_kl = approx_kl
                    optimizer.zero_grad(set_to_none=True)
                    break

                nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()

                count = len(indices)
                weights.append(count)
                stats["policy_loss"].append(accumulated["policy_loss"])
                stats["value_loss"].append(accumulated["value_loss"])
                stats["entropy"].append(accumulated["entropy"])
                stats["kl"].append(approx_kl)
                stats["clip_frac"].append(clip_frac)
                minibatches_completed += 1

            epochs_completed = epoch + 1
            if stopped_early:
                break

        if not weights:
            raise RuntimeError("PPO không cập nhật được minibatch nào; kiểm tra target_kl và dữ liệu.")

        def weighted_mean(name: str) -> float:
            return float(np.average(stats[name], weights=weights))

        return {
            "LossPi": weighted_mean("policy_loss"),
            "LossV": weighted_mean("value_loss"),
            "Entropy": weighted_mean("entropy"),
            "KL": weighted_mean("kl"),
            "ClipFrac": weighted_mean("clip_frac"),
            "ExplainedVar": float(explained_variance),
            "FirstKL": float(first_kl),
            "StopKL": float(stop_kl),
            "MaxKL": float(max_kl),
            "StopIter": epochs_completed,
            "NumMinibatches": minibatches_completed,
            "NumMinibatchesAttempted": minibatches_attempted,
            "NumMicrobatches": microbatches_completed,
            "StoppedEarly": float(stopped_early),
        }

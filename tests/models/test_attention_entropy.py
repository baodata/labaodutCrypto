"""Regression coverage for entropy aggregation, normalization and gradients."""
import math

import pytest
import torch

from src.models.gnn.gat_encoder import compute_attention_entropy_loss


def uniform_attention(nodes, degree, heads):
    target = torch.arange(nodes).repeat_interleave(degree)
    source = (target + torch.arange(degree).repeat(nodes)) % nodes
    edges = torch.stack((source, target))
    alpha = torch.full((len(target), heads), 1 / degree, dtype=torch.float64, requires_grad=True)
    return alpha, edges


@pytest.mark.parametrize('degree', [2, 8])
@pytest.mark.parametrize('heads', [1, 4])
def test_uniform_entropy_is_mean_per_node_and_head(degree, heads):
    alpha, edges = uniform_attention(8, degree, heads)
    entropy = compute_attention_entropy_loss(alpha, edges, num_nodes=8)
    assert entropy.item() == pytest.approx(math.log(degree))
    entropy.backward()
    assert torch.isfinite(alpha.grad).all()
    normalized = compute_attention_entropy_loss(alpha, edges, num_nodes=8, normalize_by_degree=True)
    assert normalized.item() == pytest.approx(1.)


def test_degree_one_and_isolated_nodes():
    edges = torch.tensor([[0, 1, 2], [0, 0, 1]])
    alpha = torch.tensor([[.5], [.5], [1.]], dtype=torch.float64)
    assert compute_attention_entropy_loss(alpha, edges, 4).item() == pytest.approx(math.log(2) / 2)
    assert compute_attention_entropy_loss(alpha, edges, 4, normalize_by_degree=True).item() == pytest.approx(.5)


def test_zero_probability_and_empty_graph_have_finite_gradients():
    alpha = torch.tensor([[1.], [0.]], requires_grad=True)
    entropy = compute_attention_entropy_loss(alpha, torch.tensor([[0, 1], [0, 0]]), 2)
    assert entropy.item() == 0
    entropy.backward()
    assert torch.isfinite(alpha.grad).all()
    empty = torch.empty((0, 1), requires_grad=True)
    result = compute_attention_entropy_loss(empty, torch.empty((2, 0), dtype=torch.long), 3)
    assert result.item() == 0
    result.backward()
    assert empty.grad is not None


def test_rejects_attention_that_is_not_normalized_per_destination():
    with pytest.raises(ValueError, match='sum to one'):
        compute_attention_entropy_loss(torch.tensor([[1.], [1.]]), torch.tensor([[0, 1], [0, 0]]))


def test_requires_edges_returned_with_attention_including_self_loops():
    with pytest.raises(ValueError, match='aligned'):
        compute_attention_entropy_loss(torch.tensor([[.5], [.5]]), torch.tensor([[0], [0]]))

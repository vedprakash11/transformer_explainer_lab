"""
Unit tests for head_analysis.

Covers head_similarity and prune_heads with synthetic similarity matrices.
"""

import pytest
import numpy as np
import torch

from visualizer.head_analysis import head_similarity, prune_heads


@pytest.fixture
def dummy_attentions_layer0_4heads() -> tuple:
    """One layer, batch=1, 4 heads, seq=3. Used for head_similarity."""
    # (1, 4, 3, 3) per layer
    layer = torch.randn(1, 4, 3, 3).abs()
    layer = layer / layer.sum(dim=-1, keepdim=True)
    return (layer,)


class TestHeadSimilarity:
    """Tests for head_similarity."""

    def test_output_shape(self, dummy_attentions_layer0_4heads: tuple) -> None:
        sim = head_similarity(dummy_attentions_layer0_4heads, layer=0)
        assert sim.shape == (4, 4)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)

    def test_layer_out_of_range_raises(self, dummy_attentions_layer0_4heads: tuple) -> None:
        with pytest.raises((ValueError, RuntimeError), match="range|Layer"):
            head_similarity(dummy_attentions_layer0_4heads, layer=1)


class TestPruneHeads:
    """Tests for prune_heads."""

    def test_returns_redundant_pairs(self) -> None:
        # Sim matrix with one high-similarity pair
        sim = np.eye(4)
        sim[0, 1] = sim[1, 0] = 0.96
        redundant = prune_heads(sim, threshold=0.95)
        assert (0, 1) in redundant or (1, 0) in redundant

    def test_no_redundant_below_threshold(self) -> None:
        sim = np.eye(4) * 0.9 + np.random.RandomState(42).rand(4, 4) * 0.05
        sim = (sim + sim.T) / 2
        np.fill_diagonal(sim, 1.0)
        redundant = prune_heads(sim, threshold=0.99)
        assert len(redundant) == 0

    def test_threshold_bounds_raises(self) -> None:
        sim = np.eye(3)
        with pytest.raises((ValueError, RuntimeError), match="Threshold|0 and 1|redundant"):
            prune_heads(sim, threshold=1.5)

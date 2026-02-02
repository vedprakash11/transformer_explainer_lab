"""
Pytest fixtures and configuration.

Shared fixtures for attention tensors and model config so tests stay DRY.
"""

import pytest
import torch


@pytest.fixture
def dummy_attentions_2layer_2head_3seq() -> tuple:
    """
    Minimal attention tuple: 2 layers, batch=1, 2 heads, seq_len=3.
    Shape per layer: (1, 2, 3, 3).
    """
    # Row-stochastic (roughly) so rollout is valid
    a1 = torch.tensor([
        [[[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]],
        [[[0.4, 0.4, 0.2], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]]],
    ])  # (2, 1, 3, 3) -> we need (1, 2, 3, 3) per layer
    a1 = a1.permute(1, 0, 2, 3)  # (1, 2, 3, 3)
    a2 = a1.clone()
    return (a1, a2)


@pytest.fixture
def dummy_tokens_3() -> list:
    """Three tokens for use with dummy_attentions_2layer_2head_3seq."""
    return ["[CLS]", "hello", "[SEP]"]

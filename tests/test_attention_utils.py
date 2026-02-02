"""
Unit tests for attention_utils.

Covers attention_rollout, token_contribution, and attention_entropy with
small synthetic attention tensors.
"""

import pytest
import torch

from visualizer.attention_utils import (
    attention_entropy,
    attention_rollout,
    token_contribution,
)


class TestAttentionRollout:
    """Tests for attention_rollout."""

    def test_output_shape(self, dummy_attentions_2layer_2head_3seq: tuple) -> None:
        out = attention_rollout(dummy_attentions_2layer_2head_3seq)
        assert out.shape == (3, 3)

    def test_single_layer(self) -> None:
        # One layer, batch=1, 1 head, seq=2
        a = torch.eye(2).unsqueeze(0).unsqueeze(0)  # (1, 1, 2, 2)
        out = attention_rollout((a,))
        assert out.shape == (2, 2)
        torch.testing.assert_close(out, torch.eye(2), atol=1e-5, rtol=1e-5)

    def test_empty_or_wrong_input_raises(self) -> None:
        with pytest.raises(ValueError, match="Error computing attention rollout"):
            attention_rollout(())  # type: ignore


class TestTokenContribution:
    """Tests for token_contribution."""

    def test_output_length_matches_tokens(
        self,
        dummy_attentions_2layer_2head_3seq: tuple,
        dummy_tokens_3: list,
    ) -> None:
        result = token_contribution(
            dummy_attentions_2layer_2head_3seq,
            dummy_tokens_3,
            remove_cls=False,
            remove_sep=False,
        )
        assert len(result) == 3
        for t, pct in result:
            assert isinstance(t, str)
            assert isinstance(pct, (int, float))
            assert 0 <= pct <= 100

    def test_remove_cls_sep(
        self,
        dummy_attentions_2layer_2head_3seq: tuple,
        dummy_tokens_3: list,
    ) -> None:
        result = token_contribution(
            dummy_attentions_2layer_2head_3seq,
            dummy_tokens_3,
            remove_cls=True,
            remove_sep=True,
        )
        tokens_only = [t for t, _ in result]
        assert "[CLS]" not in tokens_only
        assert "[SEP]" not in tokens_only
        assert len(result) == 1  # only "hello"

    def test_empty_tokens_raises(
        self,
        dummy_attentions_2layer_2head_3seq: tuple,
    ) -> None:
        with pytest.raises((ValueError, RuntimeError), match="empty|Token"):
            token_contribution(dummy_attentions_2layer_2head_3seq, [])


class TestAttentionEntropy:
    """Tests for attention_entropy."""

    def test_returns_scalar(self, dummy_attentions_2layer_2head_3seq: tuple) -> None:
        ent = attention_entropy(dummy_attentions_2layer_2head_3seq)
        assert isinstance(ent, (int, float))
        assert ent >= 0

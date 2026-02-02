"""
Unit tests for visualizer.config.

Config is the single source of truth for model configs and env; tests ensure
we don't break supported model types or get_model_config contract.
"""

import pytest

from visualizer import config


class TestGetModelConfig:
    """Tests for get_model_config."""

    def test_bert_returns_expected_keys(self) -> None:
        cfg = config.get_model_config("bert")
        assert "model_name" in cfg
        assert "max_layers" in cfg
        assert "max_heads" in cfg
        assert cfg["model_name"] == "bert-base-uncased"
        assert cfg["max_layers"] == 12
        assert cfg["max_heads"] == 12

    def test_llama_returns_expected_keys(self) -> None:
        cfg = config.get_model_config("llama")
        assert cfg["model_name"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert cfg["max_layers"] == 22
        assert cfg["max_heads"] == 32

    def test_returns_copy_not_mutable_global(self) -> None:
        cfg = config.get_model_config("bert")
        cfg["max_layers"] = 999
        assert config.get_model_config("bert")["max_layers"] == 12

    def test_unsupported_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported model type"):
            config.get_model_config("gpt99")


class TestGetEnv:
    """Tests for get_env."""

    def test_missing_key_returns_default(self) -> None:
        assert config.get_env("NONEXISTENT_KEY_XYZ") == ""
        assert config.get_env("NONEXISTENT_KEY_XYZ", default="fallback") == "fallback"

    def test_returns_stripped(self) -> None:
        import os
        os.environ["_TEST_STRIP"] = "  ab  "
        try:
            assert config.get_env("_TEST_STRIP") == "ab"
        finally:
            os.environ.pop("_TEST_STRIP", None)

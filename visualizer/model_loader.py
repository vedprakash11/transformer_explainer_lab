"""
Model loading and caching for transformer analysis.

Loads BERT and LLaMA-style models with Streamlit resource caching.
Config lives in config.py; this module stays testable by depending on it.
"""

import logging
from typing import Any, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from . import config

logger = logging.getLogger(__name__)


def _load_bert(model_name: str) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Load BERT tokenizer and model with attention outputs enabled."""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(
        model_name,
        output_attentions=True,
        output_hidden_states=False,
    )
    model.eval()
    return tokenizer, model


def _load_llama(model_name: str) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Load LLaMA tokenizer and model with attention outputs enabled."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
        output_hidden_states=False,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def load_model(
    model_type: str = "bert",
    _show_spinner: bool = True,
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Load and cache a transformer model and tokenizer.

    Uses Streamlit's cache_resource when run in a Streamlit context; otherwise
    loads fresh each time (e.g. in tests).

    Args:
        model_type: "bert" or "llama".
        _show_spinner: Reserved for Streamlit UI; ignored elsewhere.

    Returns:
        (tokenizer, model) with model in eval mode and output_attentions=True.

    Raises:
        ValueError: If model_type is not supported.
        RuntimeError: If download or loading fails.
    """
    cfg = config.get_model_config(model_type)
    model_name = cfg["model_name"]

    try:
        if model_type == "bert":
            tokenizer, model = _load_bert(model_name)
        elif model_type == "llama":
            tokenizer, model = _load_llama(model_name)
        else:
            raise ValueError(
                f"Unsupported model type: {model_type!r}. "
                f"Supported: {sorted(config.SUPPORTED_MODEL_TYPES)}"
            )
        logger.info("Loaded model: type=%s, name=%s", model_type, model_name)
        return tokenizer, model
    except Exception as e:
        logger.exception("Model load failed: type=%s", model_type)
        raise RuntimeError(f"Failed to load {model_type} model: {e}") from e


def get_model_config(model_type: str) -> dict[str, Any]:
    """
    Return configuration dict for a supported model type.

    Args:
        model_type: "bert" or "llama".

    Returns:
        Dict with model_name, max_layers, max_heads.

    Raises:
        ValueError: If model_type is not supported.
    """
    return config.get_model_config(model_type)


# When running in Streamlit, cache model loads to avoid repeated downloads.
try:
    import streamlit as st
    load_model = st.cache_resource(load_model)
except Exception:
    pass

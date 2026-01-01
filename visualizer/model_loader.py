"""
Model loading utilities with caching for efficient model management.

This module handles loading and caching of transformer models for analysis.
"""

import streamlit as st
from transformers import (
    BertTokenizer,
    BertModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "bert": {
        "model_name": "bert-base-uncased",
        "max_layers": 12,
        "max_heads": 12,
    },
    "llama": {
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_layers": 22,
        "max_heads": 32,
    },
}


@st.cache_resource
def load_model(
    model_type: str = "bert",
    _show_spinner: bool = True
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Load and cache a transformer model and tokenizer.
    
    Args:
        model_type: Type of model to load ("bert" or "llama")
        _show_spinner: Internal parameter for Streamlit spinner control
    
    Returns:
        Tuple of (tokenizer, model)
    
    Raises:
        ValueError: If model_type is not supported
        RuntimeError: If model loading fails
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: {list(MODEL_CONFIGS.keys())}"
        )
    
    config = MODEL_CONFIGS[model_type]
    model_name = config["model_name"]
    
    try:
        if model_type == "bert":
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(
                model_name,
                output_attentions=True,
                output_hidden_states=False,
            )
        
        elif model_type == "llama":
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
                trust_remote_code=True,
            )
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                output_attentions=True,
                output_hidden_states=False,
                trust_remote_code=True,
            )
        
        model.eval()
        logger.info(f"Successfully loaded {model_type} model: {model_name}")
        
        return tokenizer, model
    
    except Exception as e:
        error_msg = f"Failed to load {model_type} model: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def get_model_config(model_type: str) -> dict:
    """
    Get configuration for a model type.
    
    Args:
        model_type: Type of model ("bert" or "llama")
    
    Returns:
        Dictionary with model configuration
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return MODEL_CONFIGS[model_type]

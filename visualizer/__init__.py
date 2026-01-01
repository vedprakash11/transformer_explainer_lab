"""
Transformer Explainability Lab - Core Package

A comprehensive package for visualizing and analyzing transformer attention mechanisms.
"""

__version__ = "1.0.0"
__author__ = "Transformer Explainability Lab"

from .model_loader import load_model, get_model_config
from .attention_utils import token_contribution, attention_entropy, attention_rollout
from .attention_visualizer import plot_attention, plot_head_similarity
from .head_analysis import head_similarity, prune_heads
from .qkv_extractor import extract_qkv
from .transformer_visualizer import (
    visualize_transformer_architecture,
    visualize_encoder_block,
    visualize_self_attention_mechanism,
    visualize_multi_head_attention,
    visualize_attention_formula,
)
from .explainability import explain_sentence, demo_attention_on_sentence

__all__ = [
    "load_model",
    "get_model_config",
    "token_contribution",
    "attention_entropy",
    "attention_rollout",
    "plot_attention",
    "plot_head_similarity",
    "head_similarity",
    "prune_heads",
    "extract_qkv",
    "visualize_transformer_architecture",
    "visualize_encoder_block",
    "visualize_self_attention_mechanism",
    "visualize_multi_head_attention",
    "visualize_attention_formula",
    "explain_sentence",
    "demo_attention_on_sentence",
]


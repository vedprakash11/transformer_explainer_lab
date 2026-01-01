"""
Head analysis utilities for transformer attention heads.

This module provides functions to analyze similarity between attention heads
and identify redundant heads for potential pruning.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from typing import List, Tuple, Optional


def head_similarity(
    attentions: Tuple[torch.Tensor, ...],
    layer: int,
) -> np.ndarray:
    """
    Compute cosine similarity matrix between attention heads in a given layer.
    
    Args:
        attentions: Tuple of attention tensors from all layers
        layer: Layer index to analyze
    
    Returns:
        Similarity matrix of shape (num_heads, num_heads)
    """
    try:
        if layer < 0 or layer >= len(attentions):
            raise ValueError(
                f"Layer {layer} out of range. "
                f"Available layers: 0-{len(attentions) - 1}"
            )
        
        # Get attention heads for the specified layer
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        heads = attentions[layer][0]  # Remove batch dimension
        
        # Flatten each head's attention matrix
        # Shape: (num_heads, seq_len * seq_len)
        flat = heads.reshape(heads.shape[0], -1).detach().cpu().numpy()
        
        # Compute cosine similarity between heads
        sim = cosine_similarity(flat)
        
        return sim
    
    except Exception as e:
        raise RuntimeError(f"Error computing head similarity: {str(e)}") from e


def prune_heads(
    similarity: np.ndarray,
    threshold: float = 0.95,
) -> List[Tuple[int, int]]:
    """
    Identify redundant attention heads based on similarity threshold.
    
    Heads with similarity above the threshold are considered redundant
    and could potentially be pruned without significant performance loss.
    
    Args:
        similarity: Head similarity matrix of shape (num_heads, num_heads)
        threshold: Similarity threshold for considering heads redundant (0-1)
    
    Returns:
        List of tuples (head_i, head_j) representing redundant head pairs
    """
    try:
        if not (0 <= threshold <= 1):
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        
        if similarity.ndim != 2:
            raise ValueError(f"Expected 2D similarity matrix, got {similarity.ndim}D")
        
        redundant = []
        n = similarity.shape[0]
        
        # Check all pairs of heads
        for i in range(n):
            for j in range(i + 1, n):
                if similarity[i, j] > threshold:
                    redundant.append((i, j))
        
        return redundant
    
    except Exception as e:
        raise RuntimeError(f"Error identifying redundant heads: {str(e)}") from e

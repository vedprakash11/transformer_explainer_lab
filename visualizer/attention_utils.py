"""
Attention utility functions for analyzing transformer attention mechanisms.

This module provides functions for computing attention rollout, token contributions,
and attention entropy metrics.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional


def attention_rollout(attentions: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """
    Compute attention rollout to visualize how attention flows through layers.
    
    Attention rollout is a method to aggregate attention weights across all layers
    to understand the flow of information through the transformer.
    
    Args:
        attentions: Tuple of attention tensors from all layers.
                   Shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
    
    Returns:
        Rollout attention matrix of shape (seq_len, seq_len)
    """
    try:
        attn = torch.stack(attentions)
        # attn shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
        attn = attn.mean(dim=2)  # Average over heads
        # attn shape: (num_layers, batch_size, seq_len, seq_len)
        
        # Select batch 0 (assuming single batch)
        if attn.dim() == 4:
            attn = attn[:, 0, :, :]  # Remove batch dimension
        # attn shape: (num_layers, seq_len, seq_len)
        
        # Add identity matrix to preserve direct connections
        eye = torch.eye(attn.size(-1)).to(attn.device)
        attn = attn + eye
        attn = attn / attn.sum(dim=-1, keepdim=True)
        
        # Compute rollout by matrix multiplication across layers
        rollout = attn[0]
        for i in range(1, attn.size(0)):
            rollout = attn[i] @ rollout
        
        # rollout shape: (seq_len, seq_len)
        return rollout
    except Exception as e:
        raise ValueError(f"Error computing attention rollout: {str(e)}") from e


def token_contribution(
    attentions: Tuple[torch.Tensor, ...],
    tokens: List[str],
    remove_cls: bool = False,
    remove_sep: bool = False
) -> List[Tuple[str, float]]:
    """
    Calculate the contribution percentage of each token using attention rollout.
    
    Args:
        attentions: Tuple of attention tensors from all layers
        tokens: List of token strings
        remove_cls: Whether to exclude [CLS] token from results
        remove_sep: Whether to exclude [SEP] token from results
    
    Returns:
        List of tuples (token, contribution_percentage)
    """
    try:
        if not tokens:
            raise ValueError("Token list cannot be empty")
        
        rollout = attention_rollout(attentions)
        rollout_np = rollout.detach().cpu().numpy()
        
        # Handle different rollout shapes (remove batch dimension if present)
        if rollout_np.ndim == 3:
            # Shape: (batch, seq_len, seq_len) - select batch 0
            if rollout_np.shape[0] == 1:
                rollout_np = rollout_np[0]  # Remove batch dimension
            else:
                raise ValueError(f"Multiple batches not supported. Got shape: {rollout_np.shape}")
        
        # For token contribution, sum over columns to get total attention received by each token
        # This aggregates attention from all tokens to each target token
        if rollout_np.ndim == 2:
            # Sum over columns (axis=0) to get total attention received by each token
            scores = rollout_np.sum(axis=0)
        elif rollout_np.ndim == 1:
            scores = rollout_np
        else:
            raise ValueError(f"Unexpected rollout shape after processing: {rollout_np.shape}")
        
        # Ensure scores is 1D array of scalars
        scores = np.asarray(scores).flatten()
        
        if len(tokens) != len(scores):
            raise ValueError(
                f"Token count ({len(tokens)}) doesn't match score count ({len(scores)})"
            )
        
        data = list(zip(tokens, scores))
        
        # Filter tokens based on options
        filtered = []
        for t, s in data:
            if remove_cls and t == "[CLS]":
                continue
            if remove_sep and t == "[SEP]":
                continue
            # Ensure s is a scalar
            if isinstance(s, (list, np.ndarray)):
                s = float(np.asarray(s).item())
            else:
                s = float(s)
            filtered.append((t, s))
        
        if not filtered:
            raise ValueError("No tokens remaining after filtering")
        
        # Normalize contributions - ensure all values are scalars
        values = np.array([float(v) for _, v in filtered])
        if values.sum() == 0:
            raise ValueError("All attention scores are zero")
        
        norm = (values / values.sum()).tolist()
        
        # Ensure norm is a flat list of scalars
        result = []
        for t, v in zip([x[0] for x in filtered], norm):
            # Ensure v is a scalar
            if isinstance(v, (list, np.ndarray)):
                v = float(np.asarray(v).item())
            else:
                v = float(v)
            result.append((t, v * 100))
        
        return result
    
    except Exception as e:
        raise RuntimeError(f"Error computing token contribution: {str(e)}") from e


def attention_entropy(attentions: Tuple[torch.Tensor, ...]) -> float:
    """
    Calculate the entropy of attention distributions.
    
    Higher entropy indicates more uniform attention (less focused),
    while lower entropy indicates more focused attention.
    
    Args:
        attentions: Tuple of attention tensors from all layers
    
    Returns:
        Mean entropy value across all attention distributions
    """
    try:
        # Average over layers and heads
        attn = torch.stack(attentions).mean(dim=(0, 1))
        
        # Compute entropy: -sum(p * log(p))
        entropy = -(attn * torch.log(attn + 1e-9)).sum(dim=-1)
        
        return entropy.mean().item()
    except Exception as e:
        raise RuntimeError(f"Error computing attention entropy: {str(e)}") from e

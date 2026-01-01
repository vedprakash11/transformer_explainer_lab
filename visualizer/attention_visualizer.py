"""
Visualization utilities for attention mechanisms.

This module provides functions to create interactive visualizations
of transformer attention patterns using Plotly.
"""

import plotly.graph_objects as go
import numpy as np
import torch
from typing import List, Optional


def plot_attention(
    attn: torch.Tensor,
    tokens: List[str],
    title: str = "Attention Heatmap",
    colorscale: str = "Viridis",
    height: int = 650,
    show_values: bool = False,
) -> go.Figure:
    """
    Create an interactive attention heatmap visualization.
    
    Args:
        attn: Attention tensor of shape (seq_len, seq_len)
        tokens: List of token strings for axis labels
        title: Title for the plot
        colorscale: Plotly colorscale name
        height: Height of the plot in pixels
        show_values: Whether to show attention values on the heatmap
    
    Returns:
        Plotly Figure object
    """
    try:
        # Convert to numpy if tensor
        if isinstance(attn, torch.Tensor):
            attn = attn.detach().cpu().numpy()
        
        # Ensure 2D array
        if attn.ndim != 2:
            raise ValueError(f"Expected 2D attention matrix, got {attn.ndim}D")
        
        # Truncate tokens if needed
        if len(tokens) != attn.shape[0]:
            tokens = tokens[:attn.shape[0]]
        
        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=attn,
                x=tokens,
                y=tokens,
                colorscale=colorscale,
                hovertemplate=(
                    "<b>Query:</b> %{y}<br>"
                    "<b>Key:</b> %{x}<br>"
                    "<b>Attention:</b> %{z:.4f}<extra></extra>"
                ),
                showscale=True,
                colorbar=dict(title="Attention Weight"),
            )
        )
        
        # Update layout
        fig.update_layout(
            title={
                "text": title,
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 16},
            },
            xaxis_title="Key Tokens",
            yaxis_title="Query Tokens",
            height=height,
            width=None,  # Use container width
            margin=dict(l=100, r=50, t=80, b=100),
            xaxis=dict(tickangle=-45),
        )
        
        return fig
    
    except Exception as e:
        raise RuntimeError(f"Error creating attention plot: {str(e)}") from e


def plot_head_similarity(
    sim: np.ndarray,
    title: str = "Head Similarity Matrix",
    colorscale: str = "Blues",
) -> go.Figure:
    """
    Create a heatmap visualization of head similarity matrix.
    
    Args:
        sim: Similarity matrix of shape (num_heads, num_heads)
        title: Title for the plot
        colorscale: Plotly colorscale name
    
    Returns:
        Plotly Figure object
    """
    try:
        if sim.ndim != 2:
            raise ValueError(f"Expected 2D similarity matrix, got {sim.ndim}D")
        
        # Convert to numpy if needed
        if isinstance(sim, torch.Tensor):
            sim = sim.detach().cpu().numpy()
        
        num_heads = sim.shape[0]
        head_labels = [f"Head {i}" for i in range(num_heads)]
        
        fig = go.Figure(
            data=go.Heatmap(
                z=sim,
                x=head_labels,
                y=head_labels,
                colorscale=colorscale,
                hovertemplate=(
                    "<b>Head %{y}</b> vs <b>Head %{x}</b><br>"
                    "<b>Similarity:</b> %{z:.3f}<extra></extra>"
                ),
                showscale=True,
                colorbar=dict(title="Cosine Similarity"),
                zmin=0,
                zmax=1,
            )
        )
        
        fig.update_layout(
            title={
                "text": title,
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 16},
            },
            xaxis_title="Head",
            yaxis_title="Head",
            height=600,
            width=700,
            margin=dict(l=100, r=50, t=80, b=100),
        )
        
        return fig
    
    except Exception as e:
        raise RuntimeError(f"Error creating similarity plot: {str(e)}") from e

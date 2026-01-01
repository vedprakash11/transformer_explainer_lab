"""
Transformer architecture visualization based on "Attention is All You Need".

This module provides visualizations of the transformer architecture,
self-attention mechanism, and multi-head attention.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import torch
from typing import List, Tuple, Optional
import streamlit as st


def visualize_transformer_architecture() -> go.Figure:
    """
    Create a high-level visualization of the transformer architecture.
    
    Returns:
        Plotly Figure showing the transformer encoder-decoder architecture
    """
    fig = go.Figure()
    
    # Define positions for components
    # Encoder stack
    encoder_y = [0.9, 0.7, 0.5, 0.3, 0.1]
    encoder_labels = ["Encoder 1", "Encoder 2", "Encoder N", "...", "Output"]
    
    # Decoder stack
    decoder_y = [0.9, 0.7, 0.5, 0.3, 0.1]
    decoder_labels = ["Decoder 1", "Decoder 2", "Decoder N", "...", "Output"]
    
    # Add encoder stack
    for i, (y, label) in enumerate(zip(encoder_y, encoder_labels)):
        fig.add_shape(
            type="rect",
            x0=0.1, y0=y-0.08, x1=0.4, y1=y+0.08,
            fillcolor="lightblue",
            line=dict(color="blue", width=2),
        )
        fig.add_annotation(
            x=0.25, y=y,
            text=label,
            showarrow=False,
            font=dict(size=10, color="black"),
        )
    
    # Add decoder stack
    for i, (y, label) in enumerate(zip(decoder_y, decoder_labels)):
        fig.add_shape(
            type="rect",
            x0=0.6, y0=y-0.08, x1=0.9, y1=y+0.08,
            fillcolor="lightgreen",
            line=dict(color="green", width=2),
        )
        fig.add_annotation(
            x=0.75, y=y,
            text=label,
            showarrow=False,
            font=dict(size=10, color="black"),
        )
    
    # Add connections between encoders
    for i in range(len(encoder_y) - 1):
        fig.add_shape(
            type="line",
            x0=0.25, y0=encoder_y[i]-0.08,
            x1=0.25, y1=encoder_y[i+1]+0.08,
            line=dict(color="blue", width=2, dash="dash"),
        )
    
    # Add connections between decoders
    for i in range(len(decoder_y) - 1):
        fig.add_shape(
            type="line",
            x0=0.75, y0=decoder_y[i]-0.08,
            x1=0.75, y1=decoder_y[i+1]+0.08,
            line=dict(color="green", width=2, dash="dash"),
        )
    
    # Add encoder-decoder attention connections
    fig.add_annotation(
        x=0.6, y=0.5,
        ax=0.4, ay=0.5,
        xref="paper", yref="paper",
        showarrow=True,
        arrowhead=2,
        arrowsize=2,
        arrowwidth=2,
        arrowcolor="red",
        text="Encoder-Decoder<br>Attention",
        font=dict(size=9, color="red"),
        bgcolor="white",
        bordercolor="red",
        borderwidth=1,
    )
    
    fig.update_layout(
        title="Transformer Architecture (Attention is All You Need)",
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
        height=600,
        showlegend=False,
    )
    
    return fig


def visualize_encoder_block() -> go.Figure:
    """
    Visualize a single encoder block with self-attention and feed-forward layers.
    
    Returns:
        Plotly Figure showing encoder block structure
    """
    fig = go.Figure()
    
    # Components
    components = [
        ("Input Embeddings", 0.5, 0.9, "lightblue"),
        ("Positional Encoding", 0.5, 0.8, "lightcoral"),
        ("Multi-Head<br>Self-Attention", 0.3, 0.6, "lightgreen"),
        ("Add & Norm", 0.3, 0.5, "lightyellow"),
        ("Feed Forward", 0.3, 0.3, "lightpink"),
        ("Add & Norm", 0.3, 0.2, "lightyellow"),
        ("Output", 0.5, 0.1, "lightblue"),
    ]
    
    # Draw components
    for label, x, y, color in components:
        fig.add_shape(
            type="rect",
            x0=x-0.12, y0=y-0.04, x1=x+0.12, y1=y+0.04,
            fillcolor=color,
            line=dict(color="black", width=1.5),
        )
        fig.add_annotation(
            x=x, y=y,
            text=label,
            showarrow=False,
            font=dict(size=9, color="black"),
        )
    
    # Add connections with arrows
    connections = [
        (0.5, 0.86, 0.3, 0.64, "gray"),  # Input to Attention
        (0.3, 0.56, 0.3, 0.54, "gray"),  # Attention to Add&Norm
        (0.3, 0.46, 0.3, 0.34, "gray"),  # Add&Norm to FF
        (0.3, 0.26, 0.3, 0.24, "gray"),  # FF to Add&Norm
        (0.3, 0.16, 0.5, 0.14, "gray"),  # Add&Norm to Output
        (0.5, 0.86, 0.3, 0.54, "orange"),  # Residual connection 1
        (0.3, 0.46, 0.3, 0.24, "orange"),  # Residual connection 2
    ]
    
    for x0, y0, x1, y1, color in connections:
        fig.add_annotation(
            x=x1, y=y1,
            ax=x0, ay=y0,
            xref="paper", yref="paper",
            showarrow=True,
            arrowhead=1,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor=color,
        )
    
    fig.update_layout(
        title="Encoder Block Architecture",
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
        height=500,
        showlegend=False,
    )
    
    return fig


def visualize_self_attention_mechanism(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tokens: List[str],
    attention_weights: Optional[torch.Tensor] = None
) -> go.Figure:
    """
    Visualize the self-attention mechanism showing Q, K, V and attention computation.
    
    Args:
        q: Query vectors (seq_len, head_dim)
        k: Key vectors (seq_len, head_dim)
        v: Value vectors (seq_len, head_dim)
        tokens: List of token strings
        attention_weights: Pre-computed attention weights (optional)
    
    Returns:
        Plotly Figure showing self-attention computation flow
    """
    # Convert to numpy
    q_np = q.detach().cpu().numpy() if isinstance(q, torch.Tensor) else q
    k_np = k.detach().cpu().numpy() if isinstance(k, torch.Tensor) else k
    v_np = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
    
    # Compute attention if not provided
    if attention_weights is None:
        # Compute scaled dot-product attention
        scores = np.dot(q_np, k_np.T) / np.sqrt(q_np.shape[1])
        attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = attention_weights / attention_weights.sum(axis=-1, keepdims=True)
    else:
        attention_weights = attention_weights.detach().cpu().numpy() if isinstance(attention_weights, torch.Tensor) else attention_weights
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Query (Q) Vectors", "Key (K) Vectors", 
                        "Attention Weights (Q×K^T/√d)", "Output (Attention × V)"),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
               [{"type": "heatmap"}, {"type": "heatmap"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.15,
    )
    
    # Q vectors heatmap
    fig.add_trace(
        go.Heatmap(
            z=q_np,
            x=[f"Dim {i}" for i in range(q_np.shape[1])],
            y=tokens,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(x=0.45, len=0.3),
        ),
        row=1, col=1
    )
    
    # K vectors heatmap
    fig.add_trace(
        go.Heatmap(
            z=k_np,
            x=[f"Dim {i}" for i in range(k_np.shape[1])],
            y=tokens,
            colorscale="Greens",
            showscale=True,
            colorbar=dict(x=0.95, len=0.3),
        ),
        row=1, col=2
    )
    
    # Attention weights heatmap
    fig.add_trace(
        go.Heatmap(
            z=attention_weights,
            x=tokens,
            y=tokens,
            colorscale="Reds",
            showscale=True,
            colorbar=dict(x=0.45, len=0.3, y=0.25),
            hovertemplate="Query: %{y}<br>Key: %{x}<br>Attention: %{z:.4f}<extra></extra>",
        ),
        row=2, col=1
    )
    
    # Output (Attention × V)
    output = np.dot(attention_weights, v_np)
    fig.add_trace(
        go.Heatmap(
            z=output,
            x=[f"Dim {i}" for i in range(output.shape[1])],
            y=tokens,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(x=0.95, len=0.3, y=0.25),
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Self-Attention Mechanism: Attention(Q, K, V) = softmax(QK^T/√d)V",
        height=800,
        showlegend=False,
    )
    
    # Update axes
    fig.update_xaxes(title_text="Dimensions", row=1, col=1)
    fig.update_yaxes(title_text="Tokens", row=1, col=1)
    fig.update_xaxes(title_text="Dimensions", row=1, col=2)
    fig.update_yaxes(title_text="Tokens", row=1, col=2)
    fig.update_xaxes(title_text="Key Tokens", row=2, col=1)
    fig.update_yaxes(title_text="Query Tokens", row=2, col=1)
    fig.update_xaxes(title_text="Dimensions", row=2, col=2)
    fig.update_yaxes(title_text="Tokens", row=2, col=2)
    
    return fig


def visualize_multi_head_attention(
    num_heads: int = 8,
    head_dim: int = 64
) -> go.Figure:
    """
    Visualize the multi-head attention mechanism.
    
    Args:
        num_heads: Number of attention heads
        head_dim: Dimension of each head
    
    Returns:
        Plotly Figure showing multi-head attention structure
    """
    fig = go.Figure()
    
    # Input
    fig.add_shape(
        type="rect",
        x0=0.1, y0=0.8, x1=0.3, y1=0.9,
        fillcolor="lightblue",
        line=dict(color="blue", width=2),
    )
    fig.add_annotation(x=0.2, y=0.85, text="Input<br>(d_model)", showarrow=False, font=dict(size=10))
    
    # Linear projections
    fig.add_shape(
        type="rect",
        x0=0.35, y0=0.75, x1=0.45, y1=0.95,
        fillcolor="lightgreen",
        line=dict(color="green", width=2),
    )
    fig.add_annotation(x=0.4, y=0.85, text="Linear<br>Proj", showarrow=False, font=dict(size=9))
    
    # Multiple heads
    head_width = 0.08
    head_spacing = 0.12
    start_x = 0.5
    
    for i in range(num_heads):
        x_center = start_x + i * head_spacing
        fig.add_shape(
            type="rect",
            x0=x_center - head_width/2, y0=0.7, x1=x_center + head_width/2, y1=0.9,
            fillcolor="lightcoral",
            line=dict(color="red", width=1.5),
        )
        fig.add_annotation(
            x=x_center, y=0.8,
            text=f"H{i+1}",
            showarrow=False,
            font=dict(size=8),
        )
    
    # Concatenate
    fig.add_shape(
        type="rect",
        x0=start_x + (num_heads-1) * head_spacing + head_width/2 + 0.05,
        y0=0.75, x1=start_x + (num_heads-1) * head_spacing + head_width/2 + 0.15,
        y1=0.85,
        fillcolor="lightyellow",
        line=dict(color="orange", width=2),
    )
    fig.add_annotation(
        x=start_x + (num_heads-1) * head_spacing + head_width/2 + 0.1,
        y=0.8,
        text="Concat",
        showarrow=False,
        font=dict(size=9),
    )
    
    # Output projection
    fig.add_shape(
        type="rect",
        x0=start_x + (num_heads-1) * head_spacing + head_width/2 + 0.2,
        y0=0.75, x1=start_x + (num_heads-1) * head_spacing + head_width/2 + 0.3,
        y1=0.85,
        fillcolor="lightpink",
        line=dict(color="purple", width=2),
    )
    fig.add_annotation(
        x=start_x + (num_heads-1) * head_spacing + head_width/2 + 0.25,
        y=0.8,
        text="Output<br>Proj",
        showarrow=False,
        font=dict(size=9),
    )
    
    # Add connections with arrows
    fig.add_annotation(
        x=0.35, y=0.85,
        ax=0.3, ay=0.85,
        xref="paper", yref="paper",
        showarrow=True,
        arrowhead=1,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor="black",
    )
    fig.add_annotation(
        x=start_x - head_width/2, y=0.8,
        ax=0.45, ay=0.85,
        xref="paper", yref="paper",
        showarrow=True,
        arrowhead=1,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor="black",
    )
    
    for i in range(num_heads):
        x_center = start_x + i * head_spacing
        if i < num_heads - 1:
            fig.add_shape(
                type="line",
                x0=x_center + head_width/2, y0=0.8,
                x1=start_x + (num_heads-1) * head_spacing + head_width/2 + 0.05, y1=0.8,
                line=dict(color="black", width=1.5),
            )
    
    fig.update_layout(
        title=f"Multi-Head Attention (h={num_heads} heads, d_k={head_dim})",
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1.2]),
        yaxis=dict(showgrid=False, showticklabels=False, range=[0.6, 1]),
        height=300,
        showlegend=False,
    )
    
    return fig


def visualize_attention_formula() -> go.Figure:
    """
    Visualize the attention formula with annotations.
    
    Returns:
        Plotly Figure showing the attention formula
    """
    fig = go.Figure()
    
    # Formula text
    formula_text = """
    <b>Scaled Dot-Product Attention:</b><br><br>
    Attention(Q, K, V) = softmax(QK^T / √d_k) V<br><br>
    <b>Multi-Head Attention:</b><br><br>
    MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)W^O<br>
    where headᵢ = Attention(QW^Qᵢ, KW^Kᵢ, VW^Vᵢ)<br><br>
    <b>Parameters:</b><br>
    • Q: Query matrix<br>
    • K: Key matrix<br>
    • V: Value matrix<br>
    • d_k: Dimension of keys/queries<br>
    • h: Number of attention heads<br>
    • W^Q, W^K, W^V: Learned projection matrices<br>
    • W^O: Output projection matrix
    """
    
    fig.add_annotation(
        x=0.5, y=0.5,
        text=formula_text,
        showarrow=False,
        font=dict(size=14, color="black"),
        align="left",
        xref="paper", yref="paper",
        bgcolor="white",
        bordercolor="black",
        borderwidth=2,
    )
    
    fig.update_layout(
        title="Attention Mechanism Formulas",
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
        height=400,
        showlegend=False,
    )
    
    return fig


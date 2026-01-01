import plotly.graph_objects as go
import numpy as np

def visualize_attention_interactive(attentions, tokens, layer=0, head=0):
    """
    Interactive attention visualization using Plotly
    """

    attention = attentions[layer][0, head].detach().cpu().numpy()

    fig = go.Figure(
        data=go.Heatmap(
            z=attention,
            x=tokens,
            y=tokens,
            colorscale="Viridis",
            hovertemplate=
            "<b>Query:</b> %{y}<br>" +
            "<b>Key:</b> %{x}<br>" +
            "<b>Attention:</b> %{z:.4f}<extra></extra>"
        )
    )

    fig.update_layout(
        title=f"Transformer Attention | Layer {layer}, Head {head}",
        xaxis_title="Key Tokens",
        yaxis_title="Query Tokens",
        width=900,
        height=750
    )

    fig.show()

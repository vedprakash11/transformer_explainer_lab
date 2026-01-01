import streamlit as st
import torch

from model_loader import load_model
from attention_visualizer import plot_attention

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Transformer Attention Visualizer",
    layout="wide"
)

st.title("🔍 Transformer Attention Visualization")
st.markdown("Visualize **BERT self-attention** interactively.")

# -------------------------
# Load Model
# -------------------------
tokenizer, model = load_model()

# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("⚙️ Controls")

text_input = st.sidebar.text_area(
    "Enter a sentence",
    "Transformers are changing the world of AI",
    height=100
)

layer = st.sidebar.slider("Select Layer", 0, 11, 0)
head = st.sidebar.slider("Select Head", 0, 11, 0)
hide_cls = st.sidebar.checkbox("Hide [CLS] Token")

run_btn = st.sidebar.button("Visualize Attention")

# -------------------------
# Main Logic
# -------------------------
if run_btn:
    with st.spinner("Running model..."):
        inputs = tokenizer(
            text_input,
            return_tensors="pt",
            add_special_tokens=True
        )

        with torch.no_grad():
            outputs = model(**inputs)

        attentions = outputs.attentions
        tokens = tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0]
        )

        fig = plot_attention(
            attentions=attentions,
            tokens=tokens,
            layer=layer,
            head=head,
            hide_cls=hide_cls
        )

        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Enter text and click **Visualize Attention**")

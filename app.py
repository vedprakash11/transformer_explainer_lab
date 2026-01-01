"""
Transformer Explainability Lab - Main Application

A comprehensive Streamlit application for visualizing and analyzing
transformer attention mechanisms.
"""

import streamlit as st
import torch
import logging
from typing import Optional

from visualizer import (
    load_model,
    get_model_config,
    plot_attention,
    plot_head_similarity,
    token_contribution,
    attention_entropy,
    head_similarity,
    prune_heads,
    extract_qkv,
    visualize_transformer_architecture,
    visualize_encoder_block,
    visualize_self_attention_mechanism,
    visualize_multi_head_attention,
    visualize_attention_formula,
    explain_sentence,
    demo_attention_on_sentence,
)



import os
from typing import Dict, Optional

# Try to import Groq, but make it optional
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

import os
from typing import Dict, Optional

# Try to import dotenv for .env file support (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    # dotenv not installed, skip .env file loading
    pass

# Try to import Groq, but make it optional
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

# Initialize Groq client (will be None if API key not set or Groq not available)
groq_api_key = os.environ.get("GROQ_API_KEY")
if GROQ_AVAILABLE and groq_api_key:
    try:
        groq_client = Groq(api_key=groq_api_key)
    except Exception:
        groq_client = None
else:
    groq_client = None
def llama_groq_explain(prompt: str) -> str:
    """Use LLaMA via Groq API to generate human-readable explanation."""
    if not GROQ_AVAILABLE:
        return "[Groq API Error] Groq package not installed. Install it with: pip install groq"
    
    if not groq_client:
        return "[Groq API Error] GROQ_API_KEY environment variable not set. Please set it to use AI-generated explanations."
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an expert in natural language processing and transformer models. Explain attention patterns and token relationships in a clear, human-readable way."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as ex:
        return f"[Groq API Error] {str(ex)}"



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Transformer Explainability Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
def auto_height(df, row_height=35, min_height=150, max_height=600):
    """
    Dynamically calculate dataframe height based on row count.
    """
    return min(
        max(len(df) * row_height, min_height),
        max_height
    )


def validate_inputs(text: str, layer: int, head: int, max_layer: int, max_head: int) -> Optional[str]:
    """Validate user inputs and return error message if invalid."""
    if not text or not text.strip():
        return "Please enter some text to analyze."
    
    if len(text.strip()) < 2:
        return "Input text must be at least 2 characters long."
    
    if layer < 0 or layer > max_layer:
        return f"Layer must be between 0 and {max_layer}."
    
    if head < 0 or head > max_head:
        return f"Head must be between 0 and {max_head}."
    
    return None


def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">🧠 Transformer Explainability Lab</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center; color: #666; margin-bottom: 2rem;">
            Visualize and analyze attention mechanisms in transformer models
        </div>
        """,
        unsafe_allow_html=True,
    )

# Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_type = st.selectbox(
            "Model Type",
            ["bert", "llama"],
            help="Select the transformer model to analyze",
        )
        
        # Get model configuration
        try:
            config = get_model_config(model_type)
            max_layers = config["max_layers"] - 1
            max_heads = config["max_heads"] - 1
        except Exception as e:
            st.error(f"Error loading model config: {str(e)}")
            st.stop()
        
        st.divider()
        
        # Input text
        st.subheader("📝 Input")
        text = st.text_area(
            "Input Text",
            value="Transformers are revolutionizing artificial intelligence and natural language processing.",
            height=100,
            help="Enter the text you want to analyze",
        )
        
        st.divider()
        
        # Layer and head selection
        st.subheader("🔍 Analysis Parameters")
        layer = st.slider(
            "Layer",
            0,
            max_layers,
            0,
            help=f"Select which transformer layer to visualize (0-{max_layers})",
        )
        head = st.slider(
            "Head",
            0,
            max_heads,
            0,
            help=f"Select which attention head to visualize (0-{max_heads})",
        )
        
        st.divider()
        
        # Filtering options
        st.subheader("🔧 Options")
        remove_cls = st.checkbox(
            "Remove [CLS] Token",
            value=True,
            help="Exclude [CLS] token from token contribution analysis",
        )
        remove_sep = st.checkbox(
            "Remove [SEP] Token",
            value=True,
            help="Exclude [SEP] token from token contribution analysis",
        )
        
        st.divider()
        
        # Run button
        run = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    # Load model with error handling
    try:
        with st.spinner("Loading model (this may take a moment on first run)..."):
            tokenizer, model = load_model(model_type)
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.info("💡 Tip: Make sure you have an internet connection for first-time model downloads.")
        st.stop()

    # Main content area
    if run:
        # Validate inputs
        error_msg = validate_inputs(text, layer, head, max_layers, max_heads)
        if error_msg:
            st.error(f"❌ {error_msg}")
            return
        
        try:
            # Tokenize input
            with st.spinner("Tokenizing input..."):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

            # Run model inference
            with st.spinner("Running model inference..."):
                with torch.no_grad():
                    outputs = model(**inputs)

            if not hasattr(outputs, "attentions") or outputs.attentions is None:
                st.error("❌ Model does not return attention weights. Please check model configuration.")
                return
            
            attentions = outputs.attentions
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

            # Validate layer and head indices
            if layer >= len(attentions):
                st.error(f"❌ Layer {layer} does not exist. Model has {len(attentions)} layers.")
                return
            
            num_heads = attentions[layer].shape[2]
            if head >= num_heads:
                st.error(f"❌ Head {head} does not exist. Layer {layer} has {num_heads} heads.")
                return
            
            # Display results in tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "🔍 Attention Heatmap",
                "📊 Token Analysis",
                "🧠 Head Analysis",
                "📈 Metrics",
                "🏗️ Transformer Architecture",
                "🔗 Explainability"
            ])
            
            with tab1:
                st.subheader(f"🔍 Attention Heatmap - Layer {layer}, Head {head}")
                try:
                    fig1 = plot_attention(
                        attentions[layer][0, head].cpu(),
                        tokens,
                        f"Layer {layer} | Head {head}",
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                    # Extract and display Q, K, V vectors
                    st.divider()
                    st.subheader("🔑 Query, Key, Value Vectors")
                    
                    try:
                        q, k, v = extract_qkv(model, inputs, layer, head, model_type)
                        
                        # Convert to numpy for display
                        q_np = q.detach().cpu().numpy()
                        k_np = k.detach().cpu().numpy()
                        v_np = v.detach().cpu().numpy()
                        
                        import pandas as pd
                       

                        # -------------------------------
                        # Q Matrix
                        # -------------------------------
                        st.markdown("### 🔹 Query (Q) Vectors")

                        q_df = pd.DataFrame(
                            q_np,
                            index=tokens,
                            columns=[f"Dim {i}" for i in range(q_np.shape[1])]
                        )

                        st.dataframe(
                            q_df.style.background_gradient(cmap="Blues", axis=None),
                            use_container_width=True,
                            height=auto_height(q_df)
                        )
                        st.caption(f"Shape: {q_np.shape}")

                        st.divider()

                        # -------------------------------
                        # K Matrix
                        # -------------------------------
                        st.markdown("### 🔹 Key (K) Vectors")

                        k_df = pd.DataFrame(
                            k_np,
                            index=tokens,
                            columns=[f"Dim {i}" for i in range(k_np.shape[1])]
                        )

                        st.dataframe(
                            k_df.style.background_gradient(cmap="Greens", axis=None),
                            use_container_width=True,
                            height=auto_height(k_df)
                        )
                        st.caption(f"Shape: {k_np.shape}")

                        st.divider()

                        # -------------------------------
                        # V Matrix
                        # -------------------------------
                        st.markdown("### 🔹 Value (V) Vectors")

                        v_df = pd.DataFrame(
                            v_np,
                            index=tokens,
                            columns=[f"Dim {i}" for i in range(v_np.shape[1])]
                        )

                        st.dataframe(
                            v_df.style.background_gradient(cmap="Reds", axis=None),
                            use_container_width=True,
                            height=auto_height(v_df)
                        )
                        st.caption(f"Shape: {v_np.shape}")

                        
                        # Add explanation
                        with st.expander("ℹ️ About Q, K, V Vectors"):
                            st.markdown("""
                            **Query (Q)**: Represents what information each token is looking for.
                            
                            **Key (K)**: Represents what information each token provides.
                            
                            **Value (V)**: Represents the actual content/information stored at each position.
                            
                            The attention weights are computed as: **Attention = softmax(Q × K^T / √d) × V**
                            
                            Where d is the dimension of the head.
                            """)
                    
                    except Exception as e:
                        st.warning(f"⚠️ Could not extract Q, K, V vectors: {str(e)}")
                        st.info("💡 Q, K, V extraction may not be supported for all model architectures.")
                
                except Exception as e:
                    st.error(f"❌ Error creating attention plot: {str(e)}")
            
            with tab2:
                st.subheader("📊 Token Contribution Analysis")
                try:
                    contrib = token_contribution(
                        attentions,
                        tokens,
                        remove_cls,
                        remove_sep,
                    )

                    if contrib:
                        # Create bar chart
                        contrib_dict = {t: v for t, v in contrib}
                        st.bar_chart(contrib_dict)

                        # Display table
                        st.subheader("Token Contribution Details")
                        import pandas as pd
                        df = pd.DataFrame(contrib, columns=["Token", "Contribution (%)"])
                        df = df.sort_values("Contribution (%)", ascending=False)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("No tokens to display after filtering.")
                
                except Exception as e:
                    st.error(f"❌ Error computing token contribution: {str(e)}")
            
            with tab3:
                st.subheader("🧠 Head Similarity Analysis")
                try:
                    sim = head_similarity(attentions, layer)
                    fig2 = plot_head_similarity(sim, f"Head Similarity - Layer {layer}")
                    st.plotly_chart(fig2, use_container_width=True)

                    # Head pruning suggestions
                    st.subheader("🔧 Head Pruning Suggestions")
                    redundant = prune_heads(sim, threshold=0.95)
                    if redundant:
                        st.warning(
                            f"⚠️ **{len(redundant)} redundant head pair(s) detected** "
                            f"(similarity > 0.95):"
                        )
                        for i, (h1, h2) in enumerate(redundant, 1):
                            similarity_score = sim[h1, h2]
                            st.write(f"{i}. Head {h1} ↔ Head {h2} (similarity: {similarity_score:.3f})")
                    else:
                        st.success("✅ No redundant heads detected at threshold 0.95")
                
                except Exception as e:
                    st.error(f"❌ Error analyzing heads: {str(e)}")
            
            with tab4:
                st.subheader("📈 Attention Metrics")
                try:
                    entropy = attention_entropy(attentions)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Attention Entropy",
                            f"{entropy:.4f}",
                            help="Higher entropy = more uniform attention, Lower = more focused"
                        )
                    with col2:
                        st.metric(
                            "Number of Layers",
                            len(attentions),
                        )
                    
                    # Interpretation
                    st.info(
                        f"""
                        **Entropy Interpretation:**
                        - **Current Value:** {entropy:.4f}
                        - **Range:** Typically 0-{len(attentions) * 2:.1f}
                        - Higher entropy indicates the model distributes attention more uniformly
                        - Lower entropy indicates the model focuses attention on specific tokens
                        """
                    )
                
                except Exception as e:
                    st.error(f"❌ Error computing metrics: {str(e)}")
            
            with tab5:
                st.subheader("🏗️ Transformer Architecture Visualization")
                st.markdown("Based on **'Attention is All You Need'** paper")
                
                # Architecture overview
                st.markdown("### 📐 Full Transformer Architecture")
                arch_fig = visualize_transformer_architecture()
                st.plotly_chart(arch_fig, use_container_width=True)
                
                st.divider()
                
                # Encoder block
                st.markdown("### 🔧 Encoder Block Structure")
                encoder_fig = visualize_encoder_block()
                st.plotly_chart(encoder_fig, use_container_width=True)
                
                st.divider()
                
                # Multi-head attention
                try:
                    num_heads = attentions[layer].shape[2]
                    head_dim = model.config.hidden_size // num_heads if hasattr(model.config, 'hidden_size') else 64
                    st.markdown("### 🧠 Multi-Head Attention Mechanism")
                    mha_fig = visualize_multi_head_attention(num_heads=num_heads, head_dim=head_dim)
                    st.plotly_chart(mha_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not determine head dimensions: {str(e)}")
                    mha_fig = visualize_multi_head_attention()
                    st.plotly_chart(mha_fig, use_container_width=True)
                
                st.divider()
                
                # Self-attention mechanism with actual Q, K, V
                st.markdown("### ⚡ Self-Attention Mechanism (Current Layer & Head)")
                try:
                    q, k, v = extract_qkv(model, inputs, layer, head, model_type)
                    attn_weights = attentions[layer][0, head].cpu()
                    
                    self_attn_fig = visualize_self_attention_mechanism(
                        q, k, v, tokens, attn_weights
                    )
                    st.plotly_chart(self_attn_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Could not visualize self-attention: {str(e)}")
                    st.info("💡 This visualization requires Q, K, V extraction to work.")
                
                st.divider()
                
                # Attention formulas
                st.markdown("### 📐 Attention Formulas")
                formula_fig = visualize_attention_formula()
                st.plotly_chart(formula_fig, use_container_width=True)
                
                # Additional information
                with st.expander("ℹ️ Understanding the Transformer Architecture"):
                    st.markdown("""
                    **Key Components:**
                    
                    1. **Encoder Stack**: Processes input sequences
                       - Self-attention: Allows each position to attend to all positions
                       - Feed-forward networks: Applied to each position independently
                       - Residual connections and layer normalization
                    
                    2. **Decoder Stack**: Generates output sequences
                       - Masked self-attention: Prevents attending to future positions
                       - Encoder-decoder attention: Connects encoder and decoder
                       - Feed-forward networks
                    
                    3. **Multi-Head Attention**: 
                       - Allows the model to jointly attend to information from different representation subspaces
                       - Each head learns different attention patterns
                       - Heads are concatenated and projected
                    
                    4. **Positional Encoding**: 
                       - Injects information about token positions
                       - Uses sinusoidal functions or learned embeddings
                    
                    **Why Self-Attention?**
                    - Constant path length between any two positions
                    - Parallelizable computation
                    - Interpretable attention patterns
                    """)
            

            with tab6:
                st.subheader("🔗 Explainability: Token Relationships & Coreference")
                
                # Tabs for different explainability views
                
                explain_tab1, explain_tab2 = st.tabs([
                    "📊 Model Attention Analysis",
                    "🎓 Demo: How Attention Works"
                ])
                
                with explain_tab1:
                    st.markdown(
                        """
                        This analysis identifies relationships between tokens using attention patterns from the actual model,
                        including pronoun‑antecedent relationships, entity co‑references, and human‑readable explanations 
                        generated by LLaMA via the Groq API.
                        """
                    )
                    
                    # Show API key setup instructions if not available
                    if not GROQ_AVAILABLE:
                        st.warning("⚠️ **Groq package not installed**")
                        st.info("""
                        To enable AI-generated explanations, install the Groq package:
                        ```bash
                        pip install groq
                        ```
                        """)
                    elif not groq_client:
                        with st.expander("🔑 How to Set Up Groq API Key", expanded=True):
                            st.markdown("""
                            **Step 1: Get Your API Key**
                            1. Visit [Groq Console](https://console.groq.com/keys)
                            2. Sign up or log in
                            3. Create a new API key
                            
                            **Step 2: Set the Environment Variable**
                            
                            **Windows (PowerShell):**
                            ```powershell
                            $env:GROQ_API_KEY="your_api_key_here"
                            ```
                            
                            **Windows (Command Prompt):**
                            ```cmd
                            set GROQ_API_KEY=your_api_key_here
                            ```
                            
                            **Linux/macOS:**
                            ```bash
                            export GROQ_API_KEY="your_api_key_here"
                            ```
                            
                            **Or create a `.env` file** in the project root:
                            ```
                            GROQ_API_KEY=your_api_key_here
                            ```
                            
                            **Step 3: Restart the Application**
                            After setting the environment variable, restart Streamlit for the changes to take effect.
                            
                            **Note:** The API key is optional. The tool will work without it, but AI-generated explanations will be disabled.
                            """)

                    try:
                        # Get raw text
                        sentence = text.strip()
                        if not sentence:
                            st.warning("Enter text above to generate explainability.")
                        else:
                            # Run explain_sentence logic with actual model attentions
                            with st.spinner("Analyzing token relationships..."):
                                explanation = explain_sentence(
                                    attentions=attentions,
                                    tokens=tokens,
                                    threshold=0.25,
                                    use_groq=GROQ_AVAILABLE and groq_client is not None,
                                    sentence=sentence
                                )

                            # # Display Groq textual explanation
                            # groq_explanation = explanation.get('groq_explanation')
                            # if groq_explanation:
                            #     st.markdown("### 🧾 AI-Generated Explanation (via Groq)")
                            #     # Check if it's an error message
                            #     if groq_explanation.startswith("[Groq API") or groq_explanation.startswith("[Groq API not available"):
                            #         st.warning(groq_explanation)
                            #     else:
                            #         st.info(groq_explanation)
                            # elif not GROQ_AVAILABLE:
                            #     st.info("💡 Install the `groq` package to enable AI-generated explanations: `pip install groq`")
                            # elif not groq_client:
                            #     st.info("💡 Set GROQ_API_KEY environment variable to enable AI-generated explanations.")

                            # st.divider()

                            # Visualize token relationship graph
                            st.markdown("### 🌐 Relationship Network Graph")
                            st.plotly_chart(explanation['network_figure'], use_container_width=True)

                            st.divider()

                            # Display pronoun‑antecedent relationships
                            st.markdown("### 👤 Pronoun‑Antecedent Relationships")
                            pronoun_ants = explanation.get('pronoun_antecedents', [])
                            if pronoun_ants:
                                for pronoun, antecedent, score in pronoun_ants:
                                    st.success(
                                        f"**{pronoun}** → **{antecedent}** (Confidence: {score:.3f})"
                                    )
                            else:
                                st.info("No pronoun‑antecedent relationships detected.")

                            st.divider()

                            # Display entity relationships
                            st.markdown("### 🔗 Entity Relationships")
                            entity_rels = explanation.get('entity_relations', [])
                            if entity_rels:
                                from collections import defaultdict
                                by_type = defaultdict(list)
                                for e1, e2, rel_type, score in entity_rels:
                                    by_type[rel_type].append((e1, e2, score))

                                for rel_type, rels in by_type.items():
                                    st.markdown(f"#### {rel_type.title()} Relationships")
                                    for e1, e2, score in sorted(rels, key=lambda x: x[2], reverse=True)[:5]:
                                        st.write(f"- **{e1}** ↔ **{e2}** (Confidence: {score:.3f})")
                            else:
                                st.info("No strong entity relationships detected.")

                            st.divider()

                            # Detailed relationship table
                            st.markdown("### 📋 Detailed Relationship Table")
                            table_data = explanation.get('table_data', [])
                            if table_data:
                                import pandas as pd
                                df = pd.DataFrame(table_data)
                                st.dataframe(df, use_container_width=True, hide_index=True)
                            else:
                                st.info("No relationships found above threshold.")

                    except Exception as e:
                        st.error(f"❌ Error generating explainability: {str(e)}")
                        st.info("💡 Please ensure Groq API key is set and internet connection is available.")
                
                with explain_tab2:
                    st.markdown(
                        """
                        **Educational Demo**: This section demonstrates how self-attention works step-by-step
                        using dummy embeddings. Perfect for understanding the attention mechanism!
                        """
                    )
                    
                    # Show API key setup instructions if not available
                    if not GROQ_AVAILABLE:
                        st.warning("⚠️ **Groq package not installed**")
                        st.info("""
                        To enable AI-generated explanations, install the Groq package:
                        ```bash
                        pip install groq
                        ```
                        """)
                    elif not groq_client:
                        with st.expander("🔑 How to Set Up Groq API Key", expanded=False):
                            st.markdown("""
                            **Quick Setup:**
                            1. Get API key from [Groq Console](https://console.groq.com/keys)
                            2. Set environment variable:
                               - **Windows (PowerShell):** `$env:GROQ_API_KEY="your_key"`
                               - **Linux/macOS:** `export GROQ_API_KEY="your_key"`
                            3. Restart the application
                            
                            See `GROQ_SETUP.md` for detailed instructions.
                            """)
                    
                    try:
                        sentence = text.strip()
                        if not sentence:
                            st.warning("Enter text above to see the demo.")
                        else:
                            with st.spinner("Computing attention with dummy embeddings..."):
                                demo_results = demo_attention_on_sentence(
                                    sentence=sentence,
                                    embedding_dim=8,
                                    use_groq=True
                                )
                            
                            # Overall explanation
                            st.markdown("### 📖 Overall Explanation")
                            st.info(demo_results['overall_explanation'])
                            
                            st.divider()
                            
                            # Attention weight matrix
                            st.markdown("### 🔹 Attention Weight Matrix")
                            import pandas as pd
                            attn_df = pd.DataFrame(
                                demo_results['attention_weights'],
                                index=demo_results['tokens'],
                                columns=demo_results['tokens']
                            )
                            st.dataframe(
                                attn_df.style.background_gradient(cmap='Blues', axis=None),
                                use_container_width=True
                            )
                            
                            st.divider()
                            
                            # Pronoun-specific attention
                            if demo_results['pronoun_attention']:
                                st.markdown("### 👤 Pronoun-Specific Attention Analysis")
                                for pronoun, explanation in demo_results['pronoun_explanations'].items():
                                    with st.expander(f"🔍 Attention for '{pronoun}'"):
                                        st.write(explanation)
                                        
                                        # Show attention scores
                                        pronoun_idx = demo_results['tokens'].index(pronoun)
                                        attn_scores = demo_results['attention_weights'][pronoun_idx]
                                        
                                        # Create a bar chart
                                        score_dict = {
                                            demo_results['tokens'][i]: float(attn_scores[i])
                                            for i in range(len(demo_results['tokens']))
                                        }
                                        st.bar_chart(score_dict)
                            
                            st.divider()
                            
                            # Q, K, V matrices
                            st.markdown("### 🔑 Query, Key, Value Matrices")
                            qkv_tab1, qkv_tab2, qkv_tab3 = st.tabs(["Query (Q)", "Key (K)", "Value (V)"])
                            
                            with qkv_tab1:
                                q_df = pd.DataFrame(
                                    demo_results['Q'],
                                    index=demo_results['tokens'],
                                    columns=[f"Dim {i}" for i in range(demo_results['Q'].shape[1])]
                                )
                                st.dataframe(q_df.style.background_gradient(cmap='Blues', axis=None))
                            
                            with qkv_tab2:
                                k_df = pd.DataFrame(
                                    demo_results['K'],
                                    index=demo_results['tokens'],
                                    columns=[f"Dim {i}" for i in range(demo_results['K'].shape[1])]
                                )
                                st.dataframe(k_df.style.background_gradient(cmap='Greens', axis=None))
                            
                            with qkv_tab3:
                                v_df = pd.DataFrame(
                                    demo_results['V'],
                                    index=demo_results['tokens'],
                                    columns=[f"Dim {i}" for i in range(demo_results['V'].shape[1])]
                                )
                                st.dataframe(v_df.style.background_gradient(cmap='Reds', axis=None))
                            
                            # How it works
                            with st.expander("ℹ️ How This Demo Works"):
                                st.markdown("""
                                **Step-by-Step Process:**
                                
                                1. **Tokenization**: Sentence is split into tokens
                                2. **Embeddings**: Each token gets a random embedding vector
                                3. **Q, K, V Creation**: Embeddings are multiplied by learned weight matrices
                                4. **Attention Scores**: Computed as Q × K^T / √d_k
                                5. **Softmax**: Scores are normalized to create attention weights
                                6. **Output**: Attention weights × Value vectors = final output
                                
                                **Why This Matters:**
                                - Shows how attention allows tokens to "look at" other tokens
                                - Demonstrates the mathematical operations behind attention
                                - Helps understand how transformers process relationships
                                """)
                    
                    except Exception as e:
                        st.error(f"❌ Error in demo: {str(e)}")
                        st.info("💡 This is an educational demo using dummy embeddings.")
            
            st.success("✅ Analysis Complete!")
        
        except Exception as e:
            st.error(f"❌ An error occurred during analysis: {str(e)}")
            logger.exception("Error in main analysis")
            st.info("💡 Try adjusting your input or parameters and run again.")
    
    else:
        # Welcome message
        st.info("👈 **Configure your analysis in the sidebar and click 'Run Analysis' to begin.**")
        
        # Feature overview
        with st.expander("📖 About This Tool", expanded=True):
            st.markdown(
                """
                ### Features
                
                - **🔍 Attention Heatmap**: Visualize attention patterns between tokens
                - **📊 Token Contribution**: See which tokens contribute most to the model's understanding
                - **🧠 Head Similarity**: Analyze relationships between attention heads
                - **📈 Metrics**: Compute attention entropy and other statistics
                
                ### Supported Models
                
                - **BERT**: Bidirectional Encoder Representations from Transformers
                - **TinyLlama**: A compact Llama model for causal language modeling
                
                ### Use Cases
                
                - Understanding model behavior
                - Debugging attention mechanisms
                - Identifying redundant heads for pruning
                - Research and education
                """
            )


if __name__ == "__main__":
    main()

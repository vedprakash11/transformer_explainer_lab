# 🧠 Transformer Explainability Lab

A production-ready, interactive web application for visualizing and analyzing attention mechanisms in transformer models. This tool provides comprehensive insights into how transformers process and understand text through attention patterns.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📁 Project Structure

```
visualizer/
├── app.py                      # Main Streamlit application
├── visualizer/               # Core package
│   ├── __init__.py           # Package initialization
│   ├── model_loader.py       # Model loading and caching
│   ├── attention_utils.py    # Attention computation functions
│   ├── attention_visualizer.py  # Visualization utilities
│   ├── head_analysis.py      # Head similarity and pruning
│   ├── qkv_extractor.py      # Q, K, V vector extraction
│   ├── transformer_visualizer.py  # Architecture visualizations
│   └── explainability.py     # Token relationship analysis
├── requirements.txt           # Python dependencies
├── setup.py                  # Package setup script
├── README.md                 # This file
├── QUICKSTART.md             # Quick start guide
├── LICENSE                   # MIT License
└── .gitignore               # Git ignore rules
```

## ✨ Features

### 🔍 **Attention Visualization**
- Interactive heatmaps showing attention patterns between tokens
- Layer-by-layer and head-by-head analysis
- Customizable visualization parameters

### 📊 **Token Contribution Analysis**
- Quantify how much each token contributes to the model's understanding
- Percentage-based contribution scores
- Filterable results (exclude [CLS], [SEP] tokens)

### 🧠 **Head Similarity Analysis**
- Identify redundant attention heads
- Cosine similarity matrix visualization
- Head pruning recommendations

### 📈 **Attention Metrics**
- Entropy calculations for attention distributions
- Model architecture insights
- Statistical analysis

### 🏗️ **Transformer Architecture Visualization**
- Full encoder-decoder architecture diagrams
- Self-attention mechanism visualization
- Multi-head attention structure
- Based on "Attention is All You Need" paper

### 🔗 **Explainability & Coreference Resolution**
- Identify pronoun-antecedent relationships
- Entity relationship detection
- Interactive network graph visualization
- Token relationship analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd visualizer
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

   The application will open in your default web browser at `http://localhost:8501`

## 📖 Usage Guide

### Basic Workflow

1. **Select Model**: Choose between BERT or TinyLlama from the sidebar
2. **Enter Text**: Input the text you want to analyze
3. **Configure Parameters**:
   - Select the layer to visualize (0 to max layers)
   - Select the attention head (0 to max heads)
   - Toggle filtering options for special tokens
4. **Run Analysis**: Click "Run Analysis" to generate visualizations
5. **Explore Results**: Navigate through tabs to view different analyses

### Understanding the Visualizations

#### Attention Heatmap
- **X-axis**: Key tokens (what the model is attending to)
- **Y-axis**: Query tokens (what is doing the attending)
- **Color intensity**: Attention weight strength
- **Darker colors**: Higher attention weights

#### Token Contribution
- Shows percentage contribution of each token
- Higher percentages indicate more important tokens
- Useful for understanding model focus

#### Head Similarity
- Identifies redundant attention heads
- Similarity > 0.95 suggests potential pruning candidates
- Helps optimize model architecture

#### Explainability
- Network graph showing token relationships
- Pronoun-antecedent resolution
- Entity relationship detection

## 🔧 Configuration

### Model Configuration

Models are configured in `visualizer/model_loader.py`:

```python
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
```

### Adding New Models

To add a new model:

1. Add configuration to `MODEL_CONFIGS` in `visualizer/model_loader.py`
2. Add loading logic in the `load_model()` function
3. Update the model selection dropdown in `app.py`

## 🧪 Technical Details

### Attention Rollout

The tool uses attention rollout to aggregate attention across layers:
- Averages attention over heads
- Adds identity matrix to preserve direct connections
- Multiplies attention matrices across layers

### Token Contribution

Computed using:
- Attention rollout scores
- Normalized to percentages
- Filterable by token type

### Head Similarity

Uses cosine similarity:
- Flattens attention matrices
- Computes pairwise similarity
- Identifies redundant pairs above threshold

### Explainability

Uses attention patterns to identify:
- Coreference chains
- Pronoun-antecedent relationships
- Entity relationships
- Semantic connections

## 🐛 Troubleshooting

### Model Loading Issues

- **First-time download**: Models are downloaded on first use (requires internet)
- **Memory errors**: Use smaller models or reduce input length
- **CUDA errors**: Models default to CPU; GPU support requires proper PyTorch CUDA setup

### Common Errors

- **"Layer out of range"**: Adjust layer slider to valid range
- **"Head out of range"**: Adjust head slider to valid range
- **Empty token list**: Ensure input text is not empty

## 📚 Dependencies

- **torch**: Deep learning framework
- **transformers**: Hugging Face transformers library
- **streamlit**: Web application framework
- **plotly**: Interactive visualizations
- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning utilities
- **networkx**: Network graph analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Hugging Face for the transformers library
- Streamlit for the web framework
- Plotly for visualization capabilities
- "Attention is All You Need" paper authors

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Made with ❤️ for the AI/ML community**

# Project Structure

This document describes the organization of the Transformer Explainability Lab project.

## Directory Layout

```
visualizer/
├── app.py                      # Main Streamlit application entry point
├── visualizer/                 # Core package directory
│   ├── __init__.py            # Package initialization and exports
│   ├── model_loader.py        # Model loading and caching utilities
│   ├── attention_utils.py     # Attention computation functions
│   ├── attention_visualizer.py  # Basic attention visualizations
│   ├── head_analysis.py       # Head similarity and pruning analysis
│   ├── qkv_extractor.py       # Query, Key, Value vector extraction
│   ├── transformer_visualizer.py  # Transformer architecture visualizations
│   └── explainability.py      # Token relationship and coreference analysis
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation script
├── README.md                  # Main project documentation
├── QUICKSTART.md              # Quick start guide
├── LICENSE                    # MIT License
├── .gitignore                # Git ignore rules
└── PROJECT_STRUCTURE.md       # This file
```

## File Descriptions

### Root Level

- **app.py**: Main Streamlit application. This is the entry point for running the web application.
- **requirements.txt**: Lists all Python package dependencies with version constraints.
- **setup.py**: Allows the package to be installed via pip.
- **README.md**: Comprehensive project documentation including features, installation, and usage.
- **QUICKSTART.md**: Quick reference guide for getting started.
- **LICENSE**: MIT License file.
- **.gitignore**: Specifies files and directories to ignore in version control.

### visualizer/ Package

- **__init__.py**: Package initialization file that exports all public functions and classes for easy importing.
- **model_loader.py**: Handles loading and caching of transformer models (BERT, Llama, etc.).
- **attention_utils.py**: Core attention computation functions including attention rollout, token contribution, and entropy.
- **attention_visualizer.py**: Creates interactive Plotly visualizations for attention heatmaps and head similarity.
- **head_analysis.py**: Analyzes attention head similarity and identifies redundant heads for pruning.
- **qkv_extractor.py**: Extracts Query, Key, and Value vectors from transformer models for analysis.
- **transformer_visualizer.py**: Creates visualizations of transformer architecture based on "Attention is All You Need" paper.
- **explainability.py**: Implements explainability features including coreference resolution and entity relationship detection.

## Import Structure

The package is designed to be imported easily:

```python
from visualizer import (
    load_model,
    plot_attention,
    token_contribution,
    explain_sentence,
    # ... etc
)
```

Or import specific modules:

```python
from visualizer import model_loader
from visualizer import explainability
```

## Running the Application

From the root directory:

```bash
streamlit run app.py
```

## Installation as Package

The package can be installed in development mode:

```bash
pip install -e .
```

This allows importing the `visualizer` package from anywhere.

## Notes

- All core functionality is in the `visualizer/` package directory
- The main application (`app.py`) imports from the package
- This structure allows the code to be used both as a standalone app and as an importable package
- Future tests should be added in a `tests/` directory at the root level


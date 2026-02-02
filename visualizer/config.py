"""
Central configuration for Transformer Explainability Lab.

Holds model configs and environment loading. Single source of truth for
settings used by model_loader, app, and explainability.
"""

import os
from typing import Any

# -----------------------------------------------------------------------------
# Model configurations
# -----------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
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

SUPPORTED_MODEL_TYPES = frozenset(MODEL_CONFIGS.keys())


def get_model_config(model_type: str) -> dict[str, Any]:
    """
    Return config dict for a supported model type.

    Args:
        model_type: One of "bert" or "llama".

    Returns:
        Dict with keys: model_name, max_layers, max_heads.

    Raises:
        ValueError: If model_type is not supported.
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(
            f"Unsupported model type: {model_type!r}. "
            f"Supported: {sorted(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[model_type].copy()


# -----------------------------------------------------------------------------
# Environment (e.g. API keys)
# -----------------------------------------------------------------------------

def load_env_from_project_root(*, override: bool = True) -> None:
    """
    Load .env from the project root (parent of visualizer/).

    Safe to call multiple times. No-op if python-dotenv is not installed.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(root, ".env"), override=override)


def get_env(key: str, default: str = "") -> str:
    """Get environment variable, stripped. Never returns None."""
    return (os.environ.get(key) or default).strip()

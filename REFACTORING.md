# Refactoring Summary

This document summarizes the FAANG-style refactor applied to Transformer Explainability Lab: what changed, why, and suggested next steps.

## 1. Code Quality

| Area | Before | After |
|------|--------|--------|
| **Naming** | Mixed; some globals (`MODEL_CONFIGS`) duplicated | Single source of truth in `config.py`; `get_model_config()`, `get_env()` |
| **Types** | Sparse type hints | Consistent type hints (e.g. `Tuple[PreTrainedTokenizer, PreTrainedModel]`, `dict[str, Any]`) |
| **Docstrings** | Some present | Module and key functions documented (what, args, returns, raises) |
| **Duplication** | `MODEL_CONFIGS` and `.env` loading in multiple places; three `nlp = spacy.load(...)` blocks in explainability | Central config; lazy spacy loader `_get_spacy_nlp()` |

## 2. Architecture

| Change | Why |
|--------|-----|
| **`visualizer/config.py`** | Single place for model configs and env. App and explainability use `config.get_env("GROQ_API_KEY")`; model_loader uses `config.get_model_config()`. |
| **Model loading** | `model_loader` imports config and splits BERT/LLama into `_load_bert` / `_load_llama`. Streamlit cache applied at module level when Streamlit is available so tests can run without it. |
| **Explainability** | Groq client and spacy model are lazy (no load at import). Removed debug `print`; use `logger`. Env via `config.load_env_from_project_root()` and `config.get_env()`. |
| **App** | Loads env via `config.load_env_from_project_root()` before other imports; uses `config.get_env()` for Groq key. No duplicate dotenv block. |

## 3. Error Handling & Logging

- **Logging**: `logger = logging.getLogger(__name__)` in model_loader, explainability, app. Explainability uses `logger.exception` / `logger.warning` for Groq and spacy.
- **Exceptions**: Config and model_loader raise `ValueError` for bad input, `RuntimeError` for load failures. Errors are chained (`raise ... from e`).
- **Failures**: Missing Groq key or spacy model yields clear messages; tests can import the package without Groq or spacy model installed.

## 4. Performance & Reliability

- **Caching**: Model load remains cached via `st.cache_resource` when run from Streamlit; no change in behavior.
- **Spacy**: Loaded once on first use via `_get_spacy_nlp()` instead of three module-level loads.
- **Config**: `get_model_config()` returns a copy so callers cannot mutate global config.

## 5. Security & Inputs

- **Secrets**: No hardcoded keys; `.env` and `get_env()` only.
- **Validation**: App keeps `validate_inputs()` for text, layer, head; config validates model type.

## 6. Testing

- **`tests/`** added with pytest:
  - `test_config.py`: `get_model_config()` (bert/llama, copy, unsupported), `get_env()` (default, strip).
  - `test_attention_utils.py`: `attention_rollout` (shape, single layer), `token_contribution` (length, remove_cls/sep, empty tokens), `attention_entropy` (scalar).
  - `test_head_analysis.py`: `head_similarity` (shape, layer bounds), `prune_heads` (redundant pairs, threshold).
- **Fixtures**: `conftest.py` provides small dummy attention tensors and token lists so tests are fast and deterministic.
- **No network**: Tests use synthetic data; no model download or Streamlit.

## 7. Documentation

- **README**: Updated project structure, added Architecture and Testing sections, Groq `.env` note, corrected clone path.
- **REFACTORING.md**: This file (rationale and before/after).

---

## Suggested Next Steps for Scaling

1. **CI**: Add a GitHub Actions (or similar) workflow to run `pytest tests/` on push/PR.
2. **Coverage**: Run `pytest --cov=visualizer` and add a coverage threshold; cover `qkv_extractor` and critical explainability paths with mocks.
3. **Integration tests**: Optional: one or two tests that run `streamlit run app.py` in headless mode and assert the app loads (e.g. with `streamlit run ... --server.headless true` and a simple HTTP check).
4. **Type checking**: Add `py.typed` and run `pyright` or `mypy` in CI.
5. **Logging level**: Allow configuring log level via env (e.g. `LOG_LEVEL=DEBUG`) in app and config.
6. **More models**: When adding new model types, extend `config.MODEL_CONFIGS` and model_loader only; keep config as the single source of truth.

# CLAUDE.md - AI Assistant Instructions

## Project Overview

**Inference Server MVP** - An OpenRouter-compatible API wrapper for vLLM inference with LMCache KV offloading.

- **Version:** 0.3.0
- **Language:** Python 3.10+
- **Framework:** FastAPI + Pydantic
- **Backend:** vLLM with optional LMCache

## Quick Start Commands

```bash
# Install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn pydantic pydantic-settings httpx

# Run server (mock mode - no GPU needed)
uvicorn src.main:app --host 0.0.0.0 --port 8000

# Run tests
pip install pytest pytest-asyncio
pytest tests/ -v

# Lint and type check
pip install ruff mypy
ruff check src/ tests/ --fix
ruff format src/ tests/
mypy src/ --ignore-missing-imports
```

## Architecture

```
src/
├── main.py         # FastAPI app, endpoints, lifespan
├── config.py       # Settings via pydantic-settings
├── models.py       # Pydantic request/response models
├── vllm_client.py  # Async HTTP client for vLLM
└── __init__.py     # Version info

tests/
├── conftest.py     # Pytest fixtures
├── test_*.py       # Test modules (140 tests)
```

## Key Files to Understand

1. **src/main.py** - Entry point, 3 endpoints:
   - `GET /health` - Health check
   - `GET /api/v1/models` - Model discovery
   - `POST /api/v1/chat/completions` - Chat API

2. **src/config.py** - All configuration via environment variables

3. **src/vllm_client.py** - `VLLMClient` (real) and `MockVLLMClient` (testing)

4. **src/models.py** - OpenAI-compatible Pydantic models

## Code Patterns

### Dual Import Pattern (main.py)
```python
# Supports both `uvicorn src.main:app` and `python src/main.py`
try:
    from .config import get_settings
except ImportError:
    from config import get_settings
```

### Lifespan Pattern (main.py)
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize client
    app.state.vllm_client = VLLMClient(...)
    yield
    # Shutdown: Cleanup
    await app.state.vllm_client.close()
```

### Lazy Client Initialization (vllm_client.py)
```python
async def _get_client(self) -> httpx.AsyncClient:
    if self._client is None or self._client.is_closed:
        self._client = httpx.AsyncClient(...)
    return self._client
```

## Configuration

All settings are in `src/config.py`. Key ones:

| Variable | Env Var | Default | Purpose |
|----------|---------|---------|---------|
| `vllm_base_url` | VLLM_BASE_URL | http://localhost:8080 | vLLM server |
| `model_id` | MODEL_ID | your-org/your-model | Model identifier |
| `max_context_length` | MAX_CONTEXT_LENGTH | 131072 | Context window |
| `price_per_prompt_token` | PRICE_PER_PROMPT_TOKEN | 0.000008 | Pricing |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_chat_completions.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

Tests use `MockVLLMClient` which echoes user messages.

## Common Tasks

### Adding a New Endpoint
1. Define Pydantic models in `src/models.py`
2. Add endpoint in `src/main.py`
3. Add tests in `tests/`

### Adding a Configuration Option
1. Add attribute to `Settings` class in `src/config.py`
2. Document the env var name
3. Update `.env.example`

### Modifying Response Format
1. Update models in `src/models.py`
2. Update response construction in `src/main.py`
3. Update tests

## Important Notes

- **Request counters are NOT thread-safe** - For high concurrency, use atomic counters
- **CORS is permissive** (`allow_origins=["*"]`) - Tighten for production
- **Rate limiting is NOT implemented** - Config exists but logic missing
- **Metrics endpoint NOT implemented** - Config exists but endpoint missing

## Error Handling

All errors return OpenAI-compatible format:
```json
{
  "error": {
    "message": "Error description",
    "type": "api_error",
    "code": 500
  }
}
```

## SSE Streaming Format

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...}\n\n
: keep-alive\n\n
data: [DONE]\n\n
```

## Dependencies

**Production:**
- fastapi, uvicorn, pydantic, pydantic-settings, httpx

**Development:**
- pytest, pytest-asyncio, pytest-cov, ruff, mypy

## Related Documentation

- `docs/ENTERPRISE-INFRASTRUCTURE.md` - Weka/enterprise deployment research
- `docs/DEPLOYMENT.md` - v0.3+ deployment roadmap
- `README.md` - User-facing documentation

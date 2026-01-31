# Inference Server MVP

> **Version 0.3.0** | OpenRouter-compatible inference endpoint with vLLM + LMCache

[![CI](https://github.com/CyberBerto/inference-server-mvp/actions/workflows/ci.yaml/badge.svg)](https://github.com/CyberBerto/inference-server-mvp/actions/workflows/ci.yaml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OpenAI-compatible API wrapper for vLLM with LMCache KV offloading. Designed for fast deployment on RunPod with a clear upgrade path to enterprise infrastructure.

## Features

- **OpenAI-compatible API** - Drop-in replacement for OpenAI's chat completions
- **SSE Streaming** - Server-Sent Events with keep-alive for long requests
- **OpenRouter Integration** - Model discovery, pricing, and health endpoints
- **vLLM Backend** - High-performance inference with continuous batching
- **LMCache Support** - KV cache offloading for 100K+ token contexts
- **140 Tests** - Comprehensive test suite with OpenRouter compatibility tests

## Quick Start

### 1. Install

```bash
git clone https://github.com/CyberBerto/inference-server-mvp.git
cd inference-server-mvp
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run (Mock Mode)

```bash
# Start server (no GPU needed - uses mock client)
uvicorn src.main:app --host 0.0.0.0 --port 8000

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/models
```

### 3. Run (With vLLM)

```bash
# Terminal 1: Start vLLM
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8080

# Terminal 2: Start API
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with uptime, error rate, vLLM status |
| `/api/v1/models` | GET | Model discovery for OpenRouter |
| `/api/v1/chat/completions` | POST | OpenAI-compatible chat completions |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |

### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-org/your-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "stream": false
  }'
```

## Configuration

Copy `.env.example` to `.env` and configure:

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_BASE_URL` | `http://localhost:8080` | vLLM server URL |
| `MODEL_ID` | `your-org/your-model` | OpenRouter model identifier |
| `MODEL_DISPLAY_NAME` | `Your Model` | Human-readable name |
| `MAX_CONTEXT_LENGTH` | `131072` | Maximum context (tokens) |
| `PRICE_PER_PROMPT_TOKEN` | `0.000008` | Cost per input token (USD) |
| `PRICE_PER_COMPLETION_TOKEN` | `0.000024` | Cost per output token (USD) |

See [docs/SPECIFICATION.md](docs/SPECIFICATION.md) for complete configuration reference.

## Project Structure

```
inference-server-mvp/
├── src/
│   ├── main.py          # FastAPI application & endpoints
│   ├── config.py        # Environment configuration
│   ├── models.py        # Pydantic request/response models
│   └── vllm_client.py   # Async HTTP client for vLLM
├── tests/               # 140 tests (pytest)
├── configs/             # vLLM and LMCache configs
├── docs/                # Detailed documentation
│   ├── SPECIFICATION.md # Technical specification
│   └── ENTERPRISE-INFRASTRUCTURE.md # Enterprise research
├── CLAUDE.md            # AI assistant instructions
└── DEPLOYMENT.md        # Deployment guide
```

## Testing

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Lint and type check
ruff check src/ tests/
mypy src/ --ignore-missing-imports
```

## Deployment

### RunPod (Recommended)

1. Deploy H100 80GB pod ($1.99-2.69/hr)
2. Start vLLM with your model
3. Start this API server
4. Submit to [OpenRouter](https://openrouter.ai/how-to-list)

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

### Docker

```bash
# With GPU
docker-compose up

# Development (mock backend)
docker-compose -f docker-compose.yaml -f docker-compose.dev.yaml up
```

## OpenRouter Integration

This server implements the OpenRouter provider API:

- **Model Discovery**: `/api/v1/models` with pricing metadata
- **OpenAI Format**: Compatible request/response structure
- **SSE Streaming**: With keep-alive for long requests
- **Health Monitoring**: For uptime tracking (95%+ required)

## Documentation

- [CLAUDE.md](CLAUDE.md) - Instructions for AI assistants
- [docs/SPECIFICATION.md](docs/SPECIFICATION.md) - Complete technical spec
- [docs/ENTERPRISE-INFRASTRUCTURE.md](docs/ENTERPRISE-INFRASTRUCTURE.md) - Enterprise research (Weka, CoreWeave)
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide

## Changelog

### v0.3.0 (Current)
- Fixed lint and type check errors for CI
- Added comprehensive documentation (CLAUDE.md, SPECIFICATION.md)
- Added enterprise infrastructure research
- All 140 tests passing

### v0.2.0
- Comprehensive code review and annotations
- Expanded test suite with OpenRouter compatibility tests
- Added interactive API documentation

### v0.1.0
- Initial release with OpenAI-compatible API

## License

MIT

## Links

- **Repository**: https://github.com/CyberBerto/inference-server-mvp
- **vLLM**: https://github.com/vllm-project/vllm
- **LMCache**: https://github.com/LMCache/LMCache
- **OpenRouter**: https://openrouter.ai

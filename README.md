# Inference Server MVP

> **Ship an OpenRouter-compatible inference endpoint in 2-5 days for <$200/month**

OpenAI-compatible API wrapper for vLLM with LMCache KV offloading. Designed for fast deployment on RunPod with a clear upgrade path to enterprise WEKA infrastructure.

## Quick Start

### 1. Clone and Configure

```bash
git clone https://github.com/YOUR_ORG/inference-server-mvp.git
cd inference-server-mvp
cp .env.example .env
# Edit .env with your model details
```

### 2. Deploy on RunPod (Recommended)

1. Create a [RunPod account](https://runpod.io)
2. Deploy an H100 80GB pod ($1.99-2.69/hr)
3. Use the vLLM template or run manually:

```bash
# SSH into your RunPod instance
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu124
pip install vllm==0.11.0 lmcache==0.3.9

# Start vLLM with LMCache
LMCACHE_CONFIG_FILE=configs/lmcache.yaml \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8080 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.9 \
    --enable-chunked-prefill
```

4. In a second terminal, start the API:

```bash
pip install -r requirements.txt
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### 3. Test Your Endpoint

```bash
curl http://localhost:8000/health

curl http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-org/your-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### 4. Submit to OpenRouter

1. Go to [openrouter.ai/how-to-list](https://openrouter.ai/how-to-list)
2. Submit your endpoint URL
3. Wait 1-4 weeks for approval

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   OpenRouter    │────▶│   FastAPI API   │────▶│  vLLM + LMCache │
│   (traffic)     │     │   (port 8000)   │     │   (port 8080)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

- **FastAPI API**: OpenAI-compatible wrapper with health checks and metrics
- **vLLM**: High-performance inference engine with continuous batching
- **LMCache**: KV cache offloading to CPU RAM for 100K+ contexts

## Project Structure

```
inference-server-mvp/
├── src/
│   ├── main.py          # FastAPI application
│   ├── config.py        # Environment configuration
│   ├── models.py        # Pydantic request/response models
│   └── vllm_client.py   # vLLM HTTP client
├── tests/
│   ├── test_health.py              # Health endpoint tests
│   ├── test_chat_completions.py    # Chat API tests
│   ├── test_vllm_client.py         # Client unit tests
│   ├── test_config.py              # Configuration tests
│   └── test_openrouter_compatibility.py  # OpenRouter format tests
├── configs/
│   ├── lmcache.yaml     # LMCache configuration
│   └── vllm_args.yaml   # vLLM server arguments
├── .github/workflows/
│   ├── ci.yaml          # Test + lint pipeline
│   └── deploy.yaml      # Docker build + push
├── Dockerfile           # API container
├── Dockerfile.vllm      # vLLM container (GPU)
└── docker-compose.yaml  # Local development
```

## Configuration

All settings via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_BASE_URL` | `http://localhost:8080` | vLLM backend URL |
| `MODEL_ID` | `your-org/your-model` | OpenRouter model identifier |
| `MAX_CONTEXT_LENGTH` | `131072` | Maximum context (tokens) |
| `PRICE_PER_PROMPT_TOKEN` | `0.000008` | Pricing for OpenRouter |
| `PRICE_PER_COMPLETION_TOKEN` | `0.000024` | Pricing for OpenRouter |

## Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_openrouter_compatibility.py -v
```

## Deployment Options

### RunPod Serverless (Recommended for MVP)

- Sub-200ms cold starts
- Pay per request
- Auto-scaling included

### RunPod Pods

- Persistent GPU instances
- Better for consistent traffic
- $1.99/hr for H100

### Docker Compose (Development)

```bash
# With GPU (requires nvidia-docker)
docker-compose up

# Without GPU (mock client)
docker-compose -f docker-compose.yaml -f docker-compose.dev.yaml up
```

## API Endpoints

### Health Check

```
GET /health
```

Returns uptime, request count, error rate, and vLLM connection status.

### Model Discovery

```
GET /api/v1/models
```

Returns model metadata for OpenRouter discovery.

### Chat Completions

```
POST /api/v1/chat/completions
```

OpenAI-compatible chat completions with streaming support.

## OpenRouter Integration

This server implements the OpenRouter provider API:

1. **Model discovery** at `/api/v1/models` with pricing metadata
2. **OpenAI-compatible** request/response format
3. **SSE streaming** with keep-alive for long requests
4. **Health endpoint** for uptime monitoring

### Uptime Requirements

| Uptime | Routing Priority |
|--------|------------------|
| 95%+ | Normal |
| 80-94% | Degraded |
| <80% | Fallback only |

## Upgrade Path

When you outgrow MVP:

1. **Multi-GPU**: Add tensor parallelism in vLLM
2. **Multi-Node**: Use Lambda Labs 1-Click Clusters
3. **Enterprise**: See [inference-server-enterprise](../inference-server-enterprise/) for WEKA/CoreWeave path

## Upstream Dependencies

| Project | Version | Repository |
|---------|---------|------------|
| vLLM | 0.11.x | [vllm-project/vllm](https://github.com/vllm-project/vllm) |
| LMCache | 0.3.9 | [LMCache/LMCache](https://github.com/LMCache/LMCache) |
| FastAPI | 0.109+ | [tiangolo/fastapi](https://github.com/tiangolo/fastapi) |

## Cost Estimate

| Configuration | Hourly | Monthly (24/7) |
|---------------|--------|----------------|
| 1x H100 (RunPod) | $1.99 | ~$1,430 |
| 1x A100 80GB | $1.64 | ~$1,180 |
| Serverless (per request) | Variable | $50-500 |

## License

MIT

---

**Ready for enterprise scale?** See [inference-server-enterprise](../inference-server-enterprise/) for WEKA Token Warehouse integration.

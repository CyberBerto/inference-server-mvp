# Technical Specification - Inference Server MVP v0.3.0

## Table of Contents
1. [Classes](#1-classes)
2. [Functions](#2-functions)
3. [Pydantic Models](#3-pydantic-models)
4. [Configuration Variables](#4-configuration-variables)
5. [Environment Variables](#5-environment-variables)
6. [API Endpoints](#6-api-endpoints)
7. [Constants and Magic Values](#7-constants-and-magic-values)
8. [Internal State](#8-internal-state)

---

## 1. Classes

### 1.1 Settings (`config.py`)
**Purpose:** Server and model configuration management

| Attribute | Type | Default | Env Var |
|-----------|------|---------|---------|
| `host` | `str` | `"0.0.0.0"` | HOST |
| `port` | `int` | `8000` | PORT |
| `workers` | `int` | `1` | WORKERS |
| `debug` | `bool` | `False` | DEBUG |
| `vllm_base_url` | `str` | `"http://localhost:8080"` | VLLM_BASE_URL |
| `request_timeout` | `float` | `300.0` | REQUEST_TIMEOUT |
| `model_id` | `str` | `"your-org/your-model"` | MODEL_ID |
| `model_display_name` | `str` | `"Your Model Display Name"` | MODEL_DISPLAY_NAME |
| `organization_id` | `str` | `"your-org"` | ORGANIZATION_ID |
| `max_context_length` | `int` | `131072` | MAX_CONTEXT_LENGTH |
| `quantization` | `str` | `"fp16"` | QUANTIZATION |
| `supported_features` | `list[str]` | `["tools", "json_mode", "streaming"]` | SUPPORTED_FEATURES |
| `price_per_prompt_token` | `str` | `"0.000008"` | PRICE_PER_PROMPT_TOKEN |
| `price_per_completion_token` | `str` | `"0.000024"` | PRICE_PER_COMPLETION_TOKEN |
| `lmcache_enabled` | `bool` | `True` | LMCACHE_ENABLED |
| `lmcache_config_path` | `str` | `"configs/lmcache.yaml"` | LMCACHE_CONFIG_PATH |
| `rate_limit_requests_per_minute` | `int` | `60` | RATE_LIMIT_REQUESTS_PER_MINUTE |
| `rate_limit_tokens_per_minute` | `int` | `100000` | RATE_LIMIT_TOKENS_PER_MINUTE |
| `enable_metrics` | `bool` | `True` | ENABLE_METRICS |
| `metrics_port` | `int` | `9090` | METRICS_PORT |

**Config:**
```python
model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
)
```

---

### 1.2 VLLMClient (`vllm_client.py`)
**Purpose:** Async HTTP client for vLLM backend

| Attribute | Type | Default |
|-----------|------|---------|
| `base_url` | `str` | `"http://localhost:8080"` |
| `timeout` | `float` | `300.0` |
| `_client` | `httpx.AsyncClient \| None` | `None` |

**Methods:**

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `__init__` | `base_url: str`, `timeout: float` | `None` | Initialize client |
| `_get_client` | - | `httpx.AsyncClient` | Lazy init HTTP client |
| `close` | - | `None` | Close connections |
| `is_healthy` | - | `bool` | Check vLLM health |
| `generate` | `messages`, `model`, `max_tokens`, `temperature`, `top_p`, `stop`, `**kwargs` | `dict[str, Any]` | Non-streaming completion |
| `generate_stream` | `messages`, `model`, `max_tokens`, `temperature`, `top_p`, `stop`, `keep_alive_interval`, `**kwargs` | `AsyncGenerator[dict, None]` | Streaming completion |

---

### 1.3 MockVLLMClient (`vllm_client.py`)
**Purpose:** Mock client for testing without GPU

Inherits from `VLLMClient`. Overrides:
- `is_healthy()` → Always returns `True`
- `generate()` → Returns `"Mock response to: {user_message[:50]}"`
- `generate_stream()` → Yields response word-by-word

---

## 2. Functions

### 2.1 Configuration

| Function | Location | Returns | Purpose |
|----------|----------|---------|---------|
| `get_settings()` | `config.py` | `Settings` | Get cached settings (uses `@lru_cache`) |

---

### 2.2 Application Lifecycle

| Function | Location | Parameters | Returns | Purpose |
|----------|----------|------------|---------|---------|
| `lifespan` | `main.py` | `app: FastAPI` | `AsyncGenerator` | Manage startup/shutdown |

**Startup Actions:**
- Initialize `VLLMClient`
- Store settings in `app.state`
- Set `start_time`, `request_count=0`, `error_count=0`

**Shutdown Actions:**
- Call `await app.state.vllm_client.close()`

---

### 2.3 Endpoint Handlers

| Function | Location | Parameters | Returns |
|----------|----------|------------|---------|
| `health_check` | `main.py` | - | `dict` |
| `list_models` | `main.py` | - | `dict` |
| `chat_completions` | `main.py` | `request: ChatCompletionRequest` | `ChatCompletionResponse \| StreamingResponse` |

---

### 2.4 Internal Helpers

| Function | Location | Parameters | Returns | Purpose |
|----------|----------|------------|---------|---------|
| `generate_chat_completion` | `main.py` | `request`, `request_id` | `ChatCompletionResponse` | Non-streaming generation |
| `stream_chat_completion` | `main.py` | `request`, `request_id` | `AsyncGenerator[str, None]` | SSE streaming |

---

### 2.5 Error Handlers

| Function | Location | Handles | Returns |
|----------|----------|---------|---------|
| `http_exception_handler` | `main.py` | `HTTPException` | `JSONResponse` |
| `general_exception_handler` | `main.py` | `Exception` | `JSONResponse` |

---

## 3. Pydantic Models

### 3.1 Request Models (`models.py`)

#### Message
| Field | Type | Default | Validation |
|-------|------|---------|------------|
| `role` | `Literal["system", "user", "assistant", "tool"]` | - | Required |
| `content` | `str \| None` | `None` | - |
| `name` | `str \| None` | `None` | - |
| `tool_calls` | `list[dict[str, Any]] \| None` | `None` | - |
| `tool_call_id` | `str \| None` | `None` | - |

#### ToolFunction
| Field | Type | Default |
|-------|------|---------|
| `name` | `str` | Required |
| `description` | `str \| None` | `None` |
| `parameters` | `dict[str, Any] \| None` | `None` |

#### Tool
| Field | Type | Default |
|-------|------|---------|
| `type` | `Literal["function"]` | `"function"` |
| `function` | `ToolFunction` | Required |

#### ResponseFormat
| Field | Type | Default |
|-------|------|---------|
| `type` | `Literal["text", "json_object"]` | `"text"` |

#### ChatCompletionRequest
| Field | Type | Default | Validation |
|-------|------|---------|------------|
| `model` | `str` | Required | - |
| `messages` | `list[Message]` | Required | - |
| `max_tokens` | `int \| None` | `4096` | `ge=1, le=131072` |
| `temperature` | `float \| None` | `0.7` | `ge=0.0, le=2.0` |
| `top_p` | `float \| None` | `1.0` | `ge=0.0, le=1.0` |
| `top_k` | `int \| None` | `None` | `ge=1` |
| `frequency_penalty` | `float \| None` | `0.0` | `ge=-2.0, le=2.0` |
| `presence_penalty` | `float \| None` | `0.0` | `ge=-2.0, le=2.0` |
| `repetition_penalty` | `float \| None` | `1.0` | `ge=0.0` |
| `stop` | `str \| list[str] \| None` | `None` | - |
| `stream` | `bool \| None` | `False` | - |
| `tools` | `list[Tool] \| None` | `None` | - |
| `tool_choice` | `str \| dict[str, Any] \| None` | `None` | - |
| `response_format` | `ResponseFormat \| None` | `None` | - |
| `user` | `str \| None` | `None` | - |
| `best_of` | `int \| None` | `None` | - |
| `use_beam_search` | `bool \| None` | `False` | - |
| `skip_special_tokens` | `bool \| None` | `True` | - |

---

### 3.2 Response Models (`models.py`)

#### UsageInfo
| Field | Type |
|-------|------|
| `prompt_tokens` | `int` |
| `completion_tokens` | `int` |
| `total_tokens` | `int` |

#### Choice
| Field | Type | Default |
|-------|------|---------|
| `index` | `int` | Required |
| `message` | `Message` | Required |
| `finish_reason` | `str \| None` | `None` |
| `logprobs` | `dict[str, Any] \| None` | `None` |

#### ChatCompletionResponse
| Field | Type | Default |
|-------|------|---------|
| `id` | `str` | Required |
| `object` | `Literal["chat.completion"]` | `"chat.completion"` |
| `created` | `int` | Required |
| `model` | `str` | Required |
| `choices` | `list[Choice]` | Required |
| `usage` | `UsageInfo \| None` | `None` |
| `system_fingerprint` | `str \| None` | `None` |

---

### 3.3 Streaming Models (`models.py`)

#### DeltaMessage
| Field | Type | Default |
|-------|------|---------|
| `role` | `str \| None` | `None` |
| `content` | `str \| None` | `None` |
| `tool_calls` | `list[dict[str, Any]] \| None` | `None` |

#### StreamChoice
| Field | Type | Default |
|-------|------|---------|
| `index` | `int` | Required |
| `delta` | `DeltaMessage` | Required |
| `finish_reason` | `str \| None` | `None` |

#### ChatCompletionChunk
| Field | Type | Default |
|-------|------|---------|
| `id` | `str` | Required |
| `object` | `Literal["chat.completion.chunk"]` | `"chat.completion.chunk"` |
| `created` | `int` | Required |
| `model` | `str` | Required |
| `choices` | `list[StreamChoice]` | Required |

---

### 3.4 Model Discovery Models (`models.py`)

#### ModelPricing
| Field | Type |
|-------|------|
| `prompt` | `str` |
| `completion` | `str` |

#### ModelInfo
| Field | Type | Default |
|-------|------|---------|
| `id` | `str` | Required |
| `object` | `str` | `"model"` |
| `created` | `int` | Required |
| `owned_by` | `str` | Required |
| `name` | `str` | Required |
| `context_length` | `int` | Required |
| `pricing` | `ModelPricing` | Required |
| `quantization` | `str` | Required |
| `supported_features` | `list[str]` | Required |

---

## 4. Configuration Variables

### 4.1 Load Priority
1. **Environment variables** (highest)
2. **`.env` file**
3. **Default values** (lowest)

### 4.2 Full Variable List

| Category | Variable | Type | Default | Implemented |
|----------|----------|------|---------|-------------|
| Server | `host` | str | 0.0.0.0 | ✅ |
| Server | `port` | int | 8000 | ✅ |
| Server | `workers` | int | 1 | ✅ |
| Server | `debug` | bool | False | ✅ |
| vLLM | `vllm_base_url` | str | http://localhost:8080 | ✅ |
| vLLM | `request_timeout` | float | 300.0 | ✅ |
| Model | `model_id` | str | your-org/your-model | ✅ |
| Model | `model_display_name` | str | Your Model Display Name | ✅ |
| Model | `organization_id` | str | your-org | ✅ |
| Model | `max_context_length` | int | 131072 | ✅ |
| Model | `quantization` | str | fp16 | ✅ |
| Model | `supported_features` | list[str] | [tools, json_mode, streaming] | ✅ |
| Pricing | `price_per_prompt_token` | str | 0.000008 | ✅ |
| Pricing | `price_per_completion_token` | str | 0.000024 | ✅ |
| LMCache | `lmcache_enabled` | bool | True | ⚠️ Config only |
| LMCache | `lmcache_config_path` | str | configs/lmcache.yaml | ⚠️ Config only |
| Rate Limit | `rate_limit_requests_per_minute` | int | 60 | ❌ Not implemented |
| Rate Limit | `rate_limit_tokens_per_minute` | int | 100000 | ❌ Not implemented |
| Metrics | `enable_metrics` | bool | True | ❌ Not implemented |
| Metrics | `metrics_port` | int | 9090 | ❌ Not implemented |

---

## 5. Environment Variables

### 5.1 Required for Production

| Variable | Example | Purpose |
|----------|---------|---------|
| `MODEL_ID` | `my-org/llama-3.1-70b` | Model identifier for OpenRouter |
| `VLLM_BASE_URL` | `http://vllm:8080` | vLLM server address |
| `PRICE_PER_PROMPT_TOKEN` | `0.000015` | Pricing for billing |
| `PRICE_PER_COMPLETION_TOKEN` | `0.000045` | Pricing for billing |

### 5.2 Optional

| Variable | Default | Purpose |
|----------|---------|---------|
| `HOST` | 0.0.0.0 | Bind address |
| `PORT` | 8000 | HTTP port |
| `REQUEST_TIMEOUT` | 300 | Timeout for long contexts |
| `MAX_CONTEXT_LENGTH` | 131072 | Advertised context window |
| `QUANTIZATION` | fp16 | Model precision |

### 5.3 External (vLLM)

| Variable | Purpose |
|----------|---------|
| `LMCACHE_CONFIG_FILE` | Path to LMCache YAML for vLLM |

---

## 6. API Endpoints

### 6.1 GET /health

**Purpose:** Health check for load balancers and OpenRouter

**Response:**
```json
{
  "status": "healthy",
  "uptime_seconds": 3600.0,
  "total_requests": 1000,
  "error_rate": 0.02,
  "vllm_connected": true
}
```

---

### 6.2 GET /api/v1/models

**Purpose:** Model discovery for OpenRouter

**Response:**
```json
{
  "object": "list",
  "data": [{
    "id": "your-org/your-model",
    "object": "model",
    "created": 1699000000,
    "owned_by": "your-org",
    "name": "Your Model Display Name",
    "context_length": 131072,
    "pricing": {"prompt": "0.000008", "completion": "0.000024"},
    "quantization": "fp16",
    "supported_features": ["tools", "json_mode", "streaming"]
  }]
}
```

---

### 6.3 POST /api/v1/chat/completions

**Purpose:** OpenAI-compatible chat completions

**Request:**
```json
{
  "model": "your-org/your-model",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 4096,
  "temperature": 0.7,
  "stream": false
}
```

**Non-Streaming Response:**
```json
{
  "id": "chatcmpl-abc123...",
  "object": "chat.completion",
  "created": 1699000000,
  "model": "your-org/your-model",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello!"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
}
```

**Streaming Response (SSE):**
```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":..., "model":"...","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":..., "model":"...","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

: keep-alive

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":..., "model":"...","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}

data: [DONE]
```

---

### 6.4 Error Responses

**Format:**
```json
{
  "error": {
    "message": "Error description",
    "type": "api_error",
    "code": 500
  }
}
```

**Status Codes:**
- `200` - Success
- `422` - Validation error
- `500` - Internal server error

---

## 7. Constants and Magic Values

### 7.1 Request ID
- **Format:** `chatcmpl-{uuid.hex[:24]}`
- **Example:** `chatcmpl-1a2b3c4d5e6f7g8h9i0j1k2l`
- **Length:** 33 characters

### 7.2 Timeouts
| Timeout | Value | Purpose |
|---------|-------|---------|
| Request | 300.0s | Long-context inference |
| Connect | 10.0s | Fail fast on connection issues |
| Health Check | 5.0s | Quick health checks |

### 7.3 Keep-Alive
- **Interval:** 15.0 seconds
- **Format:** `: keep-alive\n\n`

### 7.4 Generation Parameter Ranges
| Parameter | Min | Max | Default |
|-----------|-----|-----|---------|
| max_tokens | 1 | 131072 | 4096 |
| temperature | 0.0 | 2.0 | 0.7 |
| top_p | 0.0 | 1.0 | 1.0 |
| frequency_penalty | -2.0 | 2.0 | 0.0 |
| presence_penalty | -2.0 | 2.0 | 0.0 |
| repetition_penalty | 0.0 | ∞ | 1.0 |

### 7.5 Quantization Options
`fp32`, `fp16`, `bf16`, `fp8`, `int8`, `int4`, `awq`, `gptq`

### 7.6 Finish Reasons
`stop`, `length`, `tool_calls`, `content_filter`

---

## 8. Internal State

### 8.1 Application State (`app.state`)

| Variable | Type | Initialized |
|----------|------|-------------|
| `vllm_client` | `VLLMClient` | Startup |
| `settings` | `Settings` | Startup |
| `start_time` | `float` | Startup |
| `request_count` | `int` | Startup (0) |
| `error_count` | `int` | Startup (0) |

### 8.2 Derived Metrics

```python
uptime_seconds = time.time() - app.state.start_time
error_rate = error_count / request_count if request_count > 0 else 0.0
```

---

## Appendix A: Test Coverage

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_config.py` | 16 | Configuration |
| `test_models.py` | 42 | Pydantic models |
| `test_health.py` | 10 | Health endpoint |
| `test_chat_completions.py` | 15 | Chat API |
| `test_vllm_client.py` | 13 | VLLMClient |
| `test_error_handling.py` | 20 | Error cases |
| `test_openrouter_compatibility.py` | 24 | OpenRouter format |
| **Total** | **140** | - |

---

## Appendix B: Dependencies

**Production:**
```
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
httpx>=0.26.0
```

**Development:**
```
pytest>=7.4.0
pytest-asyncio>=0.23.0
pytest-cov>=4.1.0
ruff>=0.1.0
mypy>=1.8.0
```

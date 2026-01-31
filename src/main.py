"""
Inference Server MVP - OpenRouter-compatible API.

FastAPI wrapper for vLLM with LMCache KV offloading. This module provides
the main application entry point and all HTTP endpoints.

Architecture Overview:
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │ OpenRouter  │────▶│  FastAPI    │────▶│    vLLM     │
    │  (client)   │     │  (this app) │     │  (backend)  │
    └─────────────┘     └─────────────┘     └─────────────┘

Endpoints:
    GET  /health                  - Health check with uptime/error metrics
    GET  /api/v1/models           - Model discovery for OpenRouter integration
    POST /api/v1/chat/completions - OpenAI-compatible chat completions

Features:
    - Server-Sent Events (SSE) streaming with keep-alive pings
    - Request counting and error rate tracking for observability
    - Graceful shutdown with client cleanup

Security Considerations:
    - CORS is configured permissively (allow_origins=["*"]) for OpenRouter
      callbacks. Restrict origins in production if serving directly to browsers.
    - Request counting uses simple integers; not atomic. For high-concurrency
      production, consider using thread-safe counters or external metrics.

Version: 0.3.0
License: MIT
"""

import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# Dual import pattern supports both package and direct execution modes:
#   - Package: `python -m uvicorn src.main:app` (uses relative imports)
#   - Direct:  `python src/main.py` (uses absolute imports for debugging)
# mypy: disable-error-code="no-redef"
try:
    from .config import get_settings
    from .models import (
        ChatCompletionRequest,
        ChatCompletionResponse,
        Choice,
        Message,
        UsageInfo,
    )
    from .vllm_client import VLLMClient
except ImportError:
    from config import get_settings
    from models import (
        ChatCompletionRequest,
        ChatCompletionResponse,
        Choice,
        Message,
        UsageInfo,
    )
    from vllm_client import VLLMClient


# =============================================================================
# Application Lifespan Management
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown lifecycle.

    Startup:
        - Load configuration from environment variables
        - Initialize async HTTP client for vLLM communication
        - Set up request counters and start time for metrics

    Shutdown:
        - Gracefully close HTTP client connections
        - Release any held resources

    Note:
        The vLLM client uses lazy initialization internally, so the actual
        HTTP connection is established on first request, not at startup.
    """
    settings = get_settings()

    # Initialize vLLM client with configured timeout
    # Timeout is set high (default 300s) to accommodate long-context inference
    app.state.vllm_client = VLLMClient(
        base_url=settings.vllm_base_url,
        timeout=settings.request_timeout,
    )
    app.state.settings = settings
    app.state.start_time = time.time()

    # Simple counters for observability
    # WARNING: These are not thread-safe for concurrent writes
    app.state.request_count = 0
    app.state.error_count = 0

    yield  # Application runs here

    # Cleanup: close HTTP client connections gracefully
    await app.state.vllm_client.close()


# =============================================================================
# FastAPI Application Instance
# =============================================================================

app = FastAPI(
    title="Inference Server MVP",
    description="OpenRouter-compatible inference API with vLLM + LMCache",
    version="0.3.0",
    lifespan=lifespan,
    # OpenAPI docs available at /docs (Swagger) and /redoc
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware configuration
# Required for OpenRouter callbacks and browser-based API explorers
# NOTE: In production, consider restricting allow_origins to specific domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permissive for OpenRouter compatibility
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health & Metadata Endpoints
# =============================================================================


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for load balancers and OpenRouter uptime monitoring.

    Returns server status including:
        - status: Always "healthy" if endpoint is reachable
        - uptime_seconds: Time since server start
        - total_requests: Count of chat completion requests
        - error_rate: Ratio of failed requests (0.0 to 1.0)
        - vllm_connected: Whether backend vLLM server is responding

    OpenRouter Uptime Requirements:
        - 95%+ uptime: Normal routing priority
        - 80-94% uptime: Degraded routing
        - <80% uptime: Fallback only

    Example Response:
        {
            "status": "healthy",
            "uptime_seconds": 3600.0,
            "total_requests": 1000,
            "error_rate": 0.02,
            "vllm_connected": true
        }
    """
    uptime = time.time() - app.state.start_time

    # Calculate error rate, avoiding division by zero
    error_rate = (
        app.state.error_count / app.state.request_count if app.state.request_count > 0 else 0.0
    )

    return {
        "status": "healthy",
        "uptime_seconds": round(uptime, 2),
        "total_requests": app.state.request_count,
        "error_rate": round(error_rate, 4),
        "vllm_connected": await app.state.vllm_client.is_healthy(),
    }


@app.get("/api/v1/models", tags=["Models"])
async def list_models():
    """
    OpenRouter model discovery endpoint.

    Returns model metadata including pricing and capabilities in OpenAI-compatible
    format. OpenRouter uses this endpoint to discover available models and display
    them in their model selector.

    Response follows OpenAI's /v1/models format with OpenRouter extensions:
        - pricing: Token costs for billing (as strings, e.g., "0.000008")
        - context_length: Maximum supported context window
        - supported_features: List of capabilities (tools, json_mode, streaming)

    Example Response:
        {
            "object": "list",
            "data": [{
                "id": "your-org/your-model",
                "object": "model",
                "name": "Your Model Display Name",
                "context_length": 131072,
                "pricing": {"prompt": "0.000008", "completion": "0.000024"},
                "supported_features": ["tools", "json_mode", "streaming"]
            }]
        }
    """
    settings = app.state.settings

    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_id,
                "object": "model",
                "created": int(app.state.start_time),
                "owned_by": settings.organization_id,
                "name": settings.model_display_name,
                "context_length": settings.max_context_length,
                "pricing": {
                    "prompt": settings.price_per_prompt_token,
                    "completion": settings.price_per_completion_token,
                },
                "quantization": settings.quantization,
                "supported_features": settings.supported_features,
            }
        ],
    }


# =============================================================================
# Chat Completions API
# =============================================================================


@app.post("/api/v1/chat/completions", tags=["Chat"])
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.

    Supports both streaming (SSE) and non-streaming responses. For long-context
    requests, streaming is recommended to avoid gateway timeouts.

    Args:
        request: ChatCompletionRequest with messages, model, and generation params

    Returns:
        - Non-streaming: ChatCompletionResponse with complete response
        - Streaming: Server-Sent Events stream of ChatCompletionChunks

    Request Format:
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

    Streaming Notes:
        - Each chunk is prefixed with "data: " per SSE spec
        - Stream ends with "data: [DONE]"
        - Keep-alive comments (": keep-alive") prevent proxy timeouts
    """
    # Increment request counter (not thread-safe, see module docstring)
    app.state.request_count += 1

    # Generate unique request ID in OpenAI format
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    try:
        if request.stream:
            # Return SSE stream for real-time token generation
            return StreamingResponse(
                stream_chat_completion(request, request_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Request-ID": request_id,
                },
            )
        else:
            # Return complete response after generation finishes
            return await generate_chat_completion(request, request_id)

    except Exception as e:
        # Track error for health metrics
        app.state.error_count += 1
        raise HTTPException(status_code=500, detail=str(e))


async def generate_chat_completion(
    request: ChatCompletionRequest,
    request_id: str,
) -> ChatCompletionResponse:
    """
    Generate a non-streaming chat completion response.

    Sends request to vLLM backend and waits for complete response before
    returning. Suitable for short responses; for long contexts, use streaming.

    Args:
        request: Validated chat completion request
        request_id: Unique identifier for this request

    Returns:
        ChatCompletionResponse with message, usage stats, and finish reason
    """
    response = await app.state.vllm_client.generate(
        messages=request.messages,
        model=request.model,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,
    )

    return ChatCompletionResponse(
        id=request_id,
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[
            Choice(
                index=0,
                message=Message(
                    role="assistant",
                    content=response["content"],
                ),
                finish_reason=response.get("finish_reason", "stop"),
            )
        ],
        usage=UsageInfo(
            prompt_tokens=response.get("prompt_tokens", 0),
            completion_tokens=response.get("completion_tokens", 0),
            total_tokens=response.get("total_tokens", 0),
        ),
    )


async def stream_chat_completion(
    request: ChatCompletionRequest,
    request_id: str,
) -> AsyncGenerator[str, None]:
    """
    Stream chat completion tokens via Server-Sent Events (SSE).

    Yields tokens as they're generated by vLLM, formatted per SSE specification.
    Includes periodic keep-alive comments to prevent proxy/gateway timeouts
    during long-running inference.

    Args:
        request: Validated chat completion request
        request_id: Unique identifier for this request

    Yields:
        SSE-formatted strings: "data: {json}\\n\\n" for content chunks,
        ": keep-alive\\n\\n" for connection maintenance,
        "data: [DONE]\\n\\n" to signal stream end

    SSE Format Reference:
        - Data lines: "data: {json payload}\\n\\n"
        - Comments: ": comment text\\n\\n" (ignored by clients, keeps connection alive)
        - Terminator: "data: [DONE]\\n\\n" (OpenAI convention)
    """
    async for chunk in app.state.vllm_client.generate_stream(
        messages=request.messages,
        model=request.model,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,
    ):
        # Format chunk as OpenAI-compatible SSE data
        data = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk.get("content", "")},
                    "finish_reason": chunk.get("finish_reason"),
                }
            ],
        }
        json_body = JSONResponse(content=data).body
        yield f"data: {json_body.decode() if isinstance(json_body, bytes) else json_body}\n\n"

        # Emit keep-alive comment to prevent proxy timeouts
        # vLLM client emits keep_alive=True periodically during long inference
        if chunk.get("keep_alive"):
            yield ": keep-alive\n\n"

    # Signal end of stream per OpenAI convention
    yield "data: [DONE]\n\n"


# =============================================================================
# Error Handlers
# =============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle HTTP exceptions with OpenAI-compatible error format.

    Returns structured error response matching OpenAI API error schema,
    which OpenRouter and compatible clients expect.

    Error Response Format:
        {
            "error": {
                "message": "Error description",
                "type": "api_error",
                "code": 400
            }
        }
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "api_error",
                "code": exc.status_code,
            }
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Catch-all handler for unexpected exceptions.

    Logs error for debugging while returning a safe, generic error message
    to clients. Increments error counter for health metrics.

    Security Note:
        Exception details are not exposed to clients to prevent information
        leakage. In production, consider logging to external service.
    """
    app.state.error_count += 1

    # TODO: Add structured logging here for production debugging
    # logger.exception("Unhandled exception", exc_info=exc)

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_error",
                "code": 500,
            }
        },
    )

"""
Inference Server MVP - OpenRouter-compatible API
FastAPI wrapper for vLLM with LMCache KV offloading
"""

import os
import time
import uuid
import asyncio
from typing import Optional, List, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx

try:
    from .config import Settings, get_settings
    from .vllm_client import VLLMClient
    from .models import (
        ChatCompletionRequest,
        ChatCompletionResponse,
        ModelInfo,
        UsageInfo,
        Choice,
        Message,
    )
except ImportError:
    from config import Settings, get_settings
    from vllm_client import VLLMClient
    from models import (
        ChatCompletionRequest,
        ChatCompletionResponse,
        ModelInfo,
        UsageInfo,
        Choice,
        Message,
    )


# Lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize vLLM client on startup, cleanup on shutdown."""
    settings = get_settings()
    app.state.vllm_client = VLLMClient(
        base_url=settings.vllm_base_url,
        timeout=settings.request_timeout,
    )
    app.state.settings = settings
    app.state.start_time = time.time()
    app.state.request_count = 0
    app.state.error_count = 0

    yield

    # Cleanup
    await app.state.vllm_client.close()


app = FastAPI(
    title="Inference Server MVP",
    description="OpenRouter-compatible inference API with vLLM + LMCache",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for OpenRouter callbacks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Health & Metadata Endpoints ============

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers and OpenRouter."""
    uptime = time.time() - app.state.start_time
    error_rate = (
        app.state.error_count / app.state.request_count
        if app.state.request_count > 0
        else 0.0
    )

    return {
        "status": "healthy",
        "uptime_seconds": round(uptime, 2),
        "total_requests": app.state.request_count,
        "error_rate": round(error_rate, 4),
        "vllm_connected": await app.state.vllm_client.is_healthy(),
    }


@app.get("/api/v1/models")
async def list_models():
    """
    OpenRouter model discovery endpoint.
    Returns model metadata including pricing and capabilities.
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


# ============ Chat Completions API ============

@app.post("/api/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    Supports streaming via SSE for long-running requests.
    """
    app.state.request_count += 1
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    try:
        if request.stream:
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
            return await generate_chat_completion(request, request_id)

    except Exception as e:
        app.state.error_count += 1
        raise HTTPException(status_code=500, detail=str(e))


async def generate_chat_completion(
    request: ChatCompletionRequest,
    request_id: str,
) -> ChatCompletionResponse:
    """Non-streaming chat completion."""

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
    Streaming chat completion with SSE.
    Includes keep-alive pings for OpenRouter timeout prevention.
    """

    async for chunk in app.state.vllm_client.generate_stream(
        messages=request.messages,
        model=request.model,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,
    ):
        # Format as SSE
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
        yield f"data: {JSONResponse(content=data).body.decode()}\n\n"

        # Keep-alive for long requests
        if chunk.get("keep_alive"):
            yield ": keep-alive\n\n"

    yield "data: [DONE]\n\n"


# ============ Error Handlers ============

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
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
    app.state.error_count += 1
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

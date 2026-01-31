"""
Async HTTP client for vLLM inference backend.

This module provides an async client for communicating with vLLM's OpenAI-compatible
server. It handles both streaming (SSE) and non-streaming requests with proper
connection management and keep-alive support.

Architecture:
    FastAPI App -> VLLMClient -> vLLM Server (port 8080)
                               -> /v1/chat/completions
                               -> /health

Features:
    - Lazy HTTP client initialization (connects on first request)
    - Automatic keep-alive pings during long streaming responses
    - Graceful connection cleanup
    - Mock client for testing without vLLM backend

Usage:
    client = VLLMClient(base_url="http://localhost:8080", timeout=300.0)

    # Non-streaming
    response = await client.generate(messages=[...], model="my-model")
    print(response["content"])

    # Streaming
    async for chunk in client.generate_stream(messages=[...], model="my-model"):
        print(chunk["content"], end="")

    await client.close()

vLLM Server Setup:
    Basic:
        vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8080

    With LMCache KV offloading:
        LMCACHE_CONFIG_FILE=configs/lmcache.yaml vllm serve <model> --port 8080

Upstream References:
    - vLLM: https://github.com/vllm-project/vllm
    - LMCache: https://github.com/LMCache/LMCache

Version: 0.3.0
License: MIT
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, AsyncGenerator

import httpx


class VLLMClient:
    """
    Async HTTP client for vLLM's OpenAI-compatible server.

    Provides a clean async interface for chat completions with support for
    both streaming and non-streaming generation modes.

    Attributes:
        base_url: vLLM server URL (default: http://localhost:8080)
        timeout: Request timeout in seconds (default: 300 for long contexts)

    Connection Management:
        The HTTP client is lazily initialized on first request and reused
        for subsequent requests. Call close() to release connections.

    Error Handling:
        - Connection errors raise httpx.ConnectError
        - HTTP errors (4xx, 5xx) raise httpx.HTTPStatusError
        - is_healthy() catches all exceptions and returns False

    Example:
        async with VLLMClient() as client:
            response = await client.generate(
                messages=[{"role": "user", "content": "Hello!"}],
                model="llama-3.1-8b"
            )
            print(response["content"])
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 300.0,
    ):
        """
        Initialize the vLLM client.

        Args:
            base_url: URL of the vLLM server. Trailing slashes are stripped.
                     Examples: "http://localhost:8080", "http://vllm:8080"
            timeout: Maximum time in seconds to wait for responses.
                    Set high (300s) to accommodate long-context inference.

        Note:
            The actual HTTP connection is not established until the first
            request. This allows creating the client at import time.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """
        Get or create the HTTP client (lazy initialization).

        Creates a new client if one doesn't exist or if the existing
        client has been closed. Uses httpx.Timeout with a shorter
        connect timeout (10s) to fail fast on connection issues.

        Returns:
            Configured httpx.AsyncClient instance

        Note:
            Thread-safety: This is NOT thread-safe. In FastAPI with
            uvicorn, each worker has its own event loop and client.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    self.timeout,  # Total timeout for request
                    connect=10.0,  # Fail fast if can't connect
                ),
            )
        return self._client

    async def close(self) -> None:
        """
        Close the HTTP client and release connections.

        Should be called during application shutdown to cleanly close
        any open connections. Safe to call multiple times.

        Usage in FastAPI lifespan:
            @asynccontextmanager
            async def lifespan(app):
                yield
                await app.state.vllm_client.close()
        """
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def is_healthy(self) -> bool:
        """
        Check if the vLLM backend is responding.

        Makes a GET request to /health with a short timeout (5s).
        Used by the FastAPI /health endpoint to report backend status.

        Returns:
            True if vLLM responds with 200, False otherwise

        Note:
            All exceptions are caught and result in False, including:
            - Connection refused (vLLM not running)
            - Timeout (vLLM overloaded)
            - HTTP errors (vLLM unhealthy)
        """
        try:
            client = await self._get_client()
            response = await client.get("/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            # Catch all exceptions to prevent health check from crashing
            return False

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_tokens: Optional[int] = 4096,
        temperature: Optional[float] = 0.7,
        top_p: Optional[float] = 1.0,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a non-streaming chat completion.

        Sends a request to vLLM and waits for the complete response.
        Suitable for short responses; use generate_stream() for long outputs.

        Args:
            messages: List of chat messages in OpenAI format
                     [{"role": "user", "content": "..."}]
            model: Model identifier (must match vLLM's loaded model)
            max_tokens: Maximum tokens to generate (default: 4096)
            temperature: Sampling temperature 0.0-2.0 (default: 0.7)
            top_p: Nucleus sampling threshold 0.0-1.0 (default: 1.0)
            stop: Stop sequences to end generation
            **kwargs: Additional parameters (ignored, for interface compat)

        Returns:
            Dict containing:
                - content: Generated text
                - finish_reason: Why generation stopped ("stop", "length")
                - prompt_tokens: Input token count
                - completion_tokens: Output token count
                - total_tokens: Sum of prompt + completion tokens

        Raises:
            httpx.HTTPStatusError: If vLLM returns an error status
            httpx.ConnectError: If unable to connect to vLLM

        Example:
            response = await client.generate(
                messages=[
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello!"}
                ],
                model="llama-3.1-8b",
                max_tokens=100
            )
            print(response["content"])  # "Hello! How can I help you today?"
        """
        client = await self._get_client()

        # Build request payload
        # Handle both Pydantic models and plain dicts for messages
        payload = {
            "model": model,
            "messages": [
                m.model_dump() if hasattr(m, 'model_dump') else m
                for m in messages
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        # Make request to vLLM
        response = await client.post(
            "/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        # Extract response data
        choice = data["choices"][0]
        usage = data.get("usage", {})

        return {
            "content": choice["message"]["content"],
            "finish_reason": choice.get("finish_reason", "stop"),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    async def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_tokens: Optional[int] = 4096,
        temperature: Optional[float] = 0.7,
        top_p: Optional[float] = 1.0,
        stop: Optional[List[str]] = None,
        keep_alive_interval: float = 15.0,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate a streaming chat completion.

        Streams tokens as they're generated, yielding chunks via SSE format.
        Includes periodic keep-alive signals to prevent proxy timeouts.

        Args:
            messages: List of chat messages in OpenAI format
            model: Model identifier
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            stop: Stop sequences
            keep_alive_interval: Seconds between keep-alive signals (default: 15)
            **kwargs: Additional parameters (ignored)

        Yields:
            Dict containing:
                - content: Token text (may be empty string)
                - finish_reason: Only set on final chunk ("stop", "length", None)
                - keep_alive: True if this is a keep-alive ping

        Keep-Alive Mechanism:
            Proxies like nginx, Cloudflare, and OpenRouter may timeout idle
            connections. This method emits keep-alive signals periodically
            during long inference to prevent connection drops.

            The caller should emit SSE comments for keep-alive chunks:
                if chunk.get("keep_alive"):
                    yield ": keep-alive\\n\\n"

        Example:
            async for chunk in client.generate_stream(
                messages=[{"role": "user", "content": "Write a story"}],
                model="llama-3.1-8b"
            ):
                if not chunk.get("keep_alive"):
                    print(chunk["content"], end="", flush=True)
                if chunk.get("finish_reason"):
                    print("\\n[Done]")

        Raises:
            httpx.HTTPStatusError: If vLLM returns an error status
            httpx.ConnectError: If unable to connect to vLLM
        """
        client = await self._get_client()

        # Build streaming request payload
        payload = {
            "model": model,
            "messages": [
                m.model_dump() if hasattr(m, 'model_dump') else m
                for m in messages
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,  # Enable streaming
        }
        if stop:
            payload["stop"] = stop

        # Track time for keep-alive emissions
        last_keep_alive = asyncio.get_event_loop().time()

        # Stream response from vLLM
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                # Check if we should emit a keep-alive signal
                now = asyncio.get_event_loop().time()
                if now - last_keep_alive > keep_alive_interval:
                    yield {"content": "", "keep_alive": True}
                    last_keep_alive = now

                # Skip empty lines and non-data lines
                if not line or not line.startswith("data: "):
                    continue

                # Extract JSON from SSE data line
                data_str = line[6:]  # Strip "data: " prefix

                # Check for stream termination
                if data_str == "[DONE]":
                    break

                # Parse and yield chunk
                try:
                    data = json.loads(data_str)
                    choice = data["choices"][0]
                    delta = choice.get("delta", {})

                    yield {
                        "content": delta.get("content", ""),
                        "finish_reason": choice.get("finish_reason"),
                        "keep_alive": False,
                    }
                except json.JSONDecodeError:
                    # Skip malformed chunks (shouldn't happen with vLLM)
                    continue


class MockVLLMClient(VLLMClient):
    """
    Mock vLLM client for testing without a real backend.

    Provides predictable responses for unit and integration testing.
    Echoes back the user's message with a prefix, simulating real behavior.

    Features:
        - Always reports healthy
        - Echoes user message in response
        - Simulates token counts
        - Streams word-by-word for streaming tests

    Usage:
        # In conftest.py
        @pytest.fixture
        def mock_client():
            return MockVLLMClient()

        # In app setup for testing
        app.state.vllm_client = MockVLLMClient()

    Note:
        This client does NOT make any network requests. It's purely
        for testing the API layer without a vLLM dependency.
    """

    async def is_healthy(self) -> bool:
        """
        Always returns True for mock client.

        Returns:
            True (mock is always healthy)
        """
        return True

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a mock response echoing the user's message.

        Finds the last user message and creates a response that includes
        a portion of it, making test assertions predictable.

        Args:
            messages: List of chat messages
            model: Model identifier (ignored)
            **kwargs: Additional parameters (ignored)

        Returns:
            Dict with mock response data:
                - content: "Mock response to: {first 50 chars of user message}"
                - finish_reason: "stop"
                - prompt_tokens: 10 (fixed)
                - completion_tokens: word count of content
                - total_tokens: prompt + completion
        """
        # Find the last user message
        # Handle both Pydantic models and plain dicts
        def get_role(m):
            return m.role if hasattr(m, 'role') else m.get("role")

        def get_content(m):
            return m.content if hasattr(m, 'content') else m.get("content", "")

        last_user = next(
            (m for m in reversed(messages) if get_role(m) == "user"),
            {"content": "Hello!"},
        )
        user_content = get_content(last_user) or ""
        content = f"Mock response to: {user_content[:50]}"

        return {
            "content": content,
            "finish_reason": "stop",
            "prompt_tokens": 10,
            "completion_tokens": len(content.split()),
            "total_tokens": 10 + len(content.split()),
        }

    async def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate a mock streaming response.

        Calls generate() to get the full response, then yields it
        word-by-word to simulate streaming behavior.

        Args:
            messages: List of chat messages
            model: Model identifier (ignored)
            **kwargs: Additional parameters (ignored)

        Yields:
            Dict with:
                - content: Single word + space
                - finish_reason: "stop" on last word, None otherwise
                - keep_alive: False (no keep-alive needed for mock)
        """
        response = await self.generate(messages, model, **kwargs)
        words = response["content"].split()

        for i, word in enumerate(words):
            yield {
                "content": word + " ",
                "finish_reason": "stop" if i == len(words) - 1 else None,
                "keep_alive": False,
            }

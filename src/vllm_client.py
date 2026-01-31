"""
vLLM HTTP client for inference backend.
Handles both streaming and non-streaming requests.

Upstream reference: https://github.com/vllm-project/vllm
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, AsyncGenerator

import httpx


class VLLMClient:
    """
    Async HTTP client for vLLM's OpenAI-compatible server.

    vLLM should be running with:
        vllm serve <model> --port 8080

    Or with LMCache:
        LMCACHE_CONFIG_FILE=configs/lmcache.yaml vllm serve <model> ...
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 300.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-initialize the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def is_healthy(self) -> bool:
        """Check if vLLM backend is responding."""
        try:
            client = await self._get_client()
            response = await client.get("/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
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
        Non-streaming generation request to vLLM.

        Returns dict with:
            - content: str
            - finish_reason: str
            - prompt_tokens: int
            - completion_tokens: int
            - total_tokens: int
        """
        client = await self._get_client()

        payload = {
            "model": model,
            "messages": [m.model_dump() if hasattr(m, 'model_dump') else m for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        response = await client.post(
            "/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

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
        Streaming generation request to vLLM.

        Yields dicts with:
            - content: str (delta)
            - finish_reason: Optional[str]
            - keep_alive: bool (for SSE keep-alive pings)
        """
        client = await self._get_client()

        payload = {
            "model": model,
            "messages": [m.model_dump() if hasattr(m, 'model_dump') else m for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }
        if stop:
            payload["stop"] = stop

        last_keep_alive = asyncio.get_event_loop().time()

        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                # Check for keep-alive interval
                now = asyncio.get_event_loop().time()
                if now - last_keep_alive > keep_alive_interval:
                    yield {"content": "", "keep_alive": True}
                    last_keep_alive = now

                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]  # Strip "data: " prefix
                if data_str == "[DONE]":
                    break

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
                    continue


class MockVLLMClient(VLLMClient):
    """
    Mock client for testing without a real vLLM backend.
    """

    async def is_healthy(self) -> bool:
        return True

    async def generate(self, messages, model, **kwargs) -> Dict[str, Any]:
        # Echo back the last user message
        last_user = next(
            (m for m in reversed(messages) if m.get("role") == "user"),
            {"content": "Hello!"},
        )
        content = f"Mock response to: {last_user.get('content', '')[:50]}"

        return {
            "content": content,
            "finish_reason": "stop",
            "prompt_tokens": 10,
            "completion_tokens": len(content.split()),
            "total_tokens": 10 + len(content.split()),
        }

    async def generate_stream(self, messages, model, **kwargs):
        response = await self.generate(messages, model, **kwargs)
        words = response["content"].split()

        for i, word in enumerate(words):
            yield {
                "content": word + " ",
                "finish_reason": "stop" if i == len(words) - 1 else None,
                "keep_alive": False,
            }

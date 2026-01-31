"""
Tests for vLLM client.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vllm_client import MockVLLMClient, VLLMClient


class TestMockVLLMClient:
    """Tests for the mock client used in testing."""

    @pytest.fixture
    def mock_client(self):
        return MockVLLMClient()

    @pytest.mark.asyncio
    async def test_is_healthy_returns_true(self, mock_client):
        """Mock client should always report healthy."""
        assert await mock_client.is_healthy() is True

    @pytest.mark.asyncio
    async def test_generate_returns_response(self, mock_client):
        """Mock client should return a valid response."""
        messages = [{"role": "user", "content": "Hello!"}]
        response = await mock_client.generate(
            messages=messages,
            model="test-model",
        )

        assert "content" in response
        assert "finish_reason" in response
        assert "prompt_tokens" in response
        assert "completion_tokens" in response
        assert "total_tokens" in response

    @pytest.mark.asyncio
    async def test_generate_echoes_user_message(self, mock_client):
        """Mock client should echo part of the user message."""
        messages = [{"role": "user", "content": "Testing 123"}]
        response = await mock_client.generate(
            messages=messages,
            model="test-model",
        )

        assert "Testing 123" in response["content"]

    @pytest.mark.asyncio
    async def test_generate_stream_yields_chunks(self, mock_client):
        """Mock client streaming should yield chunks."""
        messages = [{"role": "user", "content": "Hello world test"}]

        chunks = []
        async for chunk in mock_client.generate_stream(
            messages=messages,
            model="test-model",
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert all("content" in c for c in chunks)

    @pytest.mark.asyncio
    async def test_generate_stream_last_chunk_has_finish_reason(self, mock_client):
        """Last streaming chunk should have finish_reason."""
        messages = [{"role": "user", "content": "Hello"}]

        chunks = []
        async for chunk in mock_client.generate_stream(
            messages=messages,
            model="test-model",
        ):
            chunks.append(chunk)

        assert chunks[-1]["finish_reason"] == "stop"


class TestVLLMClientHealth:
    """Tests for VLLMClient health checking."""

    @pytest.fixture
    def client(self):
        return VLLMClient(base_url="http://localhost:8080")

    @pytest.mark.asyncio
    async def test_is_healthy_returns_true_on_200(self, client):
        """Health check should return True when backend returns 200."""
        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_http

            result = await client.is_healthy()
            assert result is True

    @pytest.mark.asyncio
    async def test_is_healthy_returns_false_on_error(self, client):
        """Health check should return False on connection error."""
        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_get.return_value = mock_http

            result = await client.is_healthy()
            assert result is False

    @pytest.mark.asyncio
    async def test_is_healthy_returns_false_on_500(self, client):
        """Health check should return False when backend returns 500."""
        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_http

            result = await client.is_healthy()
            assert result is False


class TestVLLMClientConfiguration:
    """Tests for VLLMClient configuration."""

    def test_default_base_url(self):
        """Default base URL should be localhost:8080."""
        client = VLLMClient()
        assert client.base_url == "http://localhost:8080"

    def test_custom_base_url(self):
        """Custom base URL should be respected."""
        client = VLLMClient(base_url="http://vllm:9000")
        assert client.base_url == "http://vllm:9000"

    def test_base_url_trailing_slash_stripped(self):
        """Trailing slash should be stripped from base URL."""
        client = VLLMClient(base_url="http://vllm:8080/")
        assert client.base_url == "http://vllm:8080"

    def test_default_timeout(self):
        """Default timeout should be 300 seconds."""
        client = VLLMClient()
        assert client.timeout == 300.0

    def test_custom_timeout(self):
        """Custom timeout should be respected."""
        client = VLLMClient(timeout=600.0)
        assert client.timeout == 600.0

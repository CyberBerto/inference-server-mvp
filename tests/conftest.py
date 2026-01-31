"""
Pytest fixtures and configuration.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    from config import Settings

    return Settings(
        model_id="test-org/test-model",
        model_display_name="Test Model",
        organization_id="test-org",
        max_context_length=131072,
        vllm_base_url="http://localhost:8080",
    )


@pytest.fixture
def mock_vllm_client():
    """Mock vLLM client that returns predictable responses."""
    from vllm_client import MockVLLMClient

    return MockVLLMClient()


@pytest.fixture
def app_with_mock_client(mock_vllm_client, mock_settings):
    """FastAPI app with mocked vLLM client."""
    from main import app

    # Override the lifespan-initialized client
    app.state.vllm_client = mock_vllm_client
    app.state.settings = mock_settings
    app.state.start_time = 1000000
    app.state.request_count = 0
    app.state.error_count = 0

    return app


@pytest.fixture
def client(app_with_mock_client):
    """Synchronous test client."""
    return TestClient(app_with_mock_client)


@pytest.fixture
async def async_client(app_with_mock_client):
    """Async test client for streaming tests."""
    transport = ASGITransport(app=app_with_mock_client)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]


@pytest.fixture
def sample_chat_request(sample_messages):
    """Sample chat completion request."""
    return {
        "model": "test-org/test-model",
        "messages": sample_messages,
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False,
    }


@pytest.fixture
def sample_streaming_request(sample_messages):
    """Sample streaming chat completion request."""
    return {
        "model": "test-org/test-model",
        "messages": sample_messages,
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": True,
    }

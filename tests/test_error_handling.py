"""
Tests for error handling across the application.

These tests verify that errors are properly caught, formatted according
to OpenAI error response schema, and that appropriate HTTP status codes
are returned.
"""

import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestErrorResponseFormat:
    """Tests for OpenAI-compatible error response format."""

    def test_validation_error_format(self, client, sample_messages):
        """Validation errors should return structured error response."""
        # Invalid temperature (out of range)
        request = {
            "model": "test-model",
            "messages": sample_messages,
            "temperature": 5.0,  # Invalid: max is 2.0
        }
        response = client.post("/api/v1/chat/completions", json=request)

        assert response.status_code == 422
        # FastAPI returns validation errors in its own format
        data = response.json()
        assert "detail" in data

    def test_missing_required_field_error(self, client):
        """Missing required fields should return 422."""
        # Missing 'model' field
        request = {"messages": [{"role": "user", "content": "test"}]}
        response = client.post("/api/v1/chat/completions", json=request)

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_invalid_json_error(self, client):
        """Invalid JSON should return 422."""
        response = client.post(
            "/api/v1/chat/completions",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422


class TestVLLMClientErrorHandling:
    """Tests for vLLM client error scenarios."""

    @pytest.fixture
    def real_client(self):
        """Create a real VLLMClient for testing."""
        from vllm_client import VLLMClient

        return VLLMClient(base_url="http://localhost:9999")  # Non-existent

    @pytest.mark.asyncio
    async def test_health_check_connection_refused(self, real_client):
        """Health check should return False when connection refused."""
        result = await real_client.is_healthy()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_timeout(self):
        """Health check should return False on timeout."""
        from vllm_client import VLLMClient

        client = VLLMClient(base_url="http://10.255.255.1:8080")  # Non-routable
        result = await client.is_healthy()
        assert result is False

    @pytest.mark.asyncio
    async def test_generate_connection_error(self, real_client):
        """Generate should raise on connection error."""
        with pytest.raises(httpx.ConnectError):
            await real_client.generate(messages=[{"role": "user", "content": "test"}], model="test")


class TestHealthEndpointEdgeCases:
    """Edge case tests for health endpoint."""

    def test_health_with_zero_requests(self, client):
        """Health should handle zero requests gracefully."""
        response = client.get("/health")
        data = response.json()

        assert data["error_rate"] == 0.0
        assert data["total_requests"] == 0

    def test_health_after_multiple_requests(self, client, sample_chat_request):
        """Health should track request count correctly."""
        # Make some requests
        client.post("/api/v1/chat/completions", json=sample_chat_request)
        client.post("/api/v1/chat/completions", json=sample_chat_request)
        client.post("/api/v1/chat/completions", json=sample_chat_request)

        response = client.get("/health")
        data = response.json()

        assert data["total_requests"] == 3


class TestChatCompletionsEdgeCases:
    """Edge case tests for chat completions endpoint."""

    def test_empty_content_message(self, client):
        """Message with empty content should be accepted."""
        request = {"model": "test-model", "messages": [{"role": "user", "content": ""}]}
        response = client.post("/api/v1/chat/completions", json=request)
        # Should succeed (vLLM will handle empty content)
        assert response.status_code == 200

    def test_very_long_message(self, client):
        """Very long messages should be accepted (up to vLLM limits)."""
        long_content = "a" * 10000
        request = {"model": "test-model", "messages": [{"role": "user", "content": long_content}]}
        response = client.post("/api/v1/chat/completions", json=request)
        assert response.status_code == 200

    def test_multiple_messages(self, client):
        """Multiple messages in conversation should work."""
        request = {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
        }
        response = client.post("/api/v1/chat/completions", json=request)
        assert response.status_code == 200

    def test_unicode_content(self, client):
        """Unicode content should be handled correctly."""
        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello! 你好 🌍 مرحبا"}],
        }
        response = client.post("/api/v1/chat/completions", json=request)
        assert response.status_code == 200

        # Check response contains original content reference
        data = response.json()
        assert "Hello!" in data["choices"][0]["message"]["content"]

    def test_special_characters_in_content(self, client):
        """Special characters should not break parsing."""
        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": 'Test with "quotes" and \\n newlines'}],
        }
        response = client.post("/api/v1/chat/completions", json=request)
        assert response.status_code == 200

    def test_max_tokens_boundary(self, client, sample_messages):
        """Boundary values for max_tokens should work."""
        # Minimum valid value
        request = {"model": "test-model", "messages": sample_messages, "max_tokens": 1}
        response = client.post("/api/v1/chat/completions", json=request)
        assert response.status_code == 200

    def test_temperature_boundary(self, client, sample_messages):
        """Boundary values for temperature should work."""
        # Minimum
        request = {"model": "test-model", "messages": sample_messages, "temperature": 0.0}
        response = client.post("/api/v1/chat/completions", json=request)
        assert response.status_code == 200

        # Maximum
        request["temperature"] = 2.0
        response = client.post("/api/v1/chat/completions", json=request)
        assert response.status_code == 200


class TestModelsEndpointEdgeCases:
    """Edge case tests for models endpoint."""

    def test_models_response_is_stable(self, client):
        """Models endpoint should return consistent data."""
        response1 = client.get("/api/v1/models")
        response2 = client.get("/api/v1/models")

        data1 = response1.json()
        data2 = response2.json()

        # Model ID should be the same
        assert data1["data"][0]["id"] == data2["data"][0]["id"]
        # Pricing should be the same
        assert data1["data"][0]["pricing"] == data2["data"][0]["pricing"]

    def test_models_has_openrouter_fields(self, client):
        """Models response should have all OpenRouter-required fields."""
        response = client.get("/api/v1/models")
        model = response.json()["data"][0]

        # Required by OpenRouter
        assert "id" in model
        assert "name" in model
        assert "context_length" in model
        assert "pricing" in model
        assert "prompt" in model["pricing"]
        assert "completion" in model["pricing"]


class TestRequestIDGeneration:
    """Tests for request ID generation."""

    def test_request_id_format(self, client, sample_chat_request):
        """Request ID should follow OpenAI format."""
        response = client.post("/api/v1/chat/completions", json=sample_chat_request)
        data = response.json()

        request_id = data["id"]
        assert request_id.startswith("chatcmpl-")
        # UUID hex is 32 chars, we use 24
        assert len(request_id) == len("chatcmpl-") + 24

    def test_request_ids_are_unique(self, client, sample_chat_request):
        """Each request should get a unique ID."""
        ids = set()

        for _ in range(10):
            response = client.post("/api/v1/chat/completions", json=sample_chat_request)
            ids.add(response.json()["id"])

        assert len(ids) == 10  # All unique

    def test_streaming_request_id_header(self, client, sample_streaming_request):
        """Streaming response should have X-Request-ID header."""
        response = client.post("/api/v1/chat/completions", json=sample_streaming_request)

        assert "X-Request-ID" in response.headers
        assert response.headers["X-Request-ID"].startswith("chatcmpl-")

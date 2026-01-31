"""
Tests for chat completions endpoint.
"""

import pytest
import json


class TestChatCompletionsNonStreaming:
    """Tests for non-streaming chat completions."""

    def test_chat_completions_returns_200(self, client, sample_chat_request):
        """Chat completions should return 200 OK."""
        response = client.post("/api/v1/chat/completions", json=sample_chat_request)
        assert response.status_code == 200

    def test_chat_completions_response_format(self, client, sample_chat_request):
        """Response should match OpenAI format."""
        response = client.post("/api/v1/chat/completions", json=sample_chat_request)
        data = response.json()

        assert "id" in data
        assert data["id"].startswith("chatcmpl-")
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        assert "usage" in data

    def test_chat_completions_has_choices(self, client, sample_chat_request):
        """Response should have at least one choice."""
        response = client.post("/api/v1/chat/completions", json=sample_chat_request)
        data = response.json()

        assert len(data["choices"]) > 0
        choice = data["choices"][0]
        assert "index" in choice
        assert "message" in choice
        assert "finish_reason" in choice

    def test_chat_completions_message_format(self, client, sample_chat_request):
        """Choice message should have role and content."""
        response = client.post("/api/v1/chat/completions", json=sample_chat_request)
        data = response.json()

        message = data["choices"][0]["message"]
        assert message["role"] == "assistant"
        assert "content" in message
        assert isinstance(message["content"], str)

    def test_chat_completions_usage_stats(self, client, sample_chat_request):
        """Response should include token usage statistics."""
        response = client.post("/api/v1/chat/completions", json=sample_chat_request)
        data = response.json()

        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_chat_completions_increments_request_count(self, client, sample_chat_request):
        """Request count should increment after each request."""
        # Get initial count
        health1 = client.get("/health").json()
        initial_count = health1["total_requests"]

        # Make a request
        client.post("/api/v1/chat/completions", json=sample_chat_request)

        # Check incremented
        health2 = client.get("/health").json()
        assert health2["total_requests"] == initial_count + 1


class TestChatCompletionsValidation:
    """Tests for request validation."""

    def test_missing_model_returns_422(self, client, sample_messages):
        """Missing model field should return 422."""
        request = {"messages": sample_messages}
        response = client.post("/api/v1/chat/completions", json=request)
        assert response.status_code == 422

    def test_missing_messages_returns_422(self, client):
        """Missing messages field should return 422."""
        request = {"model": "test-model"}
        response = client.post("/api/v1/chat/completions", json=request)
        assert response.status_code == 422

    def test_empty_messages_allowed(self, client):
        """Empty messages array should be accepted."""
        request = {"model": "test-model", "messages": []}
        response = client.post("/api/v1/chat/completions", json=request)
        # Should either succeed or return specific error, not 422 validation
        assert response.status_code in [200, 400, 500]

    def test_temperature_range_validation(self, client, sample_messages):
        """Temperature outside 0-2 range should return 422."""
        request = {
            "model": "test-model",
            "messages": sample_messages,
            "temperature": 3.0,
        }
        response = client.post("/api/v1/chat/completions", json=request)
        assert response.status_code == 422

    def test_max_tokens_negative_returns_422(self, client, sample_messages):
        """Negative max_tokens should return 422."""
        request = {
            "model": "test-model",
            "messages": sample_messages,
            "max_tokens": -1,
        }
        response = client.post("/api/v1/chat/completions", json=request)
        assert response.status_code == 422


class TestChatCompletionsStreaming:
    """Tests for streaming chat completions."""

    def test_streaming_returns_sse(self, client, sample_streaming_request):
        """Streaming should return text/event-stream content type."""
        response = client.post(
            "/api/v1/chat/completions",
            json=sample_streaming_request,
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    def test_streaming_format(self, client, sample_streaming_request):
        """Streaming response should be SSE formatted."""
        response = client.post(
            "/api/v1/chat/completions",
            json=sample_streaming_request,
        )

        content = response.text
        lines = [l for l in content.split("\n") if l.startswith("data: ")]

        # Should have at least one data line
        assert len(lines) > 0

        # Last should be [DONE]
        assert "data: [DONE]" in content

    def test_streaming_chunks_are_valid_json(self, client, sample_streaming_request):
        """Each streaming chunk should be valid JSON."""
        response = client.post(
            "/api/v1/chat/completions",
            json=sample_streaming_request,
        )

        content = response.text
        for line in content.split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                json_str = line[6:]  # Strip "data: "
                try:
                    data = json.loads(json_str)
                    assert "id" in data
                    assert "choices" in data
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON in chunk: {json_str}")

    def test_streaming_chunk_format(self, client, sample_streaming_request):
        """Streaming chunks should have correct format."""
        response = client.post(
            "/api/v1/chat/completions",
            json=sample_streaming_request,
        )

        content = response.text
        for line in content.split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                data = json.loads(line[6:])
                assert data["object"] == "chat.completion.chunk"
                assert "delta" in data["choices"][0]

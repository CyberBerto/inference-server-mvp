"""
Tests for OpenRouter API compatibility.

These tests verify that the API responses match OpenRouter's expected format.
Reference: https://openrouter.ai/docs
"""

import json


class TestOpenRouterModelDiscovery:
    """
    Tests for model discovery endpoint.
    OpenRouter uses this to discover available models and their capabilities.
    """

    def test_models_endpoint_exists(self, client):
        """GET /api/v1/models should exist and return 200."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200

    def test_models_response_structure(self, client):
        """Response should have object and data fields."""
        response = client.get("/api/v1/models")
        data = response.json()

        assert data["object"] == "list"
        assert isinstance(data["data"], list)

    def test_model_has_required_fields(self, client):
        """Each model should have OpenRouter-required fields."""
        response = client.get("/api/v1/models")
        model = response.json()["data"][0]

        required_fields = [
            "id",
            "name",
            "context_length",
            "pricing",
        ]
        for field in required_fields:
            assert field in model, f"Missing required field: {field}"

    def test_pricing_has_prompt_and_completion(self, client):
        """Pricing should have prompt and completion rates."""
        response = client.get("/api/v1/models")
        pricing = response.json()["data"][0]["pricing"]

        assert "prompt" in pricing
        assert "completion" in pricing

    def test_pricing_values_are_strings(self, client):
        """Pricing values should be strings (OpenRouter format)."""
        response = client.get("/api/v1/models")
        pricing = response.json()["data"][0]["pricing"]

        assert isinstance(pricing["prompt"], str)
        assert isinstance(pricing["completion"], str)

    def test_context_length_is_integer(self, client):
        """Context length should be an integer."""
        response = client.get("/api/v1/models")
        model = response.json()["data"][0]

        assert isinstance(model["context_length"], int)
        assert model["context_length"] > 0


class TestOpenRouterChatFormat:
    """
    Tests for chat completion format compatibility.
    """

    def test_chat_endpoint_exists(self, client, sample_chat_request):
        """POST /api/v1/chat/completions should exist."""
        response = client.post("/api/v1/chat/completions", json=sample_chat_request)
        assert response.status_code == 200

    def test_response_has_id_with_prefix(self, client, sample_chat_request):
        """Response ID should start with 'chatcmpl-'."""
        response = client.post("/api/v1/chat/completions", json=sample_chat_request)
        data = response.json()

        assert data["id"].startswith("chatcmpl-")

    def test_response_object_is_chat_completion(self, client, sample_chat_request):
        """Response object should be 'chat.completion'."""
        response = client.post("/api/v1/chat/completions", json=sample_chat_request)
        data = response.json()

        assert data["object"] == "chat.completion"

    def test_response_has_created_timestamp(self, client, sample_chat_request):
        """Response should have created timestamp."""
        response = client.post("/api/v1/chat/completions", json=sample_chat_request)
        data = response.json()

        assert "created" in data
        assert isinstance(data["created"], int)

    def test_response_has_model_field(self, client, sample_chat_request):
        """Response should include model field."""
        response = client.post("/api/v1/chat/completions", json=sample_chat_request)
        data = response.json()

        assert "model" in data

    def test_choices_is_array(self, client, sample_chat_request):
        """Choices should be an array."""
        response = client.post("/api/v1/chat/completions", json=sample_chat_request)
        data = response.json()

        assert isinstance(data["choices"], list)

    def test_choice_has_index(self, client, sample_chat_request):
        """Each choice should have an index."""
        response = client.post("/api/v1/chat/completions", json=sample_chat_request)
        choice = response.json()["choices"][0]

        assert "index" in choice
        assert choice["index"] == 0

    def test_choice_has_message(self, client, sample_chat_request):
        """Each choice should have a message object."""
        response = client.post("/api/v1/chat/completions", json=sample_chat_request)
        choice = response.json()["choices"][0]

        assert "message" in choice
        assert "role" in choice["message"]
        assert "content" in choice["message"]

    def test_message_role_is_assistant(self, client, sample_chat_request):
        """Response message role should be 'assistant'."""
        response = client.post("/api/v1/chat/completions", json=sample_chat_request)
        message = response.json()["choices"][0]["message"]

        assert message["role"] == "assistant"

    def test_choice_has_finish_reason(self, client, sample_chat_request):
        """Each choice should have a finish_reason."""
        response = client.post("/api/v1/chat/completions", json=sample_chat_request)
        choice = response.json()["choices"][0]

        assert "finish_reason" in choice


class TestOpenRouterStreaming:
    """
    Tests for streaming format compatibility.
    OpenRouter requires specific SSE formatting.
    """

    def test_streaming_content_type(self, client, sample_streaming_request):
        """Streaming should return text/event-stream."""
        response = client.post(
            "/api/v1/chat/completions",
            json=sample_streaming_request,
        )

        assert "text/event-stream" in response.headers["content-type"]

    def test_streaming_uses_sse_format(self, client, sample_streaming_request):
        """Streaming should use SSE data: prefix."""
        response = client.post(
            "/api/v1/chat/completions",
            json=sample_streaming_request,
        )

        content = response.text
        assert "data: " in content

    def test_streaming_ends_with_done(self, client, sample_streaming_request):
        """Streaming should end with data: [DONE]."""
        response = client.post(
            "/api/v1/chat/completions",
            json=sample_streaming_request,
        )

        content = response.text
        assert "data: [DONE]" in content

    def test_streaming_chunk_object_type(self, client, sample_streaming_request):
        """Streaming chunks should have object 'chat.completion.chunk'."""
        response = client.post(
            "/api/v1/chat/completions",
            json=sample_streaming_request,
        )

        for line in response.text.split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                assert chunk["object"] == "chat.completion.chunk"

    def test_streaming_chunks_have_delta(self, client, sample_streaming_request):
        """Streaming chunks should have delta instead of message."""
        response = client.post(
            "/api/v1/chat/completions",
            json=sample_streaming_request,
        )

        for line in response.text.split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                assert "delta" in chunk["choices"][0]


class TestOpenRouterUptimeRequirements:
    """
    Tests related to OpenRouter uptime thresholds.
    95%+ uptime required for normal routing.
    """

    def test_health_endpoint_exists(self, client):
        """Health endpoint should exist for uptime monitoring."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status(self, client):
        """Health should return status field."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data

    def test_health_tracks_error_rate(self, client):
        """Health should track error rate for uptime calculation."""
        response = client.get("/health")
        data = response.json()

        assert "error_rate" in data
        assert isinstance(data["error_rate"], (int, float))

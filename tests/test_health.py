"""
Tests for health check and metadata endpoints.
"""


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_contains_required_fields(self, client):
        """Health response should contain all required fields."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "uptime_seconds" in data
        assert "total_requests" in data
        assert "error_rate" in data
        assert "vllm_connected" in data

    def test_health_status_is_healthy(self, client):
        """Health status should be 'healthy' when backend is up."""
        response = client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"

    def test_health_vllm_connected_with_mock(self, client):
        """vLLM should appear connected when using mock client."""
        response = client.get("/health")
        data = response.json()

        assert data["vllm_connected"] is True

    def test_health_error_rate_starts_at_zero(self, client):
        """Error rate should be 0 with no requests."""
        response = client.get("/health")
        data = response.json()

        assert data["error_rate"] == 0.0


class TestModelsEndpoint:
    """Tests for /api/v1/models endpoint."""

    def test_models_returns_200(self, client):
        """Models endpoint should return 200 OK."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200

    def test_models_returns_list(self, client):
        """Models endpoint should return a list of models."""
        response = client.get("/api/v1/models")
        data = response.json()

        assert "object" in data
        assert data["object"] == "list"
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_models_contains_model_info(self, client):
        """Each model should have required OpenRouter fields."""
        response = client.get("/api/v1/models")
        data = response.json()

        model = data["data"][0]
        assert "id" in model
        assert "name" in model
        assert "context_length" in model
        assert "pricing" in model
        assert "supported_features" in model

    def test_models_pricing_format(self, client):
        """Pricing should have prompt and completion rates."""
        response = client.get("/api/v1/models")
        data = response.json()

        pricing = data["data"][0]["pricing"]
        assert "prompt" in pricing
        assert "completion" in pricing

    def test_models_context_length(self, client, mock_settings):
        """Context length should match settings."""
        response = client.get("/api/v1/models")
        data = response.json()

        assert data["data"][0]["context_length"] == mock_settings.max_context_length

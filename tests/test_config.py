"""
Tests for configuration management.
"""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Settings, get_settings


class TestSettingsDefaults:
    """Tests for default configuration values."""

    def test_default_host(self):
        """Default host should be 0.0.0.0."""
        settings = Settings()
        assert settings.host == "0.0.0.0"

    def test_default_port(self):
        """Default port should be 8000."""
        settings = Settings()
        assert settings.port == 8000

    def test_default_vllm_url(self):
        """Default vLLM URL should be localhost:8080."""
        settings = Settings()
        assert settings.vllm_base_url == "http://localhost:8080"

    def test_default_context_length(self):
        """Default context length should be 128K."""
        settings = Settings()
        assert settings.max_context_length == 131072

    def test_default_timeout(self):
        """Default timeout should be 300 seconds."""
        settings = Settings()
        assert settings.request_timeout == 300.0

    def test_default_features(self):
        """Default features should include tools, json_mode, streaming."""
        settings = Settings()
        assert "tools" in settings.supported_features
        assert "json_mode" in settings.supported_features
        assert "streaming" in settings.supported_features


class TestSettingsEnvironment:
    """Tests for environment variable configuration."""

    def test_env_overrides_host(self):
        """HOST env var should override default."""
        with patch.dict(os.environ, {"HOST": "127.0.0.1"}):
            settings = Settings()
            assert settings.host == "127.0.0.1"

    def test_env_overrides_port(self):
        """PORT env var should override default."""
        with patch.dict(os.environ, {"PORT": "9000"}):
            settings = Settings()
            assert settings.port == 9000

    def test_env_overrides_model_id(self):
        """MODEL_ID env var should override default."""
        with patch.dict(os.environ, {"MODEL_ID": "my-org/my-model"}):
            settings = Settings()
            assert settings.model_id == "my-org/my-model"

    def test_env_overrides_vllm_url(self):
        """VLLM_BASE_URL env var should override default."""
        with patch.dict(os.environ, {"VLLM_BASE_URL": "http://vllm:8080"}):
            settings = Settings()
            assert settings.vllm_base_url == "http://vllm:8080"

    def test_env_overrides_pricing(self):
        """Pricing env vars should override defaults."""
        with patch.dict(
            os.environ,
            {
                "PRICE_PER_PROMPT_TOKEN": "0.00001",
                "PRICE_PER_COMPLETION_TOKEN": "0.00003",
            },
        ):
            settings = Settings()
            assert settings.price_per_prompt_token == "0.00001"
            assert settings.price_per_completion_token == "0.00003"


class TestSettingsValidation:
    """Tests for configuration validation."""

    def test_supported_features_is_list(self):
        """Supported features should be a list."""
        settings = Settings()
        assert isinstance(settings.supported_features, list)

    def test_context_length_is_positive(self):
        """Context length should be positive."""
        settings = Settings()
        assert settings.max_context_length > 0

    def test_timeout_is_positive(self):
        """Request timeout should be positive."""
        settings = Settings()
        assert settings.request_timeout > 0


class TestGetSettings:
    """Tests for the cached settings getter."""

    def test_get_settings_returns_settings(self):
        """get_settings should return a Settings instance."""
        # Clear cache first
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_is_cached(self):
        """get_settings should return the same instance on subsequent calls."""
        get_settings.cache_clear()
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

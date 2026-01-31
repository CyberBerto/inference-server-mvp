"""
Configuration management for the Inference Server MVP.

This module provides centralized configuration using pydantic-settings,
which automatically loads values from environment variables and .env files.

Configuration Hierarchy (highest to lowest priority):
    1. Environment variables (e.g., MODEL_ID=my-org/my-model)
    2. .env file in project root
    3. Default values defined in Settings class

Usage:
    from config import get_settings

    settings = get_settings()
    print(settings.model_id)  # Reads from MODEL_ID env var or default

Environment Variable Naming:
    - All settings use UPPER_SNAKE_CASE in environment
    - Nested objects not supported (flat structure only)
    - Example: model_id -> MODEL_ID, vllm_base_url -> VLLM_BASE_URL

Caching:
    get_settings() is cached with lru_cache, so settings are loaded once
    and reused. To reload settings, call get_settings.cache_clear() first.

Version: 0.2.0
License: MIT
"""

from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Server configuration loaded from environment variables.

    All settings have sensible defaults for local development. In production,
    override via environment variables or .env file.

    See .env.example for all available options with documentation.

    Attributes:
        Server Settings:
            host: Bind address for the server (default: 0.0.0.0 for all interfaces)
            port: HTTP port to listen on (default: 8000)
            workers: Number of uvicorn workers (default: 1, increase for production)
            debug: Enable debug mode with auto-reload (default: False)

        vLLM Backend:
            vllm_base_url: URL of the vLLM server (default: http://localhost:8080)
            request_timeout: Max seconds to wait for vLLM response (default: 300)

        Model Metadata:
            model_id: Unique identifier in org/model format for OpenRouter
            model_display_name: Human-readable name shown in UIs
            organization_id: Your organization identifier
            max_context_length: Maximum tokens in context window
            quantization: Model precision (fp16, fp8, int8, etc.)
            supported_features: List of capabilities exposed to OpenRouter

        Pricing:
            price_per_prompt_token: Cost per input token (USD as string)
            price_per_completion_token: Cost per output token (USD as string)
            Note: Prices are strings to preserve decimal precision

        LMCache:
            lmcache_enabled: Whether LMCache KV offloading is active
            lmcache_config_path: Path to LMCache YAML configuration

        Rate Limiting (reserved for future implementation):
            rate_limit_requests_per_minute: Max requests per minute
            rate_limit_tokens_per_minute: Max tokens per minute

        Monitoring (reserved for future implementation):
            enable_metrics: Whether to expose Prometheus metrics
            metrics_port: Port for metrics endpoint
    """

    # =========================================================================
    # Server Settings
    # =========================================================================
    host: str = "0.0.0.0"
    """Bind address. Use 0.0.0.0 for all interfaces, 127.0.0.1 for localhost only."""

    port: int = 8000
    """HTTP port. Standard ports: 8000 (dev), 80/443 (prod behind proxy)."""

    workers: int = 1
    """Uvicorn worker count. Set to CPU cores for production."""

    debug: bool = False
    """Enable debug mode. NEVER enable in production (exposes stack traces)."""

    # =========================================================================
    # vLLM Backend Connection
    # =========================================================================
    vllm_base_url: str = "http://localhost:8080"
    """
    URL of the vLLM OpenAI-compatible server.

    For Docker Compose: http://vllm:8080
    For Kubernetes: http://vllm-service:8080
    For RunPod: Use the pod's internal URL
    """

    request_timeout: float = 300.0
    """
    Maximum seconds to wait for vLLM response.

    Set high (300s = 5 min) to accommodate:
    - Long context processing (100K+ tokens)
    - High-latency generation with beam search
    - Model loading delays on first request
    """

    # =========================================================================
    # Model Metadata (exposed to OpenRouter)
    # =========================================================================
    model_id: str = "your-org/your-model"
    """
    Unique model identifier in organization/model format.

    Examples:
    - meta-llama/Llama-3.1-8B-Instruct
    - anthropic/claude-3-opus
    - your-company/custom-fine-tune

    This is used by OpenRouter for routing and billing.
    """

    model_display_name: str = "Your Model Display Name"
    """Human-readable name shown in OpenRouter's model selector."""

    organization_id: str = "your-org"
    """Your organization ID, used in owned_by field."""

    max_context_length: int = 131072
    """
    Maximum context window in tokens (128K default).

    Common values:
    - 4096: GPT-3.5 era models
    - 8192: Standard instruction-tuned models
    - 32768: Extended context models
    - 131072: Long-context models (Llama 3.1, Claude 3)
    """

    quantization: str = "fp16"
    """
    Model precision/quantization level.

    Options: fp32, fp16, bf16, fp8, int8, int4, awq, gptq
    Lower precision = faster inference, less memory, slightly lower quality.
    """

    supported_features: List[str] = ["tools", "json_mode", "streaming"]
    """
    Capabilities exposed to OpenRouter.

    Common features:
    - tools: Function calling support
    - json_mode: Guaranteed JSON output
    - streaming: SSE streaming support
    - vision: Image input support
    """

    # =========================================================================
    # Pricing Configuration
    # =========================================================================
    price_per_prompt_token: str = "0.000008"
    """
    Cost per input/prompt token in USD.

    Stored as string to preserve decimal precision for billing.
    OpenRouter uses these for cost estimation and billing.

    Example pricing (per 1M tokens):
    - "0.000008" = $8/M tokens (budget tier)
    - "0.000015" = $15/M tokens (mid tier)
    - "0.000060" = $60/M tokens (premium tier)
    """

    price_per_completion_token: str = "0.000024"
    """
    Cost per output/completion token in USD.

    Typically 2-4x the prompt token price since generation
    is more compute-intensive than prefill.
    """

    # =========================================================================
    # LMCache Settings
    # =========================================================================
    lmcache_enabled: bool = True
    """
    Enable LMCache KV cache offloading.

    When enabled, KV cache is offloaded to CPU RAM (and optionally NVMe)
    to support longer contexts than GPU memory alone allows.

    Requires LMCACHE_CONFIG_FILE environment variable for vLLM.
    """

    lmcache_config_path: str = "configs/lmcache.yaml"
    """Path to LMCache YAML configuration file."""

    # =========================================================================
    # Rate Limiting (reserved for future implementation)
    # =========================================================================
    rate_limit_requests_per_minute: int = 60
    """
    Maximum requests per minute per client.

    NOTE: Not currently implemented. Reserved for future rate limiting.
    """

    rate_limit_tokens_per_minute: int = 100000
    """
    Maximum tokens per minute per client.

    NOTE: Not currently implemented. Reserved for future rate limiting.
    """

    # =========================================================================
    # Monitoring (reserved for future implementation)
    # =========================================================================
    enable_metrics: bool = True
    """
    Enable Prometheus metrics endpoint.

    NOTE: Not currently implemented. Reserved for future monitoring.
    """

    metrics_port: int = 9090
    """Port for Prometheus metrics scraping."""

    # =========================================================================
    # Pydantic Settings Configuration
    # =========================================================================
    class Config:
        """Pydantic settings configuration."""

        env_file = ".env"
        """Load settings from .env file if present."""

        env_file_encoding = "utf-8"
        """Encoding for .env file."""

        extra = "ignore"
        """
        Ignore unknown environment variables.

        This prevents errors when extra vars are set in .env
        that aren't defined in Settings.
        """


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are loaded only once. This improves
    performance and ensures consistent configuration across the application.

    To reload settings (e.g., in tests):
        get_settings.cache_clear()
        new_settings = get_settings()

    Returns:
        Settings instance with values from environment/defaults
    """
    return Settings()

"""
Configuration management for inference server.
Uses pydantic-settings for environment variable parsing.
"""

from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Server configuration loaded from environment variables.
    See .env.example for all available options.
    """

    # ============ Server Settings ============
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    debug: bool = False

    # ============ vLLM Backend ============
    vllm_base_url: str = "http://localhost:8080"
    request_timeout: float = 300.0  # 5 min for long contexts

    # ============ Model Metadata (for OpenRouter) ============
    model_id: str = "your-org/your-model"
    model_display_name: str = "Your Model Display Name"
    organization_id: str = "your-org"
    max_context_length: int = 131072  # 128K
    quantization: str = "fp16"
    supported_features: List[str] = ["tools", "json_mode", "streaming"]

    # ============ Pricing (per token, in USD) ============
    price_per_prompt_token: str = "0.000008"
    price_per_completion_token: str = "0.000024"

    # ============ LMCache Settings ============
    lmcache_enabled: bool = True
    lmcache_config_path: str = "configs/lmcache.yaml"

    # ============ Rate Limiting ============
    rate_limit_requests_per_minute: int = 60
    rate_limit_tokens_per_minute: int = 100000

    # ============ Monitoring ============
    enable_metrics: bool = True
    metrics_port: int = 9090

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()

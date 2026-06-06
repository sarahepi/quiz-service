"""
Configuration module for the AI Quiz Generator microservice.
Loads and validates all environment variables and application settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    """LLM-specific configuration."""
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    max_retries: int
    retry_delay: float


@dataclass(frozen=True)
class ChunkingConfig:
    """Text chunking configuration."""
    max_chunk_tokens: int
    overlap_tokens: int
    chars_per_token: float  # approximate


@dataclass(frozen=True)
class AppConfig:
    """Root application configuration."""
    llm: LLMConfig
    chunking: ChunkingConfig
    host: str
    port: int
    debug: bool
    log_level: str
    max_questions_per_request: int
    min_questions_per_request: int


def load_config() -> AppConfig:
    """
    Load and validate configuration from environment variables.

    Returns:
        AppConfig: Validated application configuration.

    Raises:
        ValueError: If required environment variables are missing.
    """
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is required.")

    return AppConfig(
        llm=LLMConfig(
            api_key=groq_api_key,
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("LLM_RETRY_DELAY", "2.0")),
        ),
        chunking=ChunkingConfig(
            max_chunk_tokens=int(os.getenv("MAX_CHUNK_TOKENS", "3000")),
            overlap_tokens=int(os.getenv("OVERLAP_TOKENS", "200")),
            chars_per_token=float(os.getenv("CHARS_PER_TOKEN", "4.0")),
        ),
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "8000")),
        debug=os.getenv("APP_DEBUG", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        max_questions_per_request=int(os.getenv("MAX_QUESTIONS", "20")),
        min_questions_per_request=int(os.getenv("MIN_QUESTIONS", "1")),
    )


# Singleton config instance
config: AppConfig = load_config()
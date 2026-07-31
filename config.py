"""
config.py
Centralized configuration for the Advanced GenAI Customer Service Bot.

All environment-dependent or tunable values live here. Nothing in the
application code should hardcode API keys, model names, paths, or
thresholds -- they should be imported from this module instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file if present (never committed to git).
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LLMConfig:
    """Settings for the OpenAI-compatible LLM provider."""

    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.3"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "800"))
    )
    request_timeout: int = field(
        default_factory=lambda: int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    )


@dataclass(frozen=True)
class SentimentConfig:
    """Settings for the sentiment analysis module."""

    # Model used for local (no-API-key-required) sentiment scoring.
    model_name: str = field(
        default_factory=lambda: os.getenv(
            "SENTIMENT_MODEL_NAME",
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
        )
    )
    # Below this confidence, we surface the result as "uncertain" rather
    # than asserting a label.
    confidence_floor: float = field(
        default_factory=lambda: float(os.getenv("SENTIMENT_CONFIDENCE_FLOOR", "0.4"))
    )
    # Negative-sentiment threshold at which we flag a message for
    # escalation / human handoff in a real customer-service pipeline.
    escalation_threshold: float = field(
        default_factory=lambda: float(os.getenv("SENTIMENT_ESCALATION_THRESHOLD", "0.75"))
    )
    use_transformer_model: bool = field(
        default_factory=lambda: _get_bool("SENTIMENT_USE_TRANSFORMER", True)
    )


@dataclass(frozen=True)
class PathsConfig:
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    database_dir: Path = BASE_DIR / "database"
    uploads_dir: Path = BASE_DIR / "uploads"
    vector_store_dir: Path = BASE_DIR / "vector_store"
    assets_dir: Path = BASE_DIR / "assets"
    docs_dir: Path = BASE_DIR / "docs"
    sqlite_path: Path = BASE_DIR / "database" / "app.db"
    log_path: Path = BASE_DIR / "data" / "app.log"


@dataclass(frozen=True)
class AppConfig:
    app_title: str = field(
        default_factory=lambda: os.getenv("APP_TITLE", "Advanced GenAI Customer Service Bot")
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))


llm_config = LLMConfig()
sentiment_config = SentimentConfig()
paths_config = PathsConfig()
app_config = AppConfig()


def ensure_directories() -> None:
    """Create all data/storage directories if they don't already exist."""
    for path in (
        paths_config.data_dir,
        paths_config.database_dir,
        paths_config.uploads_dir,
        paths_config.vector_store_dir,
        paths_config.assets_dir,
        paths_config.docs_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

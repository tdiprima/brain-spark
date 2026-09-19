"""Configuration loaded from environment variables with sane defaults."""

import json
import logging
import os
import time
from dataclasses import dataclass

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "gemma4:latest"
DEFAULT_PROFILE_PATH = os.path.join(
    os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
    "brain-spark",
    "user_profile.json",
)
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300
DEFAULT_LOG_LEVEL = "WARNING"

RESEARCH_MAX_TOKENS = 800
TEACH_MAX_TOKENS = 1200
FILENAME_MAX_TOKENS = 30

RESEARCH_WORD_LIMIT = 400
TEACH_WORD_LIMIT = 600

MAX_TOPIC_LENGTH = 500
MAX_NAME_LENGTH = 100

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    ollama_url: str
    ollama_model: str
    profile_path: str
    request_timeout_seconds: int
    log_level: str


def _read_positive_int(variable_name: str, default: int) -> int:
    raw_value = os.environ.get(variable_name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        parsed_value = int(raw_value)
    except ValueError as error:
        raise ConfigError(f"{variable_name} must be an integer, got {raw_value!r}") from error
    if parsed_value <= 0:
        raise ConfigError(f"{variable_name} must be positive, got {parsed_value}")
    return parsed_value


def _read_non_empty(variable_name: str, default: str) -> str:
    raw_value = os.environ.get(variable_name, default).strip()
    if not raw_value:
        raise ConfigError(f"{variable_name} must not be empty")
    return raw_value


def _validate_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        raise ConfigError(f"OLLAMA_URL must start with http:// or https://, got {url!r}")
    return url


def _validate_log_level(level: str) -> str:
    upper_level = level.upper()
    if upper_level not in _VALID_LOG_LEVELS:
        raise ConfigError(f"BRAIN_SPARK_LOG_LEVEL must be one of {sorted(_VALID_LOG_LEVELS)}, got {level!r}")
    return upper_level


def load_config() -> Config:
    """Build and validate configuration from environment variables."""
    return Config(
        ollama_url=_validate_url(_read_non_empty("OLLAMA_URL", DEFAULT_OLLAMA_URL)),
        ollama_model=_read_non_empty("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        profile_path=_read_non_empty("BRAIN_SPARK_PROFILE", DEFAULT_PROFILE_PATH),
        request_timeout_seconds=_read_positive_int(
            "OLLAMA_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT_SECONDS
        ),
        log_level=_validate_log_level(
            _read_non_empty("BRAIN_SPARK_LOG_LEVEL", DEFAULT_LOG_LEVEL)
        ),
    )


class JsonLogFormatter(logging.Formatter):
    """One JSON object per line; json.dumps escapes newlines and control characters."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "component": record.name,
            "event": record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=True)


def configure_logging(log_level: str) -> None:
    """Send structured JSON logs to stderr at the configured level."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    logging.basicConfig(level=getattr(logging, log_level), handlers=[handler], force=True)

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import os
import re

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    config_path: str = "config.yaml"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" | "console"

    # Concurrency — max model calls to run in parallel within a single request
    max_concurrent_per_request: int = 4


settings = Settings()


@dataclass(frozen=True)
class ModelConfig:
    name: str
    type: str          # "ml" | "llm"
    url: str
    for_: str | None = None
    model_id: str | None = None
    labels: dict[str, str] | None = None


_ENV_VAR_RE = re.compile(r"\$\{(\w+)(?::-(.*?))?\}")


def _expand_env(value: Any) -> Any:
    """Expand ${VAR:-default} placeholders in string values."""
    if not isinstance(value, str):
        return value
    return _ENV_VAR_RE.sub(
        lambda m: os.environ.get(m.group(1), m.group(2) or ""), value
    )


def load_model_configs(path: str | None = None) -> list[ModelConfig]:
    config_file = Path(path or settings.config_path)
    with config_file.open() as f:
        data: dict[str, Any] = yaml.safe_load(f)

    configs = []
    for entry in data["models"]:
        configs.append(ModelConfig(
            name=entry["name"],
            type=entry["type"],
            url=_expand_env(entry["url"]),
            for_=entry.get("for"),
            model_id=_expand_env(entry.get("model_id")),
            labels=entry.get("labels"),
        ))
    return configs

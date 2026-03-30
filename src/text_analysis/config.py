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

VALID_DOMAINS: frozenset[str] = frozenset({
    "intelligence", "cybersecurity", "networking", "medicine",
    "finance", "logistics", "military", "human_resources", "legal",
})


_DISCOVERED_PATH = Path("discovered_labels.json")


class CandidateStore:
    """In-memory topic candidate labels — config labels + per-domain discovered labels persisted to disk."""

    def __init__(self, config_labels: list[str]) -> None:
        self._config: set[str] = set(config_labels)
        # domain → list of discovered labels for that domain
        self._discovered: dict[str, list[str]] = {}
        if _DISCOVERED_PATH.exists():
            try:
                import json
                raw = json.loads(_DISCOVERED_PATH.read_text())
                # migrate old flat list format
                if isinstance(raw, list):
                    self._discovered = {}
                else:
                    self._discovered = raw
            except Exception:
                self._discovered = {}

    def all(self, domain: str | None = None) -> list[str]:
        """Return config labels plus discovered labels.

        If domain is given, only include discovered labels for that domain.
        Otherwise include all discovered labels across all domains.
        """
        if domain:
            discovered = self._discovered.get(domain, [])
        else:
            seen: set[str] = set()
            discovered = []
            for labels in self._discovered.values():
                for l in labels:
                    if l not in seen:
                        seen.add(l)
                        discovered.append(l)
        return list(self._config) + [l for l in discovered if l not in self._config]

    def add(self, label: str, domain: str | None = None) -> None:
        key = domain or "_unknown"
        if label not in self._config:
            bucket = self._discovered.setdefault(key, [])
            if label not in bucket:
                bucket.append(label)
                import json
                _DISCOVERED_PATH.write_text(json.dumps(self._discovered, indent=2))

    def is_discovered(self, label: str) -> bool:
        return label not in self._config and any(
            label in bucket for bucket in self._discovered.values()
        )

    def discovered_domain(self, label: str) -> str | None:
        for domain, bucket in self._discovered.items():
            if label in bucket:
                return domain if domain != "_unknown" else None
        return None

    def __contains__(self, label: str) -> bool:
        return label in self._config or self.is_discovered(label)


candidate_store: CandidateStore = CandidateStore([])


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


def load_domain_candidates(path: str | None = None) -> dict[str, list[str]]:
    config_file = Path(path or settings.config_path)
    with config_file.open() as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return data.get("domain_candidates") or {}


def load_all_candidates(path: str | None = None) -> list[str]:
    """Flatten all domain_candidates into a deduplicated list."""
    domain_map = load_domain_candidates(path)
    seen: set[str] = set()
    result: list[str] = []
    for labels in domain_map.values():
        for label in labels:
            if label not in seen:
                seen.add(label)
                result.append(label)
    return result


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

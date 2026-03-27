import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import structlog

from sentiment.observability import metrics

logger = structlog.get_logger(__name__)


class SentimentClientError(Exception):
    """Raised when a sentiment client fails to produce a result."""


@dataclass(frozen=True)
class SentimentResult:
    label: str  # "positive" | "neutral" | "negative"
    score: float  # 0.0–1.0 confidence
    raw: dict[str, Any]


class SentimentClient(ABC):
    """Abstract base for all sentiment model clients."""

    model_name: str

    @abstractmethod
    async def _predict(self, text: str) -> SentimentResult:
        """Perform the actual prediction. Implement in subclasses."""

    async def predict(self, text: str) -> SentimentResult:
        """Public entry point — wraps _predict with metrics recording."""
        metrics.record_request(self.model_name)
        start = time.perf_counter()
        try:
            result = await self._predict(text)
        except SentimentClientError:
            elapsed = time.perf_counter() - start
            metrics.record_error(self.model_name, "client_error")
            metrics.record_latency(self.model_name, "unknown", elapsed)
            logger.warning("sentiment_client_error", model=self.model_name)
            raise
        except Exception as exc:
            elapsed = time.perf_counter() - start
            error_type = type(exc).__name__.lower()
            metrics.record_error(self.model_name, error_type)
            metrics.record_latency(self.model_name, "unknown", elapsed)
            logger.exception("sentiment_client_unexpected_error", model=self.model_name)
            raise SentimentClientError(str(exc)) from exc

        elapsed = time.perf_counter() - start
        metrics.record_latency(self.model_name, result.label, elapsed)
        metrics.record_confidence(self.model_name, result.label, result.score)
        logger.info(
            "sentiment_prediction",
            model=self.model_name,
            label=result.label,
            score=round(result.score, 4),
            latency_s=round(elapsed, 4),
        )
        return result

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import structlog

from text_analysis.observability import metrics

logger = structlog.get_logger(__name__)


class ModelClientError(Exception):
    """Raised when a model client fails to produce a result."""


@dataclass(frozen=True)
class PredictionResult:
    labels: list[tuple[str, float]]  # ranked list of (label, score), best first
    raw: dict[str, Any]

    @property
    def label(self) -> str:
        return self.labels[0][0]

    @property
    def score(self) -> float:
        return self.labels[0][1]


class ModelClient(ABC):
    """Abstract base for all model clients (sentiment, topic, etc.)."""

    model_name: str
    task: str

    @abstractmethod
    async def _predict(self, text: str) -> PredictionResult:
        """Perform the actual prediction. Implement in subclasses."""

    async def predict(self, text: str) -> PredictionResult:
        """Public entry point — wraps _predict with metrics recording."""
        metrics.record_request(self.model_name)
        start = time.perf_counter()
        try:
            result = await self._predict(text)
        except ModelClientError:
            elapsed = time.perf_counter() - start
            metrics.record_error(self.model_name, "client_error")
            metrics.record_latency(self.model_name, "unknown", elapsed)
            logger.warning("model_client_error", model=self.model_name)
            raise
        except Exception as exc:
            elapsed = time.perf_counter() - start
            error_type = type(exc).__name__.lower()
            metrics.record_error(self.model_name, error_type)
            metrics.record_latency(self.model_name, "unknown", elapsed)
            logger.exception("model_client_unexpected_error", model=self.model_name)
            raise ModelClientError(str(exc)) from exc

        elapsed = time.perf_counter() - start
        metrics.record_latency(self.model_name, result.label, elapsed)
        metrics.record_confidence(self.model_name, result.label, result.score)
        logger.info(
            "model_prediction",
            model=self.model_name,
            task=self.task,
            label=result.label,
            score=round(result.score, 4),
            latency_s=round(elapsed, 4),
        )
        return result

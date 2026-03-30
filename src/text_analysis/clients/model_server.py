import httpx
import structlog
from slugify import slugify

from text_analysis.clients.base import ModelClient, ModelClientError, PredictionResult

logger = structlog.get_logger(__name__)

class ModelServerClient(ModelClient):
    """Client for the internal model-server containers (HuggingFace pipeline wrapper)."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        http_client: httpx.AsyncClient,
        label_map: dict[str, str],
        task: str = "sentiment",
        candidate_labels: list[str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.task = task
        self._base_url = base_url.rstrip("/")
        self._http = http_client
        self._label_map = label_map
        self._candidate_labels = candidate_labels or []

    def _normalise_label(self, raw_label: str) -> str:
        if not self._label_map:
            return slugify(raw_label, separator="_")
        normalised = self._label_map.get(raw_label) or self._label_map.get(raw_label.lower())
        if normalised is None:
            logger.warning("unknown_label", model=self.model_name, raw_label=raw_label)
            return "neutral"
        return normalised

    async def _predict(self, text: str, candidate_labels: list[str] | None = None, **kwargs) -> PredictionResult:
        url = f"{self._base_url}/predict"
        effective_labels = candidate_labels if candidate_labels is not None else self._candidate_labels
        payload: dict = {"text": text}
        if effective_labels:
            payload["candidate_labels"] = effective_labels
        try:
            response = await self._http.post(url, json=payload, timeout=120.0)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ModelClientError(f"Timeout calling {url}") from exc
        except httpx.HTTPStatusError as exc:
            raise ModelClientError(
                f"HTTP {exc.response.status_code} from {url}"
            ) from exc
        except httpx.RequestError as exc:
            raise ModelClientError(f"Request error calling {url}: {exc}") from exc

        try:
            data: dict = response.json()
            raw_labels: list[dict] = data["labels"]
            if not raw_labels:
                raise ModelClientError(
                    f"Empty labels list from {url}: {response.text!r}"
                )
            labels = [
                (self._normalise_label(item["label"]), float(item["score"]))
                for item in raw_labels
            ]
        except ModelClientError:
            raise
        except (KeyError, ValueError, TypeError) as exc:
            raise ModelClientError(
                f"Unexpected response shape from {url}: {response.text!r}"
            ) from exc

        return PredictionResult(labels=labels, raw=data)

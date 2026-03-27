import httpx
import structlog

from sentiment.clients.base import SentimentClient, SentimentClientError, SentimentResult

logger = structlog.get_logger(__name__)

# Label maps: raw model output label → normalised label
SIEBERT_LABELS: dict[str, str] = {
    "POSITIVE": "positive",
    "NEGATIVE": "negative",
}

CARDIFFNLP_LABELS: dict[str, str] = {
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
}

DISTILBERT_LABELS: dict[str, str] = {
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
}


class ModelServerClient(SentimentClient):
    """Client for the internal model-server containers (HuggingFace pipeline wrapper)."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        http_client: httpx.AsyncClient,
        label_map: dict[str, str],
    ) -> None:
        self.model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._http = http_client
        self._label_map = label_map

    def _normalise_label(self, raw_label: str) -> str:
        normalised = self._label_map.get(raw_label) or self._label_map.get(raw_label.lower())
        if normalised is None:
            logger.warning("unknown_label", model=self.model_name, raw_label=raw_label)
            return "neutral"
        return normalised

    async def _predict(self, text: str) -> SentimentResult:
        url = f"{self._base_url}/predict"
        try:
            response = await self._http.post(url, json={"text": text}, timeout=30.0)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise SentimentClientError(f"Timeout calling {url}") from exc
        except httpx.HTTPStatusError as exc:
            raise SentimentClientError(
                f"HTTP {exc.response.status_code} from {url}"
            ) from exc
        except httpx.RequestError as exc:
            raise SentimentClientError(f"Request error calling {url}: {exc}") from exc

        try:
            data: dict = response.json()
            raw_label: str = data["label"]
            score: float = float(data["score"])
        except (KeyError, ValueError, TypeError) as exc:
            raise SentimentClientError(
                f"Unexpected response shape from {url}: {response.text!r}"
            ) from exc

        return SentimentResult(
            label=self._normalise_label(raw_label),
            score=score,
            raw=data,
        )

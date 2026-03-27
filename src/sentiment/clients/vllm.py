import json

import httpx
import structlog

from sentiment.clients.base import SentimentClient, SentimentClientError, SentimentResult

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = "You are a precise sentiment classifier that outputs only valid JSON."

USER_PROMPT_TEMPLATE = """\
Analyze the sentiment of the text below.

Respond with a JSON object containing exactly two fields:
- "label": one of "positive", "neutral", or "negative"
- "score": a float between 0.0 and 1.0 representing your confidence

Do not include any explanation or extra text — only the JSON object.

Example output: {{"label": "positive", "score": 0.85}}

Text: {text}"""

VALID_LABELS = {"positive", "neutral", "negative"}


class VllmClient(SentimentClient):
    """Client for a vLLM server using the OpenAI-compatible chat completions API."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        model_id: str,
        http_client: httpx.AsyncClient,
    ) -> None:
        self.model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._http = http_client

    async def _predict(self, text: str) -> SentimentResult:
        url = f"{self._base_url}/v1/chat/completions"
        payload = {
            "model": self._model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
            ],
            "temperature": 0.0,
            "max_tokens": 64,
            "response_format": {"type": "json_object"},
        }

        try:
            response = await self._http.post(url, json=payload, timeout=120.0)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise SentimentClientError(f"Timeout calling vLLM at {url}") from exc
        except httpx.HTTPStatusError as exc:
            raise SentimentClientError(
                f"HTTP {exc.response.status_code} from vLLM: {exc.response.text!r}"
            ) from exc
        except httpx.RequestError as exc:
            raise SentimentClientError(f"Request error calling vLLM: {exc}") from exc

        try:
            body = response.json()
            content: str = body["choices"][0]["message"]["content"]
            parsed: dict = json.loads(content)
            raw_label: str = parsed["label"]
            score: float = float(parsed.get("score", 0.9))
        except (KeyError, IndexError, json.JSONDecodeError, ValueError, TypeError) as exc:
            raise SentimentClientError(
                f"Failed to parse vLLM response: {response.text!r}"
            ) from exc

        label = raw_label.lower().strip()
        if label not in VALID_LABELS:
            logger.warning("vllm_unknown_label", raw_label=raw_label)
            label = "neutral"

        score = max(0.0, min(1.0, score))

        return SentimentResult(label=label, score=score, raw=body)

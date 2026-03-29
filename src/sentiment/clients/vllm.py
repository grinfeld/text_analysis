import json

import httpx
import structlog

from sentiment.clients.base import ModelClient, ModelClientError, PredictionResult

logger = structlog.get_logger(__name__)

# ── Sentiment prompts ──────────────────────────────────────────────────────────

SENTIMENT_SYSTEM_PROMPT = "You are a precise sentiment classifier that outputs only valid JSON."

SENTIMENT_USER_PROMPT_TEMPLATE = """\
Analyze the sentiment of the text below.

Respond with a JSON object containing exactly two fields:
- "label": one of "positive", "neutral", or "negative"
- "score": a float between 0.0 and 1.0 representing your confidence

Do not include any explanation or extra text — only the JSON object.

Example output: {{"label": "positive", "score": 0.85}}

Text: {text}"""

SENTIMENT_VALID_LABELS: frozenset[str] = frozenset({"positive", "neutral", "negative"})

# ── Topic prompts ──────────────────────────────────────────────────────────────

TOPIC_SYSTEM_PROMPT = "You are a precise topic classifier that outputs only valid JSON."

TOPIC_USER_PROMPT_TEMPLATE = """\
Classify the topic of the text below.

Use one of these example slugs if it fits, or propose your own snake_case slug if none of them applies:
arms_trafficking, financial_crime, surveillance_operation, human_intelligence,
signals_intelligence, covert_operation, money_laundering, sanctions_evasion,
recruitment, daily_reporting, network_configuration, routing, administrative,
organizational_structure, spatial_context, temporal_reference

If the text has no clear topic, set "label" to an empty string.

Respond with a JSON object containing exactly two fields:
- "label": a snake_case topic slug, or an empty string if no topic applies
- "score": a float between 0.0 and 1.0 representing your confidence

Do not include any explanation or extra text — only the JSON object.

Example output: {{"label": "financial_crime", "score": 0.88}}

Text: {text}"""

class VllmClient(ModelClient):
    """Client for a vLLM server using the OpenAI-compatible chat completions API."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        model_id: str,
        http_client: httpx.AsyncClient,
        task: str = "sentiment",
        system_prompt: str = SENTIMENT_SYSTEM_PROMPT,
        user_prompt_template: str = SENTIMENT_USER_PROMPT_TEMPLATE,
        valid_labels: frozenset[str] = SENTIMENT_VALID_LABELS,
    ) -> None:
        self.model_name = model_name
        self.task = task
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._http = http_client
        self._system_prompt = system_prompt
        self._user_prompt_template = user_prompt_template
        self._valid_labels = valid_labels

    async def _predict(self, text: str) -> PredictionResult:
        url = f"{self._base_url}/v1/chat/completions"
        payload = {
            "model": self._model_id,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": self._user_prompt_template.format(text=text)},
            ],
            "temperature": 0.0,
            "max_tokens": 64,
            "response_format": {"type": "json_object"},
        }

        try:
            response = await self._http.post(url, json=payload, timeout=120.0)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ModelClientError(f"Timeout calling vLLM at {url}") from exc
        except httpx.HTTPStatusError as exc:
            raise ModelClientError(
                f"HTTP {exc.response.status_code} from vLLM: {exc.response.text!r}"
            ) from exc
        except httpx.RequestError as exc:
            raise ModelClientError(f"Request error calling vLLM: {exc}") from exc

        try:
            body = response.json()
            content: str = body["choices"][0]["message"]["content"]
            parsed: dict = json.loads(content)
            raw_label: str = parsed["label"]
            score: float = float(parsed.get("score", 0.9))
        except (KeyError, IndexError, json.JSONDecodeError, ValueError, TypeError) as exc:
            raise ModelClientError(
                f"Failed to parse vLLM response: {response.text!r}"
            ) from exc

        label = raw_label.lower().strip()
        if self._valid_labels and label not in self._valid_labels:
            logger.warning("vllm_unknown_label", raw_label=raw_label, task=self.task, fallback="neutral")
            label = "neutral"
        # open-ended mode (empty valid_labels): pass label through as-is, including empty string

        score = max(0.0, min(1.0, score))

        return PredictionResult(label=label, score=score, raw=body)

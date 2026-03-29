import json

import httpx
import structlog

from text_analysis.clients.base import ModelClient, ModelClientError, PredictionResult

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
Classify the topic of the text below. Return the top 3 most relevant topics ranked by confidence.

Use one of these example slugs if it fits, or propose your own snake_case slug if none of them applies:
arms_trafficking, financial_crime, surveillance_operation, human_intelligence,
signals_intelligence, covert_operation, money_laundering, sanctions_evasion,
recruitment, daily_reporting, network_configuration, routing, administrative,
organizational_structure, spatial_context, temporal_reference

Respond with a JSON object containing exactly one field:
- "labels": an array of exactly 3 objects, each with "label" (snake_case slug) and "score" (float 0.0–1.0), ranked from most to least confident

Do not include any explanation or extra text — only the JSON object.

Example output: {{"labels": [{{"label": "financial_crime", "score": 0.88}}, {{"label": "money_laundering", "score": 0.65}}, {{"label": "sanctions_evasion", "score": 0.42}}]}}

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
            "max_tokens": 200,
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
        except (KeyError, IndexError, json.JSONDecodeError, ValueError, TypeError) as exc:
            raise ModelClientError(
                f"Failed to parse vLLM response: {response.text!r}"
            ) from exc

        if "labels" in parsed:
            # Topic mode — ranked list of {label, score}
            try:
                raw_list = parsed["labels"]
                if not raw_list:
                    raise ModelClientError(
                        f"vLLM returned empty labels list: {response.text!r}"
                    )
                labels = [
                    (item["label"].lower().strip(), max(0.0, min(1.0, float(item["score"]))))
                    for item in raw_list
                ]
            except ModelClientError:
                raise
            except (KeyError, ValueError, TypeError) as exc:
                raise ModelClientError(
                    f"Failed to parse vLLM labels list: {response.text!r}"
                ) from exc
        else:
            # Sentiment mode — single {label, score}
            try:
                raw_label: str = parsed["label"]
                score: float = float(parsed["score"])
            except (KeyError, ValueError, TypeError) as exc:
                raise ModelClientError(
                    f"Failed to parse vLLM response: {response.text!r}"
                ) from exc
            label = raw_label.lower().strip()
            if self._valid_labels and label not in self._valid_labels:
                logger.warning("vllm_unknown_label", raw_label=raw_label, task=self.task, fallback="neutral")
                label = "neutral"
            score = max(0.0, min(1.0, score))
            labels = [(label, score)]

        return PredictionResult(labels=labels, raw=body)

import json

import httpx
import structlog
from slugify import slugify

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
Identify the 3 most relevant subject categories that best describe what the text is about — not its tone or style, but the real-world domain, activity, or event it refers to. A topic is a concise label for the subject matter, such as an industry, threat type, operational activity, or named phenomenon.

Rules:
- A domain is the professional or industry field the text belongs to (e.g. finance, medicine, cybersecurity). A label must never equal the domain name itself.{domain_rule_line}
- Each label must be snake_case, max 2 words (e.g. wire_fraud, port_inspection).
- Labels must be specific and precise — avoid generic terms like "activity" or "event".
- At least one label must be a broad category-level topic that generalises the type of content (e.g. clinical_diagnosis, threat_assessment, financial_report) — not just a specific detail from the text.
- A label must NOT name the subject/source or object/target of the action — those are entities, not topics.{entity_rule_line}
- These are illustrative examples only, do not copy them: financial_crime, cyber_intrusion, border_control.

Respond with a JSON object containing exactly one field:
- "labels": an array of exactly 3 objects, each with "label" (snake_case, max 2 words) and "score" (float 0.0–1.0), ranked from most to least confident

Do not include any explanation or extra text — only the JSON object.
{domain_line}{entity_line}
Text: {text}"""

class VllmClient(ModelClient):
    """Client for a vLLM server using the OpenAI-compatible chat completions API."""

    _E5_SIMILARITY_THRESHOLD = 0.92

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
        candidate_labels: list[str] | None = None,
        candidate_store=None,
        e5_small_url: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.task = task
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._http = http_client
        self._system_prompt = system_prompt
        self._user_prompt_template = user_prompt_template
        self._valid_labels = valid_labels
        self._candidate_labels = candidate_labels or []
        self._candidate_store = candidate_store
        self._e5_small_url = e5_small_url

    async def _resolve_novel_labels(self, labels: list[tuple[str, float]], domain: str | None = None) -> list[tuple[str, float, str | None]]:
        if not self._candidate_store or not self._e5_small_url:
            return [(label, score, None) for label, score in labels]
        resolved = []
        candidates = self._candidate_store.all(domain=domain)
        for label, score in labels:
            if label in self._candidate_store:
                resolved.append((label, score, None))
                continue
            # Novel label — run through e5-small to find nearest candidate
            try:
                resp = await self._http.post(
                    f"{self._e5_small_url}/predict",
                    json={"text": label, "candidate_labels": candidates},
                    timeout=30.0,
                )
                resp.raise_for_status()
                top = resp.json()["labels"][0]
                if top["score"] >= self._E5_SIMILARITY_THRESHOLD:
                    logger.info("vllm_label_mapped", novel=label, mapped=top["label"], similarity=top["score"])
                    resolved.append((top["label"], score, label))
                else:
                    logger.info("vllm_label_new", label=label, top_similarity=top["score"])
                    self._candidate_store.add(label, domain=domain)
                    resolved.append((label, score, None))
            except Exception as exc:
                logger.warning("vllm_e5_lookup_failed", label=label, error=str(exc))
                resolved.append((label, score, None))
        return resolved

    async def _predict(self, text: str, candidate_labels: list[str] | None = None, domain: str | None = None,
                       source: str | None = None, relation: str | None = None, target: str | None = None, **kwargs) -> PredictionResult:
        url = f"{self._base_url}/v1/chat/completions"
        domain_line = f"Domain: {domain.capitalize()}\n" if domain else ""
        domain_rule_line = f" For this text: \"{domain}\" is not an acceptable label." if domain else ""
        entity_parts = []
        if source:
            entity_parts.append(f"source/subject: \"{source}\"")
        if relation:
            entity_parts.append(f"relation: \"{relation}\"")
        if target:
            entity_parts.append(f"object/target: \"{target}\"")
        entity_line = f"Context: {', '.join(entity_parts)}\n" if entity_parts else ""
        entity_rule_line = (
            f" For this text: do not use {' or '.join(f'\"{v}\"' for v in [source, relation, target] if v)} as a label."
            if entity_parts else ""
        )
        payload = {
            "model": self._model_id,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": self._user_prompt_template.format(
                    text=text, domain_line=domain_line, domain_rule_line=domain_rule_line,
                    entity_line=entity_line, entity_rule_line=entity_rule_line,
                )},
            ],
            "temperature": 0.0,
            "max_tokens": 200,
            "response_format": {"type": "json_object"},
        }

        logger.debug(
            "vllm_request",
            model=self._model_id,
            task=self.task,
            domain=domain,
            user_prompt=payload["messages"][1]["content"],
        )

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
            content: str = body["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("```", 2)[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            parsed: dict = json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError, ValueError, TypeError) as exc:
            raise ModelClientError(
                f"Failed to parse vLLM response: {response.text!r}"
            ) from exc

        logger.debug("vllm_response", model=self._model_id, task=self.task, content=content)

        if "labels" in parsed:
            # Topic mode — ranked list of {label, score}
            try:
                raw_list = parsed["labels"]
                if not raw_list:
                    raise ModelClientError(
                        f"vLLM returned empty labels list: {response.text!r}"
                    )
                labels = [
                    (slugify(item["label"], separator="_"), max(0.0, min(1.0, float(item["score"]))), None)
                    for item in raw_list
                ]
            except ModelClientError:
                raise
            except (KeyError, ValueError, TypeError) as exc:
                raise ModelClientError(
                    f"Failed to parse vLLM labels list: {response.text!r}"
                ) from exc
            labels = await self._resolve_novel_labels([(l, s) for l, s, _ in labels], domain=domain)
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
            labels = [(label, score, None)]

        return PredictionResult(labels=labels, raw=body)

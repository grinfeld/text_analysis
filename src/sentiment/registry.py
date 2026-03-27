import httpx

from sentiment.clients.base import SentimentClient, SentimentClientError
from sentiment.clients.model_server import (
    CARDIFFNLP_LABELS,
    DISTILBERT_LABELS,
    SIEBERT_LABELS,
    ModelServerClient,
)
from sentiment.clients.vllm import VllmClient
from sentiment.config import settings

_registry: dict[str, SentimentClient] = {}
_hf_http: httpx.AsyncClient | None = None
_vllm_http: httpx.AsyncClient | None = None


def init() -> None:
    """Create HTTP clients and populate the registry. Called from app lifespan."""
    global _registry, _hf_http, _vllm_http

    _hf_http = httpx.AsyncClient()
    _vllm_http = httpx.AsyncClient()

    _registry = {
        "siebert/sentiment-roberta-large-english": ModelServerClient(
            model_name="siebert/sentiment-roberta-large-english",
            base_url=settings.siebert_url,
            http_client=_hf_http,
            label_map=SIEBERT_LABELS,
        ),
        "cardiffnlp/twitter-roberta-base-sentiment-latest": ModelServerClient(
            model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
            base_url=settings.cardiffnlp_url,
            http_client=_hf_http,
            label_map=CARDIFFNLP_LABELS,
        ),
        "lxyuan/distilbert-base-multilingual-cased-sentiments-student": ModelServerClient(
            model_name="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            base_url=settings.distilbert_url,
            http_client=_hf_http,
            label_map=DISTILBERT_LABELS,
        ),
        "vllm": VllmClient(
            base_url=settings.vllm_url,
            model_id=settings.vllm_model,
            http_client=_vllm_http,
        ),
    }


def get_client(model_name: str) -> SentimentClient:
    client = _registry.get(model_name)
    if client is None:
        valid = ", ".join(f'"{k}"' for k in _registry)
        raise SentimentClientError(f"Unknown model {model_name!r}. Valid: {valid}")
    return client


async def close_all() -> None:
    if _hf_http is not None:
        await _hf_http.aclose()
    if _vllm_http is not None:
        await _vllm_http.aclose()

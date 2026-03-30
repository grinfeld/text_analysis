import asyncio
import time
from typing import Literal

import structlog
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field, field_validator

from text_analysis.clients.base import ModelClientError
import text_analysis.config as _config
from text_analysis.config import settings, VALID_DOMAINS
from text_analysis.registry import get_clients_for

logger = structlog.get_logger(__name__)

router = APIRouter()


class AnalyseRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str
    for_: Literal["sentiment", "topic"] = Field("sentiment", alias="for")
    domain: str | None = Field(None)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty or whitespace")
        return v

    @field_validator("domain")
    @classmethod
    def domain_must_be_valid(cls, v: str | None) -> str | None:
        if not v:
            return None
        v = v.lower().strip()
        if v not in VALID_DOMAINS:
            raise ValueError(f"domain must be one of: {sorted(VALID_DOMAINS)}")
        return v


class LabelScore(BaseModel):
    label: str
    score: float


class ModelResult(BaseModel):
    model: str
    labels: list[LabelScore]
    latency_s: float
    error: str | None = None


class AnalyseResponse(BaseModel):
    results: list[ModelResult]


async def _call_client(
    client,
    text: str,
    sem: asyncio.Semaphore,
    candidate_labels: list[str] | None = None,
    domain: str | None = None,
) -> ModelResult:
    async with sem:
        start = time.perf_counter()
        try:
            result = await client.predict(text, candidate_labels=candidate_labels, domain=domain)
            latency_s = time.perf_counter() - start
            return ModelResult(
                model=client.model_name,
                labels=[LabelScore(label=l, score=round(s, 2)) for l, s in result.labels],
                latency_s=round(latency_s, 4),
            )
        except ModelClientError as exc:
            latency_s = time.perf_counter() - start
            logger.error("model_failed", model=client.model_name, error=str(exc))
            return ModelResult(
                model=client.model_name,
                labels=[],
                latency_s=round(latency_s, 4),
                error=str(exc),
            )


class CandidateEntry(BaseModel):
    label: str
    source: str  # "config" | "discovered"
    domain: str | None = None


@router.get("/candidates")
def candidates() -> list[CandidateEntry]:
    result = []
    for label in _config.candidate_store.all():
        if _config.candidate_store.is_discovered(label):
            domain = _config.candidate_store.discovered_domain(label)
            result.append(CandidateEntry(label=label, source="discovered", domain=domain))
        else:
            result.append(CandidateEntry(label=label, source="config"))
    return result


@router.post("/analyse", response_model=AnalyseResponse)
async def analyse(req: AnalyseRequest) -> AnalyseResponse:
    candidate_override: list[str] | None = None
    if req.for_ == "topic":
        candidate_override = _config.candidate_store.all(domain=req.domain if req.domain else None)

    clients = get_clients_for(req.for_)
    sem = asyncio.Semaphore(settings.max_concurrent_per_request)
    results = await asyncio.gather(
        *[_call_client(client, req.text, sem, candidate_labels=candidate_override, domain=req.domain) for client in clients]
    )
    return AnalyseResponse(results=list(results))

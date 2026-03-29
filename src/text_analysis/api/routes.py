import asyncio
import time
from typing import Literal

import structlog
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field, field_validator

from text_analysis.clients.base import ModelClientError
from text_analysis.config import settings
from text_analysis.registry import get_clients_for

logger = structlog.get_logger(__name__)

router = APIRouter()


class AnalyseRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str
    for_: Literal["sentiment", "topic"] = Field("sentiment", alias="for")

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty or whitespace")
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


async def _call_client(client, text: str, sem: asyncio.Semaphore) -> ModelResult:
    async with sem:
        start = time.perf_counter()
        try:
            result = await client.predict(text)
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


@router.post("/analyse", response_model=AnalyseResponse)
async def analyse(req: AnalyseRequest) -> AnalyseResponse:
    clients = get_clients_for(req.for_)
    sem = asyncio.Semaphore(settings.max_concurrent_per_request)
    results = await asyncio.gather(
        *[_call_client(client, req.text, sem) for client in clients]
    )
    return AnalyseResponse(results=list(results))

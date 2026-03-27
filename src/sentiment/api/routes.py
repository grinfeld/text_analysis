import time

import structlog
from fastapi import APIRouter
from pydantic import BaseModel, field_validator

from sentiment.clients.base import SentimentClientError
from sentiment.registry import get_all_clients

logger = structlog.get_logger(__name__)

router = APIRouter()


class AnalyseRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty or whitespace")
        return v


class ModelResult(BaseModel):
    model: str
    label: str | None = None
    score: float | None = None
    latency_s: float
    error: str | None = None


class AnalyseResponse(BaseModel):
    results: list[ModelResult]


@router.post("/analyse", response_model=AnalyseResponse)
async def analyse(req: AnalyseRequest) -> AnalyseResponse:
    results = []
    for client in get_all_clients():
        start = time.perf_counter()
        try:
            result = await client.predict(req.text)
            latency_s = time.perf_counter() - start
            results.append(ModelResult(
                model=client.model_name,
                label=result.label,
                score=result.score,
                latency_s=round(latency_s, 4),
            ))
        except SentimentClientError as exc:
            latency_s = time.perf_counter() - start
            logger.error("model_failed", model=client.model_name, error=str(exc))
            results.append(ModelResult(
                model=client.model_name,
                latency_s=round(latency_s, 4),
                error=str(exc),
            ))

    return AnalyseResponse(results=results)

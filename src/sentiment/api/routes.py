import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from sentiment.clients.base import SentimentClientError
from sentiment.registry import get_client

logger = structlog.get_logger(__name__)

router = APIRouter()


class SentimentRequest(BaseModel):
    model: str
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty or whitespace")
        return v


class SentimentResponse(BaseModel):
    model: str
    label: str
    score: float
    raw: dict


@router.post("/sentiment", response_model=SentimentResponse)
async def classify_sentiment(req: SentimentRequest) -> SentimentResponse:
    try:
        client = get_client(req.model)
    except SentimentClientError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        result = await client.predict(req.text)
    except SentimentClientError as exc:
        logger.error("prediction_failed", model=req.model, error=str(exc))
        raise HTTPException(status_code=502, detail=str(exc))

    return SentimentResponse(
        model=req.model,
        label=result.label,
        score=result.score,
        raw=result.raw,
    )

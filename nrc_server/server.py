import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from nrclex import NRCLex
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_NAME = "nrc"

# Emotions that map to each sentiment class
_POSITIVE_EMOTIONS = {"joy", "trust", "anticipation", "surprise"}
_NEGATIVE_EMOTIONS = {"fear", "anger", "sadness", "disgust"}


def _analyse(text: str) -> tuple[str, float]:
    scores = NRCLex(text).raw_emotion_scores
    pos = sum(scores.get(e, 0) for e in _POSITIVE_EMOTIONS)
    neg = sum(scores.get(e, 0) for e in _NEGATIVE_EMOTIONS)
    total = pos + neg

    if total == 0:
        return "neutral", 0.0

    if pos > neg:
        label = "positive"
        score = round(pos / total, 4)
    elif neg > pos:
        label = "negative"
        score = round(neg / total, 4)
    else:
        label = "neutral"
        score = round(0.5, 4)

    return label, score


@asynccontextmanager
async def lifespan(app: FastAPI):
    _analyse("warmup")
    logger.info("NRC warmed up")
    yield


app = FastAPI(title="NRC Emotion Lexicon Server", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty or whitespace")
        return v


class PredictResponse(BaseModel):
    label: str
    score: float


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    label, score = _analyse(req.text)
    return PredictResponse(label=label, score=score)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

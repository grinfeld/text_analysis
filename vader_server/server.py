import logging
from contextlib import asynccontextmanager

import nltk
import uvicorn
from fastapi import FastAPI
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_NAME = "vader"

logger.info("Downloading VADER lexicon")
nltk.download("vader_lexicon", quiet=True)
_analyzer = SentimentIntensityAnalyzer()
logger.info("VADER ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warmup: run one inference so the first real request is not cold
    _analyzer.polarity_scores("warmup")
    logger.info("VADER warmed up")
    yield


app = FastAPI(title="VADER Sentiment Server", lifespan=lifespan)


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
    scores = _analyzer.polarity_scores(req.text)
    compound = scores["compound"]  # -1.0 to 1.0

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    # Normalize abs(compound) to a confidence score in [0, 1]
    score = round(abs(compound), 4)

    return PredictResponse(label=label, score=score)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

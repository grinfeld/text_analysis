import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from handlers import HANDLERS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TASK = os.environ.get("TASK", "text-classification")
MODEL_NAME = os.environ.get("MODEL_NAME", TASK)

if TASK not in HANDLERS:
    raise RuntimeError(
        f"Unknown TASK: {TASK!r}. Must be one of: {', '.join(HANDLERS)}"
    )

_handler = HANDLERS[TASK]()


# ── FastAPI app ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _handler.warmup()
    logger.info("Warmed up: %s (task=%s)", MODEL_NAME, TASK)
    yield


app = FastAPI(title=f"Model Server — {MODEL_NAME}", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str
    candidate_labels: list[str] = []


class LabelScore(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    labels: list[LabelScore]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    results = _handler.predict(req.text, req.candidate_labels)
    return PredictResponse(labels=[LabelScore(label=l, score=s) for l, s in results])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME, "task": TASK}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

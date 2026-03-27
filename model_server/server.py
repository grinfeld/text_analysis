import logging
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_NAME = os.environ.get("MODEL_NAME", "")
if not MODEL_NAME:
    raise RuntimeError("MODEL_NAME environment variable is required")

logger.info("Loading model: %s", MODEL_NAME)
_pipeline = pipeline("text-classification", model=MODEL_NAME, top_k=1)
logger.info("Model loaded: %s", MODEL_NAME)

app = FastAPI(title=f"Model Server — {MODEL_NAME}")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    score: float


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    results = _pipeline(req.text)
    # pipeline returns [[{"label": ..., "score": ...}]] with top_k=1
    top = results[0][0] if isinstance(results[0], list) else results[0]
    return PredictResponse(label=top["label"], score=top["score"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

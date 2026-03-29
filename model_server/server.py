import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TASK = os.environ.get("TASK", "text-classification")
MODEL_NAME = os.environ.get("MODEL_NAME", TASK)  # lexicon tasks use TASK as name if MODEL_NAME not set

_CANDIDATE_LABELS: list[str] = [
    l.strip()
    for l in os.environ.get("CANDIDATE_LABELS", "").replace(",", "\n").split("\n")
    if l.strip()
]

# ── Task initialisation ────────────────────────────────────────────────────────

if TASK == "text-classification":
    if not MODEL_NAME or MODEL_NAME == TASK:
        raise RuntimeError("MODEL_NAME is required for text-classification")
    logger.info("Loading model: %s", MODEL_NAME)
    _hf_pipeline = pipeline("text-classification", model=MODEL_NAME, top_k=1)
    logger.info("Model loaded: %s", MODEL_NAME)

elif TASK == "zero-shot-classification":
    if not MODEL_NAME or MODEL_NAME == TASK:
        raise RuntimeError("MODEL_NAME is required for zero-shot-classification")
    if not _CANDIDATE_LABELS:
        raise RuntimeError("CANDIDATE_LABELS is required for zero-shot-classification")
    logger.info("Loading model: %s", MODEL_NAME)
    _hf_pipeline = pipeline("zero-shot-classification", model=MODEL_NAME)
    logger.info("Model loaded: %s", MODEL_NAME)

elif TASK == "embedding":
    if not MODEL_NAME or MODEL_NAME == TASK:
        raise RuntimeError("MODEL_NAME is required for embedding")
    if not _CANDIDATE_LABELS:
        raise RuntimeError("CANDIDATE_LABELS is required for embedding")
    import numpy as np
    from sentence_transformers import SentenceTransformer
    logger.info("Loading model: %s", MODEL_NAME)
    _st_model = SentenceTransformer(MODEL_NAME)
    logger.info("Encoding %d candidate labels", len(_CANDIDATE_LABELS))
    _label_embeddings: np.ndarray = _st_model.encode(_CANDIDATE_LABELS, normalize_embeddings=True)
    logger.info("Model loaded: %s", MODEL_NAME)

elif TASK == "vader":
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    logger.info("Downloading VADER lexicon")
    nltk.download("vader_lexicon", quiet=True)
    _vader = SentimentIntensityAnalyzer()
    logger.info("VADER ready")

elif TASK == "nrc":
    from nrclex import NRCLex as _NRCLex
    _POSITIVE_EMOTIONS = {"joy", "trust", "anticipation", "surprise"}
    _NEGATIVE_EMOTIONS = {"fear", "anger", "sadness", "disgust"}
    logger.info("NRC ready")

else:
    raise RuntimeError(f"Unknown TASK: {TASK!r}. Must be one of: text-classification, zero-shot-classification, embedding, vader, nrc")


# ── Inference ─────────────────────────────────────────────────────────────────

def _predict(text: str) -> tuple[str, float]:
    if TASK == "text-classification":
        results = _hf_pipeline(text)
        top = results[0][0] if isinstance(results[0], list) else results[0]
        return top["label"], top["score"]

    elif TASK == "zero-shot-classification":
        result = _hf_pipeline(text, candidate_labels=_CANDIDATE_LABELS)
        return result["labels"][0], result["scores"][0]

    elif TASK == "embedding":
        import numpy as np
        text_emb = _st_model.encode([text], normalize_embeddings=True)[0]
        sims = _label_embeddings @ text_emb
        best = int(np.argmax(sims))
        return _CANDIDATE_LABELS[best], round(float(sims[best]), 4)

    elif TASK == "vader":
        compound = _vader.polarity_scores(text)["compound"]
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        return label, round(abs(compound), 4)

    elif TASK == "nrc":
        scores = _NRCLex(text).raw_emotion_scores
        pos = sum(scores.get(e, 0) for e in _POSITIVE_EMOTIONS)
        neg = sum(scores.get(e, 0) for e in _NEGATIVE_EMOTIONS)
        total = pos + neg
        if total == 0:
            return "neutral", 0.0
        if pos > neg:
            return "positive", round(pos / total, 4)
        elif neg > pos:
            return "negative", round(neg / total, 4)
        else:
            return "neutral", 0.5


def _warmup() -> None:
    if TASK == "text-classification":
        _hf_pipeline("warmup")
    elif TASK == "zero-shot-classification":
        _hf_pipeline("warmup text", candidate_labels=_CANDIDATE_LABELS)
    elif TASK == "embedding":
        _st_model.encode(["warmup"], normalize_embeddings=True)
    elif TASK == "vader":
        _vader.polarity_scores("warmup")
    elif TASK == "nrc":
        _predict("warmup")


# ── FastAPI app ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _warmup()
    logger.info("Warmed up: %s (task=%s)", MODEL_NAME, TASK)
    yield


app = FastAPI(title=f"Model Server — {MODEL_NAME}", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    score: float


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    label, score = _predict(req.text)
    return PredictResponse(label=label, score=score)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME, "task": TASK}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

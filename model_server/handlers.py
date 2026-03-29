import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("MODEL_NAME", os.environ.get("TASK", ""))
_CANDIDATE_LABELS: list[str] = [
    l.strip()
    for l in os.environ.get("CANDIDATE_LABELS", "").replace(",", "\n").split("\n")
    if l.strip()
]


class TaskHandler(ABC):
    @abstractmethod
    def warmup(self) -> None: ...

    @abstractmethod
    def predict(self, text: str) -> list[tuple[str, float]]: ...


class TextClassificationHandler(TaskHandler):
    def __init__(self) -> None:
        if not MODEL_NAME:
            raise RuntimeError("MODEL_NAME is required for text-classification")
        from transformers import pipeline
        logger.info("Loading model: %s", MODEL_NAME)
        self._pipeline = pipeline("text-classification", model=MODEL_NAME, top_k=1)
        logger.info("Model loaded: %s", MODEL_NAME)

    def warmup(self) -> None:
        self._pipeline("warmup")

    def predict(self, text: str) -> list[tuple[str, float]]:
        results = self._pipeline(text)
        top = results[0][0] if isinstance(results[0], list) else results[0]
        return [(top["label"], top["score"])]


class ZeroShotClassificationHandler(TaskHandler):
    def __init__(self) -> None:
        if not MODEL_NAME:
            raise RuntimeError("MODEL_NAME is required for zero-shot-classification")
        if not _CANDIDATE_LABELS:
            raise RuntimeError("CANDIDATE_LABELS is required for zero-shot-classification")
        from transformers import pipeline
        logger.info("Loading model: %s", MODEL_NAME)
        self._pipeline = pipeline("zero-shot-classification", model=MODEL_NAME)
        logger.info("Model loaded: %s", MODEL_NAME)

    def warmup(self) -> None:
        self._pipeline("warmup text", candidate_labels=_CANDIDATE_LABELS)

    def predict(self, text: str) -> list[tuple[str, float]]:
        result = self._pipeline(text, candidate_labels=_CANDIDATE_LABELS)
        return list(zip(result["labels"][:3], result["scores"][:3]))


class EmbeddingHandler(TaskHandler):
    def __init__(self) -> None:
        if not MODEL_NAME:
            raise RuntimeError("MODEL_NAME is required for embedding")
        if not _CANDIDATE_LABELS:
            raise RuntimeError("CANDIDATE_LABELS is required for embedding")
        import numpy as np
        from sentence_transformers import SentenceTransformer
        logger.info("Loading model: %s", MODEL_NAME)
        self._model = SentenceTransformer(MODEL_NAME)
        logger.info("Encoding %d candidate labels", len(_CANDIDATE_LABELS))
        self._label_embeddings: np.ndarray = self._model.encode(
            _CANDIDATE_LABELS, normalize_embeddings=True
        )
        logger.info("Model loaded: %s", MODEL_NAME)

    def warmup(self) -> None:
        self._model.encode(["warmup"], normalize_embeddings=True)

    def predict(self, text: str) -> list[tuple[str, float]]:
        import numpy as np
        text_emb = self._model.encode([text], normalize_embeddings=True)[0]
        sims = self._label_embeddings @ text_emb
        top3_idx = np.argsort(sims)[::-1][:3]
        return [(_CANDIDATE_LABELS[i], round(float(sims[i]), 4)) for i in top3_idx]


class VaderHandler(TaskHandler):
    def __init__(self) -> None:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        logger.info("Downloading VADER lexicon")
        nltk.download("vader_lexicon", quiet=True)
        self._analyzer = SentimentIntensityAnalyzer()
        logger.info("VADER ready")

    def warmup(self) -> None:
        self._analyzer.polarity_scores("warmup")

    def predict(self, text: str) -> list[tuple[str, float]]:
        compound = self._analyzer.polarity_scores(text)["compound"]
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        return [(label, round(abs(compound), 4))]


class NrcHandler(TaskHandler):
    _POSITIVE = {"joy", "trust", "anticipation", "surprise"}
    _NEGATIVE = {"fear", "anger", "sadness", "disgust"}

    def __init__(self) -> None:
        from nrclex import NRCLex
        self._NRCLex = NRCLex
        logger.info("NRC ready")

    def warmup(self) -> None:
        self.predict("warmup")

    def predict(self, text: str) -> list[tuple[str, float]]:
        scores = self._NRCLex(text).raw_emotion_scores
        pos = sum(scores.get(e, 0) for e in self._POSITIVE)
        neg = sum(scores.get(e, 0) for e in self._NEGATIVE)
        total = pos + neg
        if total == 0:
            return [("neutral", 0.0)]
        if pos > neg:
            return [("positive", round(pos / total, 4))]
        if neg > pos:
            return [("negative", round(neg / total, 4))]
        return [("neutral", 0.5)]


HANDLERS: dict[str, type[TaskHandler]] = {
    "text-classification": TextClassificationHandler,
    "zero-shot-classification": ZeroShotClassificationHandler,
    "embedding": EmbeddingHandler,
    "vader": VaderHandler,
    "nrc": NrcHandler,
}

# Sentiment Analysis — Learning Project

A full-stack sentiment analysis playground that lets you compare three HuggingFace models against a vLLM-backed LLM, with structured logging, OpenTelemetry metrics, and a Grafana dashboard.

## Services

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost | Single-page UI |
| Backend API | http://localhost:8000 | FastAPI — `POST /sentiment` |
| Prometheus | http://localhost:9090 | Metrics scraper |
| Grafana | http://localhost:3000 | Dashboard (admin / admin) |
| vLLM | http://localhost:8900 | OpenAI-compatible API |

## Models

| Key (use in API / dropdown) | Container | Notes |
|---|---|---|
| `siebert/sentiment-roberta-large-english` | `siebert` | Best for formal English, 2-class |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | `cardiffnlp` | 3-class, Twitter-trained |
| `lxyuan/distilbert-base-multilingual-cased-sentiments-student` | `distilbert` | Multilingual, 3-class |
| `vllm` | `vllm` | Qwen2.5-0.5B-Instruct via vLLM |

## Quick Start

```bash
docker compose up --build
```

First run downloads model weights — this can take several minutes. The backend waits for all model containers to pass their health checks before starting.

> **vLLM on CPU:** Qwen2.5-0.5B runs in `float32` on CPU. Expect 30–120 s per inference. If you have a GPU, remove `--dtype float32 --device cpu` from the `vllm` service `command` in `docker-compose.yml`.

## API

```http
POST /sentiment
Content-Type: application/json

{
  "model": "siebert/sentiment-roberta-large-english",
  "text": "The product exceeded all my expectations."
}
```

Response:

```json
{
  "model": "siebert/sentiment-roberta-large-english",
  "label": "positive",
  "score": 0.9987,
  "raw": { ... }
}
```

## Environment Variables (backend)

| Variable | Default | Description |
|----------|---------|-------------|
| `SIEBERT_URL` | `http://siebert:8000` | siebert model server |
| `CARDIFFNLP_URL` | `http://cardiffnlp:8000` | cardiffnlp model server |
| `DISTILBERT_URL` | `http://distilbert:8000` | distilbert model server |
| `VLLM_URL` | `http://vllm:8900` | vLLM endpoint |
| `VLLM_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Model ID passed to vLLM completions |
| `LOG_LEVEL` | `INFO` | Python log level |
| `LOG_FORMAT` | `json` | `json` or `console` |

## Metrics

All metrics are emitted from inside each model client's `predict()` call (not from the HTTP layer).

| Metric | Type | Labels |
|--------|------|--------|
| `sentiment_model_requests_total` | Counter | `model` |
| `sentiment_model_errors_total` | Counter | `model`, `error_type` |
| `sentiment_model_latency_seconds` | Histogram | `model`, `label` |
| `sentiment_model_confidence_score` | Histogram | `model`, `label` |

Prometheus scrapes `http://backend:8000/metrics` every 15 s.

The Grafana dashboard ("Sentiment Analysis") is pre-provisioned at http://localhost:3000 — no manual setup needed.

## Project Structure

```
src/sentiment/
  main.py               FastAPI app, lifespan, CORS, /metrics
  config.py             pydantic-settings (env vars)
  registry.py           Client factory, initialised at startup
  api/routes.py         POST /sentiment endpoint
  clients/
    base.py             SentimentClient ABC + metrics wrapping
    model_server.py     ModelServerClient (HF containers)
    vllm.py             VllmClient (OpenAI-compat)
  observability/
    logging.py          structlog configuration
    metrics.py          OTEL MeterProvider + 4 instruments

model_server/
  server.py             Tiny FastAPI app wrapping HF pipeline
  Dockerfile            python:3.12-slim + torch (CPU) + transformers

frontend/
  static/index.html     Single-page UI
  nginx.conf            Static serving + /sentiment proxy

observability/
  prometheus.yml
  grafana/              Provisioned datasource + dashboard
```

## Development

```bash
uv sync
uv run pytest
```

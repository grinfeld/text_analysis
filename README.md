# Sentiment Analysis — Learning Project

A full-stack sentiment analysis playground. Enter text and all configured models run against it simultaneously — results are returned side-by-side with label, confidence score, and inference latency per model.

Stack: FastAPI · HuggingFace Transformers · NLTK VADER · NRC Emotion Lexicon · mlx-lm (macOS) / vLLM (Linux) · OpenTelemetry · Prometheus · Grafana

---

## Quick Start

### macOS / Apple Silicon

The LLM runs natively via mlx-lm on the host (uses the Metal GPU). Start it first:

```bash
uv tool install mlx-lm
mlx_lm.server --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --port 8900
```

Then start the rest of the stack, pointing the backend at the host:

```bash
export LLM_URL=http://host.docker.internal:8900
docker compose up --build
```

### Linux

The LLM runs inside a Docker container via vLLM (CPU mode). Use the `linux` profile to include it:

```bash
docker compose --profile linux up --build
```

`LLM_URL` defaults to `http://vllm:8900` when not set, so no export needed.

---

First run downloads model weights — this can take several minutes. The backend waits for all model containers to pass their healthchecks before starting. VADER starts in seconds; neural models take longer.

Model containers run a warmup inference at startup, so the first user request is not cold.

---

## Services

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost | Single-page UI |
| Backend API | http://localhost:8000 | FastAPI — `POST /analyse` |
| Prometheus | http://localhost:9090 | Metrics scraper |
| Grafana | http://localhost:3000 | Dashboard (admin / admin) |
| mlx-lm / vLLM | http://localhost:8900 | OpenAI-compatible LLM API |

---

## Models

Defined in `config.yaml`, loaded at backend startup. Two types:

| Type | Implementation | Notes |
|------|---------------|-------|
| `ml` | `ModelServerClient` | Posts to a `/predict` HTTP endpoint |
| `llm` | `VllmClient` | Posts to an OpenAI-compatible `/v1/chat/completions` endpoint |

Default models:

| Name | Container | Notes |
|------|-----------|-------|
| `siebert/sentiment-roberta-large-english` | `siebert` | RoBERTa, 2-class, best for formal English |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | `cardiffnlp` | RoBERTa, 3-class, Twitter-trained |
| `lxyuan/distilbert-base-multilingual-cased-sentiments-student` | `distilbert` | DistilBERT, 3-class, multilingual |
| `nrc` | `nrc` | NRC Emotion Lexicon — emotion scores aggregated to 3-class, no GPU, tiny image |
| `vader` | `vader` | VADER lexicon — rule-based, 3-class, fast, no GPU, tiny image |
| `ProsusAI/finbert` | `finbert` | BERT, 3-class, finance domain |
| `nlptown/bert-base-multilingual-uncased-sentiment` | `nlptown` | BERT, 5-star → 3-class, multilingual |
| `vllm` | host / `vllm` | Qwen2.5-0.5B-Instruct via mlx-lm (macOS) or vLLM container (Linux) |

The `url` field in `config.yaml` supports `${VAR:-default}` env var substitution. The vllm entry uses `${VLLM_URL:-http://localhost:8900}`, which the backend receives as the `VLLM_URL` env var set by Docker Compose from `LLM_URL`.

---

## API

```http
POST /analyse
Content-Type: application/json

{ "text": "The product exceeded all my expectations." }
```

Response:

```json
{
  "results": [
    { "model": "siebert/sentiment-roberta-large-english",                        "label": "positive", "score": 0.9987, "latency_s": 0.31,  "error": null },
    { "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",               "label": "positive", "score": 0.91,   "latency_s": 0.19,  "error": null },
    { "model": "lxyuan/distilbert-base-multilingual-cased-sentiments-student",   "label": "positive", "score": 0.87,   "latency_s": 0.14,  "error": null },
    { "model": "nrc",                                                             "label": "positive", "score": 0.75,   "latency_s": 0.001, "error": null },
    { "model": "vader",                                                           "label": "positive", "score": 0.72,   "latency_s": 0.001, "error": null },
    { "model": "ProsusAI/finbert",                                                "label": "positive", "score": 0.95,   "latency_s": 0.21,  "error": null },
    { "model": "nlptown/bert-base-multilingual-uncased-sentiment",                "label": "positive", "score": 0.68,   "latency_s": 0.18,  "error": null },
    { "model": "vllm",                                                            "label": "positive", "score": 0.85,   "latency_s": 2.1,   "error": null }
  ]
}
```

Models run sequentially in the order defined in `config.yaml`. A failed model returns `error: "<message>"` with `label`/`score` null — it does not abort the rest.

---

## Configuration

`config.yaml` is mounted into the backend container at `/app/config.yaml`. Edit it to add, remove, or reorder models without rebuilding.

```yaml
models:
  - name: my-model/name
    type: ml               # ml | llm
    url: http://host:port  # supports ${VAR:-default}
    labels:                # ml only — maps raw model output to canonical label
      POSITIVE: positive
      NEGATIVE: negative

  - name: my-llm
    type: llm
    url: http://host:port
    model_id: org/model-name-on-hub
```

---

## Environment Variables

| Variable | Where set | Description |
|----------|-----------|-------------|
| `LLM_URL` | shell / export | URL of the LLM server, passed to the backend as `VLLM_URL` by Compose. Defaults to `http://vllm:8900`. Set to `http://host.docker.internal:8900` on macOS. |
| `CONFIG_PATH` | shell / env | Path to `config.yaml`. Defaults to `config.yaml`. Useful in tests. |
| `LOG_LEVEL` | `docker-compose.yml` | Python log level. Default: `INFO`. |
| `LOG_FORMAT` | `docker-compose.yml` | `json` or `console`. Default: `json`. |

---

## Metrics

Emitted per model client inside `predict()`, not at the HTTP layer.

| Metric | Type | Labels |
|--------|------|--------|
| `sentiment_model_requests_total` | Counter | `model` |
| `sentiment_model_errors_total` | Counter | `model`, `error_type` |
| `sentiment_model_latency_seconds` | Histogram | `model`, `label` |
| `sentiment_model_confidence_score` | Histogram | `model`, `label` |

Prometheus scrapes `http://backend:8000/metrics` every 15 s. The Grafana dashboard ("Sentiment Analysis") is pre-provisioned — no manual setup needed.

---

## Project Structure

```
config.yaml                   Model definitions (edit to add/remove models)

Dockerfile                    Backend image (FastAPI, uv)
docker-compose.yml            All services; vllm under [linux] profile

model_server/
  server.py                   FastAPI wrapper around HuggingFace pipeline
                              Loads model at startup, runs warmup inference
  Dockerfile                  python:3.12-slim + torch + transformers
  Dockerfile.vllm             vllm/vllm-openai image (Linux only)

nrc_server/
  server.py                   FastAPI NRC server — emotion scores aggregated to sentiment
                              positive emotions (joy/trust/anticipation/surprise) vs negative
  Dockerfile                  python:3.12-slim + nrclex only (~150 MB image)

vader_server/
  server.py                   FastAPI VADER server — lexicon-based, no torch
                              Compound score → positive/neutral/negative
  Dockerfile                  python:3.12-slim + nltk only (~150 MB image)

src/sentiment/
  main.py                     App factory, lifespan, CORS, /metrics
  config.py                   Settings + load_model_configs() with env var expansion
  registry.py                 Builds clients from config.yaml at startup
  api/routes.py               POST /analyse
  clients/
    base.py                   SentimentClient ABC + metrics + timing
    model_server.py           ModelServerClient (HuggingFace containers)
    vllm.py                   VllmClient (OpenAI-compatible)
  observability/
    logging.py                structlog (JSON or console)
    metrics.py                OTEL MeterProvider + Prometheus exporter

frontend/
  static/index.html           Single-page UI — textarea + results table
  nginx.conf                  Static files + /analyse proxy to backend

observability/
  prometheus.yml
  grafana/                    Pre-provisioned datasource + dashboard

tests/
  conftest.py                 Fixtures: TestClient, metrics setup, CONFIG_PATH → config.yaml
  test_routes.py              /analyse endpoint tests
  test_clients.py             ModelServerClient + VllmClient unit tests
  test_integration.py         Testcontainers — real model_server container
```

---

## Development

```bash
uv sync
uv run pytest
```

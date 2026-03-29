# Text Analysis — Learning Project

A full-stack text analysis playground. Enter text, choose a task (Sentiment or Topic), and all configured models run against it simultaneously — results are returned side-by-side with model name, label, confidence score, and inference latency.

Stack: FastAPI · HuggingFace Transformers · sentence-transformers · NLTK VADER · NRC Emotion Lexicon · mlx-lm (macOS) / vLLM (Linux) · OpenTelemetry · Prometheus · Grafana

---

## Quick Start

Sentiment model containers are grouped under the `sentiment` profile; topic model containers under the `topic` profile. Backend, frontend, Prometheus, and Grafana start regardless of profile.

A single Docker image (`sentiment/model-server:latest`) is built once and reused by all model containers.

### macOS / Apple Silicon

The LLM runs natively via mlx-lm on the host (uses the Metal GPU). Start it first:

```bash
uv tool install mlx-lm
mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8900
```

Then start the stack:

```bash
export LLM_URL=http://host.docker.internal:8900

# Sentiment models only
docker compose --profile sentiment up --build

# Topic models only
docker compose --profile topic up --build

# Both
docker compose --profile sentiment --profile topic up --build
```

### Linux

The LLM runs inside a Docker container via vLLM (CPU mode). Add the `linux` profile:

```bash
# Sentiment + LLM
docker compose --profile sentiment --profile linux up --build

# All
docker compose --profile sentiment --profile topic --profile linux up --build
```

`LLM_URL` defaults to `http://vllm:8900` when not set, so no export needed.

---

First run downloads model weights — this can take several minutes. The backend waits for all sentiment model containers to pass their healthchecks before starting. VADER and NRC start in seconds; neural models take longer.

All model containers run a warmup inference at startup, so the first user request is not cold.

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

Defined in `config.yaml`, loaded at backend startup. Each model has a `for` field that determines which task it serves.

| Type | Implementation | Notes |
|------|---------------|-------|
| `ml` | `ModelServerClient` | Posts to a `/predict` HTTP endpoint |
| `llm` | `VllmClient` | Posts to an OpenAI-compatible `/v1/chat/completions` endpoint |

### Sentiment models (`for: sentiment`, profile: `sentiment`)

| Name | Container | Notes |
|------|-----------|-------|
| `siebert/sentiment-roberta-large-english` | `siebert` | RoBERTa, 2-class, formal English |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | `cardiffnlp` | RoBERTa, 3-class, Twitter-trained |
| `lxyuan/distilbert-base-multilingual-cased-sentiments-student` | `distilbert` | DistilBERT, 3-class, multilingual |
| `nrc` | `nrc` | NRC Emotion Lexicon — positive/negative emotion aggregation |
| `vader` | `vader` | VADER — rule-based lexicon, compound score → 3-class |
| `ProsusAI/finbert` | `finbert` | BERT, 3-class, finance domain |
| `nlptown/bert-base-multilingual-uncased-sentiment` | `nlptown` | BERT, 5-star → 3-class, multilingual |

### Topic models (`for: topic`, profile: `topic`)

Zero-shot NLI classifiers — classify directly against taxonomy slugs as candidate labels:

| Name | Container | Notes |
|------|-----------|-------|
| `MoritzLaurer/deberta-v3-base-zeroshot-v1` | `deberta` | DeBERTa, 86M, recommended small zero-shot |
| `MoritzLaurer/deberta-v3-large-zeroshot-v2` | `deberta-large` | DeBERTa, 400M, higher accuracy |
| `cross-encoder/nli-deberta-v3-large` | `nli-deberta` | NLI cross-encoder, 400M |
| `facebook/bart-large-mnli` | `bart` | BART, 400M, widely-used zero-shot baseline |

Embedding similarity — embed text and find nearest topic slug by cosine similarity:

| Name | Container | Notes |
|------|-----------|-------|
| `BAAI/bge-small-en-v1.5` | `bge-small` | 33M, recommended default embedder |
| `BAAI/bge-large-en-v1.5` | `bge-large` | 335M, higher accuracy |
| `sentence-transformers/all-MiniLM-L6-v2` | `minilm` | 22M, fastest |
| `sentence-transformers/all-mpnet-base-v2` | `mpnet` | 110M, good mid-tier |
| `intfloat/e5-small-v2` | `e5-small` | 33M, retrieval-focused |
| `intfloat/e5-large-v2` | `e5-large` | 335M, best E5 variant |

### LLM (`for: llm`, shared for both tasks)

| Name | Container | Notes |
|------|-----------|-------|
| `vllm` | host / `vllm` | Qwen2.5-7B-Instruct via mlx-lm (macOS) or vLLM container (Linux) |

The LLM uses task-specific prompts — sentiment prompt for the sentiment task, topic prompt for the topic task.

### Topic taxonomy

Topic slugs used as candidate labels:
`arms_trafficking`, `financial_crime`, `surveillance_operation`, `human_intelligence`, `signals_intelligence`, `covert_operation`, `money_laundering`, `sanctions_evasion`, `recruitment`, `daily_reporting`, `network_configuration`, `routing`, `administrative`, `organizational_structure`, `spatial_context`, `temporal_reference`

The LLM may also propose novel slugs not in this list.

---

## API

```http
POST /analyse
Content-Type: application/json

{ "text": "The product exceeded all my expectations.", "for": "sentiment" }
```

`for` is optional, defaults to `"sentiment"`. Valid values: `"sentiment"`, `"topic"`.

Response:

```json
{
  "results": [
    { "model": "siebert/sentiment-roberta-large-english", "label": "positive", "score": 0.9987, "latency_s": 0.31, "error": null },
    { "model": "vllm",                                   "label": "positive", "score": 0.85,   "latency_s": 2.1,  "error": null }
  ]
}
```

Models run sequentially in the order defined in `config.yaml`. A failed model returns `error: "<message>"` with `label`/`score` null — it does not abort the rest.

---

## Configuration

`config.yaml` is mounted into the backend container at `/app/config.yaml`. Edit it to add, remove, or reorder models without rebuilding the backend.

```yaml
models:
  - name: my-model/name
    for: sentiment        # sentiment | topic | llm
    type: ml              # ml | llm
    url: http://host:port # supports ${VAR:-default}
    labels:               # ml only — maps raw model output to canonical label
      POSITIVE: positive
      NEGATIVE: negative

  - name: my-llm
    for: llm
    type: llm
    url: http://host:port
    model_id: org/model-name-on-hub
```

---

## Model Server

All model containers use a single Docker image (`sentiment/model-server:latest`) built from `model_server/`. The `TASK` environment variable selects the inference path:

| `TASK` | Description |
|--------|-------------|
| `text-classification` | HuggingFace pipeline, default |
| `zero-shot-classification` | HuggingFace zero-shot NLI; requires `CANDIDATE_LABELS` |
| `embedding` | sentence-transformers cosine similarity; requires `CANDIDATE_LABELS` |
| `vader` | NLTK VADER lexicon, no `MODEL_NAME` needed |
| `nrc` | NRC Emotion Lexicon, no `MODEL_NAME` needed |

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
docker-compose.yml            All services; sentiment/topic/linux profiles

model_server/
  server.py                   Unified model server — behaviour selected by TASK env var
                              Supports: text-classification, zero-shot-classification,
                              embedding, vader, nrc
  requirements.txt            All deps: transformers, sentence-transformers, nltk, nrclex
  Dockerfile                  Single image for all model containers
  Dockerfile.vllm             vllm/vllm-openai image (Linux only)

src/sentiment/
  main.py                     App factory, lifespan, CORS, /metrics
  config.py                   Settings + load_model_configs() with env var expansion
  registry.py                 Builds clients from config.yaml at startup
  api/routes.py               POST /analyse — accepts "for": "sentiment"|"topic"
  clients/
    base.py                   ModelClient ABC + metrics + timing
    model_server.py           ModelServerClient (all model containers)
    vllm.py                   VllmClient (OpenAI-compatible, sentiment + topic prompts)
  observability/
    logging.py                structlog (JSON or console)
    metrics.py                OTEL MeterProvider + Prometheus exporter

frontend/
  static/index.html           Single-page UI — textarea + task dropdown + results table
  nginx.conf                  Static files + /analyse proxy to backend

observability/
  prometheus.yml
  grafana/                    Pre-provisioned datasource + dashboard

tests/
  conftest.py                 Fixtures: TestClient, metrics setup, CONFIG_PATH → config.yaml
  test_routes.py              /analyse endpoint tests (sentiment + topic)
  test_clients.py             ModelServerClient + VllmClient unit tests
  test_integration.py         Testcontainers — real model_server container
```

---

## Development

```bash
uv sync
uv run pytest
```

# Text Analysis — Learning Project

A full-stack text analysis playground. Enter text, choose a task (Sentiment or Topic), and all configured models run simultaneously — results are returned side-by-side with model name, ranked labels, confidence scores, and inference latency.

Stack: FastAPI · HuggingFace Transformers · sentence-transformers · NLTK VADER · NRC Emotion Lexicon · mlx-lm (macOS) / vLLM (Linux) · OpenTelemetry · Prometheus · Grafana

---

## Quick Start

Three Docker Compose profiles control which model containers start:

| Profile | What starts |
|---------|-------------|
| `sentiment` | All sentiment model containers |
| `topic` | All topic model containers |
| `linux` | vLLM container (Linux/CPU only) |

`text-analysis-backend`, `text-analysis-frontend`, Prometheus, and Grafana always start regardless of profile.

### Using the scripts

```bash
./start.sh   # interactive: asks OS and model profiles, builds model-server image, starts detached
./down.sh    # stops all containers (all profiles)
./down.sh --volumes   # also removes the hf_cache volume
```

`start.sh` builds `text-analysis/model-server:latest` before starting and sets `LLM_URL` automatically on macOS.

---

### Manual start

#### macOS / Apple Silicon

The LLM runs natively via mlx-lm on the host (uses the Metal GPU). Start it first:

```bash
uv tool install mlx-lm
mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8900
```

Build the model-server image once, then start the stack:

```bash
docker build -t text-analysis/model-server:latest ./model_server

export LLM_URL=http://host.docker.internal:8900

# Sentiment only
docker compose --profile sentiment up -d

# Topic only
docker compose --profile topic up -d

# Both
docker compose --profile sentiment --profile topic up -d
```

#### Linux

The LLM runs inside a Docker container via vLLM (CPU mode):

```bash
docker build -t text-analysis/model-server:latest ./model_server

# Sentiment + LLM
docker compose --profile sentiment --profile linux up -d

# All
docker compose --profile sentiment --profile topic --profile linux up -d
```

`LLM_URL` defaults to `http://vllm:8900` when not set, so no export needed.

### Manual stop

```bash
# Stop all profiles
docker compose --profile sentiment --profile topic --profile linux down

# Stop and remove the model weights cache volume
docker compose --profile sentiment --profile topic --profile linux down --volumes
```

---

First run downloads model weights — this can take several minutes. VADER and NRC start in seconds; neural models take longer. All model containers run a warmup inference at startup, so the first user request is not cold.

---

## Services

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:8080 | Single-page UI |
| Backend API | http://localhost:8000 | FastAPI — `POST /analyse` |
| Prometheus | http://localhost:9090 | Metrics scraper |
| Grafana | http://localhost:3000 | Dashboard (admin / admin) |
| mlx-lm / vLLM | http://localhost:8900 | OpenAI-compatible LLM API |

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
    {
      "model": "siebert/sentiment-roberta-large-english",
      "labels": [{ "label": "positive", "score": 0.99 }],
      "latency_s": 0.31,
      "error": null
    },
    {
      "model": "MoritzLaurer/deberta-v3-base-zeroshot-v1",
      "labels": [
        { "label": "financial_crime", "score": 0.87 },
        { "label": "money_laundering", "score": 0.60 },
        { "label": "sanctions_evasion", "score": 0.40 }
      ],
      "latency_s": 1.2,
      "error": null
    }
  ]
}
```

All models return `labels` as a ranked list. Sentiment models return one entry; topic models return up to 3. A failed model returns `labels: []` and `error: "<message>"` — it does not abort the rest.

Models run sequentially in the order defined in `config.yaml`.

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
    for: llm              # registers one client for sentiment AND one for topic
    type: llm
    url: http://host:port
    model_id: org/model-name-on-hub
```

`for: llm` is special — a single config entry registers two clients automatically (one for sentiment, one for topic) using different prompts.

---

## Models

### Sentiment (`for: sentiment`, profile: `sentiment`)

| Name | Container | Notes |
|------|-----------|-------|
| `siebert/sentiment-roberta-large-english` | `siebert` | RoBERTa, 2-class, formal English |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | `cardiffnlp` | RoBERTa, 3-class, Twitter-trained |
| `lxyuan/distilbert-base-multilingual-cased-sentiments-student` | `distilbert` | DistilBERT, 3-class, multilingual |
| `ProsusAI/finbert` | `finbert` | BERT, 3-class, finance domain |
| `nlptown/bert-base-multilingual-uncased-sentiment` | `nlptown` | BERT, 5-star → 3-class, multilingual |
| `nrc` | `nrc` | NRC Emotion Lexicon — positive/negative emotion aggregation |
| `vader` | `vader` | VADER — rule-based lexicon, compound score → 3-class |

### Topic — zero-shot NLI (`for: topic`, profile: `topic`)

Classify directly against taxonomy slugs as candidate labels:

| Name | Container | Notes |
|------|-----------|-------|
| `MoritzLaurer/deberta-v3-base-zeroshot-v1` | `deberta` | DeBERTa, 86M, recommended small zero-shot |
| `MoritzLaurer/deberta-v3-large-zeroshot-v1` | `deberta_large` | DeBERTa, 400M, higher accuracy |
| `cross-encoder/nli-deberta-v3-large` | `nli_deberta` | NLI cross-encoder, 400M |
| `facebook/bart-large-mnli` | `bart` | BART, 400M, widely-used zero-shot baseline |

### Topic — embedding similarity (`for: topic`, profile: `topic`)

Embed text and find nearest topic slugs by cosine similarity (top 3 returned):

| Name | Container | Notes |
|------|-----------|-------|
| `BAAI/bge-small-en-v1.5` | `bge_small` | 33M, recommended default embedder |
| `BAAI/bge-large-en-v1.5` | `bge_large` | 335M, higher accuracy |
| `sentence-transformers/all-MiniLM-L6-v2` | `minilm` | 22M, fastest |
| `sentence-transformers/all-mpnet-base-v2` | `mpnet` | 110M, good mid-tier |
| `intfloat/e5-small-v2` | `e5_small` | 33M, retrieval-focused |
| `intfloat/e5-large-v2` | `e5_large` | 335M, best E5 variant |

### LLM (`for: llm`, shared for both tasks)

| Name | Container | Notes |
|------|-----------|-------|
| `vllm` | host / `vllm` | Qwen2.5-7B-Instruct via mlx-lm (macOS) or vLLM container (Linux) |

A single config entry registers two clients — one uses the sentiment prompt (returns single label: positive / neutral / negative), the other uses the topic prompt (returns ranked top-3 slugs).

### Topic taxonomy

Candidate slugs used by zero-shot and embedding models, and as examples for the LLM:

`arms_trafficking`, `financial_crime`, `surveillance_operation`, `human_intelligence`, `signals_intelligence`, `covert_operation`, `money_laundering`, `sanctions_evasion`, `recruitment`, `daily_reporting`, `network_configuration`, `routing`, `administrative`, `organizational_structure`, `spatial_context`, `temporal_reference`

---

## Model Server

All model containers use a single Docker image (`text-analysis/model-server:latest`) built from `model_server/`. The `TASK` environment variable selects the inference path; `MODEL_NAME` sets the HuggingFace model to load.

Handler classes are defined in `model_server/handlers.py`; `model_server/server.py` owns only the FastAPI app and routes.

| `TASK` | Handler | Required env vars |
|--------|---------|-------------------|
| `text-classification` | `TextClassificationHandler` | `MODEL_NAME` |
| `zero-shot-classification` | `ZeroShotClassificationHandler` | `MODEL_NAME`, `CANDIDATE_LABELS` |
| `embedding` | `EmbeddingHandler` | `MODEL_NAME`, `CANDIDATE_LABELS` |
| `vader` | `VaderHandler` | — |
| `nrc` | `NrcHandler` | — |

`CANDIDATE_LABELS` is a comma-separated string of classification targets.

The `/predict` endpoint always returns:

```json
{ "labels": [{ "label": "...", "score": 0.87 }, ...] }
```

Single-label handlers (text-classification, vader, nrc) return a one-item list. Zero-shot and embedding handlers return top-3.

The Docker image is built once and reused by all model containers via `image: text-analysis/model-server:latest`.

---

## Environment Variables

| Variable | Where set | Description |
|----------|-----------|-------------|
| `LLM_URL` | shell / export | URL of the LLM server. Passed to the backend as `VLLM_URL`. Defaults to `http://vllm:8900`. Set to `http://host.docker.internal:8900` on macOS. |
| `CONFIG_PATH` | shell / env | Path to `config.yaml`. Defaults to `config.yaml`. Useful in tests. |
| `LOG_LEVEL` | `docker-compose.yml` | Python log level. Default: `INFO`. |
| `LOG_FORMAT` | `docker-compose.yml` | `json` or `console`. Default: `json`. |

---

## Metrics

Emitted per model client inside `predict()`, not at the HTTP layer.

| Metric | Type | Labels |
|--------|------|--------|
| `text_analysis_model_requests_total` | Counter | `model` |
| `text_analysis_model_errors_total` | Counter | `model`, `error_type` |
| `text_analysis_model_latency_seconds` | Histogram | `model`, `label` |
| `text_analysis_model_confidence_score` | Histogram | `model`, `label` |

Prometheus scrapes `http://text-analysis-backend:8000/metrics` every 15 s. The Grafana dashboard ("Text Analysis") is pre-provisioned — no manual setup needed.

---

## Project Structure

```
config.yaml                   Model definitions (edit to add/remove models)
start.sh                      Interactive launcher — prompts OS + profiles, builds image
down.sh                       Stops all containers (all profiles)

Dockerfile                    text-analysis-backend image (FastAPI, uv)
docker-compose.yml            All services; sentiment/topic/linux profiles
                              YAML anchors (x-model-hf, x-model-zeroshot,
                              x-model-embedding, x-model-lexicon) eliminate
                              repetition across model service definitions

model_server/
  server.py                   FastAPI app + /predict and /health routes
  handlers.py                 TaskHandler ABC + concrete implementations
                              (TextClassification, ZeroShot, Embedding, Vader, Nrc)
  requirements.txt            All deps: transformers, sentence-transformers, nltk, nrclex
  Dockerfile                  Single image (text-analysis/model-server:latest)
  Dockerfile.vllm             vllm/vllm-openai image (Linux only)

src/text_analysis/
  main.py                     App factory, lifespan, CORS, /metrics
  config.py                   Settings + load_model_configs() with env var expansion
  registry.py                 Builds clients from config.yaml at startup
  api/routes.py               POST /analyse — accepts "for": "sentiment"|"topic"
  clients/
    base.py                   ModelClient ABC + PredictionResult + metrics + timing
    model_server.py           ModelServerClient (all model containers)
    vllm.py                   VllmClient (OpenAI-compatible, sentiment + topic prompts)
  observability/
    logging.py                structlog (JSON or console)
    metrics.py                OTEL MeterProvider + Prometheus exporter

frontend/
  static/index.html           Single-page UI — textarea + task dropdown + results table
  nginx.conf                  Static files + /analyse proxy to text-analysis-backend

observability/
  prometheus.yml              Scrapes text-analysis-backend:8000/metrics
  grafana/                    Pre-provisioned datasource + dashboard (Text Analysis)

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

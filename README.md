# Sentiment Analysis — Learning Project

A full-stack sentiment analysis playground that runs all configured models against your text in one request and compares results side-by-side, with structured logging, OpenTelemetry metrics, and a Grafana dashboard.

## Services

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost | Single-page UI |
| Backend API | http://localhost:8000 | FastAPI — `POST /analyse` |
| Prometheus | http://localhost:9090 | Metrics scraper |
| Grafana | http://localhost:3000 | Dashboard (admin / admin) |
| mlx-lm | http://localhost:8900 | OpenAI-compatible API (host process) |

## Models

Models are defined in `config.yaml` (or `config.macos.yaml` for macOS). Each entry has a `type`:

| Type | Class | Notes |
|------|-------|-------|
| `ml` | `ModelServerClient` | HuggingFace pipeline container |
| `llm` | `VllmClient` | OpenAI-compatible completions endpoint |

Default models:

| Name | Container | Notes |
|------|-----------|-------|
| `siebert/sentiment-roberta-large-english` | `siebert` | Best for formal English, 2-class |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | `cardiffnlp` | 3-class, Twitter-trained |
| `lxyuan/distilbert-base-multilingual-cased-sentiments-student` | `distilbert` | Multilingual, 3-class |
| `vllm` | host (mlx-lm) | Qwen2.5-0.5B-Instruct via mlx-lm on port 8900 |

## Quick Start

### macOS / Apple Silicon

mlx-lm runs natively on the host using the Metal GPU. Start it before the stack:

```bash
uv tool install mlx-lm
mlx_lm.server --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --port 8900
```

The first run downloads ~300 MB of model weights. Once the server is up, start the rest of the stack:

```bash
docker compose -f docker-compose.yml -f docker-compose.macos.yml up --build
```

`docker-compose.macos.yml` mounts `config.macos.yaml` (which points the vllm entry at `host.docker.internal:8900`) instead of `config.yaml`. The vllm model row will show an error in the UI if mlx-lm is not running.

If port 8900 is already in use, change `--port` in the mlx-lm command and update the `url` in `config.macos.yaml` to match.

### Linux / GPU

```bash
docker compose up --build
```

First run downloads model weights — this can take several minutes. The backend waits for all model containers to pass their health checks before starting.

> **Note:** The `vllm` model is macOS-only in this setup (served by mlx-lm). On Linux, the three HuggingFace models work normally; the vllm row will show an error unless you add a compatible container and update `config.yaml`.

## API

```http
POST /analyse
Content-Type: application/json

{
  "text": "The product exceeded all my expectations."
}
```

Response:

```json
{
  "results": [
    { "model": "siebert/sentiment-roberta-large-english", "label": "positive", "score": 0.9987, "latency_s": 0.312, "error": null },
    { "model": "cardiffnlp/twitter-roberta-base-sentiment-latest", "label": "positive", "score": 0.91, "latency_s": 0.198, "error": null },
    { "model": "lxyuan/distilbert-base-multilingual-cased-sentiments-student", "label": "positive", "score": 0.87, "latency_s": 0.145, "error": null },
    { "model": "vllm", "label": "positive", "score": 0.85, "latency_s": 2.1, "error": null }
  ]
}
```

A model failure returns an `error` string for that entry and does not abort the rest.

## Configuration

Models are loaded from `config.yaml` at startup. The path can be overridden with the `CONFIG_PATH` env var.

```yaml
models:
  - name: my-model/name
    type: ml          # ml | llm
    url: http://host:port
    labels:           # required for ml — raw label → normalised label
      POSITIVE: positive
      NEGATIVE: negative
  - name: vllm
    type: llm
    url: http://host:port
    model_id: model/name-on-hub
```

## Environment Variables (backend)

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG_PATH` | `config.yaml` | Path to model config file |
| `LOG_LEVEL` | `INFO` | Python log level |
| `LOG_FORMAT` | `json` | `json` or `console` |

## Metrics

All metrics are emitted from inside each model client's `predict()` call.

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
config.yaml               Model definitions (Linux/default)
config.macos.yaml         Model definitions (macOS — vllm points to host)

src/sentiment/
  main.py               FastAPI app, lifespan, CORS, /metrics
  config.py             Settings + load_model_configs() YAML loader
  registry.py           Builds clients from config at startup
  api/routes.py         POST /analyse endpoint
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
  static/index.html     Single-page UI — text field + results table
  nginx.conf            Static serving + /analyse proxy

observability/
  prometheus.yml
  grafana/              Provisioned datasource + dashboard
```

## Development

```bash
uv sync
uv run pytest
```

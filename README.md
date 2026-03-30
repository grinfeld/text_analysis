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
./start.sh            # interactive: asks OS and model profiles, builds model-server image, starts detached
./stop.sh             # stops all containers (all profiles), keeps them for restart
./down.sh             # stops and removes all containers (all profiles)
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

### Manual stop / down

```bash
# Stop containers (keep them, restart with docker compose start)
docker compose --profile sentiment --profile topic --profile linux stop

# Stop and remove containers
docker compose --profile sentiment --profile topic --profile linux down

# Stop, remove containers, and remove the model weights cache volume
docker compose --profile sentiment --profile topic --profile linux down --volumes
```

### Rebuilding after code changes

The model-server image is pre-built and reused by all model containers. Rebuild it explicitly when `model_server/` changes:

```bash
docker build -t text-analysis/model-server:latest ./model_server
docker compose --profile topic up -d --force-recreate
```

The backend image is rebuilt automatically by `docker compose up --build`.

---

First run downloads model weights — this can take several minutes. VADER and NRC start in seconds; neural models take longer. All model containers run a warmup inference at startup, so the first user request is not cold.

---

## Services

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:8080 | Single-page UI — text input, task + domain dropdowns |
| Backend API | http://localhost:8000 | FastAPI — `POST /analyse`, `GET /candidates` |
| Prometheus | http://localhost:9090 | Metrics scraper |
| Grafana | http://localhost:3000 | Dashboard (admin / admin) |
| mlx-lm / vLLM | http://localhost:8900 | OpenAI-compatible LLM API |

---

## API

### POST /analyse

```http
POST /analyse
Content-Type: application/json

{ "text": "The product exceeded all my expectations.", "for": "sentiment", "domain": "finance" }
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | required | Text to analyse |
| `for` | string | `"sentiment"` | Task: `"sentiment"` or `"topic"` |
| `domain` | string | `null` | Optional domain context: `intelligence`, `cybersecurity`, `networking`, `medicine`, `finance`, `logistics`, `military`, `human_resources`, `legal` |

**When `domain` is provided:**
- Topic models receive only domain-scoped candidate labels (instead of the full list) — improves precision and reduces latency
- The LLM receives the domain as a line immediately before the text: `Domain: Finance`
- Sentiment models are unaffected (they have fixed label sets)

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

Models run concurrently (up to `MAX_CONCURRENT_PER_REQUEST` in parallel) in the order defined in `config.yaml`.

### GET /candidates

Returns the full live topic candidate label list, including labels discovered by the LLM at runtime.

```json
[
  { "label": "financial_crime", "source": "config" },
  { "label": "wire_transfer_scheme", "source": "discovered" }
]
```

`source` is `"config"` for labels defined in `config.yaml`, `"discovered"` for novel labels proposed by the LLM and confirmed as genuinely new (similarity < 0.92 against all known candidates via `e5-small`). Discovered labels also include a `domain` field (the domain that was active when the label was found, or `null` if no domain was specified).

---

## Configuration

`config.yaml` is mounted into the backend container at `/app/config.yaml`. Edit it to add, remove, or reorder models without rebuilding the backend.

The `domain_candidates` top-level key in `config.yaml` maps domain names to their candidate slug subsets. Add or edit these without rebuilding anything.

```yaml
models:
  - name: my-model/name
    for: sentiment        # sentiment | topic | llm
    type: ml              # ml | llm
    url: http://host:port # supports ${VAR:-default}
    labels:               # ml only — maps raw model output to canonical label
      POSITIVE: positive
      NEGATIVE: negative

  - name: my-topic-model
    for: topic
    type: ml
    url: http://host:port

  - name: my-llm
    for: llm              # registers one client for sentiment AND one for topic
    type: llm
    url: http://host:port
    model_id: org/model-name-on-hub
```

`for: llm` is special — a single config entry registers two clients automatically (one for sentiment, one for topic) using different prompts.

Candidate labels are no longer defined per model entry. They are derived at startup by flattening `domain_candidates` into a deduplicated list — a single source of truth. Domain-scoped subsets are sent to topic models when a `domain` is specified in the request; the full flattened list is used otherwise.

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

Classify directly against candidate slugs as hypotheses. Labels are sent in the request payload; processed in chunks of 20 to handle large candidate lists:

| Name | Container | Notes |
|------|-----------|-------|
| `MoritzLaurer/deberta-v3-base-zeroshot-v1` | `deberta` | DeBERTa, 86M, recommended small zero-shot |
| `MoritzLaurer/deberta-v3-large-zeroshot-v1` | `deberta-large` | DeBERTa, 400M, higher accuracy |
| `cross-encoder/nli-deberta-v3-large` | `nli-deberta` | NLI cross-encoder, 400M |
| `facebook/bart-large-mnli` | `bart` | BART, 400M, widely-used zero-shot baseline |

### Topic — embedding similarity (`for: topic`, profile: `topic`)

Embed text and find nearest candidate slugs by cosine similarity (top 3 returned). Label embeddings are cached per candidate set:

| Name | Container | Notes |
|------|-----------|-------|
| `BAAI/bge-small-en-v1.5` | `bge-small` | 33M, recommended default embedder |
| `BAAI/bge-large-en-v1.5` | `bge-large` | 335M, higher accuracy |
| `sentence-transformers/all-MiniLM-L6-v2` | `minilm` | 22M, fastest |
| `sentence-transformers/all-mpnet-base-v2` | `mpnet` | 110M, good mid-tier |
| `intfloat/e5-small-v2` | `e5-small` | 33M, retrieval-focused; also used for novel label resolution |
| `intfloat/e5-large-v2` | `e5-large` | 335M, best E5 variant |

### LLM (`for: llm`, shared for both tasks)

| Name | Container | Notes |
|------|-----------|-------|
| `vllm` | host / `vllm` | Qwen2.5-7B-Instruct via mlx-lm (macOS) or vLLM container (Linux) |

A single config entry registers two clients — one uses the sentiment prompt (returns single label: `positive` / `neutral` / `negative`), the other uses the topic prompt (returns ranked top-3 slugs).

**Topic prompt behaviour:** The LLM is instructed to return 3 ranked topic labels, at least one of which must be a broad category-level generalisation (e.g. `clinical_diagnosis`, `threat_assessment`). When a domain is specified, it is included in the prompt as `Domain: <Name>` immediately before the text. Novel labels are post-processed via `e5-small`: if cosine similarity ≥ 0.92 against a known candidate the novel label is replaced with the match; otherwise it is added to the in-memory discovered list and persisted to `discovered_labels.json` under the active domain key.

### Topic taxonomy

All candidate slugs are defined in `domain_candidates` in `config.yaml` — the single source of truth. The full flattened list is used when no domain is specified; domain-scoped subsets are used when a domain is provided.

**Intelligence (15):** `human_intelligence`, `signals_intelligence`, `covert_operation`, `surveillance_operation`, `counterintelligence`, `open_source_intelligence`, `geospatial_intelligence`, `covert_communication`, `technical_intelligence`, `imagery_intelligence`, `cyber_intelligence`, `all_source_analysis`, `intelligence_sharing`, `agent_handling`, `intelligence_operation`

**Cybersecurity (19):** `malware`, `phishing`, `vulnerability_exploitation`, `data_exfiltration`, `denial_of_service`, `command_and_control`, `ransomware`, `zero_day_exploit`, `botnet`, `web_application_attack`, `iot_attack`, `cyber_espionage`, `identity_theft`, `supply_chain_attack`, `incident_response`, `cyber_intrusion`, `cryptojacking`, `deepfake`, `dark_web`

**Networking (10):** `network_configuration`, `routing`, `network_monitoring`, `vpn_tunneling`, `firewall_policy`, `dns_management`, `bandwidth_management`, `network_access_control`, `network_segmentation`, `network_infrastructure`

**Finance (11):** `financial_crime`, `money_laundering`, `sanctions_evasion`, `tax_evasion`, `insider_trading`, `bribery_corruption`, `asset_seizure`, `trade_finance_fraud`, `cryptocurrency_crime`, `ponzi_scheme`, `financial_intelligence`

**Logistics (10):** `logistics`, `maritime_transport`, `air_transport`, `ground_transport`, `smuggling_route`, `customs_evasion`, `fuel_supply`, `logistics_supply`, `vehicle_surveillance`, `border_control`

**Military (10):** `military_operation`, `weapons_system`, `troop_movement`, `command_structure`, `electronic_warfare`, `psychological_operations`, `force_protection`, `military_intelligence`, `arms_trafficking`, `covert_operation`

**Medicine (5):** `clinical_diagnosis`, `pharmaceutical_trial`, `surgical_procedure`, `patient_triage`, `medical_imaging`

**Human Resources (5):** `employee_onboarding`, `performance_review`, `workforce_planning`, `disciplinary_action`, `compensation_benefits`

**Legal (5):** `contract_dispute`, `regulatory_compliance`, `litigation_filing`, `intellectual_property`, `criminal_prosecution`

---

## Label Discovery

The LLM topic client implements self-learning taxonomy expansion:

1. LLM returns 3 ranked topic labels (at least one broad category-level label per prompt instructions)
2. Each novel label (not in the known candidate list) is sent to `e5-small` for cosine similarity against known candidates for the active domain (or all candidates if no domain)
3. If similarity ≥ 0.92 → replaced with the matching known candidate (near-duplicate)
4. If similarity < 0.92 → genuinely new; added to the in-memory `CandidateStore` under the active domain key and persisted to `discovered_labels.json`
5. Discovered labels are loaded from `discovered_labels.json` on backend startup and included in all subsequent model requests

`discovered_labels.json` stores labels as a dict keyed by domain (e.g. `{"medicine": ["respiratory_issue"], "cybersecurity": ["firmware_backdoor"]}`). Labels discovered without a domain are stored under `_unknown`.

`GET /candidates` returns all labels with `"source": "config"` or `"source": "discovered"`, plus a `"domain"` field for discovered labels.

---

## Model Server

All model containers use a single Docker image (`text-analysis/model-server:latest`) built from `model_server/`. The `TASK` environment variable selects the inference path; `MODEL_NAME` sets the HuggingFace model to load.

Handler classes are defined in `model_server/handlers.py`; `model_server/server.py` owns only the FastAPI app and routes.

| `TASK` | Handler | Required env vars |
|--------|---------|-------------------|
| `text-classification` | `TextClassificationHandler` | `MODEL_NAME` |
| `zero-shot-classification` | `ZeroShotClassificationHandler` | `MODEL_NAME` |
| `embedding` | `EmbeddingHandler` | `MODEL_NAME` |
| `vader` | `VaderHandler` | — |
| `nrc` | `NrcHandler` | — |

Candidate labels are not baked into the container — they are sent in the `/predict` request payload by the backend, sourced from `config.yaml`. Zero-shot models process them in chunks of 20 to bound per-request latency. Embedding models cache label embeddings per candidate set.

All labels returned by the model server are normalized to `snake_case` via `python-slugify`.

The `/predict` endpoint always returns:

```json
{ "labels": [{ "label": "...", "score": 0.87 }, ...] }
```

Single-label handlers (`text-classification`, `vader`, `nrc`) return a one-item list. Zero-shot and embedding handlers return top-3.

---

## Environment Variables

| Variable | Where set | Description |
|----------|-----------|-------------|
| `LLM_URL` | shell / export | URL of the LLM server. Passed to the backend as `VLLM_URL`. Defaults to `http://vllm:8900`. Set to `http://host.docker.internal:8900` on macOS. |
| `CONFIG_PATH` | shell / env | Path to `config.yaml`. Defaults to `config.yaml`. Useful in tests. |
| `LOG_LEVEL` | `docker-compose.yml` | Python log level. Default: `INFO`. |
| `LOG_FORMAT` | `docker-compose.yml` | `json` or `console`. Default: `json`. |
| `MAX_CONCURRENT_PER_REQUEST` | `docker-compose.yml` | Max model calls run in parallel within a single request. Default: `4`. |

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
config.yaml                   Model definitions and candidate label taxonomy
discovered_labels.json        LLM-discovered topic labels (persisted across restarts)
start.sh                      Interactive launcher — prompts OS + profiles, builds image
stop.sh                       Stops all containers (all profiles), keeps them for restart
down.sh                       Stops and removes all containers (all profiles)

Dockerfile                    text-analysis-backend image (FastAPI, uv)
docker-compose.yml            All services; sentiment/topic/linux profiles
                              YAML anchors (x-model-hf, x-model-zeroshot,
                              x-model-embedding, x-model-lexicon) eliminate
                              repetition across model service definitions

model_server/
  server.py                   FastAPI app + /predict and /health routes
  handlers.py                 TaskHandler ABC + concrete implementations
                              (TextClassification, ZeroShot, Embedding, Vader, Nrc)
                              ZeroShot processes candidate_labels in chunks of 20
                              Embedding caches label embeddings per candidate set
  requirements.txt            All deps: transformers, sentence-transformers, nltk, nrclex, python-slugify
  Dockerfile                  Single image (text-analysis/model-server:latest)
  Dockerfile.vllm             vllm/vllm-openai image (Linux only)

src/text_analysis/
  main.py                     App factory, lifespan, CORS, /metrics
  config.py                   Settings + load_model_configs() + load_domain_candidates() + load_all_candidates() + CandidateStore
  registry.py                 Builds clients from config.yaml at startup
  api/routes.py               POST /analyse · GET /candidates
  clients/
    base.py                   ModelClient ABC + PredictionResult + metrics + timing
    model_server.py           ModelServerClient — normalizes labels via python-slugify
    vllm.py                   VllmClient — sentiment + topic prompts, novel label resolution
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

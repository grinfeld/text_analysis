"""
Integration tests using testcontainers.

Spins up a real model_server container against a tiny HuggingFace model
(hf-internal-testing/tiny-random-RobertaForSequenceClassification, ~1 MB)
so tests run fast without downloading large weights.

The tiny model outputs LABEL_0 / LABEL_1 — not meaningful sentiment labels,
but that is intentional: we are testing the HTTP contract and client plumbing,
not model quality.

Requires Docker to be running. Tests are automatically skipped if Docker is
unavailable (e.g. in a CI environment without a Docker daemon).
"""

import os
import platform
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs

from text_analysis.clients.base import ModelClientError
from text_analysis.clients.model_server import ModelServerClient

# ---------------------------------------------------------------------------
# Tiny model used in all integration tests.
# ~1 MB, loads in under 10 seconds, labels are LABEL_0 / LABEL_1.
# ---------------------------------------------------------------------------
TINY_MODEL = "hf-internal-testing/tiny-random-RobertaForSequenceClassification"

TINY_LABEL_MAP = {"LABEL_0": "negative", "LABEL_1": "positive"}

_MODEL_SERVER_CONTEXT = os.path.join(os.path.dirname(__file__), "..", "model_server")

# LLM_URL set based on OS — matches what .env / docker-compose.yml does at runtime.
# macOS: mlx-lm runs on the host (localhost from test process perspective).
# Linux: vllm container is started via --profile linux, reachable at localhost:8900.
_IS_MACOS = platform.system() == "Darwin"
_LLM_URL = "http://localhost:8900"  # same on both: tests run on the host, not inside Docker


# ---------------------------------------------------------------------------
# Skip marker — applied automatically when Docker is not available
# ---------------------------------------------------------------------------
def _docker_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


skip_no_docker = pytest.mark.skipif(
    not _docker_available(), reason="Docker daemon not available"
)


# ---------------------------------------------------------------------------
# Session-scoped container fixture — starts once, shared across all tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def model_server_container():
    """
    Build and start the model_server container with the tiny model.
    Waits until uvicorn logs 'Application startup complete' before yielding.
    Stopped automatically at end of session.
    """
    container = (
        DockerContainer(image="text-analysis-model-server-test")
        .with_env("MODEL_NAME", TINY_MODEL)
        .with_env("HF_HOME", "/tmp/hf_cache")
        .with_exposed_ports(8000)
    )

    import docker as docker_sdk

    client = docker_sdk.from_env()
    client.images.build(path=_MODEL_SERVER_CONTEXT, tag="text-analysis-model-server-test", rm=True)

    with container:
        wait_for_logs(container, "Application startup complete", timeout=120)
        yield container


@pytest.fixture(scope="session")
def model_server_url(model_server_container) -> str:
    host = model_server_container.get_container_host_ip()
    port = model_server_container.get_exposed_port(8000)
    return f"http://{host}:{port}"


@pytest.fixture
def hf_integration_client(model_server_url) -> ModelServerClient:
    return ModelServerClient(
        model_name=TINY_MODEL,
        base_url=model_server_url,
        http_client=httpx.AsyncClient(),
        label_map=TINY_LABEL_MAP,
    )


# ---------------------------------------------------------------------------
# Tests: model_server HTTP contract
# ---------------------------------------------------------------------------
@skip_no_docker
class TestModelServerContainer:
    def test_health_endpoint(self, model_server_url):
        resp = httpx.get(f"{model_server_url}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model"] == TINY_MODEL

    def test_predict_returns_labels(self, model_server_url):
        resp = httpx.post(f"{model_server_url}/predict", json={"text": "I love this!"})
        assert resp.status_code == 200
        data = resp.json()
        assert "labels" in data
        assert len(data["labels"]) >= 1
        top = data["labels"][0]
        assert isinstance(top["label"], str)
        assert 0.0 <= top["score"] <= 1.0

    def test_predict_empty_text_returns_422(self, model_server_url):
        resp = httpx.post(f"{model_server_url}/predict", json={"text": "   "})
        assert resp.status_code == 422

    def test_predict_missing_text_returns_422(self, model_server_url):
        resp = httpx.post(f"{model_server_url}/predict", json={})
        assert resp.status_code == 422

    def test_predict_different_inputs_return_scores(self, model_server_url):
        for text in ["Great!", "Terrible.", "The sky is blue."]:
            resp = httpx.post(f"{model_server_url}/predict", json={"text": text})
            assert resp.status_code == 200
            data = resp.json()
            assert 0.0 <= data["labels"][0]["score"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: ModelServerClient against the live container
# ---------------------------------------------------------------------------
@skip_no_docker
class TestModelServerClientIntegration:
    async def test_predict_returns_sentiment_result(self, hf_integration_client):
        result = await hf_integration_client.predict("I really enjoyed this.")
        assert result.label in ("positive", "negative", "neutral")
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.raw, dict)

    async def test_predict_raw_contains_original_response(self, hf_integration_client):
        result = await hf_integration_client.predict("Hello world")
        assert "labels" in result.raw
        assert isinstance(result.raw["labels"], list)

    async def test_predict_normalises_raw_label(self, hf_integration_client):
        result = await hf_integration_client.predict("Some text to classify.")
        assert result.label in TINY_LABEL_MAP.values()

    async def test_predict_bad_url_raises_client_error(self, model_server_url):
        bad_client = ModelServerClient(
            model_name=TINY_MODEL,
            base_url="http://localhost:19999",
            http_client=httpx.AsyncClient(),
            label_map=TINY_LABEL_MAP,
        )
        with pytest.raises(ModelClientError):
            await bad_client.predict("hello")


# ---------------------------------------------------------------------------
# Tests: full backend stack with real model_server container
# Uses a temporary config.yaml pointing siebert at the test container.
# Compose file used: docker-compose-macos.yml or docker-compose-linux.yml
# depending on the OS the tests run on (_COMPOSE_FILE).
# ---------------------------------------------------------------------------
@skip_no_docker
class TestBackendWithRealModelServer:
    """
    Bootstraps the FastAPI app with a temporary config.yaml that routes
    the siebert model at the live tiny-model container.
    """

    @pytest.fixture
    def client_with_real_model(self, model_server_url, monkeypatch, tmp_path):
        # Write a temporary config.yaml pointing siebert at the test container
        cfg = {
            "models": [
                {
                    "name": "siebert/sentiment-roberta-large-english",
                    "type": "ml",
                    "url": model_server_url,
                    "labels": {"LABEL_0": "negative", "LABEL_1": "positive"},
                }
            ]
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(cfg))

        monkeypatch.setenv("CONFIG_PATH", str(config_file))
        monkeypatch.setenv("LLM_URL", _LLM_URL)

        import importlib

        import text_analysis.config as cfg_module
        import text_analysis.registry as reg_module

        importlib.reload(cfg_module)
        importlib.reload(reg_module)

        from text_analysis.main import app
        from text_analysis.observability import metrics

        metrics.setup()
        reg_module.init()

        with TestClient(app) as c:
            yield c

        metrics.shutdown()

    async def test_analyse_endpoint_with_real_model(self, client_with_real_model):
        resp = client_with_real_model.post(
            "/analyse",
            json={"text": "This is an integration test."},
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 1
        r = results[0]
        assert r["model"] == "siebert/sentiment-roberta-large-english"
        assert len(r["labels"]) == 1
        assert r["labels"][0]["label"] in ("positive", "negative", "neutral")
        assert 0.0 <= r["labels"][0]["score"] <= 1.0
        assert r["error"] is None

    async def test_empty_text_still_422_with_real_model(self, client_with_real_model):
        resp = client_with_real_model.post("/analyse", json={"text": ""})
        assert resp.status_code == 422

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

import httpx
import pytest
from fastapi.testclient import TestClient
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs

from sentiment.clients.base import SentimentClientError
from sentiment.clients.model_server import ModelServerClient

# ---------------------------------------------------------------------------
# Tiny model used in all integration tests.
# ~1 MB, loads in under 10 seconds, labels are LABEL_0 / LABEL_1.
# ---------------------------------------------------------------------------
TINY_MODEL = "hf-internal-testing/tiny-random-RobertaForSequenceClassification"

# Label map for the tiny model's raw output
TINY_LABEL_MAP = {"LABEL_0": "negative", "LABEL_1": "positive"}

# Path to the built model_server image context
_MODEL_SERVER_CONTEXT = os.path.join(os.path.dirname(__file__), "..", "model_server")


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
        DockerContainer(image="sentiment-model-server-test")
        .with_env("MODEL_NAME", TINY_MODEL)
        .with_env("HF_HOME", "/tmp/hf_cache")
        .with_exposed_ports(8000)
    )

    # Build the image from the model_server directory
    import docker as docker_sdk

    client = docker_sdk.from_env()
    client.images.build(path=_MODEL_SERVER_CONTEXT, tag="sentiment-model-server-test", rm=True)

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
    # Function-scoped so each async test gets a fresh AsyncClient bound to its event loop.
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

    def test_predict_returns_label_and_score(self, model_server_url):
        resp = httpx.post(f"{model_server_url}/predict", json={"text": "I love this!"})
        assert resp.status_code == 200
        data = resp.json()
        assert "label" in data
        assert "score" in data
        assert isinstance(data["label"], str)
        assert 0.0 <= data["score"] <= 1.0

    def test_predict_empty_text_returns_422(self, model_server_url):
        resp = httpx.post(f"{model_server_url}/predict", json={"text": "   "})
        assert resp.status_code == 422

    def test_predict_missing_text_returns_422(self, model_server_url):
        resp = httpx.post(f"{model_server_url}/predict", json={})
        assert resp.status_code == 422

    def test_predict_different_inputs_return_scores(self, model_server_url):
        """Model returns a score for each input — not asserting on label value,
        just that the shape is consistent regardless of input text."""
        for text in ["Great!", "Terrible.", "The sky is blue."]:
            resp = httpx.post(f"{model_server_url}/predict", json={"text": text})
            assert resp.status_code == 200
            data = resp.json()
            assert 0.0 <= data["score"] <= 1.0


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
        # raw must have at least label and score from the model server
        assert "label" in result.raw
        assert "score" in result.raw

    async def test_predict_normalises_raw_label(self, hf_integration_client):
        """The tiny model returns LABEL_0 or LABEL_1 — client must normalise these."""
        result = await hf_integration_client.predict("Some text to classify.")
        # After normalisation, label must be one of our canonical values
        assert result.label in TINY_LABEL_MAP.values()

    async def test_predict_bad_url_raises_client_error(self, model_server_url):
        """Pointing to a non-existent path should raise SentimentClientError."""
        bad_client = ModelServerClient(
            model_name=TINY_MODEL,
            base_url="http://localhost:19999",  # nothing listening here
            http_client=httpx.AsyncClient(),
            label_map=TINY_LABEL_MAP,
        )
        with pytest.raises(SentimentClientError):
            await bad_client.predict("hello")


# ---------------------------------------------------------------------------
# Tests: full backend stack with overridden model URL
# ---------------------------------------------------------------------------
@skip_no_docker
class TestBackendWithRealModelServer:
    """
    Uses FastAPI TestClient with the real model_server container.
    Overrides SIEBERT_URL so the backend routes siebert requests to the
    tiny-model container instead of the real siebert service.
    """

    @pytest.fixture
    def client_with_real_model(self, model_server_url, monkeypatch):
        # Override the siebert URL in settings to point at our test container
        monkeypatch.setenv("SIEBERT_URL", model_server_url)

        # Re-import settings and re-init registry so the new URL takes effect
        import importlib

        import sentiment.config as cfg_module
        import sentiment.registry as reg_module

        importlib.reload(cfg_module)
        importlib.reload(reg_module)

        from sentiment.main import app
        from sentiment.observability import metrics

        metrics.setup()
        reg_module.init()

        with TestClient(app) as c:
            yield c

        metrics.shutdown()

    async def test_sentiment_endpoint_with_real_model(self, client_with_real_model):
        resp = client_with_real_model.post(
            "/sentiment",
            json={
                "model": "siebert/sentiment-roberta-large-english",
                "text": "This is an integration test.",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] in ("positive", "negative", "neutral")
        assert 0.0 <= data["score"] <= 1.0
        assert data["model"] == "siebert/sentiment-roberta-large-english"

    async def test_empty_text_still_422_with_real_model(self, client_with_real_model):
        resp = client_with_real_model.post(
            "/sentiment",
            json={"model": "siebert/sentiment-roberta-large-english", "text": ""},
        )
        assert resp.status_code == 422

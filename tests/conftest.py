import os
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from sentiment.observability import metrics

# Use the real config.yaml from the project root
os.environ.setdefault("CONFIG_PATH", str(Path(__file__).parent.parent / "config.yaml"))

from sentiment.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def setup_metrics():
    """Ensure OTEL metrics are initialised for every test."""
    metrics.setup()
    yield
    metrics.shutdown()


@pytest.fixture
def client():
    """Synchronous FastAPI TestClient."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_http():
    """Active respx mock router for all HTTPX calls."""
    with respx.mock(assert_all_called=False) as router:
        yield router

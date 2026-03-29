import importlib
import os
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from sentiment.observability import metrics

_CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")

# Use the real config.yaml from the project root
os.environ.setdefault("CONFIG_PATH", _CONFIG_PATH)

from sentiment.main import app  # noqa: E402
from sentiment import registry as registry_module  # noqa: E402


@pytest.fixture(autouse=True)
def setup_metrics():
    """Ensure OTEL metrics are initialised for every test."""
    metrics.setup()
    yield
    metrics.shutdown()


@pytest.fixture
def client():
    """Synchronous FastAPI TestClient. Reloads config + registry modules so
    that integration tests which reload those modules with a temp config don't
    leak their state into route tests."""
    import sentiment.config as cfg_module

    os.environ["CONFIG_PATH"] = _CONFIG_PATH
    importlib.reload(cfg_module)
    importlib.reload(registry_module)
    registry_module.init()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_http():
    """Active respx mock router for all HTTPX calls."""
    with respx.mock(assert_all_called=False) as router:
        yield router

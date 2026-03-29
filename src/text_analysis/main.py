from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from text_analysis.api.routes import router
from text_analysis.config import settings
from text_analysis.observability import metrics
from text_analysis.observability.logging import configure_logging
from text_analysis import registry as registry_module

configure_logging(settings.log_level, settings.log_format)
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: metrics first (PrometheusMetricReader must exist before /metrics is hit),
    # then build HTTP clients and registry
    metrics.setup()
    registry_module.init()
    logger.info("startup", log_level=settings.log_level, log_format=settings.log_format)
    yield
    # Shutdown: flush metrics, close HTTP clients
    logger.info("shutdown")
    await registry_module.close_all()
    metrics.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(title="Sentiment API", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    app.include_router(router)

    # Mount Prometheus /metrics endpoint
    # must be called after metrics.setup() has been invoked — the ASGI app reads
    # from the global prometheus_client registry which is populated by PrometheusMetricReader
    app.mount("/metrics", metrics.make_metrics_app())

    return app


app = create_app()

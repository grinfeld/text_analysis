from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader  # noqa: F401 (re-exported for typing)
from opentelemetry.sdk.resources import Resource
from prometheus_client import make_asgi_app

# Module-level singletons — initialised once in app lifespan
_meter_provider: MeterProvider | None = None
_meter: metrics.Meter | None = None

# Instruments (populated after setup())
_request_counter: metrics.Counter | None = None
_error_counter: metrics.Counter | None = None
_latency_histogram: metrics.Histogram | None = None
_confidence_histogram: metrics.Histogram | None = None


def setup() -> None:
    """Initialise the OTEL MeterProvider with a Prometheus exporter.

    Must be called before any record_* functions are used.
    Call make_metrics_app() AFTER this to get the ASGI app for /metrics.
    """
    global _meter_provider, _meter
    global _request_counter, _error_counter, _latency_histogram, _confidence_histogram

    reader = PrometheusMetricReader()
    resource = Resource.create({"service.name": "sentiment-backend"})
    _meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(_meter_provider)

    _meter = _meter_provider.get_meter("sentiment.model", version="0.1.0")

    _request_counter = _meter.create_counter(
        name="sentiment_model_requests_total",
        unit="{request}",
        description="Total number of sentiment prediction calls",
    )
    _error_counter = _meter.create_counter(
        name="sentiment_model_errors_total",
        unit="{error}",
        description="Total number of failed sentiment prediction calls",
    )
    _latency_histogram = _meter.create_histogram(
        name="sentiment_model_latency_seconds",
        unit="s",
        description="Time taken for a sentiment model to return a result",
        explicit_bucket_boundaries_advisory=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
    )
    _confidence_histogram = _meter.create_histogram(
        name="sentiment_model_confidence_score",
        unit="{score}",
        description="Confidence score returned by the sentiment model",
        explicit_bucket_boundaries_advisory=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
    )


def make_metrics_app():
    """Return the Prometheus ASGI app for mounting at /metrics.

    Must be called AFTER setup().
    """
    return make_asgi_app()


def shutdown() -> None:
    if _meter_provider is not None:
        _meter_provider.shutdown()


def record_request(model: str) -> None:
    if _request_counter is not None:
        _request_counter.add(1, {"model": model})


def record_error(model: str, error_type: str) -> None:
    if _error_counter is not None:
        _error_counter.add(1, {"model": model, "error_type": error_type})


def record_latency(model: str, label: str, latency_seconds: float) -> None:
    if _latency_histogram is not None:
        _latency_histogram.record(latency_seconds, {"model": model, "label": label})


def record_confidence(model: str, label: str, score: float) -> None:
    if _confidence_histogram is not None:
        _confidence_histogram.record(score, {"model": model, "label": label})

import httpx

from sentiment.clients.base import SentimentClient
from sentiment.clients.model_server import ModelServerClient
from sentiment.clients.vllm import VllmClient
from sentiment.config import load_model_configs

_clients: list[SentimentClient] = []
_http_clients: list[httpx.AsyncClient] = []


def init() -> None:
    """Build all clients from config.yaml. Called from app lifespan."""
    global _clients, _http_clients

    _clients = []
    _http_clients = []

    for model_cfg in load_model_configs():
        http = httpx.AsyncClient()
        _http_clients.append(http)

        if model_cfg.type == "ml":
            _clients.append(ModelServerClient(
                model_name=model_cfg.name,
                base_url=model_cfg.url,
                http_client=http,
                label_map=model_cfg.labels or {},
            ))
        elif model_cfg.type == "llm":
            _clients.append(VllmClient(
                model_name=model_cfg.name,
                base_url=model_cfg.url,
                model_id=model_cfg.model_id or model_cfg.name,
                http_client=http,
            ))


def get_all_clients() -> list[SentimentClient]:
    return list(_clients)


async def close_all() -> None:
    for http in _http_clients:
        await http.aclose()

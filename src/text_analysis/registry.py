import httpx

from text_analysis.clients.base import ModelClient
from text_analysis.clients.model_server import ModelServerClient
from text_analysis.clients.vllm import (
    VllmClient,
    SENTIMENT_SYSTEM_PROMPT,
    SENTIMENT_USER_PROMPT_TEMPLATE,
    SENTIMENT_VALID_LABELS,
    TOPIC_SYSTEM_PROMPT,
    TOPIC_USER_PROMPT_TEMPLATE,
)
from text_analysis.config import load_model_configs, load_all_candidates, load_domain_candidates, candidate_store

_VALID_TASKS = {"sentiment", "topic", "llm"}

_clients: list[ModelClient] = []
_http_clients: list[httpx.AsyncClient] = []


def init() -> None:
    """Build all clients from config.yaml. Called from app lifespan."""
    global _clients, _http_clients

    _clients = []
    _http_clients = []

    all_configs = load_model_configs()
    topic_candidate_labels = load_all_candidates()
    domain_map = load_domain_candidates()
    candidate_store.__init__(topic_candidate_labels, domain_config=domain_map)

    e5_small_url = next(
        (c.url for c in all_configs if c.name == "intfloat/e5-small-v2"),
        None,
    )

    for model_cfg in all_configs:
        task = model_cfg.for_ or "sentiment"
        if task not in _VALID_TASKS:
            raise ValueError(
                f"Invalid 'for' value {task!r} in config for model {model_cfg.name!r}. "
                f"Must be one of: {sorted(_VALID_TASKS)}"
            )

        http = httpx.AsyncClient()
        _http_clients.append(http)

        if model_cfg.type == "ml":
            _clients.append(ModelServerClient(
                model_name=model_cfg.name,
                base_url=model_cfg.url,
                http_client=http,
                label_map=model_cfg.labels or {},
                task=task,
                candidate_labels=topic_candidate_labels if task == "topic" else [],
            ))
        elif model_cfg.type == "llm":
            # for: llm means the same server is used for both sentiment and topic.
            # Register one client per task, sharing the same http client.
            tasks_to_register = ["sentiment", "topic"] if task == "llm" else [task]
            for llm_task in tasks_to_register:
                if llm_task == "topic":
                    system_prompt = TOPIC_SYSTEM_PROMPT
                    user_prompt_template = TOPIC_USER_PROMPT_TEMPLATE
                    valid_labels = frozenset()  # open-ended — model may propose any slug
                else:
                    system_prompt = SENTIMENT_SYSTEM_PROMPT
                    user_prompt_template = SENTIMENT_USER_PROMPT_TEMPLATE
                    valid_labels = SENTIMENT_VALID_LABELS

                _clients.append(VllmClient(
                    model_name=model_cfg.name,
                    base_url=model_cfg.url,
                    model_id=model_cfg.model_id or model_cfg.name,
                    http_client=http,
                    task=llm_task,
                    system_prompt=system_prompt,
                    user_prompt_template=user_prompt_template,
                    valid_labels=valid_labels,
                    candidate_labels=topic_candidate_labels if llm_task == "topic" else [],
                    candidate_store=candidate_store if llm_task == "topic" else None,
                    e5_small_url=e5_small_url if llm_task == "topic" else None,
                ))


def get_clients_for(task: str) -> list[ModelClient]:
    return [c for c in _clients if c.task == task]


async def close_all() -> None:
    for http in _http_clients:
        await http.aclose()

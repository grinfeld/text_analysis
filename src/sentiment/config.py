from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Model server URLs (internal Docker hostnames)
    siebert_url: str = "http://siebert:8000"
    cardiffnlp_url: str = "http://cardiffnlp:8000"
    distilbert_url: str = "http://distilbert:8000"

    # vLLM service
    vllm_url: str = "http://vllm:8900"
    vllm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" | "console"


settings = Settings()

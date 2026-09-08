from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_path: Path = Path("data/curated/credit_risk.duckdb")
    schema_registry: Path = Path("configs/schema_registry.yaml")
    schema_registry_version: str = "1.1.0"
    architecture_policy: Path = Path("configs/architecture_policy.yaml")
    architecture_policy_version: str = "1.1.0"
    retrieval_policy: Path = Path("configs/retrieval.yaml")
    service_role: str = "api_gateway"
    service_token: str = "local-dev-only-change-me"
    data_service_url: str = "http://data-service:8081"
    # oMLX on the target Mac is configured for 9905, not the upstream default 8000.
    omlx_base_url: str = "http://127.0.0.1:9905/v1"
    omlx_model: str = "credit-risk-qwen3.5-9b"
    feedback_path: Path = Path("data/feedback/feedback.jsonl")
    query_timeout_seconds: int = 15
    max_context_tokens: int = 16_384  # read by the inference layer (phase 3)
    min_evidence_score: float = 0.72

    model_config = SettingsConfigDict(env_prefix="CR_", env_file=".env", extra="ignore")


settings = Settings()

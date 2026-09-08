from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_path: Path = Path("data/curated/credit_risk.duckdb")
    schema_registry: Path = Path("configs/schema_registry.yaml")
    schema_registry_version: str = "1.1.0"
    architecture_policy: Path = Path("configs/architecture_policy.yaml")
    architecture_policy_version: str = "1.0.0"
    service_role: str = "api_gateway"
    service_token: str = "local-dev-only-change-me"
    data_service_url: str = "http://data-service:8081"
    omlx_base_url: str = "http://127.0.0.1:8000/v1"
    omlx_model: str = "Qwen3.5-9B-4bit"
    query_timeout_seconds: int = 15
    max_result_rows: int = 120
    max_context_tokens: int = 16_384
    min_evidence_score: float = 0.72

    model_config = SettingsConfigDict(env_prefix="CR_", env_file=".env", extra="ignore")


settings = Settings()

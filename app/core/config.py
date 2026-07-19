from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    app_name: str = "roadbogo-ai"
    environment: str = "local"
    model_name: str = "roadbogo-baseline"

    backend_cors_origins: list[str] = [
        "http://localhost:3000"
    ]

    internal_api_v1_prefix: str = "/api/internal/v1"
    internal_api_key: SecretStr | None = None


settings = Settings()

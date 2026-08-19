from functools import lru_cache
from pathlib import Path

from pydantic import (
    Field,
)

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


class Settings(BaseSettings):

    # ======================================================
    # Application
    # ======================================================

    app_name: str = (
        "LLM Cost Autopilot"
    )

    app_version: str = (
        "0.1.0"
    )

    app_env: str = (
        "development"
    )


    # ======================================================
    # Gemini
    # ======================================================

    gemini_api_key: str


    # ======================================================
    # Verification
    # ======================================================

    verification_auto_accept_score: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
    )


    # ======================================================
    # Database
    # ======================================================

    database_path: str = str(
        PROJECT_ROOT
        /
        "data"
        /
        "autopilot.db"
    )


    # ======================================================
    # .env configuration
    # ======================================================

    model_config = SettingsConfigDict(

        env_file=
            PROJECT_ROOT / ".env",

        env_file_encoding=
            "utf-8",

        extra=
            "ignore",
    )


@lru_cache
def get_settings() -> Settings:

    return Settings()
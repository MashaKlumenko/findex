"""Process configuration. Values come from the environment, never from code."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """12-factor settings.

    ``INDEX_PATH`` is required for a useful server. When it is unset the
    process still starts and ``/health`` returns 503, which is what a
    platform probe should see before the index is loaded.

    ``FINDEX_SEARCH_ASYNC`` selects the blocking ``async def`` search route.
    It exists so the Lab 7 load table can be reproduced. Leave it unset:
    the default ``def`` route runs in a threadpool and does not stall the
    event loop.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    index_path: Path | None = None
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000
    findex_search_async: bool = Field(default=False)

    @field_validator("index_path", mode="before")
    @classmethod
    def empty_path_is_unset(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("log_level")
    @classmethod
    def normalize_level(cls, value: str) -> str:
        return value.strip().upper() or "INFO"


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Settings singleton for the process entrypoint."""
    return Settings()

"""Application settings loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    alpaca_api_key: str = Field(default="", alias="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(default="", alias="ALPACA_SECRET_KEY")
    fred_api_key: str = Field(default="", alias="FRED_API_KEY")
    fdq_data_dir: Path = Field(default=Path("data"), alias="FDQ_DATA_DIR")

    @property
    def data_dir(self) -> Path:
        return self.fdq_data_dir

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def quality_dir(self) -> Path:
        return self.data_dir / "quality"

    @property
    def config_dir(self) -> Path:
        return Path("config")


def get_settings() -> Settings:
    return Settings()

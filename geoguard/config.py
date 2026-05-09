from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ReasoningEffort = Literal["minimal", "low", "medium", "high"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GEOGUARD_",
        extra="ignore",
    )

    model: str = "openai:gpt-5.2"
    reasoning_effort: ReasoningEffort = "medium"


settings = Settings()

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ReasoningEffort = Literal["minimal", "low", "medium", "high"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GEOGUARD_",
        extra="ignore",
    )

    # LLM defaults — used by every block when its `model` / `reasoning_effort`
    # constructor params are None.
    model: str = "openai:gpt-5.2"
    reasoning_effort: ReasoningEffort = "medium"

    # Claim extraction — applied by GeoGuard.from_config().
    max_claims: int | None = 15

    # Rubric — applied by GeoGuard.from_config().
    questions_per_claim_min: int = 5
    questions_per_claim_max: int = 10

    # HTTP timeout (seconds) — used by tools that make external API calls.
    http_timeout_seconds: float = 30.0

    # Max tool calls per claim verification — caps the verifier agent's
    # tool-use budget so it commits to a verdict rather than over-sampling.
    verification_tool_usage_limit: int = 7


settings = Settings()

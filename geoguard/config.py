from typing import Literal

from pydantic import Field
from pydantic_ai.models import Model, infer_model, infer_provider_class
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

    # Optional explicit API key. When set, blocks construct providers with
    # this key directly — concurrency-safe (no env mutation), required for
    # multi-tenant deployments (e.g. an HF Space accepting BYOK). When None
    # (default), pydantic-ai reads the provider's standard env var
    # (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …).
    api_key: str | None = None

    # Claim extraction — applied by GeoGuard.from_config().
    max_claims: int | None = 15

    # Rubric — applied by GeoGuard.from_config().
    questions_per_claim_min: int = 5
    questions_per_claim_max: int = 10

    # HTTP timeout (seconds) — used by tools that make external API calls.
    http_timeout_seconds: float = 30.0

    # NASA VEDA endpoints — the public deployment by default; override
    # (GEOGUARD_VEDA_STAC_URL / GEOGUARD_VEDA_RASTER_URL) for staging or a
    # private deployment. The raster URL is the titiler API that computes
    # zonal statistics server-side, so no raster ever needs downloading.
    veda_stac_url: str = "https://openveda.cloud/api/stac"
    veda_raster_url: str = "https://openveda.cloud/api/raster"

    # STAC exploration sub-agent budget. The sub-agent runs nested inside a
    # single verifier tool call, so its cost multiplies the verifier's —
    # these caps are mandatory, not optional (None is deliberately not
    # allowed here, unlike the verifier's limits). Hitting either limit is
    # handled gracefully: the sub-agent returns a partial "no usable
    # evidence" summary instead of killing the verification.
    stac_agent_request_limit: int = Field(default=12, gt=0)
    stac_agent_tool_calls_limit: int = Field(default=8, gt=0)

    # Max tool calls per claim verification. None (default) = no cap.
    # Set to a positive int (e.g. 15) to bound the verifier's tool-use
    # budget when over-sampling becomes a problem. Hitting the limit
    # currently terminates the verification (pydantic-ai raises
    # UsageLimitExceeded) rather than producing a partial verdict.
    verification_tool_usage_limit: int | None = Field(default=None, gt=0)

    # Max LLM API requests per claim verification — runaway-loop guard,
    # not a budget. A normal run is 5-20 requests; 100 is generous
    # headroom. None = unlimited. Hitting it raises pydantic-ai's
    # UsageLimitExceeded.
    verification_request_limit: int | None = Field(default=100, gt=0)

    # Pydantic-ai output-validation retries (applies to MetadataExtractor
    # and Verifier agents). Number of times the agent is re-prompted when
    # its structured output fails validation before giving up.
    output_retries: int = Field(default=2, ge=0)


settings = Settings()


def build_model(
    model: str | None = None,
    api_key: str | None = None,
) -> Model | str:
    """Resolve a `provider:name` model string to a pydantic-ai Model.

    Provider-agnostic: works for `openai:`, `anthropic:`, etc. — anything
    pydantic-ai's `infer_provider_class` knows about.

    - `api_key=None` → return the string unchanged; pydantic-ai's normal
      env-driven path constructs the provider and reads its standard env
      var (`OPENAI_API_KEY`, …) at agent-construction time.
    - `api_key="..."` → construct the provider explicitly with that key.
      Concurrency-safe: multiple GeoGuard instances with different keys
      coexist without env-var races.

    Defaults to `settings.model` / `settings.api_key` when either argument
    is None.
    """
    resolved_model = model or settings.model
    resolved_key = api_key or settings.api_key
    if resolved_key is None:
        return resolved_model

    def _factory(provider_name: str):
        return infer_provider_class(provider_name)(api_key=resolved_key)

    return infer_model(resolved_model, provider_factory=_factory)

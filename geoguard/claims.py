from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

from .config import ReasoningEffort, settings
from .schemas import Input

DEFAULT_INSTRUCTIONS = (
    "Extract atomic, decontextualized, individually verifiable factual claims "
    "from the input. Each claim must be self-contained and verifiable in "
    "isolation: include all proper nouns (e.g. 'Hurricane Beryl' not 'the "
    "storm'), absolute dates ('April 15, 2026' not 'that day'), and explicit "
    "locations ('Houston, Texas' not 'the city'). Avoid pronouns ('it', "
    "'they') and deictic references ('the flood', 'that area') that require "
    "surrounding context to interpret. "
    "Skip opinions, hedges, and meta-commentary."
)


class Claim(BaseModel):
    """A single factual claim extracted from input text — one unit of verification."""

    claim: str = Field(
        description=(
            "An atomic, decontextualized, individually verifiable factual "
            "statement. Self-contained and verifiable in isolation: include "
            "all proper nouns (e.g. 'Hurricane Beryl' not 'the storm'), "
            "absolute dates ('April 15, 2026' not 'that day'), and explicit "
            "locations ('Houston, Texas' not 'the city'). Avoid pronouns "
            "and deictic references that require surrounding context."
        ),
    )


class ClaimExtractor:
    def __init__(
        self,
        model: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        instructions: str | None = None,
    ):
        self._agent = Agent(
            model=model or settings.model,
            output_type=list[Claim],
            capabilities=[
                Thinking(effort=reasoning_effort or settings.reasoning_effort),
            ],
            instructions=instructions or DEFAULT_INSTRUCTIONS,
        )

    async def __call__(self, inp: Input) -> list[Claim]:
        result = await self._agent.run(inp.text)
        return result.output

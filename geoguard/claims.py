from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

from .config import ReasoningEffort, settings
from .schemas import Input

DEFAULT_INSTRUCTIONS = (
    "Extract atomic, individually verifiable factual claims from the input. "
    "Each claim should be self-contained and check-able against an external source. "
    "Skip opinions, hedges, and meta-commentary."
)


class Claim(BaseModel):
    claim: str


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

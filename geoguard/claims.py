from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

from .config import ReasoningEffort, settings
from .schemas import Input

DEFAULT_MAX_CLAIMS = 15


CLAIM_RULES = """\
Each claim must be:
- ATOMIC: one factual assertion per claim
- DECONTEXTUALIZED: include all proper nouns ('Hurricane Beryl' not 'the storm'), \
absolute dates ('April 15, 2026' not 'that day'), and explicit locations \
('Houston, Texas' not 'the city'). Avoid pronouns and deictic references that \
require surrounding context.
- DISTINCT: no claim may overlap with or be a sub-statement of another. \
If multiple facts overlap (same event from different angles, or one fact implying \
another), MERGE them into one comprehensive claim.
- FAITHFUL: every fact in a claim must be explicitly stated somewhere in the input. \
You MAY combine facts from different parts of the input (e.g., to decontextualize a \
claim with a date mentioned elsewhere). You MUST NOT add facts the input doesn't \
state, infer beyond the text, or use prior knowledge to supplement claims.
- VERIFIABLE: checkable against an external source.\
"""


def _cap_rule(max_claims: int | None) -> str:
    if max_claims is None:
        return "Extract as many distinct claims as the input warrants."
    return (
        f"Extract AT MOST {max_claims} claims. When the input contains more facts "
        f"than the cap, prioritize central, load-bearing claims; skip trivia and "
        f"asides."
    )


DEFAULT_INSTRUCTIONS = (
    "Extract atomic, decontextualized, individually verifiable factual claims "
    "from the input.\n\n"
    + CLAIM_RULES
    + "\n\nSkip opinions, hedges, and meta-commentary."
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
            "and deictic references that require surrounding context. "
            "Every fact in the claim must be explicitly stated in the input."
        ),
    )


class ClaimExtractor:
    def __init__(
        self,
        model: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        instructions: str | None = None,
        max_claims: int | None = None,
    ):
        instructions = instructions or DEFAULT_INSTRUCTIONS
        if max_claims is not None:
            instructions += f"\n\nLimits:\n{_cap_rule(max_claims)}"
        self._agent = Agent(
            model=model or settings.model,
            output_type=list[Claim],
            capabilities=[
                Thinking(effort=reasoning_effort or settings.reasoning_effort),
            ],
            instructions=instructions,
        )

    async def __call__(self, inp: Input) -> list[Claim]:
        result = await self._agent.run(inp.text)
        return result.output

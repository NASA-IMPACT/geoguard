from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

from geoguard.claims import Claim
from geoguard.config import ReasoningEffort, build_model, settings
from geoguard.schemas import Input
from geoguard.verifications import VerifierResult


class RubricItem(BaseModel):
    """A single yes/no question + answer about one claim's verification."""

    question: str
    answer: bool
    reasoning: str | None = None


class ClaimRubric(BaseModel):
    """Rubric questions for one specific claim, with per-claim score."""

    claim: Claim
    items: list[RubricItem]

    @property
    def score(self) -> float:
        if not self.items:
            return 0.0
        return sum(1 for it in self.items if it.answer) / len(self.items)


class Rubric(BaseModel):
    """Holistic rubric across all claims; aggregated to one confidence score."""

    per_claim: list[ClaimRubric]

    @property
    def confidence(self) -> float:
        if not self.per_claim:
            return 0.0
        return sum(cr.score for cr in self.per_claim) / len(self.per_claim)


DEFAULT_INSTRUCTIONS = """\
For each claim, generate between {min_q} and {max_q} atomic yes/no questions, \
calibrated to that claim's specific assertions and the evidence gathered in its \
tool_calls. Each question must be answerable from the tool_call evidence alone.

WITHIN a single claim's rubric:
- Every question must test a DISTINCT aspect of that claim
- No duplicates or near-duplicates
- Cover (where applicable): location verifiability, time verifiability, \
quantity / magnitude consistency, causal attribution, source attribution

ACROSS claims:
- Prefer questions that target each claim's UNIQUE verifiable details
- Don't ask the same generic question across every claim — instead ask questions \
specific to each claim's quantitative or qualitative content

For each `answer=true`:
- The `reasoning` field MUST cite the specific tool_call result (tool name + \
relevant value) that supports the answer

When evidence is ambiguous, partial, or absent, answer `false` (conservative \
default). False is the safe choice for a guardrail.

Produce one ClaimRubric per input claim, in the same order as the claims appear \
in the prompt. Use the original claim text verbatim.\
"""


class Rubricator:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        instructions: str | None = None,
        questions_per_claim: tuple[int, int] = (5, 10),
        **agent_kwargs,
    ):
        self._questions_per_claim = questions_per_claim
        self._agent = Agent(
            model=build_model(model, api_key),
            output_type=Rubric,
            capabilities=[
                Thinking(effort=reasoning_effort or settings.reasoning_effort),
            ],
            instructions=instructions
            or DEFAULT_INSTRUCTIONS.format(
                min_q=questions_per_claim[0],
                max_q=questions_per_claim[1],
            ),
            **agent_kwargs,
        )

    @staticmethod
    def _build_prompt(inp: Input, verifications: list[VerifierResult]) -> str:
        """Serialize the original input + per-claim verification trace for the agent."""
        lines = ["# Original Input", "", inp.text, "", "# Per-Claim Verifications", ""]
        for i, vr in enumerate(verifications, 1):
            v = vr.verification
            lines.append(f"## Claim {i}: {v.claim.claim}")
            lines.append("")
            lines.append(f"- Verdict: {v.verdict.value}")
            lines.append(f"- Verifier rationale: {v.rationale}")
            if vr.tool_calls:
                lines.append(f"- Tool calls ({len(vr.tool_calls)}):")
                for tc in vr.tool_calls:
                    lines.append(f"  - {tc.name}({tc.args}) -> {tc.result}")
            else:
                lines.append("- Tool calls: (none)")
            lines.append("")
        return "\n".join(lines)

    async def __call__(
        self,
        inp: Input,
        verifications: list[VerifierResult],
        **run_kwargs,
    ) -> Rubric:
        prompt = self._build_prompt(inp, verifications)
        result = await self._agent.run(prompt, **run_kwargs)
        return result.output

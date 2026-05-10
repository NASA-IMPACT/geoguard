from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

from geoguard.claims import Claim
from geoguard.config import ReasoningEffort, settings
from geoguard.metadata import Metadata
from geoguard.tools.registry import registry


class Verdict(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    INCONCLUSIVE = "inconclusive"


class ClaimVerification(BaseModel):
    claim: Claim
    metadata: Metadata
    verdict: Verdict
    rationale: str


class ToolCall(BaseModel):
    name: str
    args: str  # JSON-serialized args from the agent (human-readable)
    result: Any


@dataclass
class VerifierResult:
    verification: ClaimVerification
    tool_calls: list[ToolCall]


DEFAULT_INSTRUCTIONS = (
    "Verify the given claim using the tools attached. Reason over the claim "
    "and metadata, call the tools you need to gather evidence, and return a "
    "ClaimVerification with one of three verdicts: SUPPORTS (tool evidence "
    "directly confirms the claim), CONTRADICTS (tool evidence directly "
    "disconfirms the claim), or INCONCLUSIVE (insufficient or ambiguous "
    "evidence). Cite the evidence you used in the rationale. If no tools "
    "are available, the verdict must be INCONCLUSIVE."
)


def _extract_tool_calls(messages) -> list[ToolCall]:
    pending: dict[str, tuple[str, str]] = {}
    calls: list[ToolCall] = []
    for msg in messages:
        for part in getattr(msg, "parts", []):
            if isinstance(part, ToolCallPart):
                if part.tool_name == "final_result":
                    continue  # pydantic-ai's synthetic structured-output tool
                pending[part.tool_call_id] = (
                    part.tool_name,
                    part.args_as_json_str(),
                )
            elif isinstance(part, ToolReturnPart):
                key = part.tool_call_id
                if key in pending:
                    name, args = pending.pop(key)
                    calls.append(ToolCall(name=name, args=args, result=part.content))
    return calls


class Verifier:
    def __init__(
        self,
        model: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        instructions: str | None = None,
        **agent_kwargs,
    ):
        self._model = model or settings.model
        self._reasoning_effort = reasoning_effort or settings.reasoning_effort
        self._instructions = instructions or DEFAULT_INSTRUCTIONS
        self._agent_kwargs = agent_kwargs

    async def __call__(
        self,
        claim: Claim,
        metadata: Metadata,
        tools: list[Callable],
        **run_kwargs,
    ) -> VerifierResult:
        toolsets = [registry.build_toolset(tools, id="verifier")] if tools else []
        agent = Agent(
            model=self._model,
            output_type=ClaimVerification,
            toolsets=toolsets,
            capabilities=[Thinking(effort=self._reasoning_effort)],
            instructions=self._instructions,
            **self._agent_kwargs,
        )
        prompt = f"Claim: {claim.claim}\n\nMetadata: {metadata.model_dump_json()}"
        result = await agent.run(prompt, **run_kwargs)
        return VerifierResult(
            verification=result.output,
            tool_calls=_extract_tool_calls(result.all_messages()),
        )

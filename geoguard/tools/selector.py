import inspect
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

from geoguard.claims import Claim
from geoguard.config import ReasoningEffort, build_model, settings
from geoguard.metadata import Metadata
from geoguard.tools.registry import registry


class ToolSelection(BaseModel):
    chosen: list[str]
    reasoning: str | None = None


@dataclass
class SelectedTools:
    tools: list[Callable]
    reasoning: str | None = None
    claim: Claim | None = None


DEFAULT_INSTRUCTIONS = (
    "Given a claim and its metadata context, plus a list of candidate tools "
    "(name, full signature with parameter and return types, and docstring), "
    "pick the subset best suited to verify the claim. Match the tool's "
    "parameter types against the context available in the metadata — only "
    "pick tools whose params can be satisfied. Return the names of selected "
    "tools. Skip tools that would not provide useful evidence.\n\n"
    "IMPORTANT: For any flood-related claim, ALWAYS include precipitation "
    "and streamflow tools as baseline corroborating evidence — even when "
    "the claim's specific metric (e.g. flood extent in km², zone counts, "
    "satellite detections) cannot be directly verified. Heavy rainfall or "
    "elevated river discharge at the stated location and date supports "
    "the occurrence of flooding, which indirectly supports the claim."
)


def _describe_tool(fn: Callable) -> str:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        sig = "(...)"
    name = getattr(fn, "__name__", fn.__class__.__name__)
    doc = (fn.__doc__ or "").strip().split("\n")[0] or "(no description)"
    return f"- {name}{sig}\n  {doc}"


class ToolSelector:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        instructions: str | None = None,
        **agent_kwargs,
    ):
        self._agent = Agent(
            model=build_model(model, api_key),
            output_type=ToolSelection,
            capabilities=[
                Thinking(effort=reasoning_effort or settings.reasoning_effort),
            ],
            instructions=instructions or DEFAULT_INSTRUCTIONS,
            **agent_kwargs,
        )

    async def __call__(
        self, claim: Claim, metadata: Metadata, **run_kwargs
    ) -> SelectedTools:
        candidates = registry.get_candidates(metadata.event_type)
        if not candidates:
            return SelectedTools(tools=[], claim=claim)
        descriptions = "\n".join(_describe_tool(c) for c in candidates)
        prompt = (
            f"Claim: {claim.claim}\n\n"
            f"Metadata: {metadata.model_dump_json()}\n\n"
            f"Available tools:\n{descriptions}"
        )
        result = await self._agent.run(prompt, **run_kwargs)
        chosen = set(result.output.chosen)
        selected = [c for c in candidates if c.__name__ in chosen]

        # Fallback: if the selector returned no tools but candidates exist,
        # use all candidates. This prevents silent verification skips when
        # the model fails to populate the chosen list.
        if not selected and candidates:
            selected = candidates

        return SelectedTools(
            tools=selected,
            reasoning=result.output.reasoning,
            claim=claim,
        )

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.usage import UsageLimits

from geoguard.claims import Claim
from geoguard.config import ReasoningEffort, build_model, settings
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


NARRATIVE_RULE = (
    "Disaster-reporting language uses summary phrasing that is not a strict "
    'measurement. Phrases like "up to X", "approximately X", or "around X" '
    "name a representative or peak value, not a hard bound. Phrases like "
    '"X to Y across [region]" give a regional summary that tolerates '
    "individual readings outside the stated range. Words like "
    '"record", "historic", "catastrophic", "major", or "widespread" mean '
    "unusually large relative to context, not strictly the all-time "
    "maximum. When tool evidence matches the claim's direction and "
    "magnitude but lands marginally outside its literal wording — a "
    "reading slightly past a stated peak, a few readings outside an "
    "areal range, or a near-top ranking on a superlative claim — return "
    "SUPPORTS. Reserve CONTRADICTS for evidence that disagrees with the "
    "claim's core assertion, not its phrasing."
)


FLOOD_LOOKBACK_RULE = (
    "IMPORTANT — flood persistence: Flooding often persists days or weeks "
    "after rainfall ends. When verifying a flood claim, ALWAYS use a "
    "14-day lookback window (from 14 days before the claimed date up to "
    "the claimed date) for BOTH precipitation AND streamflow tools. "
    "Pass this full date range as start_date/end_date so you can see "
    "the peak, not just the event date. If heavy rainfall or high river "
    "stages occurred in the days preceding the event — even if levels "
    "dropped by the exact date — that supports the flood claim. "
    "For streamflow, do NOT reduce max_gauges below the default (5) — "
    "the best evidence often comes from gauges 15-25 km away."
)

COORDINATE_RULE = (
    "IMPORTANT — use claim-stated coordinates, not just geocoded ones: "
    "The metadata's location lat/lon comes from geocoding a place name, "
    "which may resolve to a city center far from the actual event. When "
    "the claim text states specific coordinates (e.g. '38.89°N, 121.78°W') "
    "or the metadata contains an analysis_bbox, use THOSE coordinates for "
    "tool calls — especially for streamflow and precipitation, where the "
    "search point determines which gauges are found. For streamflow, "
    "call get_streamflow_history at the claim's stated center point OR "
    "the center of the analysis_bbox, NOT the geocoded location. If both "
    "are available, prefer the claim-stated coordinates."
)

BBOX_RULE = (
    "IMPORTANT — spatial extent matching: When the metadata contains "
    "an analysis_bbox, ALWAYS pass those bounds to the satellite tool "
    "as bbox_lon_min, bbox_lat_min, bbox_lon_max, bbox_lat_max. Also "
    "check the claim text for bounding box coordinates. Comparing "
    "statistics from different spatial extents (e.g. a full 10°×10° "
    "tile vs. a specific valley) produces spurious mismatches in area."
)

RESOLUTION_RULE = (
    "Cross-sensor zone comparison rule: Zone count and per-zone area "
    "are physically determined by sensor resolution (pixel size, noise "
    "floor, classification threshold). A 250 m sensor and a 10 m sensor "
    "will ALWAYS disagree on zone count and individual zone area for the "
    "same flood event — this is expected physics, not evidence of error. "
    "Therefore: when the claim's zone count or largest-zone size comes "
    "from a DIFFERENT sensor/model than your tool evidence, ignore zone "
    "metrics entirely and return INCONCLUSIVE for that claim. Base flood "
    "verification ONLY on total inundated area, which IS comparable "
    "across sensors. Example: if a claim says 'Prithvi-EO detected 57 "
    "zones, largest 1616 km²' and your MODIS tool returns 456 zones, "
    "largest 588 km² — that is INCONCLUSIVE (expected cross-sensor "
    "disagreement), NOT CONTRADICTS."
)


DEFAULT_INSTRUCTIONS = (
    "Verify the given claim using the tools attached. Reason over the claim "
    "and metadata, call the tools you need to gather evidence, and return a "
    "ClaimVerification with one of three verdicts: SUPPORTS (tool evidence "
    "directly confirms the claim), CONTRADICTS (tool evidence directly "
    "disconfirms the claim), or INCONCLUSIVE (insufficient or ambiguous "
    "evidence). If no tools are available, the verdict must be INCONCLUSIVE.\n\n"
    "EVIDENCE GATHERING: Call EVERY tool available to you — do not stop "
    "after one or two tools. Each tool provides a different line of "
    "evidence (satellite imagery, precipitation, river gage height, "
    "streamflow). More evidence makes the verdict more robust.\n\n"
    "RATIONALE: Your rationale MUST cite specific numbers from EVERY "
    "tool you called. For each tool, extract the most significant "
    "finding and state the number:\n"
    "  - Satellite: total flood area in km².\n"
    "  - Precipitation: total mm over the lookback window, peak daily mm.\n"
    "  - Streamflow: name the gauge, cite gage height in ft and/or "
    "discharge in cfs, note event peak rank if available (e.g. "
    "'all-time record', 'rank #3 of 76 years').\n"
    "Example of a GOOD rationale: 'MODIS detected 1830 km² flooded "
    "within the bbox. In the 14 days prior, 198 mm of rain fell "
    "(peak 47 mm/day on Jan 14). Yolo Bypass gage height peaked at "
    "27.5 ft with discharge of 49,700 cfs; Cache Creek set an "
    "all-time record peak (806 cfs, rank #1 of 17 years).'\n"
    "A rationale that says 'satellite data confirms flooding' without "
    "numbers is unacceptable.\n\n"
    "MANDATORY RULES (apply in order before choosing a verdict):\n\n"
    + COORDINATE_RULE
    + "\n\n"
    + BBOX_RULE
    + "\n\n"
    + RESOLUTION_RULE
    + "\n\n"
    + FLOOD_LOOKBACK_RULE
    + "\n\n"
    + NARRATIVE_RULE
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
        api_key: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        instructions: str | None = None,
        tool_calls_limit: int | None = None,
        request_limit: int | None = None,
        output_retries: int | None = None,
        **agent_kwargs,
    ):
        self._model = build_model(model, api_key)
        self._reasoning_effort = reasoning_effort or settings.reasoning_effort
        self._instructions = instructions or DEFAULT_INSTRUCTIONS
        self._tool_calls_limit = (
            tool_calls_limit
            if tool_calls_limit is not None
            else settings.verification_tool_usage_limit
        )
        self._request_limit = (
            request_limit
            if request_limit is not None
            else settings.verification_request_limit
        )
        self._output_retries = (
            output_retries if output_retries is not None else settings.output_retries
        )
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
            output_retries=self._output_retries,
            **self._agent_kwargs,
        )
        tool_names = [t.__name__ for t in tools] if tools else []
        tool_line = (
            f"\n\nYou MUST call each of these tools before rendering a verdict: "
            f"{', '.join(tool_names)}."
            if tool_names
            else ""
        )
        prompt = (
            f"Claim: {claim.claim}\n\n"
            f"Metadata: {metadata.model_dump_json()}"
            f"{tool_line}"
        )
        result = await agent.run(
            prompt,
            usage_limits=UsageLimits(
                request_limit=self._request_limit,
                tool_calls_limit=self._tool_calls_limit,
            ),
            **run_kwargs,
        )
        return VerifierResult(
            verification=result.output,
            tool_calls=_extract_tool_calls(result.all_messages()),
        )

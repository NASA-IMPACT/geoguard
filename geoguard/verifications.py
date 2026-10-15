import json
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
    "ONE DAY AFTER the claimed date) for BOTH precipitation AND river tools (USGS "
    "streamflow, GloFAS discharge, European gauges). "
    "Pass this full date range as start_date/end_date so you can see "
    "the peak, not just the event date. If heavy rainfall or high river "
    "stages occurred in the days preceding the event — even if levels "
    "dropped by the exact date — that supports the flood claim. "
    "For streamflow, do NOT reduce max_gauges below the default (5) — "
    "the best evidence often comes from gauges 15-25 km away."
)

TIMING_RULE = (
    "IMPORTANT — timing tolerance: tool values are daily totals on UTC "
    "days, while events are reported in local time and often begin in the "
    "evening, so an event can fall on the UTC day AFTER the claimed date. "
    "Always end every query window at least ONE DAY AFTER the claimed "
    "date. For storm and wind claims use the window from one day before to "
    "one day after the claimed date."
)

FLOOD_EVIDENCE_RULE = (
    "IMPORTANT — flood-level evidence: rain, or a rise in a river, is not "
    "by itself a flood. SUPPORT a flood claim only when the evidence shows "
    "flood-level conditions: river discharge or stage far above normal (a "
    "peak among the highest in the gauge's multi-year record, or flow many "
    "times its baseline — an ordinary annual high is not enough), rainfall "
    "that is extreme for the location, "
    "or a disaster-catalog record of the event. If the evidence shows only "
    "ordinary wet weather or a modest rise, return INCONCLUSIVE. Flash "
    "floods can come from a single storm cell that daily gridded rainfall "
    "and distant gauges miss, so unremarkable evidence is INCONCLUSIVE, "
    "not CONTRADICTS. Return CONTRADICTS only when the window was clearly "
    "dry (little or no rain anywhere in it) and rivers stayed at baseline."
)

STORM_EVIDENCE_RULE = (
    "IMPORTANT — storm severity: words like 'severe storm', 'major storm' "
    "or 'violent storm' assert damaging winds or extreme rain. SUPPORT such "
    "a claim only when the evidence shows damaging conditions near the place "
    "and date: measured or reanalysis gusts of roughly 90 km/h or more "
    "(storm force), or rainfall that is extreme for the location. Ordinary "
    "windy or wet weather (gusts of 40-70 km/h, moderate rain) does not "
    "support a severe-storm claim — return INCONCLUSIVE. Return CONTRADICTS "
    "only when conditions around the date were clearly calm and dry."
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
    "  - River discharge outside the US (GloFAS, European gauges): peak "
    "m³/s and date, the rise over baseline, and the multi-year rank if "
    "computed (e.g. '825 m³/s on Sep 16, 9× baseline, highest of 21 years').\n"
    "  - Weather stations: the station name and distance, its rainfall "
    "total and/or MEASURED peak gust in km/h.\n"
    "Example of a GOOD rationale: 'MODIS detected 1830 km² flooded "
    "within the bbox. In the 14 days prior, 198 mm of rain fell "
    "(peak 47 mm/day on Jan 14). Yolo Bypass gage height peaked at "
    "27.5 ft with discharge of 49,700 cfs; Cache Creek set an "
    "all-time record peak (806 cfs, rank #1 of 17 years).'\n"
    "A rationale that says 'satellite data confirms flooding' without "
    "numbers is unacceptable.\n\n"
    "MANDATORY RULES (apply in order before choosing a verdict):\n\n"
    + TIMING_RULE
    + "\n\n"
    + COORDINATE_RULE
    + "\n\n"
    + BBOX_RULE
    + "\n\n"
    + RESOLUTION_RULE
    + "\n\n"
    + FLOOD_LOOKBACK_RULE
    + "\n\n"
    + FLOOD_EVIDENCE_RULE
    + "\n\n"
    + STORM_EVIDENCE_RULE
    + "\n\n"
    + NARRATIVE_RULE
)


def _canonical_args(args: str) -> str:
    """Normalize a tool call's JSON args so formatting variance doesn't defeat dedup.

    The same logical call is serialized inconsistently across emissions: the
    first tool call in a turn arrives as the model's raw argument string (with
    spaces, e.g. ``{"lat": 1, "lon": 2}``) while later ones are normalized
    compact (``{"lat":1,"lon":2}``). Raw-string comparison would treat those
    as distinct. Canonicalizing (parse + re-dump with sorted keys) collapses
    them. Falls back to the raw string when args aren't valid JSON.
    """
    try:
        return json.dumps(json.loads(args), sort_keys=True)
    except (ValueError, TypeError):
        return args


def _extract_tool_calls(messages) -> list[ToolCall]:
    """Pair tool calls with their returns, in call order, collapsing duplicates.

    A verifier LLM routinely emits the same ``(tool, args)`` call many times in
    one run. pydantic-ai records each emission separately, and the tool-layer
    dedup cache (see ``ToolRegistry.build_toolset``) already short-circuits the
    redundant network request — returning an identical payload every time.
    Recording every emission anyway balloons this result, and everything built
    from it (the rubricator prompt, the MCP wrapper's response), with redundant
    copies of large evidence payloads — the main driver of context overload
    downstream.

    Deduping by ``(name, canonical_args)`` and keeping the first occurrence is
    lossless: identical calls are guaranteed to carry identical results. Calls
    with genuinely different args (e.g. a narrower date window) have a distinct
    key and are preserved.
    """
    pending: dict[str, tuple[str, str]] = {}
    calls: list[ToolCall] = []
    seen: set[tuple[str, str]] = set()
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
                    dedup_key = (name, _canonical_args(args))
                    if dedup_key in seen:
                        continue  # exact duplicate — identical result, drop it
                    seen.add(dedup_key)
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
        self._reasoning_effort: ReasoningEffort = (
            reasoning_effort or settings.reasoning_effort
        )
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
            retries={"output": self._output_retries},
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
        verification = result.output
        tool_calls = _extract_tool_calls(result.all_messages())
        # A verdict must rest on tool evidence. Without any, SUPPORTS or
        # CONTRADICTS can only come from the model's own knowledge (or the
        # claim's metadata restating itself), so it is downgraded.
        if not tool_calls and verification.verdict is not Verdict.INCONCLUSIVE:
            verification = verification.model_copy(
                update={
                    "verdict": Verdict.INCONCLUSIVE,
                    "rationale": (
                        "No tool evidence was gathered, so the claim cannot be "
                        "grounded. Ungrounded model verdict was "
                        f"{verification.verdict.value}: {verification.rationale}"
                    ),
                }
            )
        return VerifierResult(verification=verification, tool_calls=tool_calls)

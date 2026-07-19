"""VEDA STAC exploration wrapped as an agentic sub-agent tool.

Finding the right evidence in a broad catalog is iterative — pick candidate
datasets, search, inspect what came back, refine, choose the best item, then
analyze it. Doing that inside the main verifier bloats its context with raw
catalog listings. This module encapsulates the loop in a dedicated sub-agent
(`StacExplorer`) and exposes it to the verifier as ONE registered tool
(`find_veda_evidence`): one call in, one compact structured summary out
(NASA-IMPACT/geoguard#7).

From everything upstream this behaves exactly like any other tool. The
sub-agent's internal steps never surface as pipeline events — the nested
`Agent.run` is just an awaited coroutine inside the tool call — and its
raw catalog traffic never enters the verifier's message history; only the
`StacEvidence` dict does.

The sub-agent runs nested inside a verifier that is itself billed per
request, so its budget (`UsageLimits`) is mandatory and exhaustion is
handled gracefully: `UsageLimitExceeded` becomes a "no usable evidence"
result instead of killing the parent verification.
"""

from __future__ import annotations

import functools
from typing import Callable

from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import UsageLimits

from geoguard.config import ReasoningEffort, build_model, settings
from geoguard.schemas import EventType
from geoguard.tools.registry import _deduplicated, registry
from geoguard.tools.stac.veda import build_veda_tools


class StacEvidence(BaseModel):
    """Compact synthesized result of one VEDA catalog exploration.

    This is the ONLY thing the main verifier ever sees from the sub-agent —
    it must stay small regardless of how much catalog iteration happened.
    """

    found: bool
    summary: str
    collection_id: str | None = None
    item_ids: list[str] = []
    statistics: dict[str, float] = {}
    units: str | None = None
    source: str | None = None
    caveats: str | None = None


DEFAULT_INSTRUCTIONS = (
    "You are a NASA VEDA catalog exploration specialist. Given a claim with "
    "an area and time of interest, find the single best piece of VEDA "
    "evidence for or against it and return a compact StacEvidence synthesis."
    "\n\n"
    "WORKFLOW:\n"
    "1. Derive search keywords from the claim: the phenomenon (flood, "
    "precipitation, fire, ...), the measured variable, instruments, and any "
    "named event (e.g. a hurricane name). Call search_veda_collections — "
    "trying two keyword sets is fine, but do not burn budget rephrasing the "
    "same terms.\n"
    "2. Judge candidates by their temporal_interval and bbox against the "
    "claim's time and area — a keyword match with the wrong year or region "
    "is not evidence. Note what each collection measures and its units from "
    "the description.\n"
    "3. Call search_veda_items on the most promising collection. Zero items "
    "is a real catalog gap — move to the next candidate rather than "
    "retrying minor variations.\n"
    "4. Pick the best item (closest in time, covering the area) and call "
    "get_veda_item_statistics over the claim's bounding box. If you only "
    "have a point, use a box of roughly ±0.5 degrees around it.\n"
    "5. Synthesize.\n\n"
    "BUDGET: your tool-call budget is strict. Every tool result carries a "
    "tool_calls_remaining countdown; once it reaches 0, further calls "
    "return nothing useful, so synthesize BEFORE it runs out. Spend the "
    "budget on breadth (different collections) rather than depth "
    "(re-searching the same thing).\n\n"
    "OUTPUT CONTRACT (StacEvidence):\n"
    "- summary: 2-5 sentences stating what you found, the actual numbers "
    "with units, and how it bears on the claim. Never paste catalog "
    "listings, item lists, or raw JSON into it.\n"
    "- statistics: the handful of headline numbers (e.g. mean, max) you "
    "would cite, as flat name->value pairs.\n"
    "- units: the physical units of those numbers, taken from the "
    "collection description (the raster API does not report units). If "
    "genuinely unstated, say so in caveats.\n"
    "- source: cite collection and item ids.\n"
    "- caveats: resolution, temporal offset from the claimed date, low "
    "valid_percent (cloud/no-data), or unit uncertainty.\n"
    "- No usable evidence after honest exploration -> found=False and a "
    "1-3 sentence summary of what you searched and why nothing matched. "
    "That is a first-class answer, not a failure."
)


def _budgeted(fn: Callable, budget: dict) -> Callable:
    """Enforce a soft tool-call budget the model can see and react to.

    pydantic-ai's own `tool_calls_limit` is a hard kill: hitting it raises
    UsageLimitExceeded and the run dies without a synthesis. This wrapper
    spends `budget["remaining"]` per call, stamps the countdown into every
    dict result so the model knows where it stands, and — once exhausted —
    answers with a "synthesize now" notice instead of data. The model's
    only remaining move is to produce its final StacEvidence, typically a
    partial summary of whatever it saw. The hard limit stays on as a
    runaway backstop above this.

    Charged per call on purpose, even for dedup-cache hits: the budget
    bounds loop iterations, not network cost.
    """

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        if budget["remaining"] <= 0:
            return {
                "error": (
                    "Tool budget exhausted — no more catalog calls will "
                    "succeed. Produce your final StacEvidence now from what "
                    "you have already seen."
                )
            }
        budget["remaining"] -= 1
        result = await fn(*args, **kwargs)
        if isinstance(result, dict):
            result = {**result, "tool_calls_remaining": budget["remaining"]}
        return result

    return wrapper


class StacExplorer:
    """Runs a bounded STAC catalog exploration and returns StacEvidence.

    Defaults to the NASA VEDA deployment; point it at any other eoAPI-shaped
    deployment with `stac_url`/`raster_url`, or swap the toolset entirely
    with `tools=` for catalogs with a different API shape (the two are
    mutually exclusive). Mirrors `Verifier`'s construction (build_model /
    Thinking / UsageLimits) so it is configured the same way and testable
    the same way. The pydantic-ai Agent is built per call — construction is
    cheap and per-call toolsets keep the dedup cache and the call budget
    scoped to one exploration.
    """

    def __init__(
        self,
        stac_url: str | None = None,
        raster_url: str | None = None,
        model: str | None = None,
        model_api_key: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        instructions: str | None = None,
        request_limit: int | None = None,
        tool_calls_limit: int | None = None,
        output_retries: int | None = None,
        tools: list | None = None,
        **agent_kwargs,
    ):
        # model_api_key is the LLM provider key (what sibling blocks call
        # api_key) — named to distinguish it from catalog auth, which the
        # public VEDA endpoints don't need.
        if tools is not None and (stac_url or raster_url):
            raise ValueError(
                "pass either tools= (your own toolset) or stac_url/raster_url "
                "(endpoints for the default VEDA toolset), not both — "
                "endpoints would be silently ignored"
            )
        self._model = build_model(model, model_api_key)
        self._reasoning_effort = reasoning_effort or settings.reasoning_effort
        self._request_limit = (
            request_limit
            if request_limit is not None
            else settings.stac_agent_request_limit
        )
        self._tool_calls_limit = (
            tool_calls_limit
            if tool_calls_limit is not None
            else settings.stac_agent_tool_calls_limit
        )
        self._output_retries = (
            output_retries if output_retries is not None else settings.output_retries
        )
        self._tools = (
            tools if tools is not None else build_veda_tools(stac_url, raster_url)
        )
        self._instructions = instructions or (
            DEFAULT_INSTRUCTIONS
            + f"\n\nYour budget: at most {self._tool_calls_limit} tool calls "
            f"across the whole exploration."
        )
        self._agent_kwargs = agent_kwargs

    async def __call__(self, query: str, **run_kwargs) -> StacEvidence:
        # Budget outside dedup: identical repeat calls hit the cache but
        # still spend budget — repetition is the loop the budget bounds.
        budget = {"remaining": self._tool_calls_limit}
        toolset = FunctionToolset(
            tools=[_budgeted(_deduplicated(fn), budget) for fn in self._tools],
            id="stac-explorer",
        )
        agent = Agent(
            model=self._model,
            output_type=StacEvidence,
            toolsets=[toolset],
            capabilities=[Thinking(effort=self._reasoning_effort)],
            instructions=self._instructions,
            output_retries=self._output_retries,
            **self._agent_kwargs,
        )
        try:
            result = await agent.run(
                query,
                usage_limits=UsageLimits(
                    request_limit=self._request_limit,
                    # Hard backstop only — the soft budget above handles
                    # normal exhaustion with a synthesis instead of a kill.
                    tool_calls_limit=self._tool_calls_limit + 4,
                ),
                **run_kwargs,
            )
        except UsageLimitExceeded as e:
            # The exploration budget ran out before the sub-agent produced
            # its synthesis. Degrade to an honest empty-handed result — the
            # verifier treats it like any other no-evidence tool answer.
            return StacEvidence(
                found=False,
                summary=(
                    "VEDA catalog exploration hit its budget before reaching "
                    f"a conclusion ({e}). No synthesized evidence available."
                ),
                caveats="Exploration terminated by usage limit; partial search only.",
            )
        return result.output


@registry(EventType.OTHER)
async def find_veda_evidence(
    claim: str,
    start_date: str,
    end_date: str,
    lat: float | None = None,
    lon: float | None = None,
    bbox_lon_min: float | None = None,
    bbox_lat_min: float | None = None,
    bbox_lon_max: float | None = None,
    bbox_lat_max: float | None = None,
) -> dict:
    """Search NASA VEDA's Earth-science catalog for evidence about a claim.

    Delegates to an autonomous exploration agent that searches VEDA's ~250
    curated collections (precipitation, floods, fires, emissions, hydrology,
    event-specific disaster imagery, ...), inspects candidate datasets for
    the claim's place and time, and computes summary statistics over the
    area of interest from the best matching raster. Returns one compact
    synthesized finding — never raw catalog listings.

    Coverage: global, but dataset-dependent — VEDA curates many one-off
    event datasets alongside long time-series (e.g. precipitation from
    2001). Resolution and units vary per dataset; the result states its
    source, units, and caveats. A clean "no usable evidence" answer means
    the catalog genuinely lacks coverage for that place/time — treat it
    like any other no-data tool result, not as contradiction.

    This is a slow tool (it runs a multi-step exploration internally):
    call it ONCE with the claim's full context, not repeatedly with
    variations.

    Args:
        claim: The claim text being verified, verbatim.
        start_date: Evidence window start, YYYY-MM-DD (use the same
            lookback window as other tools, e.g. 14 days for floods).
        end_date: Evidence window end, YYYY-MM-DD.
        lat: Latitude of the point of interest.
        lon: Longitude of the point of interest.
        bbox_lon_min: Western bound of the analysis area (degrees).
        bbox_lat_min: Southern bound.
        bbox_lon_max: Eastern bound.
        bbox_lat_max: Northern bound. Prefer passing the analysis_bbox
            when the metadata has one; otherwise lat/lon suffices.

    Returns: dict with keys:
        found: True if usable evidence was synthesized.
        summary: Short narrative with the key numbers.
        collection_id / item_ids / statistics / units / source / caveats.
    """
    bbox = (bbox_lon_min, bbox_lat_min, bbox_lon_max, bbox_lat_max)
    if all(v is not None for v in bbox):
        area = f"bounding box (lon/lat): {list(bbox)}"
    elif lat is not None and lon is not None:
        area = f"point: lat={lat}, lon={lon}"
    else:
        return {
            "found": False,
            "summary": (
                "No area of interest provided — pass either lat/lon or a "
                "full bounding box."
            ),
        }
    query = (
        f"Claim: {claim}\n"
        f"Area of interest — {area}\n"
        f"Time window: {start_date} to {end_date}\n\n"
        "Find and analyze the best VEDA evidence for this claim's area and time."
    )
    try:
        evidence = await StacExplorer()(query)
    except Exception as e:  # noqa: BLE001 — a nested-agent failure must not
        # kill the parent verification; degrade like a network error would.
        logger.exception("STAC exploration sub-agent failed")
        return {
            "found": False,
            "summary": f"VEDA catalog exploration failed: {type(e).__name__}: {e}",
        }
    return evidence.model_dump(mode="json")

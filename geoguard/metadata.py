from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Annotated, Literal

from geopy.adapters import AioHTTPAdapter
from geopy.geocoders import Photon
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

from .claims import CLAIM_RULES, DEFAULT_MAX_CLAIMS, Claim
from .config import ReasoningEffort, settings
from .schemas import EventType, Input


class GeoLocation(BaseModel):
    name: str | None = None
    lat: float | None = None
    lon: float | None = None


class TimeRange(BaseModel):
    start: datetime | None = None
    end: datetime | None = None


class Entity(BaseModel):
    name: str
    kind: str | None = None


class GeneralMetadata(BaseModel):
    event_type: Literal[EventType.OTHER] = EventType.OTHER
    location: GeoLocation | None = None
    time_range: TimeRange | None = None
    entities: list[Entity] = []


class FloodMetadata(GeneralMetadata):
    event_type: Literal[EventType.FLOOD] = EventType.FLOOD
    affected_area_km2: float | None = None
    water_depth_m: float | None = None
    river_basin: str | None = None


Metadata = Annotated[
    FloodMetadata | GeneralMetadata,
    Field(discriminator="event_type"),
]


class ClaimGroup(BaseModel):
    metadata: Metadata
    claims: list[Claim]


async def geocode(name: str) -> dict:
    """Resolve a place name to coordinates via OpenStreetMap (Photon endpoint).

    Returns a dict with keys: found (bool), lat (float or None),
    lon (float or None), display_name (str).
    """
    async with Photon(adapter_factory=AioHTTPAdapter) as geocoder:
        loc = await geocoder.geocode(name, timeout=10)
    if loc is None:
        return {"found": False, "lat": None, "lon": None, "display_name": name}
    return {
        "found": True,
        "lat": loc.latitude,
        "lon": loc.longitude,
        "display_name": loc.address,
    }


GEOCODE_RULE = (
    "For any GeoLocation, lat and lon MUST come from the `geocode` tool. "
    "Never produce coordinates from prior knowledge. Call `geocode` once "
    "per distinct place name in the input. When calling `geocode`, pass "
    "the FULLEST place name available from the input — include state, "
    "country, or region whenever the input provides them (e.g., call "
    "`geocode('Galveston, Texas, USA')` not `geocode('Galveston')`) to "
    "avoid resolving to a wrong same-named place. If `geocode` returns "
    "`found: false`, leave lat and lon as None on that GeoLocation."
)


DEFAULT_INSTRUCTIONS = (
    "Extract structured geospatial metadata from the input. "
    "Identify each distinct event described in the input and return one entry "
    "per event. For each, classify the event_type (flood or other) and fill in "
    "the fields relevant to that event type. Use 'other' with only the base "
    "fields when you cannot confidently classify the event. "
    "Leave any field you cannot confidently extract as None.\n\n" + GEOCODE_RULE
)


def _group_cap_rule(max_claims: int | None) -> str:
    if max_claims is None:
        return (
            "Across all event groups, extract as many distinct claims as the "
            "input warrants."
        )
    return (
        f"Across ALL event groups, extract AT MOST {max_claims} claims TOTAL. "
        f"When the input contains more facts than the cap, prioritize central, "
        f"load-bearing claims; skip trivia and asides."
    )


def claim_group_instructions(max_claims: int | None = DEFAULT_MAX_CLAIMS) -> str:
    return (
        "Identify each distinct event described in the input. Information that "
        "shares location, time, and a causal chain describes the SAME event — "
        "group all such claims under ONE ClaimGroup. Do not create separate "
        "groups for different aspects of the same event (cause, impact, response).\n\n"
        "For each event, extract:\n"
        "1. Structured metadata (event_type, location, time_range, entities, "
        "and event-specific fields).\n"
        "2. Atomic, decontextualized claims about it.\n\n"
        + CLAIM_RULES
        + "\n\nLimits:\n"
        + _group_cap_rule(max_claims)
        + "\n\n"
        + GEOCODE_RULE
        + "\n\nSkip opinions, hedges, and meta-commentary. "
        "Leave any metadata field you cannot confidently extract as None."
    )


CLAIM_GROUP_INSTRUCTIONS = claim_group_instructions()


class MetadataExtractor:
    def __init__(
        self,
        model: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        instructions: str | None = None,
        output_type=list[Metadata],
        max_claims: int | None = DEFAULT_MAX_CLAIMS,
        geocoder: Callable[[str], Awaitable[dict]] = geocode,
    ):
        self._agent = Agent(
            model=model or settings.model,
            output_type=output_type,
            tools=[geocoder],
            capabilities=[
                Thinking(effort=reasoning_effort or settings.reasoning_effort),
            ],
            instructions=instructions
            or (
                claim_group_instructions(max_claims)
                if output_type == list[ClaimGroup]
                else DEFAULT_INSTRUCTIONS
            ),
        )

    async def __call__(self, inp: Input) -> list[Metadata]:
        result = await self._agent.run(inp.text)
        return result.output

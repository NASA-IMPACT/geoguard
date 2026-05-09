from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

from .claims import Claim
from .config import ReasoningEffort, settings
from .schemas import Input


class EventType(StrEnum):
    FLOOD = "flood"
    OTHER = "other"


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


DEFAULT_INSTRUCTIONS = (
    "Extract structured geospatial metadata from the input. "
    "Identify each distinct event described in the input and return one entry "
    "per event. For each, classify the event_type (flood or other) and fill in "
    "the fields relevant to that event type. Use 'other' with only the base "
    "fields when you cannot confidently classify the event. "
    "Leave any field you cannot confidently extract as None."
)


class MetadataExtractor:
    def __init__(
        self,
        model: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        instructions: str | None = None,
        output_type=list[Metadata],
    ):
        self._agent = Agent(
            model=model or settings.model,
            output_type=output_type,
            capabilities=[
                Thinking(effort=reasoning_effort or settings.reasoning_effort),
            ],
            instructions=instructions or DEFAULT_INSTRUCTIONS,
        )

    async def __call__(self, inp: Input) -> list[Metadata]:
        result = await self._agent.run(inp.text)
        return result.output

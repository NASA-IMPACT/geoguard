from enum import StrEnum

from pydantic import BaseModel


class EventType(StrEnum):
    FLOOD = "flood"
    STORM = "storm"
    OTHER = "other"


class Region(StrEnum):
    """Where a tool's data source has coverage."""

    GLOBAL = "global"
    US = "us"
    EUROPE = "europe"


# Approximate coverage boxes as (lat_min, lat_max, lon_min, lon_max).
REGION_BOXES: dict[Region, list[tuple[float, float, float, float]]] = {
    Region.US: [
        (24.0, 49.6, -125.0, -66.5),  # contiguous US
        (51.0, 71.6, -180.0, -129.0),  # Alaska
        (18.5, 22.5, -160.5, -154.5),  # Hawaii
        (17.8, 18.6, -67.4, -65.2),  # Puerto Rico
    ],
    Region.EUROPE: [(34.0, 72.0, -25.0, 45.0)],
}


def region_contains(region: Region, lat: float, lon: float) -> bool:
    if region is Region.GLOBAL:
        return True
    return any(
        lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
        for lat_min, lat_max, lon_min, lon_max in REGION_BOXES[region]
    )


class BoundingBox(BaseModel):
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float


class ImageRef(BaseModel):
    path: str


class TiffRef(ImageRef):
    bbox: BoundingBox
    date: str
    region_name: str = "the study area"
    model_name: str = "a foundation model"
    input_source: str = "satellite imagery"


class Input(BaseModel):
    text: str = ""
    images: list[ImageRef | TiffRef] = []

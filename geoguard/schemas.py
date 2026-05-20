from enum import StrEnum

from pydantic import BaseModel


class EventType(StrEnum):
    FLOOD = "flood"
    STORM = "storm"
    OTHER = "other"


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

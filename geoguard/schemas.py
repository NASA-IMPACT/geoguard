from enum import StrEnum

from pydantic import BaseModel


class EventType(StrEnum):
    FLOOD = "flood"
    OTHER = "other"


class ImageRef(BaseModel):
    path: str


class Input(BaseModel):
    text: str
    images: list[ImageRef] = []

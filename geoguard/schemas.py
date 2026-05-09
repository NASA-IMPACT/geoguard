from pydantic import BaseModel


class ImageRef(BaseModel):
    path: str


class Input(BaseModel):
    text: str
    images: list[ImageRef] = []

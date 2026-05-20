from geoguard.adapters import tiff_to_claims
from geoguard.schemas import Input, TiffRef


class InputProcessor:
    """Adapt non-text inputs (TIFFs, etc.) into Input.text before extraction.

    Returns the input unchanged when no images are attached. Otherwise
    runs the type-appropriate adapter for each image and appends its
    text to inp.text. The metadata extractor sees a plain Input(text=...)
    regardless of how the original input was constructed.
    """

    async def __call__(self, inp: Input) -> Input:
        if not inp.images:
            return inp
        parts = [inp.text] if inp.text else []
        for img in inp.images:
            if isinstance(img, TiffRef):
                parts.append(self._tiff_to_text(img))
        return inp.model_copy(update={"text": "\n\n".join(parts)})

    def _tiff_to_text(self, ref: TiffRef) -> str:
        bbox = [
            ref.bbox.lon_min,
            ref.bbox.lat_min,
            ref.bbox.lon_max,
            ref.bbox.lat_max,
        ]
        return tiff_to_claims(
            tiff_path=ref.path,
            bbox=bbox,
            date=ref.date,
            region_name=ref.region_name,
            model_name=ref.model_name,
            input_source=ref.input_source,
        )

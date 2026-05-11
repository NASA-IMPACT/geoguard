import httpx

from geoguard.config import settings
from geoguard.schemas import EventType
from geoguard.tools.registry import registry


@registry(EventType.FLOOD)
async def get_elevation(lat: float, lon: float) -> dict:
    """Elevation in meters above sea level at the given lat/lon.

    Source: Open-Meteo Elevation API (SRTM-90m).
    Returns a dict with keys: elevation_m (float or None), source (str).
    """
    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as c:
        r = await c.get(
            "https://api.open-meteo.com/v1/elevation",
            params={"latitude": lat, "longitude": lon},
        )
        r.raise_for_status()
        data = r.json()
    elevations = data.get("elevation") or [None]
    return {
        "elevation_m": elevations[0],
        "source": "Open-Meteo Elevation API",
    }

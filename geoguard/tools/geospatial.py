import httpx
from geopy.distance import geodesic

from geoguard.config import settings
from geoguard.schemas import EventType
from geoguard.tools.registry import registry
from geoguard.utils import graceful_http


@registry(EventType.FLOOD, EventType.STORM)
async def get_elevation(lat: float, lon: float) -> dict:
    """Look up elevation in meters above sea level at the given coordinates.

    Source: Open-Meteo Elevation API (global SRTM-90m).

    Returns: dict with keys:
        elevation_m: Elevation in meters above sea level (float or None).
        source: Identifier of the data source.
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


@registry(EventType.FLOOD, EventType.STORM)
@graceful_http
async def find_nearest_water_body(
    lat: float,
    lon: float,
    search_radius_km: float = 10.0,
) -> dict:
    """Find the nearest river, stream, canal, lake, or coastline near a point.

    Queries OpenStreetMap via the Overpass API.

    Args:
        search_radius_km: Radius around the point to search, in kilometers.

    Returns: dict with keys:
        found: True if a water body was found within the radius, False otherwise.
        distance_m: Distance to the nearest water body in meters, or None if not found.
        name: Name of the water body (e.g. "Buffalo Bayou"), or None.
        kind: OSM tag describing the feature
            (e.g. "river", "stream", "water", "coastline"), or None.
        source: Identifier of the data source.
    """
    radius_m = int(search_radius_km * 1000)
    query = (
        "[out:json][timeout:25];\n"
        "(\n"
        f'  way["waterway"~"river|stream|canal"](around:{radius_m},{lat},{lon});\n'
        f'  way["natural"="water"](around:{radius_m},{lat},{lon});\n'
        f'  way["natural"="coastline"](around:{radius_m},{lat},{lon});\n'
        ");\n"
        "out center tags;"
    )
    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as c:
        r = await c.get(
            "https://overpass.kumi.systems/api/interpreter",
            params={"data": query},
            headers={"User-Agent": "geoguard/0.1.0"},
        )
        r.raise_for_status()
        data = r.json()

    nearest = {
        "found": False,
        "distance_m": None,
        "name": None,
        "kind": None,
        "source": "OSM Overpass API",
    }
    for el in data.get("elements", []):
        center = el.get("center") or {}
        if "lat" not in center or "lon" not in center:
            continue
        dist_m = geodesic((lat, lon), (center["lat"], center["lon"])).meters
        if not nearest["found"] or dist_m < nearest["distance_m"]:
            tags = el.get("tags", {})
            nearest = {
                "found": True,
                "distance_m": dist_m,
                "name": tags.get("name"),
                "kind": tags.get("waterway") or tags.get("natural"),
                "source": "OSM Overpass API",
            }
    return nearest

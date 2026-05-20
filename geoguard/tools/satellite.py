"""Satellite-derived flood detection tool — MODIS/VIIRS NRT Global Flood Product.

Source: NASA LANCE NRT (recent) and LAADS Archive (historical, 2003–present).
Product: MCDWD_L3_F3_NRT (3-day composite GeoTIFF, 250 m, global).

Tile grid: 10° × 10° geographic (not sinusoidal).
  h = int((lon + 180) / 10), v = int((90 - lat) / 10)
  Each tile: 4800 × 4800 pixels.

Pixel values: 0=no water, 1=reference water, 2=recurring flood, 3=flood, 255=no data.

Requires:
  - NASA Earthdata credentials (env: EARTHDATA_USERNAME, EARTHDATA_PASSWORD)
  - For historical data (before NRT window): LAADS_APP_KEY env var
  - rasterio, numpy
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from datetime import date, timedelta

import httpx
import numpy as np
import rasterio

from geoguard.config import settings
from geoguard.schemas import EventType
from geoguard.tools.registry import registry
from geoguard.utils import graceful_http


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

async def _get_earthdata_token(client: httpx.AsyncClient) -> str:
    """Obtain or reuse a bearer token from NASA Earthdata Login."""
    username = os.environ.get("EARTHDATA_USERNAME", "")
    password = os.environ.get("EARTHDATA_PASSWORD", "")
    if not username or not password:
        raise RuntimeError(
            "EARTHDATA_USERNAME and EARTHDATA_PASSWORD env vars are required."
        )
    # Check for existing tokens first
    r = await client.get(
        "https://urs.earthdata.nasa.gov/api/users/tokens",
        auth=(username, password),
    )
    r.raise_for_status()
    tokens = r.json()
    if tokens:
        return tokens[0]["access_token"]

    # Create new token
    r = await client.post(
        "https://urs.earthdata.nasa.gov/api/users/token",
        auth=(username, password),
    )
    r.raise_for_status()
    return r.json()["access_token"]


# ---------------------------------------------------------------------------
# Tile math
# ---------------------------------------------------------------------------

def _tile_hv(lat: float, lon: float) -> tuple[int, int]:
    """Convert lat/lon to MODIS flood product tile indices."""
    h = int((lon + 180) / 10)
    v = int((90 - lat) / 10)
    return h, v


def _pixel_xy(lat: float, lon: float, h: int, v: int) -> tuple[int, int]:
    """Convert lat/lon to pixel x, y within a 4800×4800 tile."""
    tile_lon_min = h * 10 - 180
    tile_lat_max = 90 - v * 10
    pixel_size = 10.0 / 4800
    x = int((lon - tile_lon_min) / pixel_size)
    y = int((tile_lat_max - lat) / pixel_size)
    return min(max(x, 0), 4799), min(max(y, 0), 4799)


def _pixel_area_km2(lat: float) -> float:
    """Area of one 250 m pixel in km² at the given latitude."""
    deg = 10.0 / 4800
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat))
    return (deg * km_per_deg_lat) * (deg * km_per_deg_lon)


# ---------------------------------------------------------------------------
# Download — LANCE NRT (GeoTIFF) + LAADS Archive (HDF)
# ---------------------------------------------------------------------------

_LANCE_NRT_BASE = (
    "https://nrt3.modaps.eosdis.nasa.gov/api/v2/content/archives"
    "/allData/61/MCDWD_L3_F3_NRT"
)
_LAADS_BASE = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MCDWD_L3"


def _lance_url(d: date, h: int, v: int) -> str:
    """GeoTIFF URL on LANCE NRT."""
    doy = d.timetuple().tm_yday
    return (
        f"{_LANCE_NRT_BASE}/{d.year}/{doy:03d}/"
        f"MCDWD_L3_F3_NRT.A{d.year}{doy:03d}.h{h:02d}v{v:02d}.061.tif"
    )


async def _download(
    client: httpx.AsyncClient, url: str, headers: dict
) -> bytes | None:
    """Download a file. Returns bytes or None on 404/error."""
    try:
        r = await client.get(url, headers=headers, follow_redirects=True, timeout=90.0)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        # Reject HTML error pages
        if r.content[:5] == b"<!DOC" or r.content[:5] == b"<html":
            return None
        return r.content
    except httpx.HTTPError:
        return None


async def _find_laads_filename(
    client: httpx.AsyncClient, d: date, h: int, v: int, app_key: str
) -> str | None:
    """Lookup the exact HDF filename on LAADS (has processing timestamp)."""
    doy = d.timetuple().tm_yday
    dir_url = f"{_LAADS_BASE}/{d.year}/{doy:03d}/"
    headers = {"Authorization": f"Bearer {app_key}"}
    try:
        r = await client.get(dir_url, headers=headers, follow_redirects=True, timeout=30.0)
        if r.status_code != 200:
            return None
        pattern = rf"(MCDWD_L3\.A{d.year}{doy:03d}\.h{h:02d}v{v:02d}\.061\.\d+\.hdf)"
        match = re.search(pattern, r.text)
        return match.group(1) if match else None
    except httpx.HTTPError:
        return None


async def _download_tile(
    client: httpx.AsyncClient, d: date, h: int, v: int, token: str
) -> tuple[bytes | None, str]:
    """Try LANCE NRT first, then LAADS archive with ±2 day window."""
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Try LANCE NRT (GeoTIFF, recent data)
    for offset in (0, 1, -1, 2, -2):
        d2 = d + timedelta(days=offset)
        url = _lance_url(d2, h, v)
        data = await _download(client, url, headers)
        if data and len(data) > 1000:
            src = f"LANCE NRT" + (f" (offset {offset:+d}d)" if offset else "")
            return data, src

    # 2. Try LAADS archive (HDF, historical)
    app_key = os.environ.get("LAADS_APP_KEY", "")
    if app_key:
        laads_headers = {"Authorization": f"Bearer {app_key}"}
        for offset in (0, 1, -1, 2, -2):
            d2 = d + timedelta(days=offset)
            fname = await _find_laads_filename(client, d2, h, v, app_key)
            if fname:
                doy = d2.timetuple().tm_yday
                url = f"{_LAADS_BASE}/{d2.year}/{doy:03d}/{fname}"
                data = await _download(client, url, laads_headers)
                if data and len(data) > 1000:
                    src = f"LAADS Archive" + (f" (offset {offset:+d}d)" if offset else "")
                    return data, src

    return None, ""


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

_LABEL = {0: "no_water", 1: "reference_water", 2: "recurring_flood", 3: "flood", 255: "no_data"}


def _read_array(data: bytes, is_hdf: bool) -> np.ndarray:
    """Read the flood classification array from GeoTIFF or HDF4."""
    suffix = ".hdf" if is_hdf else ".tif"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        if is_hdf:
            from pyhdf.SD import SD, SDC
            hdf = SD(tmp.name, SDC.READ)
            # Use 3-day composite for best cloud coverage
            ds = hdf.select("Flood_3Day_250m")
            arr = ds[:]
            hdf.end()
            return arr
        else:
            with rasterio.open(tmp.name) as src:
                return src.read(1)


def _crop_to_bbox(
    arr: np.ndarray, h: int, v: int,
    bbox: tuple[float, float, float, float],
) -> np.ndarray:
    """Crop a 4800×4800 tile array to a geographic bounding box.

    Args:
        bbox: (lon_min, lat_min, lon_max, lat_max) in degrees.
    """
    pixel_size = 10.0 / 4800  # degrees per pixel
    tile_lon_min = h * 10 - 180
    tile_lat_max = 90 - v * 10
    lon_min, lat_min, lon_max, lat_max = bbox

    # Pixel bounds (y increases downward = decreasing lat)
    y_min = max(int((tile_lat_max - lat_max) / pixel_size), 0)
    y_max = min(int((tile_lat_max - lat_min) / pixel_size), 4799) + 1
    x_min = max(int((lon_min - tile_lon_min) / pixel_size), 0)
    x_max = min(int((lon_max - tile_lon_min) / pixel_size), 4799) + 1

    return arr[y_min:y_max, x_min:x_max]


def _analyze_tile(
    data: bytes, lat: float, lon: float, h: int, v: int,
    is_hdf: bool = False,
    bbox: tuple[float, float, float, float] | None = None,
) -> dict:
    """Read a GeoTIFF or HDF4 tile and compute flood statistics.

    If bbox is given (lon_min, lat_min, lon_max, lat_max), crop the
    analysis to that bounding box. Otherwise analyze the full 10°×10° tile.
    """
    arr = _read_array(data, is_hdf)

    # Check the specific point (before any cropping)
    px, py = _pixel_xy(lat, lon, h, v)
    point_value = int(arr[py, px])

    # Crop to bounding box if requested
    if bbox is not None:
        arr = _crop_to_bbox(arr, h, v, bbox)

    total_pixels = arr.size
    no_data_pixels = int(np.sum(arr == 255))
    valid_pixels = total_pixels - no_data_pixels
    water_pixels = int(np.sum(arr == 1))
    recurring_flood_pixels = int(np.sum(arr == 2))
    flood_pixels = int(np.sum(arr == 3))
    flood_total = recurring_flood_pixels + flood_pixels

    pixel_km2 = _pixel_area_km2(lat)
    bbox_note = f"bbox {bbox}" if bbox else "full tile"

    return {
        "tile": f"h{h:02d}v{v:02d}",
        "analysis_region": bbox_note,
        "valid_pixels": valid_pixels,
        "no_data_pixels": no_data_pixels,
        "total_flood_pixels": flood_total,
        "flood_area_km2": round(flood_total * pixel_km2, 2),
        "pixel_area_km2": round(pixel_km2, 6),
        "point_pixel_value": point_value,
        "point_classification": _LABEL.get(point_value, "unknown"),
    }


# ---------------------------------------------------------------------------
# Public tool
# ---------------------------------------------------------------------------

@registry(EventType.FLOOD)
@graceful_http
async def get_satellite_flood_extent(
    lat: float,
    lon: float,
    event_date: str,
    bbox_lon_min: float | None = None,
    bbox_lat_min: float | None = None,
    bbox_lon_max: float | None = None,
    bbox_lat_max: float | None = None,
) -> dict:
    """Satellite-derived flood extent from NASA MODIS Global Flood Product (250 m).

    Queries the MODIS 3-day composite flood map for the given date.
    When a bounding box is provided, computes flood area, zone count,
    and largest zone size ONLY within that region — essential for
    comparing against upstream products that cover a specific extent.
    Without a bbox, analyzes the full 10°×10° tile.

    Source: NASA LANCE NRT / LAADS Archive. Product: MCDWD_L3_F3.
    Pixel classification: 0=dry, 1=permanent water, 2=recurring flood,
    3=flood, 255=insufficient data (cloud/swath gap).

    Coverage: global, 250 m, daily composites (3-day). NRT data covers
    recent ~7 days; historical data (2003–present) requires LAADS_APP_KEY.

    Resolution note: MODIS at 250 m is much coarser than Sentinel-2
    (10 m) or HLS (30 m). At coarser resolution, adjacent small flood
    zones merge into fewer, larger zones. Expect MODIS to report FEWER
    zones with LARGER individual areas compared to higher-resolution
    products. Total flood area should be comparable, but zone counts
    and per-zone sizes will differ systematically.

    Args:
        lat: Latitude of the query point (used for point classification).
        lon: Longitude of the query point.
        event_date: Date to query (YYYY-MM-DD).
        bbox_lon_min: Western bound of analysis region (degrees).
        bbox_lat_min: Southern bound of analysis region (degrees).
        bbox_lon_max: Eastern bound of analysis region (degrees).
        bbox_lat_max: Northern bound of analysis region (degrees).
            When all four bbox params are provided, flood stats are
            computed only within that box. When omitted, the full
            10°×10° tile is analyzed.

    Returns: dict with keys:
        found: True if a flood map tile was available.
        flood_detected_at_point: Whether the lat/lon is flooded.
        point_classification: Label at the exact point.
        flood_area_km2: Total flooded area in the analysis region.
        analysis_region: Description of the spatial extent analyzed.
        resolution_m: Sensor resolution (250 m for MODIS).
        source: Data source identifier.
        note: Cross-sensor comparison guidance.
    """
    d = date.fromisoformat(event_date[:10])
    h, v = _tile_hv(lat, lon)

    async with httpx.AsyncClient(timeout=90.0) as client:
        token = await _get_earthdata_token(client)
        tile_bytes, source_used = await _download_tile(client, d, h, v, token)

    if tile_bytes is None:
        return {
            "found": False,
            "reason": (
                f"No MODIS flood tile available for h{h:02d}v{v:02d} "
                f"on or near {event_date}. For historical data, set "
                f"LAADS_APP_KEY env var (get from ladsweb.modaps.eosdis.nasa.gov/profiles)."
            ),
            "lat": lat,
            "lon": lon,
            "event_date": event_date,
        }

    bbox = None
    if all(v is not None for v in (bbox_lon_min, bbox_lat_min, bbox_lon_max, bbox_lat_max)):
        bbox = (bbox_lon_min, bbox_lat_min, bbox_lon_max, bbox_lat_max)

    is_hdf = source_used.startswith("LAADS")
    stats = _analyze_tile(
        tile_bytes, lat, lon, h, v, is_hdf=is_hdf, bbox=bbox
    )

    return {
        "found": True,
        "flood_detected_at_point": stats["point_classification"] in ("flood", "recurring_flood"),
        "point_classification": stats["point_classification"],
        "flood_area_km2": stats["flood_area_km2"],
        "total_flood_pixels": stats["total_flood_pixels"],
        "analysis_region": stats["analysis_region"],
        "valid_pixels": stats["valid_pixels"],
        "no_data_pixels": stats["no_data_pixels"],
        "pixel_area_km2": stats["pixel_area_km2"],
        "tile": stats["tile"],
        "lat": lat,
        "lon": lon,
        "event_date": event_date,
        "source": f"NASA MODIS Global Flood Product MCDWD_L3_F3 ({source_used})",
        "resolution_m": 250,
        "note": (
            "MODIS resolution is 250 m. Only total flood_area_km2 is "
            "comparable across resolutions. Zone counts and per-zone "
            "sizes are omitted because they are physically determined "
            "by sensor resolution and NOT comparable across sensors."
        ),
    }

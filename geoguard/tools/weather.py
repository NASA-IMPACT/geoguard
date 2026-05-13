import asyncio

import httpx

from geoguard.config import settings
from geoguard.schemas import EventType
from geoguard.tools.registry import registry
from geoguard.utils import date_range, graceful_http


@registry(EventType.FLOOD, EventType.STORM)
async def get_historical_precipitation(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str | None = None,
) -> dict:
    """Daily precipitation in millimeters over a date range at the given coordinates.

    Source: Open-Meteo Historical Archive (ERA5 reanalysis). Daily
    granularity; the range is inclusive of both endpoints.

    ERA5 is a global gridded reanalysis (~9 km effective resolution),
    not gauge data. It smooths and underestimates localized intense
    rainfall, especially convective events — point gauge observations
    at the same coordinates are routinely much higher than ERA5. Treat
    ERA5 values as a lower bound when verifying point-rainfall claims.

    Args:
        start_date: Start of the inclusive range (YYYY-MM-DD).
        end_date: End of the inclusive range (YYYY-MM-DD). If None, queries
            only `start_date` (single-day result).

    Returns: dict with keys:
        daily_mm: List of daily total precipitation values in millimeters,
            one per day in the range (None for days with missing data).
        total_mm: Sum of all daily values in the range (None if all missing).
        peak_daily_mm: Maximum single-day precipitation across the range.
        dates: ISO date strings (YYYY-MM-DD) matching daily_mm.
        lat: Latitude that was queried.
        lon: Longitude that was queried.
        start_date: Start date of the queried range.
        end_date: End date of the queried range.
        source: Identifier of the data source.
    """
    end = end_date or start_date
    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as c:
        r = await c.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end,
                "daily": "precipitation_sum",
                "timezone": "UTC",
            },
        )
        r.raise_for_status()
        data = r.json()
    daily = data.get("daily", {})
    daily_mm = daily.get("precipitation_sum") or []
    dates = daily.get("time") or []
    valid = [v for v in daily_mm if v is not None]
    return {
        "daily_mm": daily_mm,
        "total_mm": sum(valid) if valid else None,
        "peak_daily_mm": max(valid) if valid else None,
        "dates": dates,
        "lat": lat,
        "lon": lon,
        "start_date": start_date,
        "end_date": end,
        "source": "Open-Meteo Historical Archive",
    }


@registry(EventType.FLOOD, EventType.STORM)
async def get_historical_winds(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str | None = None,
) -> dict:
    """Daily peak wind speed and gust (km/h) over a date range at the given coordinates.

    Source: Open-Meteo Historical Archive (global, no API key required).
    Data granularity is daily; the range is inclusive of both endpoints.

    Args:
        start_date: Start of the inclusive range (YYYY-MM-DD).
        end_date: End of the inclusive range (YYYY-MM-DD). If None, queries
            only `start_date` (single-day result).

    Returns: dict with keys:
        daily_max_speed_kmh: List of daily peak sustained wind speeds (km/h).
        daily_max_gust_kmh: List of daily peak wind gusts (km/h).
        peak_speed_kmh: Highest sustained wind speed across the range.
        peak_gust_kmh: Highest wind gust across the range.
        dates: ISO date strings (YYYY-MM-DD) matching the daily lists.
        lat: Latitude that was queried.
        lon: Longitude that was queried.
        start_date: Start date of the queried range.
        end_date: End date of the queried range.
        source: Identifier of the data source.
    """
    end = end_date or start_date
    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as c:
        r = await c.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end,
                "daily": "wind_speed_10m_max,wind_gusts_10m_max",
                "timezone": "UTC",
            },
        )
        r.raise_for_status()
        data = r.json()
    daily = data.get("daily", {})
    speeds = daily.get("wind_speed_10m_max") or []
    gusts = daily.get("wind_gusts_10m_max") or []
    dates = daily.get("time") or []
    valid_speeds = [v for v in speeds if v is not None]
    valid_gusts = [v for v in gusts if v is not None]
    return {
        "daily_max_speed_kmh": speeds,
        "daily_max_gust_kmh": gusts,
        "peak_speed_kmh": max(valid_speeds) if valid_speeds else None,
        "peak_gust_kmh": max(valid_gusts) if valid_gusts else None,
        "dates": dates,
        "lat": lat,
        "lon": lon,
        "start_date": start_date,
        "end_date": end,
        "source": "Open-Meteo Historical Archive",
    }


@registry(EventType.FLOOD, EventType.STORM)
@graceful_http
async def get_radar_gauge_precipitation(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str | None = None,
) -> dict:
    """Daily precipitation from NOAA radar+gauge gridded products (US only).

    Source: Iowa Environmental Mesonet IEMRE — point-extracted from NOAA's
    operational precipitation products. Returns two complementary estimates:

      - prism_*: PRISM Climate Group daily analysis (~4 km, gauge-anchored).
        Best for verifying multi-day rainfall totals against gauge-based
        narratives (NWS Storm Events, COOP/CoCoRaHS reports).
      - mrms_*: NOAA MRMS QPE (~1 km radar with real-time gauge correction).
        Best for storm timing and shorter-window attribution. Tends to
        underestimate gauge totals by ~20-40% for warm-season convection.

    These are observation-anchored products — fundamentally different from
    `get_historical_precipitation` (ERA5 reanalysis), which smooths intense
    rainfall. For US point-rainfall claims, prefer this tool. Outside the
    US, returns found=False.

    Args:
        start_date: Start of the inclusive range (YYYY-MM-DD).
        end_date: End of the inclusive range (YYYY-MM-DD). If None, queries
            only `start_date` (single-day result).

    Returns: dict with keys:
        found: True if data was returned for at least one day.
        reason: Short explanation when found=False (else absent).
        dates: ISO date strings (YYYY-MM-DD), one per day in the range.
        prism_daily_mm: Daily PRISM totals in millimeters (None where missing).
        prism_total_mm: Sum of PRISM daily values (None if all missing).
        mrms_daily_mm: Daily MRMS totals in millimeters (None where missing).
        mrms_total_mm: Sum of MRMS daily values (None if all missing).
        lat, lon, start_date, end_date: Echo of the query.
        source: Identifier of the data source.
    """
    end = end_date or start_date
    dates = date_range(start_date, end)

    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as c:
        responses = await asyncio.gather(
            *(
                c.get(
                    f"https://mesonet.agron.iastate.edu/iemre/daily/"
                    f"{d.isoformat()}/{lat}/{lon}/json"
                )
                for d in dates
            ),
            return_exceptions=True,
        )

    prism_in: list[float | None] = []
    mrms_in: list[float | None] = []
    any_data = False
    for resp in responses:
        prism_val: float | None = None
        mrms_val: float | None = None
        if not isinstance(resp, Exception):
            try:
                row = resp.json().get("data", [{}])[0]
                prism_val = row.get("prism_precip_in")
                mrms_val = row.get("mrms_precip_in")
            except (ValueError, IndexError):
                pass
        prism_in.append(prism_val)
        mrms_in.append(mrms_val)
        if prism_val is not None or mrms_val is not None:
            any_data = True

    if not any_data:
        return {
            "found": False,
            "reason": (
                "IEMRE returned no precipitation data (likely outside US "
                "coverage or unsupported date)."
            ),
            "lat": lat,
            "lon": lon,
            "start_date": start_date,
            "end_date": end,
        }

    in_to_mm = 25.4
    prism_mm = [round(v * in_to_mm, 2) if v is not None else None for v in prism_in]
    mrms_mm = [round(v * in_to_mm, 2) if v is not None else None for v in mrms_in]
    prism_valid = [v for v in prism_mm if v is not None]
    mrms_valid = [v for v in mrms_mm if v is not None]

    return {
        "found": True,
        "dates": [d.isoformat() for d in dates],
        "prism_daily_mm": prism_mm,
        "prism_total_mm": round(sum(prism_valid), 2) if prism_valid else None,
        "mrms_daily_mm": mrms_mm,
        "mrms_total_mm": round(sum(mrms_valid), 2) if mrms_valid else None,
        "lat": lat,
        "lon": lon,
        "start_date": start_date,
        "end_date": end,
        "source": "Iowa Environmental Mesonet IEMRE (NOAA PRISM + MRMS)",
    }

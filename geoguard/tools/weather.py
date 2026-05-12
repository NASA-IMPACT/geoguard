import httpx

from geoguard.config import settings
from geoguard.schemas import EventType
from geoguard.tools.registry import registry


@registry(EventType.FLOOD)
async def get_historical_precipitation(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str | None = None,
) -> dict:
    """Daily precipitation in millimeters over a date range at the given coordinates.

    Source: Open-Meteo Historical Archive (global, no API key required).
    Data granularity is daily; the range is inclusive of both endpoints.

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


@registry(EventType.FLOOD)
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

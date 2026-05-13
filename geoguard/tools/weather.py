import asyncio
import math

import httpx
from geopy.distance import geodesic

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


@registry(EventType.FLOOD, EventType.STORM)
@graceful_http
async def get_streamflow_history(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str | None = None,
    search_radius_km: float = 30.0,
    max_gauges: int = 5,
) -> dict:
    """Daily streamflow at nearby USGS river gauges (US only).

    Source: USGS Water Services NWIS — daily mean discharge plus the
    period-of-record annual peak history. Use for verifying river-stage,
    river-flooding, and 'record flooding' claims, and for distinguishing
    overland flooding (follows rainfall within hours) from river flooding
    (can persist days after rain stops). Outside the US returns found=False.

    For each gauge within `search_radius_km` of the point (capped at
    `max_gauges`, nearest first), returns the daily mean discharge series
    plus the all-time annual peak record and the rank of any annual peak
    that occurred during the queried window. event_rank=1 means the event
    set a new all-time record; event_rank=2 means second-highest; None
    means no annual peak was recorded in this window.

    Note on units: USGS reports streamflow in cubic feet per second (cfs),
    which is the convention in US flood narratives. m³/s is provided in
    parallel for metric audiences (1 cfs = 0.02831685 m³/s).

    Note on peaks vs daily means: USGS annual peaks are *instantaneous*
    maxima, typically higher than the daily mean of the same day. Compare
    daily series across days; compare annual peak record to the event's
    annual peak entry (not to the daily mean).

    Args:
        start_date: Start of inclusive range (YYYY-MM-DD).
        end_date: End of inclusive range (YYYY-MM-DD). None → single day.
        search_radius_km: Radius around (lat, lon) to search for gauges.
        max_gauges: Cap on returned gauges to limit response size.

    Returns: dict with keys:
        found: True if at least one gauge was located, False otherwise.
        reason: Short explanation when found=False (else absent).
        dates: ISO date strings, one per day.
        gauges: list of per-gauge dicts (nearest first):
            site_no: USGS site number.
            name: Gauge station name.
            distance_km: Distance from query point.
            daily_cfs: Daily mean discharge in cfs (None on missing).
            daily_m3s: Same series converted to m³/s.
            event_peak_cfs: Instantaneous annual peak that occurred in the
                window (None if no annual peak was recorded that period).
            event_peak_date: ISO date of event_peak_cfs.
            all_time_record_cfs: Highest annual peak in the gauge's record.
            all_time_record_date: ISO date of all_time_record_cfs.
            peaks_years_on_record: Count of annual peaks on file.
            event_rank: Rank of event_peak_cfs among all annual peaks
                (1 = new record, 2 = second-highest, etc.; None if no
                event peak).
        lat, lon, start_date, end_date: Query echo.
        source: Identifier of the data source.
    """
    end = end_date or start_date
    dates = date_range(start_date, end)
    iso_dates = [d.isoformat() for d in dates]

    deg_lat = search_radius_km / 111.32
    deg_lon = search_radius_km / (111.32 * max(math.cos(math.radians(lat)), 0.01))
    bbox = (
        f"{lon - deg_lon:.5f},{lat - deg_lat:.5f},"
        f"{lon + deg_lon:.5f},{lat + deg_lat:.5f}"
    )

    site_url = "https://waterservices.usgs.gov/nwis/site/"
    dv_url = "https://waterservices.usgs.gov/nwis/dv/"
    peak_url = "https://nwis.waterdata.usgs.gov/nwis/peak"

    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as c:
        r_sites = await c.get(
            site_url,
            params={
                "format": "rdb",
                "bBox": bbox,
                "parameterCd": "00060",
                "siteStatus": "active",
                "hasDataTypeCd": "dv",
            },
        )
        if r_sites.status_code == 404:
            return {
                "found": False,
                "reason": (
                    f"USGS has no active streamflow gauges in the search "
                    f"bbox around ({lat}, {lon}) — likely outside US "
                    f"coverage."
                ),
                "lat": lat,
                "lon": lon,
                "start_date": start_date,
                "end_date": end,
            }
        r_sites.raise_for_status()

        header: list[str] | None = None
        candidates: list[dict] = []
        for line in r_sites.text.splitlines():
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if header is None:
                header = cols
                continue
            if cols[0] == "5s":  # RDB column-spec row
                continue
            if len(cols) < len(header):
                continue
            row = dict(zip(header, cols))
            try:
                site_lat = float(row["dec_lat_va"])
                site_lon = float(row["dec_long_va"])
            except (KeyError, ValueError):
                continue
            dist_km = geodesic((lat, lon), (site_lat, site_lon)).kilometers
            if dist_km <= search_radius_km:
                candidates.append(
                    {
                        "site_no": row["site_no"],
                        "name": row["station_nm"],
                        "distance_km": round(dist_km, 2),
                    }
                )

        candidates.sort(key=lambda g: g["distance_km"])
        candidates = candidates[:max_gauges]

        if not candidates:
            return {
                "found": False,
                "reason": (
                    f"No active USGS streamflow gauges within "
                    f"{search_radius_km} km (likely outside US or no "
                    f"nearby gauges)."
                ),
                "lat": lat,
                "lon": lon,
                "start_date": start_date,
                "end_date": end,
            }

        site_ids = ",".join(g["site_no"] for g in candidates)
        r_dv = await c.get(
            dv_url,
            params={
                "format": "json",
                "sites": site_ids,
                "parameterCd": "00060",
                "statCd": "00003",
                "startDT": iso_dates[0],
                "endDT": iso_dates[-1],
            },
        )
        r_dv.raise_for_status()
        dv_data = r_dv.json()

        dv_by_site: dict[str, dict[str, float | None]] = {}
        for ts in dv_data.get("value", {}).get("timeSeries", []):
            sid = ts["sourceInfo"]["siteCode"][0]["value"]
            site_dv: dict[str, float | None] = {}
            for v in ts.get("values", [{}])[0].get("value", []):
                d = v["dateTime"][:10]
                try:
                    site_dv[d] = float(v["value"])
                except (TypeError, ValueError):
                    site_dv[d] = None
            dv_by_site[sid] = site_dv

        peak_responses = await asyncio.gather(
            *(
                c.get(
                    peak_url,
                    params={
                        "site_no": g["site_no"],
                        "format": "rdb",
                        "agency_cd": "USGS",
                    },
                )
                for g in candidates
            ),
            return_exceptions=True,
        )

    cfs_to_m3s = 0.02831685
    gauges_out: list[dict] = []
    for g, peak_resp in zip(candidates, peak_responses):
        sid = g["site_no"]
        site_dv = dv_by_site.get(sid, {})
        daily_cfs = [site_dv.get(d) for d in iso_dates]
        daily_m3s = [
            round(v * cfs_to_m3s, 2) if v is not None else None for v in daily_cfs
        ]

        peaks: list[tuple[str, float]] = []
        if not isinstance(peak_resp, Exception):
            for line in peak_resp.text.splitlines():
                if not line.startswith("USGS"):
                    continue
                cols = line.split("\t")
                if len(cols) <= 4 or not cols[4] or len(cols[2]) < 10:
                    continue
                try:
                    peaks.append((cols[2], float(cols[4])))
                except ValueError:
                    continue

        event_peak_cfs = None
        event_peak_date = None
        all_time_cfs = None
        all_time_date = None
        event_rank = None
        if peaks:
            window_peaks = [
                (d, p) for d, p in peaks if iso_dates[0] <= d <= iso_dates[-1]
            ]
            if window_peaks:
                event_peak_date, event_peak_cfs = max(window_peaks, key=lambda x: x[1])
                event_rank = sum(1 for _, p in peaks if p > event_peak_cfs) + 1
            all_time_date, all_time_cfs = max(peaks, key=lambda x: x[1])

        gauges_out.append(
            {
                "site_no": sid,
                "name": g["name"],
                "distance_km": g["distance_km"],
                "daily_cfs": daily_cfs,
                "daily_m3s": daily_m3s,
                "event_peak_cfs": event_peak_cfs,
                "event_peak_date": event_peak_date,
                "all_time_record_cfs": all_time_cfs,
                "all_time_record_date": all_time_date,
                "peaks_years_on_record": len(peaks),
                "event_rank": event_rank,
            }
        )

    return {
        "found": True,
        "dates": iso_dates,
        "gauges": gauges_out,
        "lat": lat,
        "lon": lon,
        "start_date": start_date,
        "end_date": end,
        "source": "USGS Water Services NWIS (daily values + annual peak streamflow)",
    }

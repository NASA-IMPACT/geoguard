import asyncio
import gzip
import json
import math
from datetime import date, timedelta

import httpx
from geopy.distance import geodesic

from geoguard.config import settings
from geoguard.schemas import EventType, Region
from geoguard.tools.registry import registry
from geoguard.utils import date_range, graceful_http


@registry(EventType.FLOOD, EventType.STORM)
@graceful_http
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
@graceful_http
async def get_historical_winds(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str | None = None,
) -> dict:
    """Daily peak wind speed and gust (km/h) over a date range at the given coordinates.

    Source: Open-Meteo Historical Archive (global, no API key required).
    Data granularity is daily; the range is inclusive of both endpoints.

    ERA5 is a ~25 km reanalysis: it captures large storms (cyclones,
    windstorms) but smooths out convective downbursts and thunderstorm
    gusts, which last minutes and span a few km. A low ERA5 gust does NOT
    contradict a measured thunderstorm gust — prefer station observations
    (`get_station_observations`) and treat ERA5 as a lower bound there.

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


@registry(EventType.FLOOD, EventType.STORM, coverage=Region.US)
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


@registry(EventType.FLOOD, EventType.STORM, coverage=Region.US)
@graceful_http
async def get_streamflow_history(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str | None = None,
    search_radius_km: float = 50.0,
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

    Returns both **gage height** (water level in feet, parameter 00065)
    and **discharge** (flow in cfs, parameter 00060) when available. Gage
    height is the most intuitive flood indicator — e.g. "river at 42 ft
    vs normal 28 ft" — while discharge quantifies total water volume.

    Note on units: USGS reports streamflow in cubic feet per second (cfs),
    which is the convention in US flood narratives. m³/s is provided in
    parallel for metric audiences (1 cfs = 0.02831685 m³/s). Gage height
    is reported in feet (ft).

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
            daily_gage_height_ft: Daily mean gage height in feet (None
                on missing). The most intuitive flood metric.
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

        # Fetch both discharge (00060) and gage height (00065)
        r_dv = await c.get(
            dv_url,
            params={
                "format": "json",
                "sites": site_ids,
                "parameterCd": "00060,00065",
                "statCd": "00003",
                "startDT": iso_dates[0],
                "endDT": iso_dates[-1],
            },
        )
        r_dv.raise_for_status()
        dv_data = r_dv.json()

        dv_by_site: dict[str, dict[str, float | None]] = {}
        gh_by_site: dict[str, dict[str, float | None]] = {}
        for ts in dv_data.get("value", {}).get("timeSeries", []):
            sid = ts["sourceInfo"]["siteCode"][0]["value"]
            param = ts["variable"]["variableCode"][0]["value"]
            site_vals: dict[str, float | None] = {}
            for v in ts.get("values", [{}])[0].get("value", []):
                d = v["dateTime"][:10]
                try:
                    site_vals[d] = float(v["value"])
                except (TypeError, ValueError):
                    site_vals[d] = None
            if param == "00060":
                dv_by_site[sid] = site_vals
            elif param == "00065":
                gh_by_site[sid] = site_vals

        # Fallback: fetch gage height from instantaneous values (iv)
        # for gauges that didn't return daily gage height. Many gauges
        # only publish gage height as 15-min instantaneous readings.
        iv_url = "https://nwis.waterservices.usgs.gov/nwis/iv/"
        sites_needing_gh = [
            g["site_no"] for g in candidates if g["site_no"] not in gh_by_site
        ]
        if sites_needing_gh:
            try:
                r_iv = await c.get(
                    iv_url,
                    params={
                        "format": "json",
                        "sites": ",".join(sites_needing_gh),
                        "parameterCd": "00065",
                        "startDT": iso_dates[0],
                        "endDT": iso_dates[-1],
                        "siteStatus": "all",
                    },
                )
                if r_iv.status_code == 200:
                    iv_data = r_iv.json()
                    for ts in iv_data.get("value", {}).get("timeSeries", []):
                        sid = ts["sourceInfo"]["siteCode"][0]["value"]
                        # Aggregate 15-min readings to daily max
                        daily_max: dict[str, float] = {}
                        for v in ts.get("values", [{}])[0].get("value", []):
                            d = v["dateTime"][:10]
                            try:
                                val = float(v["value"])
                                if val >= 0:
                                    daily_max[d] = max(daily_max.get(d, 0), val)
                            except (TypeError, ValueError):
                                pass
                        if daily_max:
                            gh_by_site[sid] = daily_max
            except httpx.HTTPError:
                pass  # IV fallback is best-effort

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
        site_gh = gh_by_site.get(sid, {})
        daily_cfs = [site_dv.get(d) for d in iso_dates]
        daily_m3s = [
            round(v * cfs_to_m3s, 2) if v is not None else None for v in daily_cfs
        ]
        daily_gage_height_ft = [site_gh.get(d) for d in iso_dates]

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
                "daily_gage_height_ft": daily_gage_height_ft,
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


_GLOFAS_URL = "https://flood-api.open-meteo.com/v1/flood"
_GLOFAS_BASELINE_DAYS = 60
_glofas_history: dict[tuple[float, float, int, int], list[tuple[str, float]]] = {}


async def _open_meteo_get(
    client: httpx.AsyncClient, url: str, params: dict
) -> httpx.Response:
    """GET with backoff on 429 — Open-Meteo's free tier rate-limits bursts per minute."""
    for attempt in range(4):
        r = await client.get(url, params=params)
        if r.status_code != 429:
            break
        await asyncio.sleep(15 * (attempt + 1))
    r.raise_for_status()
    return r


@registry(EventType.FLOOD, EventType.STORM)
@graceful_http
async def get_river_discharge_glofas(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str | None = None,
    history_years: int = 0,
) -> dict:
    """Daily river discharge (m³/s) from Copernicus GloFAS — global, including Europe.

    Source: Open-Meteo Flood API serving the Copernicus Emergency
    Management Service GloFAS v4 reanalysis (~5 km grid, 1984–present, no
    API key). Use this for river-flooding claims OUTSIDE the US, where
    USGS gauges (`get_streamflow_history`) are unavailable; it also works
    inside the US as a second, independent line of evidence.

    The query point is snapped to the river cell with the largest flow
    within ~5 km, so pass the claim's location — `river_cell` reports
    where the series actually comes from. Values are model-simulated
    discharge, not a gauge reading: trust them for RELATIVE magnitude
    (how far the peak rose above normal) far more than for exact m³/s,
    and expect small streams to be poorly represented.

    `peak_to_baseline_ratio` compares the peak with the median flow of the
    60 days before the range. Floods typically lift flow one to two orders
    of magnitude, but ordinary rain on a low-flow river can also give a
    several-fold rise — read the ratio together with peak_m3s, and use
    history_years when the claim's wording hinges on how extreme it was.

    Args:
        start_date: Start of the inclusive range (YYYY-MM-DD). For flood
            claims use a lookback window (e.g. 14 days before the event).
        end_date: End of the inclusive range (YYYY-MM-DD). None → single day.
        history_years: 0 (default) skips the long-term comparison. Set it
            (max 20) ONLY when the claim asserts a record or "highest in N
            years" — it is expensive — to rank the peak among that many
            years of annual maxima.

    Returns: dict with keys:
        found: True if a river cell with flow was found near the point.
        reason: Short explanation when found=False (else absent).
        river_cell: {"lat", "lon"} of the grid cell used.
        dates: ISO dates, one per day in the range.
        daily_discharge_m3s: Daily discharge values (None where missing).
        peak_m3s: Highest daily discharge in the range.
        peak_date: Date of peak_m3s.
        baseline_m3s: Median daily discharge over the 60 days before the range.
        peak_to_baseline_ratio: peak_m3s / baseline_m3s.
        event_rank: (only with history_years) rank of the peak among the
            annual maxima of the previous history_years years plus this
            event — 1 = highest on record for that period.
        years_compared: (only with history_years) annual maxima compared.
        lat, lon, start_date, end_date: Query echo.
        source: Identifier of the data source.
    """
    end = end_date or start_date
    step = 0.05  # GloFAS v4 grid spacing in degrees
    offsets = [-step, 0.0, step]
    lats = [round(lat + dy, 4) for dy in offsets for _ in offsets]
    lons = [round(lon + dx, 4) for _ in offsets for dx in offsets]

    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as c:
        r = await _open_meteo_get(
            c,
            _GLOFAS_URL,
            {
                "latitude": ",".join(map(str, lats)),
                "longitude": ",".join(map(str, lons)),
                "daily": "river_discharge",
                "start_date": start_date,
                "end_date": end,
            },
        )
        cells = r.json()
        if isinstance(cells, dict):
            cells = [cells]

        def mean_flow(cell: dict) -> float:
            vals = [
                v
                for v in cell.get("daily", {}).get("river_discharge") or []
                if v is not None
            ]
            return sum(vals) / len(vals) if vals else 0.0

        best = max(cells, key=mean_flow)
        if mean_flow(best) <= 0:
            return {
                "found": False,
                "reason": "No GloFAS river cell with flow within ~5 km of the point.",
                "lat": lat,
                "lon": lon,
                "start_date": start_date,
                "end_date": end,
            }
        cell_lat, cell_lon = best["latitude"], best["longitude"]

        first_day = date.fromisoformat(start_date[:10])
        base = await _open_meteo_get(
            c,
            _GLOFAS_URL,
            {
                "latitude": cell_lat,
                "longitude": cell_lon,
                "daily": "river_discharge",
                "start_date": (
                    first_day - timedelta(days=_GLOFAS_BASELINE_DAYS)
                ).isoformat(),
                "end_date": (first_day - timedelta(days=1)).isoformat(),
            },
        )
        base_vals = sorted(
            v
            for v in base.json().get("daily", {}).get("river_discharge") or []
            if v is not None
        )

        history: list[tuple[str, float]] = []
        years = max(0, min(int(history_years), 20))
        if years:
            key = (cell_lat, cell_lon, first_day.year, years)
            if key not in _glofas_history:
                hist = await _open_meteo_get(
                    c,
                    _GLOFAS_URL,
                    {
                        "latitude": cell_lat,
                        "longitude": cell_lon,
                        "daily": "river_discharge",
                        "start_date": f"{max(first_day.year - years, 1984)}-01-01",
                        "end_date": f"{first_day.year - 1}-12-31",
                    },
                )
                daily = hist.json().get("daily", {})
                _glofas_history[key] = [
                    (d, v)
                    for d, v in zip(
                        daily.get("time") or [], daily.get("river_discharge") or []
                    )
                    if v is not None
                ]
            history = _glofas_history[key]

    dates = best["daily"]["time"]
    series = best["daily"]["river_discharge"]
    valid = [(d, v) for d, v in zip(dates, series) if v is not None]
    peak_date, peak = max(valid, key=lambda x: x[1])
    baseline = base_vals[len(base_vals) // 2] if base_vals else None

    result = {
        "found": True,
        "river_cell": {"lat": cell_lat, "lon": cell_lon},
        "dates": dates,
        "daily_discharge_m3s": series,
        "peak_m3s": round(peak, 2),
        "peak_date": peak_date,
        "baseline_m3s": round(baseline, 2) if baseline else None,
        "peak_to_baseline_ratio": round(peak / baseline, 1) if baseline else None,
        "lat": lat,
        "lon": lon,
        "start_date": start_date,
        "end_date": end,
        "source": "Copernicus GloFAS v4 reanalysis via Open-Meteo Flood API",
    }
    if history:
        annual_max: dict[str, float] = {}
        for d, v in history:
            annual_max[d[:4]] = max(annual_max.get(d[:4], 0.0), v)
        result["event_rank"] = sum(1 for m in annual_max.values() if m > peak) + 1
        result["years_compared"] = len(annual_max) + 1
    return result


_METEOSTAT_BULK = "https://bulk.meteostat.net/v2"
_meteostat_stations: list[dict] | None = None


async def _meteostat_station_list(client: httpx.AsyncClient) -> list[dict]:
    """Meteostat's station catalog (~16k stations), downloaded once per process."""
    global _meteostat_stations
    if _meteostat_stations is None:
        r = await client.get(f"{_METEOSTAT_BULK}/stations/lite.json.gz")
        r.raise_for_status()
        _meteostat_stations = json.loads(gzip.decompress(r.content))
    return _meteostat_stations


def _float_or_none(value: str) -> float | None:
    return float(value) if value not in ("", None) else None


@registry(EventType.FLOOD, EventType.STORM)
@graceful_http
async def get_station_observations(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str | None = None,
    search_radius_km: float = 75.0,
    max_stations: int = 4,
) -> dict:
    """Daily rain gauge totals and measured wind gusts at nearby weather stations — global, incl. Europe.

    Source: Meteostat bulk archive of national weather-service station
    records (AEMET, DWD, Météo-France, Met Office, NOAA ISD, …). These are
    point observations from instruments, so they are the European
    counterpart to `get_radar_gauge_precipitation` (US-only) for rainfall
    claims, and the only source here of station-MEASURED wind gusts for
    storm claims (ERA5 winds are model output).

    Station networks are sparse (synoptic stations, often 20-100 km apart),
    and convective rain is highly localized: a nearby station recording
    little rain does NOT contradict an extreme local rainfall claim. Use
    the maximum across stations, and treat low readings at distant
    stations as inconclusive rather than contradicting.

    Args:
        start_date: Start of the inclusive range (YYYY-MM-DD).
        end_date: End of the inclusive range (YYYY-MM-DD). None → single day.
        search_radius_km: Radius around (lat, lon) to search for stations.
        max_stations: Cap on stations returned (nearest first, only stations
            with data in the range).

    Returns: dict with keys:
        found: True if at least one station had data in the range.
        reason: Short explanation when found=False (else absent).
        dates: ISO dates, one per day in the range.
        stations: list (nearest first) of {id, name, country, distance_km,
            daily_precip_mm, daily_peak_gust_kmh, daily_mean_wind_kmh,
            total_precip_mm, peak_gust_kmh}.
        max_total_precip_mm: Largest range total across stations.
        max_daily_precip_mm: Largest single-day total across stations.
        max_peak_gust_kmh: Largest measured gust across stations.
        lat, lon, start_date, end_date: Query echo.
        source: Identifier of the data source.
    """
    end = end_date or start_date
    iso_dates = [d.isoformat() for d in date_range(start_date, end)]

    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as c:
        catalog = await _meteostat_station_list(c)
        nearby = []
        for s in catalog:
            loc = s.get("location") or {}
            daily = (s.get("inventory") or {}).get("daily") or {}
            if loc.get("latitude") is None or not daily.get("start"):
                continue
            if not (
                daily["start"] <= iso_dates[-1]
                and (daily.get("end") or "") >= iso_dates[0]
            ):
                continue
            dist = geodesic((lat, lon), (loc["latitude"], loc["longitude"])).kilometers
            if dist <= search_radius_km:
                nearby.append((dist, s))
        nearby.sort(key=lambda x: x[0])

        stations_out = []
        for dist, s in nearby:
            if len(stations_out) >= max_stations:
                break
            r = await c.get(f"{_METEOSTAT_BULK}/daily/{s['id']}.csv.gz")
            if r.status_code != 200:
                continue
            rows = {}
            for line in gzip.decompress(r.content).decode().splitlines():
                cols = line.split(",")
                if cols[0] in iso_dates:
                    rows[cols[0]] = cols
            if not rows:
                continue
            precip = [
                _float_or_none(rows[d][4]) if d in rows else None for d in iso_dates
            ]
            gust = [
                _float_or_none(rows[d][8]) if d in rows else None for d in iso_dates
            ]
            wind = [
                _float_or_none(rows[d][7]) if d in rows else None for d in iso_dates
            ]
            p_valid = [v for v in precip if v is not None]
            g_valid = [v for v in gust if v is not None]
            stations_out.append(
                {
                    "id": s["id"],
                    "name": (s.get("name") or {}).get("en"),
                    "country": s.get("country"),
                    "distance_km": round(dist, 1),
                    "daily_precip_mm": precip,
                    "daily_peak_gust_kmh": gust,
                    "daily_mean_wind_kmh": wind,
                    "total_precip_mm": round(sum(p_valid), 1) if p_valid else None,
                    "peak_gust_kmh": max(g_valid) if g_valid else None,
                }
            )

    if not stations_out:
        return {
            "found": False,
            "reason": f"No station with daily data within {search_radius_km:.0f} km for this range.",
            "lat": lat,
            "lon": lon,
            "start_date": start_date,
            "end_date": end,
        }

    def _max(values):
        values = [v for v in values if v is not None]
        return max(values) if values else None

    return {
        "found": True,
        "dates": iso_dates,
        "stations": stations_out,
        "max_total_precip_mm": _max(s["total_precip_mm"] for s in stations_out),
        "max_daily_precip_mm": _max(
            v for s in stations_out for v in s["daily_precip_mm"]
        ),
        "max_peak_gust_kmh": _max(s["peak_gust_kmh"] for s in stations_out),
        "lat": lat,
        "lon": lon,
        "start_date": start_date,
        "end_date": end,
        "source": "Meteostat bulk archive (national weather-service station records)",
    }


_HUBEAU = "https://hubeau.eaufrance.fr/api/v2/hydrometrie"
_UK_EA = "https://environment.data.gov.uk/hydrology"
_FRANCE_BOX = (41.3, 51.2, -5.3, 9.7)
_UK_BOX = (49.8, 61.0, -8.7, 2.0)


def _in_box(lat: float, lon: float, box: tuple[float, float, float, float]) -> bool:
    return box[0] <= lat <= box[1] and box[2] <= lon <= box[3]


async def _france_gauges(c, lat, lon, iso_dates, radius_km, max_gauges) -> list[dict]:
    r = await c.get(
        f"{_HUBEAU}/referentiel/stations",
        params={
            "latitude": lat,
            "longitude": lon,
            "distance": radius_km,
            "en_service": "true",
            "format": "json",
            "size": 30,
        },
    )
    r.raise_for_status()
    stations = sorted(
        (
            (
                geodesic(
                    (lat, lon), (s["latitude_station"], s["longitude_station"])
                ).kilometers,
                s,
            )
            for s in r.json().get("data", [])
            if s.get("latitude_station") is not None
        ),
        key=lambda x: x[0],
    )
    out = []
    for dist, s in stations:
        if len(out) >= max_gauges:
            break
        obs = await c.get(
            f"{_HUBEAU}/obs_elab",
            params={
                "code_entite": s["code_station"],
                "grandeur_hydro_elab": "QmnJ",
                "date_debut_obs_elab": iso_dates[0],
                "date_fin_obs_elab": iso_dates[-1],
                "size": 400,
            },
        )
        if obs.status_code not in (200, 206):
            continue
        by_day = {
            o["date_obs_elab"][:10]: o["resultat_obs_elab"] / 1000.0  # l/s → m³/s
            for o in obs.json().get("data", [])
            if o.get("resultat_obs_elab") is not None
        }
        if by_day:
            out.append(
                {
                    "name": s["libelle_station"],
                    "id": s["code_station"],
                    "country": "France",
                    "distance_km": round(dist, 1),
                    "daily_m3s": [
                        round(by_day[d], 2) if d in by_day else None for d in iso_dates
                    ],
                }
            )
    return out


async def _uk_gauges(c, lat, lon, iso_dates, radius_km, max_gauges) -> list[dict]:
    r = await c.get(
        f"{_UK_EA}/id/stations",
        params={
            "lat": lat,
            "long": lon,
            "dist": radius_km,
            "observedProperty": "waterFlow",
            "_limit": 30,
        },
    )
    r.raise_for_status()
    stations = sorted(
        (
            (geodesic((lat, lon), (s["lat"], s["long"])).kilometers, s)
            for s in r.json().get("items", [])
            if s.get("lat") is not None
        ),
        key=lambda x: x[0],
    )
    end_exclusive = (date.fromisoformat(iso_dates[-1]) + timedelta(days=1)).isoformat()
    out = []
    for dist, s in stations:
        if len(out) >= max_gauges:
            break
        measure = next(
            (
                m["@id"]
                for m in s.get("measures", [])
                if "flow-m-86400-m3s" in m.get("@id", "")
            ),
            None,
        )
        if not measure:
            continue
        measure = measure.replace(
            "http://", "https://", 1
        )  # EA ids are http; API redirects
        obs = await c.get(
            f"{measure}/readings",
            params={"mineq-date": iso_dates[0], "max-date": end_exclusive},
        )
        if obs.status_code != 200:
            continue
        by_day = {
            o["date"][:10]: o["value"]
            for o in obs.json().get("items", [])
            if o.get("value") is not None
        }
        if by_day:
            out.append(
                {
                    "name": s.get("label"),
                    "id": s.get("notation"),
                    "country": "United Kingdom",
                    "distance_km": round(dist, 1),
                    "daily_m3s": [by_day.get(d) for d in iso_dates],
                }
            )
    return out


@registry(EventType.FLOOD, EventType.STORM, coverage=Region.EUROPE)
@graceful_http
async def get_european_river_gauges(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str | None = None,
    search_radius_km: float = 40.0,
    max_gauges: int = 4,
) -> dict:
    """Daily mean river discharge (m³/s) MEASURED at national river gauges in Europe.

    The European counterpart to `get_streamflow_history` (USGS, US-only):
    observed gauge readings from national hydrometric services. Currently
    integrated: France (Hub'Eau / Vigicrues network) and the United Kingdom
    (Environment Agency hydrology). Elsewhere in Europe it returns
    found=False — use `get_river_discharge_glofas` (Copernicus, pan-European,
    model-simulated) instead, which also gives the multi-year rank of a peak.

    Args:
        start_date: Start of inclusive range (YYYY-MM-DD). For flood claims
            use a lookback window (e.g. 14 days before the event).
        end_date: End of inclusive range (YYYY-MM-DD). None → single day.
        search_radius_km: Radius around (lat, lon) to search for gauges.
        max_gauges: Cap on gauges returned (nearest first, with flow data).

    Returns: dict with keys:
        found: True if at least one gauge had discharge data in the range.
        reason: Short explanation when found=False (else absent).
        dates: ISO dates, one per day.
        gauges: list (nearest first) of {name, id, country, distance_km,
            daily_m3s, peak_m3s, peak_date}.
        lat, lon, start_date, end_date: Query echo.
        source: Identifier of the data source.
    """
    end = end_date or start_date
    iso_dates = [d.isoformat() for d in date_range(start_date, end)]
    gauges: list[dict] = []
    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as c:
        if _in_box(lat, lon, _FRANCE_BOX):
            gauges += await _france_gauges(
                c, lat, lon, iso_dates, search_radius_km, max_gauges
            )
        if _in_box(lat, lon, _UK_BOX):
            gauges += await _uk_gauges(
                c, lat, lon, iso_dates, search_radius_km, max_gauges
            )

    if not gauges:
        return {
            "found": False,
            "reason": (
                "No national gauge with discharge data nearby. Gauge APIs are "
                "integrated for France and the UK; elsewhere use "
                "get_river_discharge_glofas."
            ),
            "lat": lat,
            "lon": lon,
            "start_date": start_date,
            "end_date": end,
        }
    for g in gauges:
        valid = [(d, v) for d, v in zip(iso_dates, g["daily_m3s"]) if v is not None]
        g["peak_date"], g["peak_m3s"] = max(valid, key=lambda x: x[1])
    gauges.sort(key=lambda g: g["distance_km"])
    return {
        "found": True,
        "dates": iso_dates,
        "gauges": gauges[:max_gauges],
        "lat": lat,
        "lon": lon,
        "start_date": start_date,
        "end_date": end,
        "source": "Hub'Eau hydrométrie (France) / Environment Agency hydrology (UK)",
    }

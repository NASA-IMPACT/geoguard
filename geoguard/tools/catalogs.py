"""Disaster-catalog tools — authoritative records that an event occurred.

Catalogs corroborate that an event was recorded; they do not measure it.
A match is strong support for an event-existence claim (it happened, here,
then), but a catalog's silence is weak evidence: catalogs list only events
that crossed their reporting threshold.
"""

import json
from datetime import timedelta

import httpx
from geopy.distance import geodesic

from geoguard.config import settings
from geoguard.schemas import EventType
from geoguard.tools.registry import registry
from geoguard.utils import _parse_date, graceful_http

_GDACS_URL = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"


@registry(EventType.FLOOD, EventType.STORM)
@graceful_http
async def search_gdacs_events(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str | None = None,
    radius_km: float = 400.0,
) -> dict:
    """Disasters recorded by GDACS near a point and date window — global catalog, incl. Europe.

    Source: GDACS (Global Disaster Alert and Coordination System, UN + European
    Commission JRC) event archive. Covers floods (FL) and tropical cyclones
    (TC) worldwide, with alert levels Green / Orange / Red.

    Use this to corroborate that an event OCCURRED at a place and time —
    especially outside the US. A matching event is strong support for an
    event-existence claim. A missing event is NOT evidence the event did not
    happen: GDACS records events of international significance, so local
    floods are routinely absent. Never return CONTRADICTS from this tool alone.

    Args:
        start_date: Start of the window (YYYY-MM-DD); padded by 7 days each side.
        end_date: End of the window (YYYY-MM-DD). None → single day.
        radius_km: Max distance from the point to an event's reported location.
            Flood events are reported at a country/region centroid, so keep
            this generous (default 400 km).

    Returns: dict with keys:
        found: True if at least one event matched.
        events: list (nearest first) of {eventtype, name, alertlevel, country,
            fromdate, todate, distance_km, severity, report_url}.
        searched: description of the window and radius used.
        lat, lon, start_date, end_date: Query echo.
        source: Identifier of the data source.
    """
    end = end_date or start_date
    pad = timedelta(days=7)
    window_start = _parse_date(start_date) - pad
    window_end = _parse_date(end) + pad

    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as c:
        r = await c.get(
            _GDACS_URL,
            params={
                "eventlist": "FL;TC",
                "fromdate": window_start.isoformat(),
                "todate": window_end.isoformat(),
                "alertlevel": "Green;Orange;Red",
            },
        )
        if r.status_code == 204 or not r.content:
            features = []
        else:
            r.raise_for_status()
            features = r.json().get("features", [])

    events = []
    for f in features:
        p = f.get("properties", {})
        coords = (f.get("geometry") or {}).get("coordinates") or []
        if len(coords) < 2:
            continue
        ev_lon, ev_lat = coords[0], coords[1]
        dist = geodesic((lat, lon), (ev_lat, ev_lon)).kilometers
        if dist > radius_km:
            continue
        events.append(
            {
                "eventtype": p.get("eventtype"),
                "name": p.get("name"),
                "alertlevel": p.get("alertlevel"),
                "country": p.get("country"),
                "fromdate": (p.get("fromdate") or "")[:10],
                "todate": (p.get("todate") or "")[:10],
                "distance_km": round(dist, 1),
                "severity": (p.get("severitydata") or {}).get("severitytext"),
                "report_url": (p.get("url") or {}).get("report"),
            }
        )
    events.sort(key=lambda e: e["distance_km"])

    return {
        "found": bool(events),
        "events": events[:10],
        "searched": (
            f"GDACS FL+TC events {window_start}..{window_end} "
            f"within {radius_km:.0f} km"
        ),
        "lat": lat,
        "lon": lon,
        "start_date": start_date,
        "end_date": end,
        "source": "GDACS event archive (UN / European Commission JRC)",
    }


async def get_current_gdacs_alerts(
    lat: float,
    lon: float,
    event_type: str = "FL",
    radius_km: float = 400.0,
) -> dict:
    """Disasters GDACS is tracking RIGHT NOW near a point — live alerts, global.

    Source: GDACS via the EO MCP tools server. Lists only ongoing or very
    recent events (the last few weeks), so use it for claims about a
    current disaster ("flooding is ongoing in…"). For a dated historical
    claim use `search_gdacs_events` instead. As with any catalog, a match
    corroborates the event; absence is not evidence it did not happen.

    Args:
        event_type: GDACS hazard code — "FL" flood, "TC" tropical cyclone,
            "EQ" earthquake, "WF" wildfire, "DR" drought, "VO" volcano.
        radius_km: Max distance from the point to an event's reported location.

    Returns: dict with keys:
        found: True if at least one current event matched.
        events: list (nearest first) of {eventtype, name, alertlevel,
            fromdate, todate, distance_km, report_url}.
        lat, lon: Query echo.
        source: Identifier of the data source.
    """
    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as c:
        r = await c.post(
            settings.gdacs_mcp_url,
            headers={
                "Authorization": f"Bearer {settings.gdacs_mcp_token}",
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": "2025-06-18",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "search_gdacs_events",
                    "arguments": {"eventlist": event_type},
                },
            },
        )
        r.raise_for_status()
    payload = next(
        line.removeprefix("data: ")
        for line in r.text.splitlines()
        if line.startswith("data: ")
    )
    result = json.loads(payload).get("result", {})
    collection = json.loads(
        (result.get("structuredContent") or {}).get("result") or "{}"
    )

    events = []
    for f in collection.get("features", []):
        p = f.get("properties", {})
        ev_lon, ev_lat = f["geometry"]["coordinates"][:2]
        dist = geodesic((lat, lon), (ev_lat, ev_lon)).kilometers
        if dist <= radius_km:
            events.append(
                {
                    "eventtype": p.get("eventtype"),
                    "name": p.get("name"),
                    "alertlevel": p.get("alertlevel"),
                    "fromdate": (p.get("fromdate") or "")[:10],
                    "todate": (p.get("todate") or "")[:10],
                    "distance_km": round(dist, 1),
                    "report_url": (p.get("url") or {}).get("report"),
                }
            )
    events.sort(key=lambda e: e["distance_km"])
    return {
        "found": bool(events),
        "events": events[:10],
        "lat": lat,
        "lon": lon,
        "source": "GDACS live alerts via EO MCP tools server",
    }


# Live alerts need the MCP server token; without it the tool is not offered.
if settings.gdacs_mcp_token:
    registry(EventType.FLOOD, EventType.STORM)(graceful_http(get_current_gdacs_alerts))

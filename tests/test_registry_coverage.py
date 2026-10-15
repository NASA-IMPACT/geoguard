"""Region-aware tool candidates: a claim is never routed to a source that can't cover it."""

import geoguard.tools.catalogs  # noqa: F401
import geoguard.tools.geospatial  # noqa: F401
import geoguard.tools.weather  # noqa: F401
from geoguard.schemas import EventType, Region, region_contains
from geoguard.tools.registry import registry

US_ONLY = {"get_radar_gauge_precipitation", "get_streamflow_history"}
EUROPE_ONLY = {"get_european_river_gauges"}
EUROPE_READY = {
    "get_river_discharge_glofas",
    "get_station_observations",
    "get_historical_precipitation",
    "get_historical_winds",
    "search_gdacs_events",
}


def _names(lat=None, lon=None) -> set[str]:
    return {
        fn.__name__ for fn in registry.get_candidates(EventType.FLOOD, lat=lat, lon=lon)
    }


def test_european_claim_drops_us_only_sources():
    names = _names(39.47, -0.38)  # Valencia, Spain
    assert not names & US_ONLY
    assert EUROPE_READY | EUROPE_ONLY <= names


def test_us_claim_keeps_us_and_global_sources():
    names = _names(29.76, -95.36)  # Houston, Texas
    assert US_ONLY <= names
    assert EUROPE_READY <= names
    assert not names & EUROPE_ONLY


def test_unknown_location_filters_nothing():
    assert US_ONLY | EUROPE_ONLY | EUROPE_READY <= _names()


def test_region_boxes():
    assert region_contains(Region.US, 61.2, -149.9)  # Anchorage
    assert region_contains(Region.EUROPE, 64.1, -21.9)  # Reykjavik
    assert not region_contains(Region.US, 48.85, 2.35)  # Paris
    assert region_contains(Region.GLOBAL, -33.9, 151.2)  # Sydney

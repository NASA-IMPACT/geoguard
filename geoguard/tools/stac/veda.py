"""VEDA STAC catalog primitives — search, inspect, and analyze catalog assets.

NASA VEDA (Visualization, Exploration, and Data Analysis) publishes curated
Earth-science datasets behind a standard STAC 1.0 API. The functions here are
the raw search / inspect / analyze steps used by the STAC exploration
sub-agent (`geoguard.tools.stac.agent`). They are deliberately NOT registered
as verifier tools: the sub-agent owns the messy exploration loop, and the
main verifier sees only its synthesized summary (see NASA-IMPACT/geoguard#7;
the plain registered single-call tool is NASA-IMPACT/geoguard#5).

Endpoints: `build_veda_tools(stac_url, raster_url)` binds the primitives to
an explicit deployment; the module-level defaults resolve from settings at
call time (GEOGUARD_VEDA_STAC_URL / GEOGUARD_VEDA_RASTER_URL, falling back
to the public catalog):
  - STAC API:   https://openveda.cloud/api/stac   (catalog search)
  - Raster API: https://openveda.cloud/api/raster (titiler-pgstac)

The search endpoints are plain STAC 1.0, but the statistics endpoint is
eoAPI/titiler-specific — these primitives work against any eoAPI-shaped
deployment, not arbitrary STAC catalogs. The raster API computes zonal
statistics server-side from cloud-optimized GeoTIFFs, so the analyze step
needs no raster download and no credentials — important because the
catalog's data-asset hrefs are s3:// URIs that are not anonymously
fetchable.

Every function returns a compact, JSON-serializable dict carrying a `found`
key. No-coverage is a first-class answer (`found: False` with a reason), and
HTTP failures degrade to `{tool, error}` dicts via @graceful_http — neither
ever raises into the calling agent.

Docstrings double as prompt context: the sub-agent reads them as its tool
descriptions.
"""

from __future__ import annotations

from typing import Any, Callable

import httpx

from geoguard.config import settings
from geoguard.utils import graceful_http

# How much free text to keep per record. Everything these functions return
# feeds an LLM context — truncation here is what keeps the exploration loop
# from ballooning the sub-agent's own context.
_DESCRIPTION_CHARS = 300
_MAX_ASSET_KEYS = 8

# Process-level cache of compacted collection indexes, keyed by STAC URL.
# A catalog is ~250 collections and changes rarely; one fetch per endpoint
# per process is plenty.
_collections_cache: dict[str, list[dict]] = {}


def _clear_collections_cache() -> None:
    """Drop the cached collection indexes — for tests."""
    _collections_cache.clear()


def _compact_collection(c: dict) -> dict:
    """Reduce a raw STAC collection to the fields exploration needs."""
    extent = c.get("extent") or {}
    spatial = (extent.get("spatial") or {}).get("bbox") or [[None]]
    temporal = (extent.get("temporal") or {}).get("interval") or [[None, None]]
    description = (c.get("description") or "").strip()
    if len(description) > _DESCRIPTION_CHARS:
        description = description[: _DESCRIPTION_CHARS - 1] + "…"
    return {
        "id": c.get("id"),
        "title": c.get("title") or "",
        "description": description,
        "bbox": spatial[0],
        "temporal_interval": temporal[0],
        "asset_keys": sorted(c.get("item_assets") or {})[:_MAX_ASSET_KEYS],
    }


def _compact_item(f: dict) -> dict:
    """Reduce a raw STAC item (GeoJSON feature) to an identifiable summary."""
    props = f.get("properties") or {}
    assets = f.get("assets") or {}
    data_assets = [
        k
        for k, a in assets.items()
        if "data" in (a.get("roles") or []) or "geotiff" in (a.get("type") or "")
    ]
    return {
        "id": f.get("id"),
        "collection": f.get("collection"),
        "datetime": props.get("datetime")
        or f"{props.get('start_datetime')}/{props.get('end_datetime')}",
        "bbox": [round(v, 4) for v in (f.get("bbox") or [])],
        "data_assets": sorted(data_assets)[:_MAX_ASSET_KEYS],
    }


def _compact_stats(raw: dict) -> dict:
    """Keep the headline numbers from a titiler statistics payload.

    Drops histograms and percentile noise — the sub-agent needs magnitude
    and coverage, not a full distribution.
    """
    keep = ("min", "max", "mean", "median", "std", "valid_percent", "valid_pixels")
    return {
        band: {k: round(v, 4) for k, v in stats.items() if k in keep}
        for band, stats in raw.items()
        if isinstance(stats, dict)
    }


async def _fetch_all_collections(client: httpx.AsyncClient, url: str) -> list[dict]:
    """Fetch and compact every collection at `url` (cached per endpoint)."""
    if url not in _collections_cache:
        r = await client.get(url, params={"limit": 1000})
        r.raise_for_status()
        _collections_cache[url] = [
            _compact_collection(c) for c in r.json().get("collections", [])
        ]
    return _collections_cache[url]


def _keyword_terms(keywords: str) -> list[str]:
    return [t for t in keywords.lower().replace(",", " ").split() if t]


def _score_collection(col: dict, terms: list[str]) -> int:
    """Count how many search terms appear in the collection's text fields."""
    text = f"{col['id']} {col['title']} {col['description']}".lower()
    return sum(1 for t in terms if t in text)


def build_veda_tools(
    stac_url: str | None = None,
    raster_url: str | None = None,
) -> list[Callable]:
    """Build the three catalog primitives bound to explicit endpoints.

    With the defaults (None), each call resolves the endpoint from settings
    at call time — the module-level functions below are exactly this. Pass
    explicit URLs to target a different eoAPI deployment (staging, a private
    catalog): `stac_url` must serve STAC 1.0 search, `raster_url` a
    titiler-pgstac raster API.
    """

    def _stac(path: str) -> str:
        return (stac_url or settings.veda_stac_url).rstrip("/") + path

    def _raster(path: str) -> str:
        return (raster_url or settings.veda_raster_url).rstrip("/") + path

    @graceful_http
    async def search_veda_collections(keywords: str, limit: int = 15) -> dict:
        """Find VEDA dataset collections whose name or description match keywords.

        Searches the full NASA VEDA catalog (~250 curated Earth-science
        collections: precipitation, flood imagery, land cover, emissions,
        hydrology, event-specific disaster mosaics, …). Matching is substring-
        based over collection id, title, and description; collections matching
        more terms rank first.

        Tips: try several distinct terms in one call ("flood precipitation
        inundation"), and prefer generic phenomenon words over full sentences.
        Check each result's temporal_interval and bbox against your claim —
        a keyword match with the wrong year or region is not evidence.

        Args:
            keywords: Space- or comma-separated search terms, matched
                case-insensitively ("flood", "precipitation imerg", …).
            limit: Max collections to return (default 15).

        Returns: dict with keys:
            found: True if at least one collection matched.
            matched_count: Total matches before truncation.
            collections: Up to `limit` records, each with id, title,
                description (truncated), bbox, temporal_interval, asset_keys.
        """
        terms = _keyword_terms(keywords)
        async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as client:
            index = await _fetch_all_collections(client, _stac("/collections"))
        scored = [(s, c) for c in index if (s := _score_collection(c, terms)) > 0]
        scored.sort(key=lambda sc: -sc[0])
        matches = [c for _, c in scored]
        if not matches:
            return {
                "found": False,
                "matched_count": 0,
                "reason": (
                    f"No VEDA collections matched {keywords!r}. Try different or "
                    "more generic terms (e.g. the phenomenon, instrument, or "
                    "event name)."
                ),
            }
        return {
            "found": True,
            "matched_count": len(matches),
            "collections": matches[:limit],
        }

    @graceful_http
    async def search_veda_items(
        collection_id: str,
        start_date: str,
        end_date: str,
        bbox_lon_min: float | None = None,
        bbox_lat_min: float | None = None,
        bbox_lon_max: float | None = None,
        bbox_lat_max: float | None = None,
        limit: int = 10,
    ) -> dict:
        """List a VEDA collection's items (granules) intersecting a place and time.

        An item is one dated, georeferenced asset (typically a cloud-optimized
        GeoTIFF). Some VEDA collections expose zero items — that is a real
        catalog gap, not an error; move on to another collection. When items
        exist but none intersect your window, widening the date range a few
        days often helps (composites are stamped at their period start).

        Args:
            collection_id: Collection to search (from search_veda_collections).
            start_date: Window start, YYYY-MM-DD (inclusive).
            end_date: Window end, YYYY-MM-DD (inclusive).
            bbox_lon_min: Western bound of the area of interest (degrees).
            bbox_lat_min: Southern bound.
            bbox_lon_max: Eastern bound.
            bbox_lat_max: Northern bound. All four together restrict results
                to items intersecting the box; omit all four for no spatial
                filter.
            limit: Max items to return (default 10).

        Returns: dict with keys:
            found: True if at least one item matched.
            matched_count: Total matches in the catalog (before truncation).
            items: Up to `limit` records with id, collection, datetime, bbox,
                data_assets (asset keys usable for statistics).
        """
        body: dict[str, Any] = {
            "collections": [collection_id],
            "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
            "limit": limit,
        }
        bbox = (bbox_lon_min, bbox_lat_min, bbox_lon_max, bbox_lat_max)
        if all(v is not None for v in bbox):
            body["bbox"] = list(bbox)
        async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as client:
            r = await client.post(_stac("/search"), json=body)
            r.raise_for_status()
            data = r.json()
        features = data.get("features", [])
        if not features:
            return {
                "found": False,
                "matched_count": 0,
                "collection": collection_id,
                "reason": (
                    f"No items in {collection_id!r} for {start_date}/{end_date}"
                    + (" within the bbox" if "bbox" in body else "")
                    + ". The collection may have no items at all, or none for "
                    "this place/time — try another window or collection."
                ),
            }
        return {
            "found": True,
            "matched_count": data.get("numberMatched", len(features)),
            "items": [_compact_item(f) for f in features],
        }

    @graceful_http
    async def get_veda_item_statistics(
        collection_id: str,
        item_id: str,
        bbox_lon_min: float,
        bbox_lat_min: float,
        bbox_lon_max: float,
        bbox_lat_max: float,
        asset: str = "cog_default",
    ) -> dict:
        """Compute summary statistics for one VEDA item over a bounding box.

        Fetches nothing locally: the VEDA raster API reads the item's
        cloud-optimized GeoTIFF server-side and returns zonal statistics
        (min / max / mean / median / std, valid-pixel coverage) for the exact
        area of interest. Use this to turn a promising catalog item into an
        actual number over the claim's area.

        Units are whatever the underlying dataset measures — the raster API
        does not report them, so take the variable and units from the
        collection description and say so in your synthesis. A low
        valid_percent means clouds/no-data dominate the area; treat the
        numbers with caution.

        Args:
            collection_id: The item's collection.
            item_id: The item to analyze (from search_veda_items).
            bbox_lon_min: Western bound of the analysis area (degrees).
            bbox_lat_min: Southern bound.
            bbox_lon_max: Eastern bound.
            bbox_lat_max: Northern bound.
            asset: Asset key to analyze (default "cog_default", the standard
                VEDA data layer; see the item's data_assets).

        Returns: dict with keys:
            found: True if statistics were computed.
            statistics: Per-band headline stats (min/max/mean/median/std,
                valid_percent, valid_pixels).
            bbox: The analysis bounds echoed back.
            source: Attribution string (catalog, collection, item).
        """
        geometry = {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [bbox_lon_min, bbox_lat_min],
                        [bbox_lon_max, bbox_lat_min],
                        [bbox_lon_max, bbox_lat_max],
                        [bbox_lon_min, bbox_lat_max],
                        [bbox_lon_min, bbox_lat_min],
                    ]
                ],
            },
        }
        url = _raster(f"/collections/{collection_id}/items/{item_id}/statistics")
        async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as client:
            r = await client.post(url, params={"assets": asset}, json=geometry)
            r.raise_for_status()
            data = r.json()
        stats = _compact_stats((data.get("properties") or {}).get("statistics") or {})
        if not stats:
            return {
                "found": False,
                "collection": collection_id,
                "item": item_id,
                "reason": (
                    f"The raster API returned no statistics for asset {asset!r} "
                    f"of {collection_id}/{item_id} — the asset may not be a "
                    "readable raster, or the bbox may fall outside its footprint."
                ),
            }
        return {
            "found": True,
            "statistics": stats,
            "bbox": [bbox_lon_min, bbox_lat_min, bbox_lon_max, bbox_lat_max],
            "source": f"NASA VEDA STAC: {collection_id}/{item_id} (asset {asset})",
        }

    return [search_veda_collections, search_veda_items, get_veda_item_statistics]


# Settings-bound defaults — resolve GEOGUARD_VEDA_*_URL (or the public
# catalog) at call time, exactly as before endpoints became injectable.
search_veda_collections, search_veda_items, get_veda_item_statistics = (
    build_veda_tools()
)

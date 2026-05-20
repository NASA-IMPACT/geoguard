"""GeoGuard Tiff-to-Claims Adapter.

Converts a Prithvi-EO (or any binary) flood detection mask (.tif) into
text claims that GeoGuard can verify, or into structured metadata for
reports.

Usage::

    from geoguard.adapters import tiff_to_claims
    from geoguard import GeoGuard, Input

    claims_text = tiff_to_claims(
        tiff_path="flood_detection_2023022_california.tif",
        bbox=[-122.172, 38.175, -121.381, 39.601],
        date="2023-01-22",
        region_name="Sacramento Valley, California, USA",
        model_name="Prithvi-EO",
        input_source="Sentinel-2",
    )

    guard = GeoGuard.from_config()
    report = await guard.run(Input(text=claims_text))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import label


# ---------------------------------------------------------------------------
# Shared pixel-geometry helpers
# ---------------------------------------------------------------------------

@dataclass
class _TileGeometry:
    """Pre-computed spatial constants for a georeferenced raster tile."""

    west: float
    south: float
    east: float
    north: float
    h: int  # image height (rows)
    w: int  # image width (cols)

    @property
    def lon_res(self) -> float:
        return (self.east - self.west) / self.w

    @property
    def lat_res(self) -> float:
        return (self.north - self.south) / self.h

    @property
    def mid_lat(self) -> float:
        return (self.north + self.south) / 2

    @property
    def km_per_deg_lat(self) -> float:
        return 111.0

    @property
    def km_per_deg_lon(self) -> float:
        return 111.0 * math.cos(math.radians(self.mid_lat))

    @property
    def pixel_area_km2(self) -> float:
        return abs(self.lon_res * self.km_per_deg_lon) * abs(
            self.lat_res * self.km_per_deg_lat
        )

    @property
    def pixel_res_m(self) -> float:
        return abs(self.lon_res * self.km_per_deg_lon * 1000)

    @property
    def pixel_res_m_x(self) -> float:
        return abs(self.lon_res * self.km_per_deg_lon * 1000)

    @property
    def pixel_res_m_y(self) -> float:
        return abs(self.lat_res * self.km_per_deg_lat * 1000)


def _extract_patches(
    flood_mask: np.ndarray,
    geo: _TileGeometry,
    min_patch_pixels: int,
    linear_threshold: float,
) -> list[dict]:
    """Run connected-component analysis and return sorted patch dicts."""
    labeled_arr, n_patches = label(flood_mask)
    patches: list[dict] = []

    for i in range(1, n_patches + 1):
        mask_i = labeled_arr == i
        size = int(mask_i.sum())
        if size < min_patch_pixels:
            continue

        rows, cols = np.where(mask_i)
        centroid_lat = geo.north - rows.mean() * geo.lat_res
        centroid_lon = geo.west + cols.mean() * geo.lon_res

        height_km = abs(
            (rows.max() - rows.min()) * geo.lat_res * geo.km_per_deg_lat
        )
        width_km = abs(
            (cols.max() - cols.min()) * geo.lon_res * geo.km_per_deg_lon
        )
        aspect_ratio = max(height_km, width_km) / max(
            min(height_km, width_km), 0.001
        )

        patches.append(
            {
                "pixels": size,
                "area_km2": round(size * geo.pixel_area_km2, 2),
                "centroid_lat": round(centroid_lat, 4),
                "centroid_lon": round(centroid_lon, 4),
                "height_km": round(height_km, 2),
                "width_km": round(width_km, 2),
                "aspect_ratio": round(aspect_ratio, 1),
                "shape": (
                    "linear/river"
                    if aspect_ratio > linear_threshold
                    else "compact/ponding"
                ),
            }
        )

    patches.sort(key=lambda x: x["area_km2"], reverse=True)
    return patches


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def tiff_to_claims(
    tiff_path: str | Path,
    bbox: list[float],
    date: str,
    region_name: str = "the study area",
    model_name: str = "a foundation model",
    input_source: str = "satellite imagery",
) -> str:
    """Convert a flood detection GeoTIFF to verifiable text claims.

    Generates a short paragraph containing only claims that GeoGuard's
    tools can actually check: total flood area (satellite), location
    (geocoding), and whether flooding occurred (precipitation +
    streamflow).  Unverifiable details (zone counts, aspect ratios,
    resolution, cloud cover) are omitted — use :func:`tiff_to_metadata`
    for the full breakdown.

    The returned string is designed to be passed directly to
    ``GeoGuard.run(Input(text=...))`` for end-to-end verification.

    Args:
        tiff_path:  Path to flood mask (0=no flood, 1=flood, 255=nodata).
        bbox:       [west, south, east, north] in degrees.
        date:       Event date as ``YYYY-MM-DD``.
        region_name: Human-readable region name for claims.
        model_name:  Name of the model that produced the mask.
        input_source: Input imagery type (e.g. "Sentinel-2").

    Returns:
        Generated claims text ready for GeoGuard input.
    """
    data = tifffile.imread(str(tiff_path))
    west, south, east, north = bbox
    h, w = data.shape
    geo = _TileGeometry(west, south, east, north, h, w)

    flood_mask = data == 1
    flood_pixels = int(flood_mask.sum())

    if flood_pixels == 0:
        return f"No flooding detected in {region_name} on {date}."

    total_area_km2 = flood_pixels * geo.pixel_area_km2

    all_rows, all_cols = np.where(flood_mask)
    overall_lat = geo.north - all_rows.mean() * geo.lat_res
    overall_lon = geo.west + all_cols.mean() * geo.lon_res

    return (
        f"Satellite-derived flood detection for {date} shows approximately "
        f"{total_area_km2:.0f} km² of inundated area in {region_name} "
        f"(centered near {overall_lat:.2f}°N, {abs(overall_lon):.2f}°W). "
        f"The analysis, based on {model_name} applied to "
        f"{input_source}, covers the bounding box "
        f"[{west:.3f}, {south:.3f}, {east:.3f}, {north:.3f}]."
    )


def tiff_to_metadata(
    tiff_path: str | Path,
    bbox: list[float],
    date: str,
    min_patch_pixels: int = 50,
    linear_threshold: float = 3.0,
) -> dict:
    """Extract structured metadata from a flood mask.

    Use for report metadata or for fields that tools can't verify
    (zone counts, patch shapes, etc.).

    Args:
        tiff_path:  Path to flood mask (0=no flood, 1=flood, 255=nodata).
        bbox:       [west, south, east, north] in degrees.
        date:       Event date as ``YYYY-MM-DD``.
        min_patch_pixels: Minimum patch size to count as significant.
        linear_threshold: Aspect ratio above which a patch is "linear/river".

    Returns:
        Dict with all extracted spatial and statistical metrics.
    """
    data = tifffile.imread(str(tiff_path))
    west, south, east, north = bbox
    h, w = data.shape
    geo = _TileGeometry(west, south, east, north, h, w)

    flood_mask = data == 1
    nodata_mask = data == 255
    valid_pixels = int((~nodata_mask).sum())
    flood_pixels = int(flood_mask.sum())

    patches = _extract_patches(
        flood_mask, geo, min_patch_pixels, linear_threshold
    )

    overall_lat = None
    overall_lon = None
    if flood_pixels:
        all_rows, all_cols = np.where(flood_mask)
        overall_lat = round(geo.north - all_rows.mean() * geo.lat_res, 4)
        overall_lon = round(geo.west + all_cols.mean() * geo.lon_res, 4)

    linear_patches = sum(1 for p in patches if p["shape"] == "linear/river")
    compact_patches = sum(1 for p in patches if p["shape"] == "compact/ponding")

    return {
        "date": date,
        "bbox": [west, south, east, north],
        "image_size": [w, h],
        "pixel_resolution_m": [
            round(geo.pixel_res_m_x, 1),
            round(geo.pixel_res_m_y, 1),
        ],
        "aoi_area_km2": round(valid_pixels * geo.pixel_area_km2, 0),
        "total_flood_area_km2": round(flood_pixels * geo.pixel_area_km2, 1),
        "flood_fraction_pct": round(flood_pixels / valid_pixels * 100, 1)
        if valid_pixels
        else 0.0,
        "nodata_fraction_pct": round(nodata_mask.sum() / data.size * 100, 1),
        "centroid_lat": overall_lat,
        "centroid_lon": overall_lon,
        "significant_patches": len(patches),
        "linear_patches": linear_patches,
        "compact_patches": compact_patches,
        "largest_patch_km2": patches[0]["area_km2"] if patches else 0,
        "top_patches": patches[:10],
    }

# Registered tools

This is the catalog of tools currently registered in GeoGuard's verification toolkit. Each tool is an async function decorated with `@registry(EventType.X, ...)` declaring which event types it serves. The tool selector LLM reads each tool's signature and docstring at runtime to decide which to attach to a given claim's verifier agent.

For the architectural treatment of the registry pattern, see [`architecture.md`](architecture.md). For a narrative explainer aimed at ML/NLP folks, see [`tools-explained.md`](tools-explained.md).

---

## At a glance

| # | Tool | Source | Coverage | What it verifies | Key gotcha |
|---|---|---|---|---|---|
| 1 | `geocode` | OpenStreetMap Nominatim | Global | Place name → (lat, lon) | Pass fullest place string to avoid same-name collisions |
| 2 | `get_historical_precipitation` | Open-Meteo / ERA5 reanalysis | Global, ~9 km | Daily rainfall (mm) | Smooths extremes — treat as lower bound |
| 3 | `get_radar_gauge_precipitation` | NOAA PRISM (~4 km) + MRMS (~1 km) via IEMRE | US only | Daily rainfall, gauge-anchored + radar-derived | Returns two values per day — they intentionally disagree |
| 4 | `get_historical_winds` | Open-Meteo / ERA5 | Global, ~9 km | Daily peak sustained + gust (km/h) | "Sustained" = 10-min mean (differs from US 2-min NWS convention) |
| 5 | `get_streamflow_history` | USGS NWIS | US only | River discharge (cfs + m³/s) + annual peak records | Daily means ≠ instantaneous peaks — don't compare across types |
| 6 | `get_elevation` | Open-Meteo / SRTM-90m | Global, ~90 m | Meters above sea level | Returns 0 over water; canopy/buildings inflate it |
| 7 | `find_nearest_water_body` | OpenStreetMap Overpass | Global (quality varies) | Nearest river/stream/lake/coastline | Volunteer-mapped — some features unnamed, sparse in some regions |

---

## Coverage matrix

What's verifiable today, broken down by event type and geography:

| | Rainfall amount | Wind speed | River flooding | Storm surge | Location context |
|---|---|---|---|---|---|
| **US · STORM** | #3 preferred → #2 fallback | #4 | #5 | gap (future) | #6 + #7 |
| **US · FLOOD** | #3 preferred → #2 fallback | #4 (peripheral) | #5 | n/a | #6 + #7 |
| **Outside US · STORM** | #2 only | #4 | gap | gap (future) | #6 + #7 |
| **Outside US · FLOOD** | #2 only | #4 (peripheral) | gap | n/a | #6 + #7 |

Gaps are the honest scope limitations called out on the poster: storm surge worldwide, river flooding outside the US.

---

## Tools in pipeline order

### 1. `geocode(name)` — `geoguard/metadata.py:99`

**Source:** OpenStreetMap via Nominatim (the geocoder behind openstreetmap.org search).

**Signature:** `async def geocode(name: str) -> dict`

**Returns:** `{found, lat, lon, display_name}`

**What it verifies:** every downstream tool needs `(lat, lon)` — this is what resolves "Houston, TX" to `(29.7589, -95.3597)`. Runs during metadata extraction, *before* tool selection, so every downstream verification tool can take coordinates as inputs.

**Resolution:** point. Returns a single (lat, lon) corresponding to the place's centroid (or a representative point for non-point features).

**Limitations:**
- **Same-name collisions.** "Springfield" resolves to one of dozens; the `GEOCODE_RULE` (`geoguard/metadata.py:119-128`) instructs the extractor to pass the fullest available place string ("Galveston, Texas, USA" rather than "Galveston") to disambiguate.
- **City-centroid bias.** "Houston, TX" returns a single point for a city ~50 km wide. Claims about a specific neighborhood get resolved to a representative centroid.
- **Rate limits.** Nominatim is run by volunteers and asks for ≤1 request/sec.
- **No spatial extent.** Returns a single point, no bounding box.

---

### 2. `get_historical_precipitation(lat, lon, start_date, end_date=None)` — `geoguard/tools/weather.py:13`

**Source:** Open-Meteo Historical Archive, serving ECMWF's **ERA5** global reanalysis.

**Returns:** `{daily_mm, total_mm, peak_daily_mm, dates, lat, lon, start_date, end_date, source}`

**What it verifies:** rainfall-volume claims worldwide. Good for "moderate rain over a region"; weaker for "extreme rainfall at a specific gauge."

**Resolution:** spatial ~9 km (ERA5 native grid); temporal daily (hourly native, aggregated by the tool).

**Limitations:**
- **Reanalysis, not observation.** ERA5 is the output of a numerical weather model run backward in time, constrained by observations. Where gauges were dense it tracks reality well; elsewhere the model interpolates.
- **Smooths intense rainfall.** Convective and orographic extremes are sub-grid; the docstring explicitly tells the agent to **treat ERA5 as a lower bound**.
- **UTC timezone.** Days are UTC, not local. For US locations this means a ~5–8 hour offset from a "local July 8."
- **No `found=False` path.** Open-Meteo always returns something — even for ocean points — so no failure signal for nonsense coordinates.

---

### 3. `get_radar_gauge_precipitation(lat, lon, start_date, end_date=None)` — `geoguard/tools/weather.py:144`

**Source:** Iowa Environmental Mesonet IEMRE, point-extracted from NOAA's operational precipitation products:
- **PRISM** — ~4 km grid, anchored to real gauges.
- **MRMS QPE** — ~1 km grid, weather-radar with real-time gauge correction.

**Returns:** `{found, dates, prism_daily_mm, prism_total_mm, mrms_daily_mm, mrms_total_mm, lat, lon, ...}` (or `{found: False, reason: ...}` outside US).

**What it verifies:** US point-rainfall claims with observation-anchored estimates from two independent methods.

**Resolution:** PRISM ~4 km / MRMS ~1 km; daily totals.

**Limitations:**
- **US-only.** Outside CONUS / Hawaii / Alaska returns `found=False`.
- **PRISM and MRMS disagree on purpose.** They use different methods, so they bracket the truth rather than agreeing. Docstring tells the agent how to interpret each.
- **MRMS underestimates warm-season convection** by ~20–40% (radar Z-R relationships diverge for warm-season raindrop distributions).
- **Unit conversion happens inside the tool.** IEMRE returns inches; we convert to mm at line 231.

---

### 4. `get_historical_winds(lat, lon, start_date, end_date=None)` — `geoguard/tools/weather.py:80`

**Source:** Open-Meteo Historical Archive (ERA5).

**Returns:** `{daily_max_speed_kmh, daily_max_gust_kmh, peak_speed_kmh, peak_gust_kmh, dates, lat, lon, ...}`

**What it verifies:** wind-speed and gust claims globally.

**Resolution:** spatial ~9 km; temporal daily peaks.

**Limitations:**
- **"Sustained" = 10-minute mean here.** That's the WMO convention. US NWS reports sustained winds as 2-minute means; NHC hurricane advisories use 1-minute. Definitions differ before any data does.
- **10 m elevation.** Standard surface-wind height. Not comparable to readings at tower height or aloft.
- **Smooths peaks.** Same grid-averaging issue as precipitation — hurricane eyewall and tornado-scale winds are sub-grid.
- **Gusts come from a parameterization** — ERA5 estimates gusts from turbulence statistics, not direct measurement. Often under-reports real gusts.

---

### 5. `get_streamflow_history(lat, lon, start_date, end_date=None, search_radius_km=30.0, max_gauges=5)` — `geoguard/tools/weather.py:252`

**Source:** USGS Water Services NWIS — daily mean discharge + the period-of-record annual peak streamflow history.

**Returns:** per-gauge `{site_no, name, distance_km, daily_cfs, daily_m3s, event_peak_cfs, event_peak_date, all_time_record_cfs, all_time_record_date, peaks_years_on_record, event_rank}` for up to `max_gauges` nearest gauges (or `{found: False}` outside US coverage).

**What it verifies:** river-flooding and "record flooding" claims; distinguishes overland (pluvial) flooding from river (fluvial) flooding.

**Resolution:** spatial = point measurements at gauges (density varies — dense in urban Houston, sparse in rural West); temporal = daily means + annual instantaneous peaks.

**Limitations:**
- **US only.** Outside the US returns `found=False`.
- **Daily means vs instantaneous peaks.** Two different time conventions in the same response. Annual peaks are instantaneous maxima; daily series are 24-hour means. Never compare across types — peaks will always look bigger than daily means.
- **Units.** Returns both cfs (US convention) and m³/s (1 cfs = 0.02831685 m³/s). Easy to drop a factor of 35 if you grab the wrong field.
- **`event_rank` interprets "record."** Rank 1 = new all-time record at that gauge; rank 15 of 50 = "record-breaking" is overstated.
- **Active gauges only.** `siteStatus=active` filter — decommissioned gauges are excluded, so very old historical claims may not return data.

---

### 6. `get_elevation(lat, lon)` — `geoguard/tools/geospatial.py:10`

**Source:** Open-Meteo Elevation API, backed by NASA SRTM-90m (Shuttle Radar Topography Mission, 2000).

**Returns:** `{elevation_m, source}`

**What it verifies:** physical-plausibility context — "low-lying Houston" (correct, ~10 m) vs "storm surge in Denver" (implausible, 1600 m inland).

**Resolution:** spatial ~90 m; single timepoint (Earth's surface doesn't change much on weather timescales).

**Limitations:**
- **Single point only.** No slope, no aspect, no surrounding terrain shape.
- **Canopy and buildings inflate readings.** Radar reflects off the top surface — dense forest can add 10–30 m.
- **Returns 0 over water.** No `found=False` path for "this is over water" — pair with `find_nearest_water_body` to interpret.
- **2000 vintage.** Manmade earthworks built after 2000 not represented (rare problem, but real for landfills / mining sites).

---

### 7. `find_nearest_water_body(lat, lon, search_radius_km=10.0)` — `geoguard/tools/geospatial.py:34`

**Source:** OpenStreetMap via the Overpass API.

**Returns:** `{found, distance_m, name, kind, source}` for the nearest river/stream/canal/lake/coastline.

**What it verifies:** location-context claims like "flooding along Buffalo Bayou" — confirms the named waterway exists within the search radius. Also useful in conjunction with `get_elevation` to interpret "0 m" results (ocean vs inland low spot).

**Resolution:** depends on OSM contributor density. US/Europe/Japan: well mapped. Other regions: sparser.

**Limitations:**
- **OSM coverage varies wildly by region.** A real river may not be mapped; the "nearest" may be a more distant feature.
- **Default radius 10 km.** A river 11 km away returns nothing.
- **Returns one feature.** Just the single closest one — multi-waterway environments (Houston has bayous + the Gulf + Galveston Bay) get reduced to one nearest.
- **Some features are unnamed.** OSM volunteers may have traced a drainage ditch but not named it; `name` returns `None`.
- **Coastline is a special case.** OSM coastline is the most complex feature class; Overpass can be slow or partial for coastal queries far from other water features.

---

## Cross-cutting patterns

These show up in every tool and are worth knowing as a contributor:

- **`@registry(EventType.X, ...)`** declares which event types a tool serves. Tool selection pre-filters by event type before the LLM picks — a cheap prior that shrinks the LLM's search space. See `geoguard/tools/registry.py`.
- **`@graceful_http`** wraps tools so HTTP failures and missing data become `{"found": False, "reason": "..."}` payloads instead of raised exceptions. The verification agent can reason about coverage gaps as evidence. See `geoguard/utils.py`.
- **Docstrings are part of the prompt.** Each tool's docstring describes args, returns, *and limitations* — and the tool selector LLM reads it. Writing a clear, limitations-aware docstring is the lever for getting the right tool picked.
- **Multiple tools for the same metric is intentional.** Precipitation has both ERA5 (global, smoothed) and PRISM+MRMS (US, observation-anchored). Different bias profiles, picked by the agent based on location.

---

## Adding a tool

Five steps:

```python
# geoguard/tools/my_tool.py
from geoguard.schemas import EventType
from geoguard.tools.registry import registry
from geoguard.utils import graceful_http

@registry(EventType.FLOOD, EventType.STORM)   # 1. declare event types
@graceful_http                                 # 2. graceful failure (optional)
async def get_high_water_marks(                # 3. async function, primitive params
    lat: float, lon: float, date: str
) -> dict:
    """High-water mark observations within 20 km of a point.

    Source: USGS Flood Event Viewer (US only).

    Use for verifying claims about post-event flood-extent surveys ...
    Limitations: US-only, sparse outside major events, ...

    Returns: dict with keys {found, marks, lat, lon, source}.
    """
    ...  # 4. implementation
    return {"found": True, ...}                # 5. structured return
```

That's it. Import the module (e.g. from `geoguard/tools/__init__.py`) so the decorator runs at import time, and the tool is part of the registry. The selector LLM picks it up automatically based on `EventType` + docstring fit.

Examples of the `@registry` decorator forms in `geoguard/tools/registry.py:42-46`:

```python
@registry(EventType.FLOOD, EventType.OTHER)   # tagged for multiple types
@registry(EventType.FLOOD)
@registry(EventType.OTHER)                    # always-on (OTHER is the catch-all)
```

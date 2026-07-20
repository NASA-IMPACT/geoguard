# STAC tools

GeoGuard can pull verification evidence out of STAC catalogs — standardized,
searchable archives of geospatial datasets — instead of relying only on
hand-wired, one-dataset tools. This document explains the moving parts in
`geoguard/tools/stac/`, how they fit the verification pipeline, and when to
reach for which one.

## STAC in 30 seconds

**STAC (SpatioTemporal Asset Catalog)** is a standard JSON API for describing
and searching geospatial data. Everything hangs off a three-level hierarchy:

```
Catalog                 the API root
 └─ Collection          one dataset ("landslides-imerg", "GPM_3IMERGDF")
     └─ Item            one dated, georeferenced granule of it
         └─ Asset       the actual file (usually a cloud-optimized GeoTIFF)
```

The spec's one search primitive: *"items in these collections intersecting
this bbox and this time range."* Any catalog speaking STAC is searchable with
the same code — which is why the endpoints here are injectable.

## Architecture

```
Verifier (pydantic-ai Agent)
   │  sees ONE flat tool, like any other
   ▼
find_veda_evidence            stac/agent.py — registered @registry(EventType.OTHER)
   │  claim + area + window in, compact StacEvidence dict out
   ▼
StacExplorer                  stac/agent.py — nested agent, own instructions/budget
   │  owns the messy search → judge → select → analyze loop
   ▼
3 primitives                  stac/veda.py — plain async httpx functions
   ▼
VEDA STAC API + titiler raster API
```

Two invariants hold at every layer:

- **Compaction.** The primitives never return raw API payloads (descriptions
  truncated, histograms dropped, lists capped), and only the small
  `StacEvidence` synthesis crosses back into the verifier. Raw catalog
  listings never enter the verifier's context.
- **Graceful degradation.** No-coverage returns `found: False` with a reason;
  HTTP failures return `{tool, error}` dicts; nothing raises into the calling
  agent. "The catalog has nothing for this claim" is a first-class answer.

## VEDA

[NASA VEDA](https://www.earthdata.nasa.gov/dashboard/) (Visualization,
Exploration and Data Analysis) is the first catalog wired in.
`openveda.cloud` runs the eoAPI stack: a STAC API for search plus
**titiler**, a raster API that reads cloud-optimized GeoTIFFs server-side.

Endpoints (overridable via `GEOGUARD_VEDA_STAC_URL` / `GEOGUARD_VEDA_RASTER_URL`):

| API | URL | Used for |
|---|---|---|
| STAC | `https://openveda.cloud/api/stac` | collection + item search |
| Raster | `https://openveda.cloud/api/raster` | server-side zonal statistics |

Properties that shaped the implementation:

- **Curated and sparse (~250 collections):** long time-series (LIS hydrology,
  NLDAS precipitation, EPA methane) plus event-specific "story" datasets
  (Hurricane Helene IMERG accumulation, Derna 2023 daily GPM, Beryl IR
  mosaics). Whether VEDA helps a claim depends on whether someone curated
  that event or variable — expect strong coverage for curated disasters and
  non-US locations, weak coverage where a claim needs an uncurated variable.
- **Data assets are non-public `s3://` URIs.** You cannot download the
  GeoTIFFs anonymously — but titiler reads them server-side, so the
  fetch+analyze step is a single HTTPS call, no rasterio, no credentials.
- **Catalog quirks** (all learned from live probing, all encoded in the tool
  docstrings): some collections expose zero items (`GPM_3IMERGDF`);
  composites are stamped at their period start, so widen date windows;
  the statistics API reports **no units** — read them from the collection
  description.

### The three primitives (`stac/veda.py`)

These are the sub-agent's internal tools; the verifier never sees them.
They are plain awaitables — usable directly from a notebook or script with
no LLM involved.

**`search_veda_collections(keywords, limit=15)` — "what datasets exist about X?"**
Fetches the full collection list once per endpoint (cached per process),
scores id/title/description against the keywords, returns compact records
(id, title, truncated description, bbox, temporal interval, asset keys).
Always the first call: item search needs a collection id, and the returned
extents are how wrong-time/wrong-place candidates get rejected cheaply.

**`search_veda_items(collection_id, start_date, end_date, bbox…, limit=10)` —
"does that dataset have data for my place and time?"**
Standard STAC `POST /search` scoped to one collection. Returns compact items
(id, datetime, bbox, data-asset keys). Zero items is a real catalog gap, not
an error — move to the next candidate.

**`get_veda_item_statistics(collection_id, item_id, bbox…, asset="cog_default")` —
"turn the best item into a number over my area."**
Sends the bbox as a GeoJSON polygon to titiler's statistics endpoint; the
server reads the COG and returns zonal stats (min/max/mean/median/std,
valid-pixel coverage), compacted. Low `valid_percent` means clouds/no-data
dominate — treat the numbers with caution.

Worked example (Derna, Libya dam-collapse flood):

```python
import asyncio
from geoguard.tools.stac.veda import (
    search_veda_collections, search_veda_items, get_veda_item_statistics,
)

cols  = asyncio.run(search_veda_collections("flood precipitation daniel derna"))
# → darnah-gpm-daily (GPM IMERG data of 2023 Medicane Daniel)
items = asyncio.run(search_veda_items("darnah-gpm-daily", "2023-09-05", "2023-09-22"))
# → cog_Darnah_Flood_IMERG_2023-09-11, ...
stats = asyncio.run(get_veda_item_statistics(
    "darnah-gpm-daily", "cog_Darnah_Flood_IMERG_2023-09-11",
    22.1391, 32.2648, 23.1391, 33.2648,
))
# → mean 147.1, max 541.9 (mm/day) over the Derna box on the flood date
```

Custom deployments: `build_veda_tools(stac_url, raster_url)` returns the same
three functions bound to explicit endpoints (any eoAPI-shaped deployment).

### The registered tool (`stac/agent.py`)

`find_veda_evidence(claim, start_date, end_date, lat/lon or bbox…)` is the
**only** VEDA surface the verifier sees. It hands the claim's context to
`StacExplorer` — a nested pydantic-ai agent whose toolset is the three
primitives — and returns a compact `StacEvidence` dict:

```python
{
  "found": true,
  "summary": "2-5 sentences with the actual numbers and how they bear on the claim",
  "collection_id": "...", "item_ids": [...],
  "statistics": {"mean": ..., "max": ...},
  "units": "...", "source": "...", "caveats": "..."
}
```

Behavior notes:

- **Registration:** `@registry(EventType.OTHER)` — in this registry, OTHER
  means *always-on candidate for every event type* (see
  `ToolRegistry.get_candidates`). The LLM tool selector still gates actual
  use per claim.
- **Budget:** the sub-agent runs under a soft tool-call budget
  (`GEOGUARD_STAC_AGENT_TOOL_CALLS_LIMIT`, default 8). Every tool result
  carries a `tool_calls_remaining` countdown; once spent, further calls
  return a "synthesize now" notice instead of data, so the run ends with a
  (possibly partial) synthesis instead of an exception. A hard
  `UsageLimits` backstop sits above it and is caught into a clean
  "no usable evidence" result.
- **Nesting is invisible:** the sub-agent's steps never surface as pipeline
  events and its catalog traffic never enters the verifier's message
  history; each call records as one ordinary tool call (~1–2 KB).
- Registration happens on import: `import geoguard.tools.stac.agent`
  (the `stac` package `__init__` deliberately imports nothing, so plain
  `import geoguard.tools.stac` has no side effects).

## When to use what

| You want to... | Use |
|---|---|
| Browse/discover what VEDA has on a topic | `search_veda_collections` |
| Check a dataset's coverage of a place/time | `search_veda_items` |
| Get numbers over an area from a known item | `get_veda_item_statistics` |
| "Find the best evidence for this claim" (agent context) | `find_veda_evidence` |

Rule of thumb: **primitives are for humans and scripts doing deliberate
exploration (free, no LLM); the registered tool is for agents.** Inside
GeoGuard, the verifier should only ever touch `find_veda_evidence` — the
iteration belongs out of its context.

## Configuration

| Setting (env var) | Default | Meaning |
|---|---|---|
| `GEOGUARD_VEDA_STAC_URL` | `https://openveda.cloud/api/stac` | STAC search endpoint |
| `GEOGUARD_VEDA_RASTER_URL` | `https://openveda.cloud/api/raster` | titiler statistics endpoint |
| `GEOGUARD_STAC_AGENT_REQUEST_LIMIT` | 12 | sub-agent LLM request cap |
| `GEOGUARD_STAC_AGENT_TOOL_CALLS_LIMIT` | 8 | sub-agent soft tool-call budget |

`StacExplorer` itself mirrors `Verifier` construction (`model`,
`model_api_key`, `reasoning_effort`, limits, `instructions`) and takes either
endpoint overrides (`stac_url`/`raster_url`) or a full toolset replacement
(`tools=`) — mutually exclusive.

## Extending to other catalogs

- **Same API shape (any eoAPI deployment — STAC search + titiler):** pass
  `stac_url`/`raster_url` to `StacExplorer`, or `build_veda_tools(...)` for
  the bare primitives.
- **Different shape (e.g. Planetary Computer's signed assets, CMR-STAC with
  no raster API):** supply `tools=` and `instructions=` — the exploration
  harness (budget, structured output, graceful degradation) is
  catalog-agnostic. The analyze step for such catalogs needs its own
  fetch/compute primitive (tracked conceptually under NASA-IMPACT/geoguard#5's
  raster machinery).

The plain, single-call registered STAC tool (search + analyze without the
sub-agent) remains NASA-IMPACT/geoguard#5's deliverable; this package's
primitives are deliberately unregistered internals until then.

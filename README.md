# GeoGuard

Pluggable guardrail framework for geospatial AI.

---

## What is GeoGuard?

GeoGuard wraps any upstream system that emits geographically-grounded text — foundation-model inference, agentic workflows, hand-authored copy — and audits its factual claims against authoritative external data sources. It does not modify the upstream model. It decomposes the output into atomic claims, picks tools per claim, runs verification with those tools, scores the result with a holistic rubric, and emits a structured report with verdicts, evidence, and a confidence score.

```
| upstream | → output → | GeoGuard | → Report → | consumer |
```

---

## Installation

The project ships with a `uv.lock` for reproducible installs:

```bash
git clone <repo-url>
cd geoguard
uv sync
```

(Not on PyPI yet.)

---

## Configuration

GeoGuard auto-loads a `.env` file in the project root. All settings are optional except the OpenAI key:

```bash
# .env

OPENAI_API_KEY=sk-...                     # required (used by the OpenAI SDK)

# Optional — override defaults
GEOGUARD_MODEL=openai:gpt-5.2
GEOGUARD_REASONING_EFFORT=medium          # minimal | low | medium | high
GEOGUARD_MAX_CLAIMS=15
GEOGUARD_QUESTIONS_PER_CLAIM_MIN=5
GEOGUARD_QUESTIONS_PER_CLAIM_MAX=10
GEOGUARD_HTTP_TIMEOUT_SECONDS=30
GEOGUARD_VERIFICATION_TOOL_USAGE_LIMIT=7
```

`GeoGuard.from_config()` instantiates the pipeline with these settings.

---

## Quickstart

A runnable example using free public APIs (OpenStreetMap Nominatim + Open-Meteo Historical Archive — no API keys):

```python
import asyncio
import httpx

from geoguard import GeoGuard, Input
from geoguard.schemas import EventType
from geoguard.tools import registry


@registry(EventType.OTHER)
async def geocode_location(name: str) -> dict:
    """Resolve a place name to lat/lon via OpenStreetMap Nominatim."""
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": name, "format": "json", "limit": 1},
            headers={"User-Agent": "geoguard-quickstart"},
        )
        results = r.json()
    if not results:
        return {"found": False, "name": name}
    first = results[0]
    return {"found": True, "lat": float(first["lat"]), "lon": float(first["lon"])}


@registry(EventType.FLOOD)
async def query_historical_precipitation(lat: float, lon: float, date: str) -> dict:
    """Total daily precipitation (mm) at lat/lon on date (YYYY-MM-DD)."""
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": date,
                "end_date": date,
                "daily": "precipitation_sum",
                "timezone": "UTC",
            },
        )
        data = r.json()
    return {
        "precipitation_mm": (data.get("daily", {}).get("precipitation_sum") or [None])[0]
    }


async def main():
    guard = GeoGuard.from_config()
    report = await guard.run(
        Input(
            text=(
                "Hurricane Beryl made landfall near Galveston, Texas on July 8, 2024, "
                "with heavy rainfall causing widespread flooding in Houston (over 100 "
                "mm of rain in 24 hours)."
            )
        )
    )

    print(f"Overall verdict: {report.overall_verdict}")
    print(f"Confidence:      {report.rubric.confidence:.0%}")
    for vr in report.verifications:
        print(f"  • {vr.verification.claim.claim[:80]}")
        print(f"    → {vr.verification.verdict} — {vr.verification.rationale[:120]}")


asyncio.run(main())
```

For live progress (e.g., a UI), iterate the streaming form instead:

```python
from geoguard.claims import Claim
from geoguard.metadata import ClaimGroup
from geoguard.pipeline import Report
from geoguard.rubrics import Rubric
from geoguard.tools.selector import SelectedTools
from geoguard.verifications import VerifierResult

async for event in guard.stream(Input(text="...")):
    match event:
        case ClaimGroup():     ui.show_group(event)
        case Claim():          ui.show_claim(event)
        case SelectedTools():  ui.show_tools(event)
        case VerifierResult(): ui.show_verdict(event)
        case Rubric():         ui.show_confidence(event)
        case Report():         ui.show_summary(event)
```

---

## What you get back

Every `guard.run(input)` returns a `Report`:

| Field | Contents |
|---|---|
| `input` | the original input |
| `verifications: list[VerifierResult]` | one per claim — verdict + rationale + full tool-call trace (name, args, returned data) |
| `rubric: Rubric` | dynamically generated yes/no rubric items per claim, per-claim scores, and an overall confidence value |
| `overall_verdict: Verdict` | rolled-up: `supports`, `contradicts`, or `inconclusive` |

Sample text output (compressed):

```
Overall verdict: supports
Confidence:      72%
  • Hurricane Beryl made landfall near Galveston, Texas on July 8, 2024
    → supports — Geocoded "Galveston, Texas" to (29.30, -94.80); place
      verified. Historical precipitation at the location on 2024-07-08
      returned 42 mm, consistent with a storm-related rainfall event.
  • Houston received over 100 mm of rain in 24 hours on July 8, 2024
    → supports — Tool returned 102 mm precipitation_sum for Houston
      (29.76, -95.37) on 2024-07-08.
```

---

## Extending the pipeline

The framework is fully tool-extensible. **Adding a check is one decorator on one function** — the orchestration discovers it automatically. No schema changes, no orchestration edits.

```python
from geoguard.tools import registry
from geoguard.schemas import EventType


# A tool is an async function with primitive params and a clear return value.
# Its signature + docstring are read by the agentic selector to decide
# whether the tool fits a given claim.

@registry(EventType.FLOOD)
async def query_river_gauge(basin: str, datetime: str) -> dict:
    """Latest water level (m) for the named river basin at the given datetime."""
    ...


@registry(EventType.OTHER)                              # always-on (any event)
async def query_eonet_events(category: str, days: int = 30) -> dict:
    """List active events from NASA EONET in the given category over the last N days."""
    ...


@registry(EventType.FLOOD, EventType.OTHER)             # multi-event registration
async def fetch_news_articles(query: str, since: str) -> dict:
    """News articles matching `query` since `since` (YYYY-MM-DD)."""
    ...
```

The moment you import the module that decorates these functions, they're available to the pipeline — pre-filtered by event type, picked by the agentic selector when relevant to a claim, attached to the per-claim verifier, and surfaced in tool-call traces and rubric reasoning.

A few rules of the road:

- Tools are **domain-pure**: their parameters are primitives (lat, lon, datetime, place name, basin, etc.) and their return is a `dict`. They have **no `Claim` / `Metadata`** in their signatures — the agentic verifier translates `(claim, metadata) → (tool args)`.
- Tools registered under `EventType.OTHER` are always-on — they appear in every event type's candidate pool.
- Class-based tools work the same — instantiate, then `registry(EventType.X)(instance)`.
- The selector reads the docstring (first line) and the full signature, so write descriptive docstrings and use specific parameter names / types.

A small registered set can verify flood claims comprehensively: geocoding, precipitation, wind, river gauges, tide gauges, DEMs, historical disaster databases, news APIs. Each is one decorated function.

---

## Documentation

- **[Architecture](docs/architecture.md)** — full pipeline diagram, per-block contracts, streaming events, type reference, extension points
- **[ESA-NASA 2026 poster](docs/poster-esa-nasa-workshop-II.md)** — high-level framing for the workshop session

---

## License

[Apache License 2.0](LICENSE).

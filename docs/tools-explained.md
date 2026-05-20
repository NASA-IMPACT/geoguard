# Tools, explained — for ML / NLP engineers

This document is a companion to [`registered-tools.md`](registered-tools.md). The catalog gives you the *what*; this gives you the *why* — the mental model, the resolution-and-coverage reasoning the verifier agent does at runtime, and a worked example. Audience: an ML or NLP engineer comfortable with agentic frameworks and RAG, new to weather and hydrology data.

For the architectural context, see [`architecture.md`](architecture.md).

---

## Mental model — typed RAG, for fact-checking

A normal RAG system is `text → dense retrieval → text`. Each retrieval returns a passage that might or might not be relevant.

GeoGuard's verification step is the same shape, but each "retrieval" is a **typed API call into a specialized geo database**. The tool is a function with a clear signature; the result is a structured payload with units, dates, and source attribution. The verifier agent's job is to translate a free-text claim into the right API calls and then synthesize a verdict.

You can think of every tool as one slice of a giant relational schema over Earth:

```
WHERE → location resolver (geocode)
WHEN  → expressed as a date range in the call
WHAT  → which axis: precipitation, wind, streamflow, elevation, water-body
WHICH → which source: gauge-anchored, model-derived, observed, volunteered
```

A free-text claim almost always implies a `(WHERE, WHEN, WHAT)` triple. Resolving it is the verifier agent's structured-prediction task.

---

## The orthogonal axes

No single database covers all the physical quantities a weather event generates. The tools partition the verification problem along independent physical axes:

| Axis | Example claim | Tools |
|---|---|---|
| **Where (resolution)** | "Houston, TX" → coords | `geocode` |
| **How wet (volume)** | "100 mm of rain in 24 h" | `get_radar_gauge_precipitation` (US), `get_historical_precipitation` (global) |
| **How windy** | "sustained winds of 80 mph" | `get_historical_winds` |
| **Did the river flood** | "Whiteoak Bayou peaked at 27,800 cfs" | `get_streamflow_history` |
| **Could it flood (context)** | "low-lying area near a bayou" | `get_elevation` + `find_nearest_water_body` |

Notice three things about this table:

1. **Some axes have multiple tools.** Precipitation has two — they have different bias profiles, and the agent picks based on the claim's geography.
2. **Some axes have none.** Storm surge has no tool. Non-US river flooding has no tool. These are honest gaps, called out on the poster's limitations footer.
3. **Some "tools" are only useful in combination.** Elevation alone proves little; elevation + nearest-water gives a plausibility signal.

---

## Resolution primer — two axes, both matter

This is the geo concept ML folks consistently underestimate.

### Spatial resolution = patch size

Same idea as a Vision Transformer's patch size. Smaller patches resolve finer features; bigger patches average across them.

| Patch size | ML analogue | Weather tool |
|---|---|---|
| 1 km | ViT-S/16 on a 224×224 image (~14 px per face) | MRMS QPE |
| 4 km | ViT-B/32 on a smaller image | PRISM |
| 9 km | ViT-L/16 on a low-res image | ERA5 |
| 90 m | very fine-grained encoder | SRTM-90m elevation |
| ~point | retrieval of a single embedding | USGS gauges (sparse but exact) |

ERA5 at 9 km cannot distinguish a thunderstorm cell that hits one Houston suburb and misses the next. MRMS at 1 km can. PRISM at 4 km splits the difference. The agent has to know this before it picks a tool — and that knowledge lives in the tool docstrings.

### Temporal resolution = time-window granularity

Independent of spatial resolution.

| Tool | Temporal native | What we expose |
|---|---|---|
| ERA5 (precipitation, wind) | hourly | daily totals / daily peaks |
| PRISM | daily | daily |
| MRMS | 5–15 min | daily (via IEMRE aggregation) |
| USGS NWIS daily values | daily mean | daily |
| USGS annual peaks | once per year (instantaneous) | annual peaks |

**The hidden hazard:** a claim of "100 mm in 24 hours" maps neatly to daily totals only if the 24-hour window aligns with the calendar day boundary in our query timezone. ERA5 uses UTC; PRISM/MRMS use local-day. A storm that drops 100 mm centered on midnight gets split across two daily buckets. The verifier agent has to reason about this — and the docstrings flag it where it matters.

---

## Per-tool deeper dive

Each subsection here adds *why* — the physical reasoning the catalog page doesn't. For signatures, return shapes, and limitation bullets, see [`registered-tools.md`](registered-tools.md).

### `geocode` — the one tool that runs *before* verification

It's not a verification tool. It runs during metadata extraction. Every downstream tool consumes `(lat, lon)`, so resolving place names happens once at the top of the pipeline.

The interesting design choice: a hard prompt rule (`GEOCODE_RULE` in `geoguard/metadata.py:119-128`) forbidding the LLM from producing coordinates from parametric memory. The extractor *must* call `geocode`, *must* pass the fullest place string. This guards against the known LLM failure mode of confidently inventing coordinates from training data — a particularly bad failure for a guardrail system to inherit.

### `get_historical_precipitation` (ERA5) — the global default

ERA5 is a **reanalysis**. That word matters: it's not a measurement, it's the output of a numerical weather model run backward in time, constrained by observations. Where gauges and satellites were dense, ERA5 closely tracks reality; where they were sparse, the model carries the answer.

The implication is that ERA5 systematically **smooths** local extremes. A real rain gauge might report 200 mm at a point during a convective storm; ERA5 at the same point might say 80 mm because that point is inside a 9 km cell whose average over land contained much less rain. The docstring tells the agent explicitly: **treat ERA5 as a lower bound.**

Why we still use it: it's the only option outside the US, and even inside the US it provides a cross-check.

### `get_radar_gauge_precipitation` (PRISM + MRMS) — the US-only specialist

The two products here are intentionally different:

- **PRISM** is a daily product from the PRISM Climate Group, anchored to physical rain gauges and spatially interpolated with terrain-aware methods. ~4 km grid. Gauge-truth at coarse resolution.
- **MRMS QPE** is a NOAA radar product at ~1 km, derived from the NEXRAD radar network's reflectivity through the Z-R relationship, then *gauge-corrected in real time* using gauges that report within the hour.

The tool returns both. For the same point on the same day they will differ — often by 20–40%. That's the feature: the agent gets bracketed estimates from two methods that fail in different ways. Convection is the canonical example — MRMS underestimates it (radar Z-R relationships diverge for warm-season raindrop distributions), PRISM more reliably reflects gauge totals.

If you're inside the US, **prefer this over ERA5.** The tool selector knows this from the docstring framing. Outside the US the tool returns `found=False` and the agent falls back to ERA5.

### `get_historical_winds` (ERA5) — the global wind archive

Same data source as #2, different variable. Daily peak sustained wind and daily peak gust, both at 10 m above the surface.

The subtle trap: "sustained wind" doesn't have one definition. ERA5 (following WMO convention) reports peak **10-minute** mean wind. US NWS reports sustained winds as **2-minute** means. NHC hurricane advisories use **1-minute** means. They're all called "sustained," but the averaging window changes the value — a longer average smooths peaks more.

For the Beryl claim "80 mph sustained," the data the tool returned (around 65 km/h ≈ 40 mph 10-min sustained at Matagorda) was lower than 80 mph, but the daily peak gust (~104 km/h ≈ 65 mph) was much closer. Suggests the upstream system reported a gust value as if it were sustained — a *qualitative* misclassification.

### `get_streamflow_history` (USGS NWIS) — the hydrology specialist

Floods aren't all the same physical phenomenon. There's:

- **Pluvial / urban flooding** — rain falls faster than the ground or storm drains can absorb. Follows rain within hours.
- **Fluvial / river flooding** — rivers rise above their banks. Can persist days after the rain stopped, often downstream of where it fell.

Precipitation tools alone can't distinguish these. Streamflow data can. The tool returns daily mean discharge plus the gauge's all-time annual peak record. The `event_rank` field tells the agent how the event ranks across the gauge's history: rank 1 = new record, rank 15 of 50 = "record-breaking" is overstated.

Three subtle conventions the docstring flags:

1. **Daily means ≠ instantaneous peaks.** They're both in the response but they're different metrics. Don't compare daily means to peaks — peaks are always bigger.
2. **cfs vs m³/s.** US flood narratives use cfs; the metric world uses m³/s; the conversion factor is ~35. Tool returns both.
3. **Bounding box → distance filter.** The query uses a lat/lon box, then filters by geodesic distance. Cheap approximation but the bbox-to-circle mismatch grows at high latitudes.

US-only. Outside the US returns `found=False`.

### `get_elevation` (SRTM-90m) — the topographic context tool

Single value, single point. Cheap. Almost never *directly* verifies a claim but powerfully informs plausibility:

- "Storm surge in Denver" → elevation 1600 m, 1500 km from any coast → implausible.
- "Flooding in low-lying Houston" → elevation ~10 m, coastal → plausible.

The data is from the Space Shuttle Radar Topography Mission flown in February 2000 — radar interferometry with a 60 m baseline mast. 90 m grid. Caveats: radar reflects off canopy and buildings (so forest and urban areas read 10–30 m too high), and the dataset is from 2000 (manmade earthworks built since aren't represented).

### `find_nearest_water_body` (OSM Overpass) — the named-entity check for geography

Closest geo-NLP analogue is named-entity verification: does the geographic entity referenced in the claim actually exist where the claim places it? "Buffalo Bayou near Houston" — call the tool, confirm the named waterway is within the search radius.

OSM coverage varies dramatically: US, Europe, Japan are well mapped; large parts of Africa and South Asia are patchy. The tool returns the single nearest feature, so multi-water environments (Houston has bayous, the Gulf, and Galveston Bay all within ~50 km) get collapsed to one nearest.

Pair this tool with elevation to interpret edge cases — a `0 m` elevation reading is either ocean or a coastal low spot, and the water-body proximity disambiguates.

---

## Cross-cutting patterns

The tools share four design decisions worth understanding as a contributor.

### `@registry(EventType.X, ...)` — coarse-to-fine retrieval prior

Every tool declares which event types it serves (`STORM`, `FLOOD`, `OTHER`). The tool selector pre-filters by event type before invoking the LLM picker. The LLM sees only the tools matching the claim's event type plus always-on `OTHER` tools.

Why this matters: the LLM's job is now "pick from 5–7 candidates," not "pick from all tools ever registered." Cheaper, faster, fewer mis-selections. Exactly the same pattern as a coarse-then-fine retriever in RAG.

### `@graceful_http` — failures as evidence

This decorator wraps tools so HTTP errors and missing data return `{"found": False, "reason": "..."}` instead of raising exceptions. The verification agent treats absence as a kind of evidence:

> "I called the radar+gauge tool, it returned `found=False, reason: outside US coverage`. Fell back to ERA5."

That graceful degradation is structural, not learned — the agent doesn't need to be trained to handle outages; it just sees a structured "no data" response and reasons about it.

### Docstrings as prompt context

This is the under-celebrated piece. Each tool's docstring describes args, returns, *and limitations* — and the tool selector LLM ingests the docstrings as part of its prompt. The docstring is effectively a few-shot exemplar about when (not) to call the tool.

When the ERA5 docstring says "treat as a lower bound," the selector internalizes that — and when an upstream claim asserts an extreme value, the selector will prefer the US-specialist tool when available. The limitation is the prompt.

Practical implication: **write your docstrings like prompt-engineering artifacts.** Reflect real biases, real failure modes, real units. The LLM is reading them.

### Multiple tools per metric — intentional bias diversification

Precipitation has two tools (ERA5 reanalysis + PRISM/MRMS observation-anchored). They have different bias profiles:

- ERA5: smooths extremes, global coverage.
- PRISM/MRMS: observation-anchored, US-only, MRMS underestimates convection.

For a US claim, both can be called, and the agent triangulates. For a non-US claim, ERA5 is the only option but the agent knows to interpret its values as a lower bound. This is the same idea as ensemble methods in ML: combine estimators with different bias profiles to reduce error.

---

## Worked example — verifying the Houston rainfall claim

The poster's case study leads with Hurricane Beryl. Here's the trace for *one* of its claims:

> **Claim:** "Houston received >100 mm of rain on 2024-07-08."

**Step 1 — During metadata extraction**

```
geocode("Houston, TX, USA")
  → {found: true, lat: 29.7589, lon: -95.3597, display_name: "Houston, Harris County, Texas, USA"}
```

The full place string is passed to disambiguate from Houston BC, Houston Mississippi, etc.

**Step 2 — Tool selection (per-claim, parallel with other claims)**

The selector sees:
- Claim: "Houston received >100 mm of rain on 2024-07-08."
- Metadata: `event_type=FLOOD`, location resolved.
- Candidate tools (pre-filtered to FLOOD-compatible): `get_historical_precipitation`, `get_radar_gauge_precipitation`, `get_streamflow_history`, `get_elevation`, `find_nearest_water_body`.

LLM reasoning (excerpted from the trace):
> The claim concerns total rainfall at a US location. `get_radar_gauge_precipitation` is the observation-anchored US specialist; ERA5 would smooth a convective event of this intensity. Selecting `get_radar_gauge_precipitation`.

**Step 3 — Verification**

```
get_radar_gauge_precipitation(lat=29.7589, lon=-95.3597, start_date="2024-07-08")
  → {found: true,
     prism_total_mm: 129,
     mrms_total_mm: 174,
     dates: ["2024-07-08"],
     source: "Iowa Environmental Mesonet IEMRE (NOAA PRISM + MRMS)"}
```

Agent synthesis:
> Claim asserts >100 mm. PRISM observed 129 mm; MRMS observed 174 mm. Both observation-anchored sources exceed the claimed threshold. **Verdict: SUPPORTS.**

**Step 4 — Rubric (after all claims verified)**

The holistic Rubricator generates per-claim yes/no checks. For this claim, items might include: "Is the location US-coverage for the observation product?" (yes), "Do both PRISM and MRMS exceed the claimed value?" (yes), "Is the date range correctly aligned to the event?" (yes). Score: 86% (5 of 6 items yes — one ambiguous on attribution to Beryl specifically).

**Step 5 — Final report**

This claim's verdict, rationale, raw tool-call payloads, and rubric score all roll up into the final `Report` for downstream consumption.

Total cost for this claim: one LLM call for selection, one agent loop for verification (small, since one tool call was enough), one rubric question batch. About 5–7 seconds wall-clock on GPT-4.1-mini.

---

## Extending the toolkit

What's missing today is just what hasn't been built yet — the registry pattern makes each new tool a small, isolated addition.

Natural next additions, with their data sources:

| Capability | Source | Why |
|---|---|---|
| Hurricane Best Tracks | NOAA HURDAT2 | Verify landfall coordinates, peak intensity, track timing |
| Global precipitation | NASA GPM IMERG | Complement ERA5 outside the US — observation-anchored, half-hourly |
| Disaster catalogs | GDACS, NASA EONET, EM-DAT, ReliefWeb | Cross-validate that a claimed event was officially reported |
| Storm surge | NOAA Tides & Currents | The unfilled axis in the coverage matrix |
| Wildfire perimeters | NIFC, MTBS | Open the WILDFIRE event type |
| Flood extents | Copernicus EMS, USGS Flood Event Viewer | Verify spatial extent claims, not just point values |
| Earthquake catalog | USGS Earthquakes | New event type with a single canonical authoritative source |

For each, the recipe is the same: one async function, one `@registry(EventType.X)`, one well-written docstring. No orchestration changes, no schema churn.

The architectural payoff worth landing in any conversation about extending GeoGuard: **one well-documented tool per orthogonal evidence axis.** That's the unit of extension, and it's small.

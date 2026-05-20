# GeoGuard — An Agentic Guardrails & Validation Framework for Geospatial AI

**Nishan Pantha¹ · Rohit Sahoo¹ · Sanjog Thapa¹ · Muthukumaran Ramasubramanian¹ · Rahul Ramachandran²**

¹ The University of Alabama in Huntsville · ² NASA Marshall Space Flight Center

Contact: np0069@uah.edu · rs0214@uah.edu

---

## 01 · Motivation

> **Can we trust the output before a decision maker sees it?**

Geospatial AI is moving from research into operations. Authoritative data to verify claims like precipitation, flood extents, and storm tracks often exists, but verification is manual, ad-hoc, and doesn't scale. Guardrails have become standard for LLMs, but no automated equivalent sits between geo-AI output and the decisions that depend on it.

**GeoGuard.** A model-agnostic validation layer that wraps any upstream geospatial AI system and audits its output against authoritative external data.

**Upstream can be:**
- LLMs and agentic geo-AI workflows
- Foundation-model predictions
- Hand-authored geo-text

**Properties:**
- ◆ **Model-agnostic** — wraps any upstream system
- ◆ **Transparent** — every step exposes claims, tool calls, evidence
- ◆ **Extensible** — tools registered per geospatial event type
- ◆ **Confidence-scored** — categorical verdict + holistic rubric

---

## 02 · As a black box

```
        ┌───────────────────────────┐
        │     external producer     │
        │   (FM · agent · human)    │
        └──────────────┬────────────┘
                       │  output (text ± images)
                       ▼
        ╔═══════════════════════════╗
        ║         GeoGuard          ║
        ║  decompose · verify ·     ║
        ║  score · report           ║
        ╚══════════════┬════════════╝
                       │  Report (verdicts + evidence + confidence)
                       ▼
        ┌───────────────────────────┐
        │   downstream consumer     │
        │  (gate · retry · log)     │
        └───────────────────────────┘
```

GeoGuard wraps any geo-AI producer and validates output through external tools backed by authoritative data — geocoders, weather archives, precipitation records, USGS river gauges. New tools can be registered to extend the verification process.

---

## 03 · Pipeline

Three agentic decisions: **what claims are in the input**, **which tools to use per claim**, and **how to interpret tool evidence into a verdict**. Per-claim work runs in parallel: fan-out at extraction, re-converge at rubric.

```
   Input: text (± images) from upstream
                 │
                 ▼
   ┌────────────────────────────────────────┐
   │ 01 · Claim + metadata extraction       │  LLM (one structured-output call)
   │  decomposes output into atomic,        │
   │  decontextualized claims               │
   └─────────────────┬──────────────────────┘
                     │  list of (claim, metadata) groups
                     ▼
   ┌────────────────────────────────────────┐
   │ 02 · Tool selection · PER-CLAIM        │
   │      PARALLEL                          │
   │  pre-filter by event type,             │
   │  LLM picks tools by signature          │
   └─────────────────┬──────────────────────┘
                     │  chosen tools + reasoning
                     ▼
   ┌────────────────────────────────────────┐
   │ 03 · Verification · PER-CLAIM PARALLEL │
   │  agent loop calls tools and            │
   │  synthesises verdict                   │
   └─────────────────┬──────────────────────┘
                     │  per-claim verdicts + tool-call traces
                     ▼
   ┌────────────────────────────────────────┐
   │ 04 · Rubric                            │  LLM (one holistic pass)
   │  per-claim yes/no checks, aggregated   │
   └─────────────────┬──────────────────────┘
                     │  rubric items + overall score
                     ▼
   ┌────────────────────────────────────────┐
   │ 05 · Final report                      │  typed output
   │  verdicts · evidence · rubric ·        │
   │  overall confidence                    │
   └────────────────────────────────────────┘
```

---

## 04 · Block details

### Block 01 — Claim + metadata extraction

- **In:** raw text (± images)
- **Does:** decomposes into atomic, decontextualized claims + structured event metadata, in one call
- **Out:** (claim, metadata) groups

Claims are **atomic · faithful · distinct**. *Example:* "Hurricane Beryl made landfall…" → STORM (3 claims) · FLOOD (2 claims).

### Block 02 — Tool selection (per-claim parallel)

- **In:** one claim + metadata
- **Does:** pre-filter by event type, then LLM picks tools by signature + docstring
- **Out:** chosen tools + reasoning

*Example:* "sustained winds of 80 mph" → [`get_historical_winds`].

### Block 03 — Verification (per-claim parallel)

- **In:** claims + tools
- **Does:** agent loop decides what to call, with what args, and how to interpret. Never sees the original input — only the decontextualized claim and the tool outputs.
- **Out:** verdict + rationale + tool-call trace

*Example:* winds claim + tool → ❌ **CONTRADICTS** (65–97 km/h across TX coast vs claimed ~129 km/h).

### Block 04 — Rubric

- **In:** input · verifications · traces
- **Does:** per-claim yes/no checks answered from existing evidence — no new tool calls
- **Out:** items + scores + overall confidence

*Example:* 5 verified claims → winds 71% · rain 86% · landfall 57% · … → 67% overall.

### Block 05 — Final report

- **In:** all of the above
- **Does:** rolls per-claim verdicts → one overall verdict; packages a typed report
- **Out:** downstream can gate · retry · surface · log

*Example:* 5 verdicts + rubric → ❌ **CONTRADICTS · 67%** → 1 refuted · 1 supported · 3 inconclusive.

Verdict rollup: any **CONTRADICTS** → CONTRADICTS · all **SUPPORTS** → SUPPORTS · else **INCONCLUSIVE**.

---

## 05 · Case study — Hurricane Beryl

### Input

> *"Hurricane Beryl made landfall near Matagorda Bay, Texas on July 8, 2024. The storm brought sustained winds of 80 mph and significant storm surge along the Texas coast. Heavy rainfall caused widespread flooding in Houston, with reports of over 100 mm of rain falling over a 24-hour period."*

Source: upstream geospatial AI system. Event: FLOOD · STORM.

### GeoGuard trace

**Claims**

- 📦 **STORM · Matagorda Bay · 2024-07-08** · landfall · ~80 mph sustained winds · significant storm surge
- 📦 **FLOOD · Houston · 2024-07-08** · widespread flooding · >100 mm rain in 24 h

**Tools selected**

`place-name resolver` · `historical wind` · `historical precipitation` · `elevation` · `nearest-water body` · `radar+gauge precipitation` · `streamflow history`

**Sample tool calls**

```
resolve("Houston, TX")           → 29.7589, -95.3597
radar_gauge(Houston, 2024-07-08) → PRISM 129 mm · MRMS 174 mm
streamflow(Houston, 2024-07-08)  → Whiteoak Bayou peak 27,800 cfs
winds(Galveston, 2024-07-08)     → 96.6 km/h sustained · 139.7 km/h gust
winds(Matagorda Bay, 2024-07-08) → 65 km/h sustained · 104 km/h gust
```

### Final report

**STORM · Matagorda Bay · 2024-07-08**

| Claim | Confidence |
|---|---|
| Hurricane Beryl landfall | 57% |
| sustained winds of 80 mph | 71% |
| significant storm surge | 57% |

**FLOOD · Houston · 2024-07-08**

| Claim | Confidence |
|---|---|
| widespread flooding | 62% |
| over 100 mm of rain in 24 h | 86% |

**Overall confidence: 67% · Verdict: ❌ CONTRADICTS**

The "sustained 80 mph" claim is refuted by the wind archive (peak sustained 65–97 km/h across the Texas coast — well below the implied ~129 km/h). Plausibly a gust mislabeled as sustained. The rainfall claim is supported (PRISM 129 mm, MRMS 174 mm both exceed the claimed 100 mm).

---

## 06 · Benchmark

**65 flood events · NOAA Storm Events 2024**

50 verified events drawn from the NOAA NCEI Storm Events Database, plus 15 fabricated counterparts produced by perturbing real events (wrong date · wrong location · inflated magnitude). Goal: measure both endorsement of real events and rejection of fabricated ones.

|  | SUPPORTS | CONTRADICTS | INCONCLUSIVE |
|---|---|---|---|
| **Verified events (50)** | 41 | 4 | 5 |
| **Fabricated events (15)** | 0 | 14 | 1 |

**Headline results**

- **92%** of verified events not rejected (41 SUPPORTS + 5 INCONCLUSIVE out of 50)
- **0%** of fabricated events endorsed (no SUPPORTS verdicts on the 15 fabricated counterparts)
- Of 15 fabricated events, 14 were correctly refuted as CONTRADICTS and 1 returned INCONCLUSIVE

**Cost / latency**

GPT-4.1-mini · ~$0.05 per event · ~35 seconds per event end-to-end.

---

## 07 · Validation check categories

The framework treats each validation check as a registered tool. The taxonomy below maps to what we demonstrate today vs. what slots in next.

| Category | What it answers | Status |
|---|---|---|
| **Meteorological** | Did supporting weather conditions actually occur? (rainfall, wind) | ✅ demonstrated — `get_historical_precipitation`, `get_radar_gauge_precipitation`, `get_historical_winds` |
| **Location resolution** | Does the named place exist, and where? | ✅ demonstrated — `geocode` |
| **Physical plausibility** | Is the prediction physically reasonable given terrain? | 🟡 partial — `get_elevation`, `find_nearest_water_body` registered; slope / land-cover future |
| **Event corroboration** | Do authoritative monitors confirm the event? | 🟡 partial — `get_streamflow_history` (US river flooding) registered; GDACS / EONET / EM-DAT future |
| **Historical contextualization** | Does the location have a documented history of similar events? | ⚪ future |
| **Cross-reference** | Does an independent model or reference product agree? | ⚪ future |
| **Temporal consistency** | Do prior / subsequent observations support the prediction? | ⚪ future |

Adding a new check is a **single async function registered against an event type** — no orchestration changes, no schema churn.

---

## 08 · References & future work

**References**
- *Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents*
- *Mitigating Geospatial Knowledge Hallucination in Large Language Models* (GeoHaluBench + DynamicKTO)
- *Towards LLM Agents for Earth Observation*
- *Empowering LLM Agents with Geospatial Awareness for Wildfire Response*

**Future work**
- Real-time disaster catalogs as registered tools — NASA EONET, GDACS, EM-DAT, ReliefWeb
- Global precipitation archive — NASA GPM IMERG to complement ERA5 outside the US
- Non-US streamflow / river-gauge equivalents (GloFAS, regional hydromet agencies)
- Tropical cyclone Best Tracks (HURDAT2, IBTrACS) for hurricane landfall verification
- Adapter for raster-input FMs (segmentation masks, flood polygons) — extends the input contract to carry geo-rasters; tools verify per-pixel claims against DEM, land cover, historical event extents
- Calibration study — rubric confidence vs. human-annotated ground truth

---

## 09 · Resources

> **[ GitHub Repo — QR ]**   **[ HF Space Demo — QR ]**
>
> *On the printed poster, two QR codes link to the open-source repository and the live Hugging Face Space demo.*

**Data sources:** OpenStreetMap Nominatim · Open-Meteo (ERA5) · OSM Overpass · NOAA PRISM/MRMS · USGS NWIS

**Limitations:** flood / storm only · limited tool set (extensible) · US-only for radar/gauge & streamflow · text claims only

---

*The material contained in this document is based upon work supported by a National Aeronautics and Space Administration (NASA) grant or cooperative agreement. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author and do not necessarily reflect the views of NASA.*

*This work is supported by NASA Grant 80MSFC22M004.*

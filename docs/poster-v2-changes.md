# Poster v2 — changes from the ESA–NASA workshop poster

Same layout, same six sections. Edits are listed per section; anything not
listed stays as is. **[RUN]** marks numbers that come from tonight's run.

---

## Header

- **Authors** (match the AAI4EO abstract): Nishan Pantha¹\*, Rohit Sahoo¹\*,
  Sanjog Thapa¹, Nidhi Jha¹, Ajinkya Kulkarni¹, Muthukumaran Ramasubramanian¹,
  Rahul Ramachandran² — add "\* equal contribution".
- Title unchanged.

## 01 · Motivation

- PROPERTIES → **Extensible**: "tools registered per geospatial event type"
  → **"tools registered per event type *and region* — US · Europe · global"**.
- Everything else unchanged.

## 02 · As a black box

- Last paragraph: "…weather archives, precipitation records, and USGS river
  gauges" → **"…weather archives, rain gauges, river gauges in the US and
  Europe, Copernicus GloFAS river discharge, and GDACS disaster catalogs."**

## 03 · Pipeline (figure)

Two small edits to the existing diagram:

- **02 · Tool selection** box: "LLM picks tools by signature" →
  **"pre-filter: event type + region · LLM picks by signature"**.
- **03 · Verification** box: add one line under "synthesizes verdict" —
  **"no tool evidence → INCONCLUSIVE"**.

## 04 · Block details

- **Block 02 — DOES**: "pre-filter by event type, then LLM picks tools…" →
  **"pre-filter by event type and coverage region (a claim in Poland never
  reaches USGS), then LLM picks tools by signature + docstring"**.
  e.g. → **"Oder discharge claim (Poland) → [get_river_discharge_glofas]"**.
- **Block 03 — add a line**: **"a verdict must cite tool evidence — none
  gathered → INCONCLUSIVE, enforced in code"**.
  e.g. → **"Vienna gust claim + station tool → ✗ CONTRADICTS
  → 90 km/h measured vs >150 km/h claimed"**.
- **Block 04 / 05 examples**: swap Beryl numbers for Boris numbers **[RUN]**.

## 05 · Case study — replace Beryl with **Storm Boris** (Central Europe, Sep 2024)

Why Boris: every independent source agrees, and like Beryl it contains one
planted error the system refutes. It is also the right example for an EO
audience in Europe.

**01 · Input quote**

> "Storm Boris brought extreme rainfall to Central Europe in mid-September
> 2024: between 12 and 16 September, more than 300 mm of rain fell in the
> town of Jeseník, Czechia. The Oder river at Opole, Poland rose to its
> highest level in over two decades, and flooding spread across Poland,
> Czechia, Austria and Slovakia. In Vienna, Austria, the storm brought wind
> gusts of over 150 km/h."

SOURCE Upstream geospatial AI system · EVENT FLOOD · STORM

**02 · GeoGuard trace** (from `data/case_studies/boris.md`)

CLAIMS
- 📦 **FLOOD · Jeseník · 2024-09-12** · >300 mm rain, 12–16 Sep
- 📦 **FLOOD · Opole · 2024-09-12** · Oder highest in two decades · flooding across PL, CZ, AT, SK
- 📦 **STORM · Vienna · 2024-09-12** · gusts over 150 km/h

TOOLS · place-name resolver · historical precipitation · historical wind ·
GloFAS river discharge · weather stations · European river gauges · GDACS catalog

CALLS
```
resolve("Jeseník, Czechia")        → 50.276, 17.181
era5_precip(Jeseník, 08-29→09-17)  → 486 mm · peak 171 mm/day
glofas(Oder @ Opole, 08-29→09-17)  → 825 m³/s on 09-16 · #1 of 21 yrs
gdacs(Opole, 09-10→09-20)          → Orange flood alert: AT, CZ, DE, PL, RO, SK
stations(Vienna, 09-11→09-17)      → Wien Hohe Warte: 90 km/h peak gust
```

**03 · Final report**

| Claim | Verdict | Rubric |
|---|---|---|
| >300 mm of rain in Jeseník, 12–16 Sep | ✅ SUPPORTS | 62 % |
| Oder at Opole highest in two decades | ✅ SUPPORTS | 75 % |
| Flooding across PL, CZ, AT, SK | ✅ SUPPORTS | 75 % |
| Gusts over 150 km/h in Vienna | ❌ CONTRADICTS | 43 % |

**OVERALL CONFIDENCE 64 % · VERDICT: CONTRADICTS** — 1 refuted · 3 supported.
Claude Sonnet · 289 s · 16 LLM calls. (Final code. Verdicts were identical on
an earlier run; rubric scores vary by a few points between runs.)

Report sentence (replaces the Beryl one): *"The 'gusts over 150 km/h in Vienna'
claim is refuted by a station 4.7 km away that measured 90 km/h. Plausibly a
storm-wide maximum misattributed to the city. The rainfall and river claims are
supported: ERA5 shows 446 mm in Jeseník over 12–16 Sep, and the Oder at Opole
peaked at 825 m³/s, the highest of 21 years in GloFAS."*

Block 04 / 05 examples (replace Beryl's):
- Block 04: *"4 verified claims → rain 62% · Oder 75% · flooding 75% · Vienna gusts 43% → 64% overall"*
- Block 05: *"4 verdicts + rubric → ✗ CONTRADICTS · 64% → 1 refuted · 3 supported"*

## 06 · Benchmark

Replace the single table with a US + Europe table **[RUN]**:

| | SUPPORTS | CONTRADICTS | INCONCLUSIVE |
|---|---|---|---|
| US · real (70) | | | |
| US · fabricated (35) | | | |
| Europe · real (45) | | | |
| Europe · fabricated (24) | | | |

Headline line: **"174 events · NOAA Storm Events (US) + Copernicus EMS
(Europe) · fabricated: wrong date / wrong location / inflated magnitude, easy
and hard"** · model · $/event · s/event **[RUN]**.

Footnote (keep it — reviewers respect it): **"Benchmark uses measurement tools
only; catalog tools are disabled so the event catalogs cannot confirm their
own labels. European events named only by country, and overseas territories, are excluded."**

If the run can't finish in time, keep the existing 65-event table (it's the
published abstract result) and add one line: "European evaluation (174
events, US + Europe) in progress."

### Results — run 1 (complete; fallback if run 2 doesn't beat it)

174 events · Claude Sonnet via claude-cli · $0.41/event API-equivalent · 99 s/event · 7 LLM calls/event

| | SUPPORTS | CONTRADICTS | INCONCLUSIVE |
|---|---|---|---|
| US · real (70) | 45 | 6 | 19 |
| US · fabricated (35) | 3 | 25 | 7 |
| Europe · real (45) | 26 | 8 | 11 |
| Europe · fabricated (24) | 5 | 12 | 7 |

- Real events not rejected: **88%** (101/115, 95% CI 81–93%) — US 91%, Europe 82%
- Fabricated events endorsed: **14%** (8/59, 95% CI 7–25%) — easy 10%, hard 18%
- Fabricated events refuted outright: **63%** (37/59)

### What the errors teach (worth one line on the poster)

- **Most endorsed fabrications landed on real weather.** A date moved weeks
  away, or a place moved 150–400 km, often falls inside another real event:
  "Danube flood, 15 June 2024" was fabricated from a September event, but the
  Danube really flooded in early June 2024.
- **Most wrongly rejected real events were region-scale claims or local
  convective storms** — "flooding in Southern Italy", a 75 mph thunderstorm
  gust — that a point measurement or 25 km reanalysis cannot resolve.
- Fixed between runs: claim extraction dropped stated coordinates (it
  geocoded "Southern Italy (38.4, 13.4)" to Naples, 300 km away).

## Data sources & limitations (bottom strip)

- **Data sources**: OpenStreetMap Nominatim · Open-Meteo (ERA5) · OSM Overpass
  · NOAA PRISM/MRMS · USGS NWIS · **Copernicus GloFAS · Meteostat (national
  weather services) · Hub'Eau (France) · Environment Agency (UK) · GDACS**
- **US ↔ Europe counterparts** (new, compact — worth a small box):

  | Evidence | US | Europe |
  |---|---|---|
  | Rain gauges | PRISM + MRMS | Meteostat stations |
  | River gauges | USGS NWIS | Hub'Eau · EA (UK) |
  | River discharge + rank | GloFAS | GloFAS |
  | Measured gusts | Meteostat | Meteostat |
  | Disaster catalog | GDACS | GDACS |

- **Limitations**: "flood/storm only · limited tools (can be extended) ·
  US-only for radar/gauge & streamflow · text claims only" →
  **"flood/storm only · radar rainfall US-only (EUMETNET OPERA next) ·
  measured river gauges US, France, UK (GloFAS elsewhere) · text claims and
  GeoTIFF flood masks"**.

## QR codes

Keep both. Note: the HF Space demo runs the US tool set until the European
tools are merged into the demo branch — don't promise Europe in the demo
unless that merge happens before printing.

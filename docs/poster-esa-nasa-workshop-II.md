# GeoGuard — A Guardrails Framework for Geospatial Foundation Models

*ESA – NASA 2026 Workshop · Poster Session*

*Authors / affiliation — TBD*

---

## Problem & Motivation

> **The trust gap.** Geospatial foundation models excel at segmentation, classification, and forecasting across Earth observation, weather, and climate. As these models move from research into operational use, a key question arises: **can we trust the output before it reaches a decision-maker?**

> **No automatic check exists.** In disaster response or weather forecasting, ground truth at inference time often does not exist. Today, there is no standard way to validate whether an FM's output is reliable.

> **GeoGuard.** A model-agnostic validation layer that sits between the FM's output and its downstream consumer. It does not modify the model. It decomposes the output into atomic claims, verifies each against authoritative external data, and emits a structured report with verdicts, evidence, and a confidence score.

- ◆ **Model-agnostic** — wraps any FM, agentic workflow, or hand-authored geo-text.
- ◆ **Auditable** — every claim cites the tools that verified it.
- ◆ **Confidence-scored** — categorical verdict + holistic rubric score.

---

## GeoGuard as a Black Box

```
        ┌───────────────────────────┐
        │     external producer     │
        │   (FM / agent / human)    │
        └──────────────┬────────────┘
                       │  output (text ± images)
                       ▼
        ╔═══════════════════════════╗
        ║                           ║
        ║         GeoGuard          ║   ← decompose · verify · score · report
        ║                           ║
        ╚══════════════┬════════════╝
                       │  Report
                       │  (verdicts + evidence + confidence)
                       ▼
        ┌───────────────────────────┐
        │   downstream consumer     │
        │  (gate · retry · log)     │
        └───────────────────────────┘
```

**Three deployment patterns:**

| Pattern | Flow |
|---|---|
| **Sync** | upstream output → GeoGuard inline → consumer waits for verdict |
| **Sideways** | upstream output proceeds; GeoGuard audits in parallel and emits a report |
| **Adapter** | reads inferences from a GeoFM model's outputs (e.g., flood masks) and validates them |

---

## Pipeline Overview

```
   Input
     │
     ▼
 ┌──────────────────────┐
 │ Claim + Metadata     │
 │ Extraction           │   ← agentic
 └──────────┬───────────┘
            │  list of (claim, metadata) groups
            ▼
 ┌──────────────────────┐
 │ Tool Selection       │   ← agentic, per-claim
 │ (from registered     │      (parallel)
 │   toolset)           │
 └──────────┬───────────┘
            │  per-claim selected tools
            ▼
 ┌──────────────────────┐
 │ Verification         │   ← agentic, per-claim, calls tools
 │ (agent loop with     │      (parallel)
 │  tools attached)     │
 └──────────┬───────────┘
            │  per-claim verdicts + tool-call traces
            ▼
 ┌──────────────────────┐
 │ Rubric               │   ← holistic confidence scoring
 │ (yes/no checks per   │      (one pass over all claims)
 │  claim, aggregated)  │
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ Final Report         │
 │ verdicts + evidence  │
 │ + rubric +           │
 │ overall confidence   │
 └──────────────────────┘
```

Agentic decisions happen at three points: **what claims are in the input**, **which tools to use per claim**, and **how to interpret tool evidence into a verdict**. Per-claim work runs in parallel.

---

## Block Details

### Claim + Metadata Extraction

- **In:** raw text (or text + images) from the upstream system.
- **Does:** decomposes the output into **atomic factual claims**. Each claim is *decontextualized* — every proper noun, date, and location is explicit, so the claim can be verified in isolation. Alongside the claims, structured **geospatial metadata** is extracted (event type, location, time range, entities). Both produced in one structured-output call.
- **Out:** groups of claims, each tied to its event metadata.

Three rules baked in: claims must be (1) **atomic**, (2) **faithful** — no information added beyond the input, (3) **distinct** — overlapping facts merged.

### Tool Selection (from registered toolset)

- **In:** one claim + its metadata.
- **Does:** prefilters the registered toolset by event type to a candidate pool, then an LLM picks the relevant subset based on the claim's specifics and each tool's signature (parameter types, return types, docstring).
- **Out:** the chosen tools to attach to the verifier — plus the LLM's reasoning for *why* these tools were picked.

A claim about *storm surge in Galveston on a specific date* needs different tools than a claim about *displaced population in Houston*. The LLM sees the candidate signatures and picks what fits.

### Verification (agentic tool calls)

- **In:** one claim + metadata + the tools selected for it.
- **Does:** spins up an agent with those tools attached. The agent decides which tools to call, with what parameters, and how to interpret the results. It synthesizes a verdict — *supports*, *contradicts*, or *inconclusive* — with a written rationale that cites the tool evidence it gathered.
- **Out:** a structured verification result: verdict, rationale, and the **full trace** of every tool call (name, arguments, returned data).

Per-claim verifications run **in parallel** within an event group. The verifier never sees the original input — it only knows what the upstream claimed (the decontextualized claim) and what the tools have to say.

### Rubric

- **In:** the original input + every claim's verification + every tool-call trace.
- **Does:** in one holistic pass, generates **per-claim yes/no checks** tailored to each claim (covering location, time, magnitude, attribution, etc.). Each check is answered using the evidence already gathered — *no new tool calls*. Each "yes" must cite a specific tool result; ambiguity defaults to "no".
- **Out:** a Rubric with **per-claim items + scores**, plus a single **overall confidence** value (mean of per-claim scores).

Why this is separate from verification: it's a *meta-check*. Did the verification gather appropriate evidence, and does that evidence actually support each claim? Splitting the concerns makes both layers more focused.

### Final Report

- **In:** original input, every verification result, the rubric.
- **Does:** rolls verdicts up into a single overall verdict (any contradiction → contradicts; all supports → supports; otherwise inconclusive) and packages everything into a typed report.
- **Out:** a structured Report carrying input, per-claim verifications with their tool-call traces, rubric, and overall verdict. The downstream consumer can gate, retry, surface, or log based on whichever signal matters.

---

## Use Case — Flood Verification (Hurricane Beryl)

**Input given to GeoGuard:**

> *"Hurricane Beryl made landfall near Matagorda Bay, Texas on July 8, 2024. The storm brought sustained winds of 80 mph and significant storm surge along the Texas coast. Heavy rainfall caused widespread flooding in Houston, with reports of over 100 mm of rain falling over a 24-hour period."*

**GeoGuard's trace (compressed):**

| Stage | Output (sample) |
|---|---|
| **Claims extracted** | • Hurricane Beryl made landfall near Matagorda Bay, Texas on 2024-07-08<br>• Hurricane Beryl had ~80 mph sustained winds at landfall<br>• Houston received >100 mm of rain on / around 2024-07-08 |
| **Event metadata** | Flood event · Texas Gulf Coast · 2024-07-08 |
| **Tools selected per claim** | place-name resolver · historical wind data · historical precipitation data |
| **Tool calls (sample)** | resolve("Houston, Texas") → 29.76, −95.37<br>precipitation(29.76, −95.37, 2024-07-08) → **102 mm**<br>winds(29.76, −95.37, 2024-07-08) → **84 mph max gust** |
| **Per-claim verdicts** | rainfall: **SUPPORTS** · winds: **SUPPORTS** · landfall: **INCONCLUSIVE** *(no landfall registry queried)* |
| **Rubric** | per-claim 80 % / 75 % / 50 % — **overall confidence ≈ 68 %** |

Real APIs used (no keys required): **OpenStreetMap Nominatim** for authoritative place-to-coordinates resolution; **Open-Meteo Historical Archive** for daily precipitation and wind speed.

---

## Validation Check Categories

The framework treats each validation check as a registered tool. The taxonomy below — drawn from the workshop abstract — maps to **what we demonstrate today** vs. **what slots in next**.

| Category | What it answers | Status |
|---|---|---|
| **Meteorological** | Did supporting weather conditions actually occur? (rainfall, wind, temperature) | ✅ demonstrated |
| **Location resolution** | Does the named place exist, and where? | ✅ demonstrated |
| **Physical plausibility** | Is the prediction physically reasonable given terrain? (DEM, slope, land cover) | future |
| **Event corroboration** | Do authoritative disaster monitors confirm the event? | future |
| **Historical contextualization** | Does the location have a documented history of similar events? | future |
| **Cross-reference** | Does an independent model or reference product agree? | future |
| **Temporal consistency** | Do prior / subsequent observations support the prediction? | future |

Adding a new check is a **single function registered against an event type** — no changes to orchestration, no schema churn.

---

## References & Future Work

**References:**
- *Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents*
- *Mitigating Geospatial Knowledge Hallucination in Large Language Models*
- *Towards LLM Agents for Earth Observation*

**Future work:**
- Disaster catalogs as registered tools (NASA EONET, GDACS, EM-DAT, ReliefWeb)
- Adapter for raster-input FMs (segmentation masks, flood polygons, temperature fields) — extends the input contract to carry geo-rasters; tools verify per-pixel claims against DEM, land cover, and historical event extents
- Physical plausibility checks via DEM, slope, land cover composition
- Cross-reference checks against independent reference products
- Calibration study — rubric confidence vs. human-annotated ground truth

---

## Try It Live

> **[ QR CODE PLACEHOLDER ]**
>
> *Scan to interact with the live Streamlit demo*
> *(URL: TBD)*

*Authors' contact: TBD · Logos: ESA · NASA · institutional*

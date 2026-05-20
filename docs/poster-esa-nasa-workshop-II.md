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

## 08 · Related work & where GeoGuard fits

Four lines of work touch the problem GeoGuard targets. Each is necessary background; none is a complete solution to the "audit geo-AI output against authoritative data" problem.

### General LLM guardrails — safety, not factuality

> **[1]** Rebedea, Dinu, Sreedhar, Parisien, Cohen (2023). *NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications with Programmable Rails*. EMNLP 2023 (Demo). [arXiv:2310.10501](https://arxiv.org/abs/2310.10501).

→ **Gap:** programmable rails handle topic, safety, dialogue flow, language style. They are domain-blind — no geocoders, no NOAA, no USGS. Toxicity and PII checks don't help when an upstream model claims 100 mm of rain that didn't fall.

### Automated fact-checking — general, web-grounded

> **[2]** Iqbal, Wang, Wang, Georgiev, Geng, Gurevych, Nakov (2024). *OpenFactCheck: A Unified Framework for Factuality Evaluation of LLMs*. EMNLP 2024. [arXiv:2408.11832](https://arxiv.org/abs/2408.11832).
>
> **[3]** Xie, Xing, Wang, Geng, Iqbal, Sahnan, Gurevych, Nakov (2025). *FIRE: Fact-checking with Iterative Retrieval and Verification*. NAACL 2025 Findings. [arXiv:2411.00784](https://arxiv.org/abs/2411.00784).
>
> **[4]** Trinh, Nguyen, Hy (2025). *Towards Robust Fact-Checking: A Multi-Agent System with Advanced Evidence Retrieval*. arXiv:2506.17878. [Link](https://arxiv.org/abs/2506.17878).

→ **Gap:** the architectures are the right shape — decompose claims, retrieve evidence, verify, score — but the evidence is web text. None grounds verification in authoritative geospatial data (PRISM, MRMS, USGS NWIS, OSM). A news article that mis-states rainfall counts as supporting evidence to these systems.

### Geospatial hallucination — measurement and mitigation

> **[5]** Wang, Feng, Liu, Pei, Li (2025). *Mitigating Geospatial Knowledge Hallucination in Large Language Models: Benchmarking and Dynamic Factuality Aligning*. arXiv:2507.19586. [Link](https://arxiv.org/abs/2507.19586).
>
> **[6]** Zhang, Gao, Wei, Zhao, Nie, Chen, Chen, Su, Sun (2025). *GeoAnalystBench: A GeoAI benchmark for assessing large language models for spatial analysis workflow and code generation*. Transactions in GIS 2025. [arXiv:2509.05881](https://arxiv.org/abs/2509.05881).

→ **Gap:** model-side. Either a benchmark for scoring how often LLMs hallucinate geospatial facts, or a fine-tuning method (DynamicKTO) to make them hallucinate less. Neither is a runtime auditor that wraps an unmodified upstream model and audits its output in operation.

### Geospatial AI agents — producers, not validators

> **[7]** opendatalab et al. (2026). *Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents*. ICLR 2026. [arXiv:2509.23141](https://arxiv.org/abs/2509.23141).
>
> **[8]** Luo, Lin, Xu, Wu, Mao, Wang, Feng, Huang, Du (2025). *GeoJSON Agents: A Multi-Agent LLM Architecture for Geospatial Analysis — Function Calling vs Code Generation*. arXiv:2509.08863. [Link](https://arxiv.org/abs/2509.08863).
>
> **[9]** Chen, Li, Ma, Hu, Zhu, Deng, Yu (2025). *Empowering LLM Agents with Geospatial Awareness: Toward Grounded Reasoning for Wildfire Response*. arXiv:2510.12061. [Link](https://arxiv.org/abs/2510.12061).
>
> **[10]** Sukhorukov et al. (2025). *Hierarchical AI-Meteorologist: LLM-Agent System for Multi-Scale and Explainable Weather Forecast Reporting*. arXiv:2511.23387. [Link](https://arxiv.org/abs/2511.23387).

→ **Gap:** every one of these *produces* geo-AI output — Earth-Observation task results, GeoJSON spatial analyses, wildfire resource allocations, forecast narratives. None *validates* such output. GeoGuard is the layer above them.

### Where GeoGuard fills in

GeoGuard is a **model-agnostic, post-hoc, domain-typed validation layer** that wraps any upstream producer — an LLM, a foundation model, an agent like the four above, or a human — and audits its text output against authoritative geospatial data: gauges, radar+gauge QPE, river streamflow records, place-name registries. It does not modify the upstream model (unlike [5]), does not rely on web text as evidence (unlike [2–4]), is not a benchmark for measuring how good models are (unlike [5, 6]), and is not itself a producer (unlike [7–10]). It is the validator that the four lines above leave unfilled. On an adversarial NOAA Storm Events benchmark, **0% of fabricated events are endorsed** and **92% of real events are not rejected** — see § 06.

---

### Future work

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

# GeoGuard architecture

## What is GeoGuard?

A pluggable **guardrail layer** for geo-AI. Any upstream system that emits geographically-grounded text (foundation-model inference, agentic workflows, hand-authored copy) can be wrapped so its factual claims are verified against external geospatial data sources before reaching a downstream consumer.

The framework decomposes the upstream's output into atomic claims, fetches authoritative evidence via registered tools, scores the verification with a holistic rubric, and emits a structured report with verdicts, evidence, rationale, and confidence per claim.

---

## System integration

GeoGuard lives between an upstream producer and its downstream consumer:

```
                  ┌──────────────────────────┐
                  │   external producer      │
                  │   (FM, agent, human…)    │
                  └─────────────┬────────────┘
                                │  output
                                │  (text ± images)
                                ▼
                  ┌──────────────────────────┐
                  │        geoguard          │ ← decompose → verify → score → report
                  └─────────────┬────────────┘
                                │  Report
                                │  (verdicts + evidence + rubric + confidence)
                                ▼
                  ┌──────────────────────────┐
                  │   downstream consumer    │ ← gate / retry / surface / log
                  │   (UI, policy, store)    │
                  └──────────────────────────┘
```

Same shape for any producer:

```
| external FM inference   | → output → | geoguard | → Report → consumer
| external agent workflow | → output → | geoguard | → Report → consumer
| human-authored copy     | → output → | geoguard | → Report → consumer
```

The guardrail is **format-agnostic on the way in** (just text/images) and **structured on the way out** (typed `Report` with discriminated verdicts + a confidence score) — so any consumer can decide whether to gate, retry, surface, or log based on the verification result.

---

## Internal pipeline

```
                            Input
                       { text, images }
                              │
                              ▼
                    ┌─────────────────────┐
                    │ MetadataExtractor   │ ← LLM (pydantic-ai)
                    │  (claims+metadata   │
                    │   in one call)      │
                    └──────────┬──────────┘
                               │  list[ClaimGroup]
                               │  each = { metadata, list[Claim] }
                               ▼
                    ╭══════════════════════════════════════╮
                    ║       for each claim:                ║
                    ║       (parallel within group)        ║
                    ║                                      ║
                    ║          ┌──────────────────┐        ║
                    ║          │  ToolSelector    │ ← LLM  ║
                    ║          └────────┬─────────┘        ║
                    ║                   │ SelectedTools    ║
                    ║                   │ { tools,         ║
                    ║                   │   reasoning,     ║
                    ║                   │   claim }        ║
                    ║                   ▼                  ║
                    ║          ┌──────────────────┐        ║
                    ║          │   Verifier       │ ← LLM  ║
                    ║          │   (calls tools,  │   +    ║
                    ║          │    synthesises   │ tools  ║
                    ║          │    verdict)      │        ║
                    ║          └────────┬─────────┘        ║
                    ║                   │ VerifierResult   ║
                    ║                   │ { verification,  ║
                    ║                   │   tool_calls }   ║
                    ╰══════════════════ │ ═════════════════╯
                                        │ list[VerifierResult]
                                        ▼
                              ┌──────────────────┐
                              │   Rubricator     │ ← LLM (1 holistic call)
                              │  (input + all    │
                              │   verifications  │
                              │   → yes/no       │
                              │   questions per  │
                              │   claim)         │
                              └────────┬─────────┘
                                       │ Rubric
                                       │ { per_claim: [{claim, items, score}],
                                       │   confidence (mean of scores) }
                                       ▼
                              ┌──────────────────┐
                              │     Report       │
                              │ { input,         │
                              │   verifications, │
                              │   rubric,        │
                              │   overall_verdict}
                              └────────┬─────────┘
                                       │
                                       ▼
                                   consumer
```

| Block | Role | LLM? |
|---|---|---|
| `MetadataExtractor` | Extracts claims + classifies event type + fills metadata in one structured call (via `output_type=list[ClaimGroup]`) | Yes |
| `ToolSelector` | Pre-filters tools via registry; LLM picks the relevant subset given the claim + metadata | Yes |
| `Verifier` | Per-claim agent with the selected tools attached; agent calls tools, synthesises verdict + rationale | Yes |
| `Rubricator` | One holistic call after all claims are verified; reads original input + every claim's verification + tool_calls; produces dynamic yes/no rubric items per claim with overall confidence | Yes |

The `overall_verdict` rollup (any CONTRADICTS → CONTRADICTS; all SUPPORTS → SUPPORTS; else INCONCLUSIVE) is a pure-rules helper inside `pipeline.py` — small enough to live with the `Report` construction rather than a standalone block.

---

## Streaming events

`GeoGuard.stream()` (also reachable via `__call__`) is an async generator. Each block step yields a typed event so a UI can react in real time. Per-claim work runs in parallel within a group; cross-claim events interleave by completion time, while within a single claim the order `Claim → SelectedTools → VerifierResult` is preserved.

```
ClaimGroup ──┐
             ├── Claim ──── SelectedTools ──── VerifierResult
             ├── Claim ──── SelectedTools ──── VerifierResult     (parallel,
             └── Claim ──── SelectedTools ──── VerifierResult      interleaved)
                                                       │
                                                       ▼
                                                   Rubric    (holistic, after
                                                       │      all claims done)
                                                       ▼
                                                   Report    (terminal)
```

Consumers `isinstance`-dispatch on the event type. `GeoGuard.run(input)` is a convenience that drains the stream and returns just the final `Report`.

---

## Tool registry

Tools are primitive functions or classes — domain wrappers around APIs, datasets, models. They have **no `Claim`/`Metadata` in their signatures**: they take their own params (lat, lon, datetime, place name, etc.) and return their own data. The verification agent is what bridges *(claim, metadata)* → *(tool params)*.

Authors register tools at import time via decorator:

```python
from geoguard.tools import registry
from geoguard.schemas import EventType

@registry(EventType.FLOOD)
async def query_noaa_tide(lat: float, lon: float, dt: str) -> dict:
    """Tide reading at lat/lon for the given datetime."""
    ...

@registry(EventType.FLOOD, EventType.OTHER)
async def geocode_location(name: str) -> dict:
    """Resolve a place name to lat/lon."""
    ...
```

`ToolSelector` queries `registry.get_candidates(metadata.event_type)` for the candidate pool (event-type matches + always-on `OTHER` tools, deduped by name), then an LLM picks the relevant subset based on the claim. Selected tools are wrapped in a `pydantic_ai.toolsets.FunctionToolset` and attached to the per-claim verifier agent.

The registry stays decoupled from pydantic-ai's data structures:
- Our `@register` is domain-aware (event-type tagged)
- `FunctionToolset` is the runtime container pydantic-ai consumes

Pydantic-ai compatibility activates at agent-attachment time (in the verifier), not at registration.

---

## Claim extraction rules

Both `ClaimExtractor` and `MetadataExtractor` (in `list[ClaimGroup]` mode) compose the same `CLAIM_RULES` constant from `claims.py`. Single source of truth for what a claim must satisfy:

- **ATOMIC**: one factual assertion per claim
- **DECONTEXTUALIZED**: include all proper nouns, absolute dates, explicit locations; no pronouns / deictic references that need surrounding context
- **DISTINCT**: no claim may overlap with or be a sub-statement of another; merge overlapping facts into one comprehensive claim
- **FAITHFUL**: every fact must be explicitly stated in the input (combining facts from different parts of the input is allowed; adding, inferring, or pulling from prior knowledge is not)
- **VERIFIABLE**: checkable against an external source

Both extractors also accept a `max_claims: int | None = 15` cap. When set, the prompt instructs the agent to prioritize central, load-bearing claims and skip trivia. Pass `None` for no cap.

---

## Rubric design

The Rubricator is the framework's confidence layer. After every claim has been verified, one LLM call ingests:
- the original `Input.text`
- every `VerifierResult` (claim, verdict, rationale, full tool-call trace)

…and produces a `Rubric` that contains, **per claim**, between 5 and 10 dynamically-generated yes/no questions, each answered using the evidence already gathered (no new tool calls). Per-claim score = ratio of yes-answers; overall confidence = mean of per-claim scores.

Five prompt-level constraints reduce hallucination:

1. **Distinct within a claim**: no near-duplicate questions inside one rubric
2. **Specific across claims**: each claim's questions target its unique verifiable details, not a generic checklist
3. **Mandatory citation**: every `answer=true` must reference a specific `tool_call` result in its `reasoning`
4. **Conservative-no default**: when evidence is ambiguous, answer `false` (safer for a guardrail)
5. **Tunable cap**: `questions_per_claim: tuple[int, int] = (5, 10)` controls budget; range is a soft target

The rubric questions are generated **dynamically per claim** — no hardcoded templates. Each claim's content drives what to check.

---

## Core data types

| Type | Shape | Defined in |
|---|---|---|
| `Input` | `text: str`, `images: list[ImageRef]` | `schemas.py` |
| `EventType` | `FLOOD`, `OTHER` (StrEnum) | `schemas.py` |
| `Claim` | `claim: str` (atomic, decontextualized, faithful) | `claims.py` |
| `GeneralMetadata` | base + `OTHER` fallback variant | `metadata.py` |
| `FloodMetadata` | extends `GeneralMetadata`, adds flood-specific fields | `metadata.py` |
| `Metadata` | discriminated union over the variants | `metadata.py` |
| `ClaimGroup` | `metadata: Metadata`, `claims: list[Claim]` | `metadata.py` |
| `SelectedTools` | `tools: list[Callable]`, `reasoning: str`, `claim: Claim` | `tools/selector.py` |
| `Verdict` | `SUPPORTS`, `CONTRADICTS`, `INCONCLUSIVE` (StrEnum) | `verifications.py` |
| `ClaimVerification` | `claim`, `metadata`, `verdict`, `rationale` | `verifications.py` |
| `ToolCall` | `name`, `args` (JSON str), `result` | `verifications.py` |
| `VerifierResult` | `verification: ClaimVerification`, `tool_calls: list[ToolCall]` | `verifications.py` |
| `RubricItem` | `question: str`, `answer: bool`, `reasoning: str` | `rubrics.py` |
| `ClaimRubric` | `claim: Claim`, `items: list[RubricItem]`, `.score` property | `rubrics.py` |
| `Rubric` | `per_claim: list[ClaimRubric]`, `.confidence` property | `rubrics.py` |
| `Report` | `input`, `verifications`, `rubric`, `overall_verdict` | `pipeline.py` |
| `PipelineEvent` | `ClaimGroup \| Claim \| SelectedTools \| VerifierResult \| Rubric \| Report` | `pipeline.py` |

---

## Extension points

| To add… | How |
|---|---|
| A new tool | `@registry(EventType.X) async def my_tool(...): ...` — see `tools/` |
| A new event type | Add value to `EventType` enum (`schemas.py`), add a metadata subclass (e.g., `BurnMetadata`) extending `GeneralMetadata`, add it to the `Metadata` discriminated union |
| Custom block (selector / verifier / rubricator / metadata extractor) | Replace via constructor injection: `GeoGuard(verifier=MyVerifier())`, `GeoGuard(rubricator=MyRubricator())`, etc. |
| Custom output schema for any block | Pass `output_type=...` to the block's constructor — `MetadataExtractor` is generic over the structured output type |
| Tighter / looser claim caps | `MetadataExtractor(max_claims=N)` or `max_claims=None` for no cap |
| More / fewer rubric questions | `Rubricator(questions_per_claim=(low, high))` |

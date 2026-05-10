# GeoGuard architecture

## What is GeoGuard?

A pluggable **guardrail layer** for geo-AI. Any upstream system that emits geographically-grounded text (foundation-model inference, agentic workflows, hand-authored copy) can be wrapped so its factual claims are verified against external geospatial data sources before reaching a downstream consumer.

The framework decomposes the upstream's output into atomic claims, fetches authoritative evidence via registered tools, and emits a structured report with verdicts, evidence, and rationale per claim.

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
                  │        geoguard          │ ← decompose → verify → report
                  └─────────────┬────────────┘
                                │  Report
                                │  (verdicts + evidence + rationale)
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

The guardrail is **format-agnostic on the way in** (just text/images) and **structured on the way out** (typed `Report` with discriminated verdicts) — so any consumer can decide whether to gate, retry, surface, or log based on the verification result.

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
                    ║          for each claim:             ║
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
                                        │
                                        ▼
                              ┌──────────────────┐
                              │   Aggregator     │  pure rules:
                              │  (roll up        │  any CONTRADICTS → CONTRADICTS
                              │   verdicts)      │  all SUPPORTS    → SUPPORTS
                              └────────┬─────────┘  else            → INCONCLUSIVE
                                       │
                                       ▼
                              ┌──────────────────┐
                              │     Report       │
                              │ { input,         │
                              │   verifications, │
                              │   overall }      │
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
| Aggregator | Roll-up of per-claim verdicts | No |

---

## Streaming events

`GeoGuard.__call__` is an async generator. Each block step yields a typed event so a UI can react in real time:

```
ClaimGroup ──┐
             ├── Claim ──── SelectedTools ──── VerifierResult
             ├── Claim ──── SelectedTools ──── VerifierResult
             └── Claim ──── SelectedTools ──── VerifierResult
                                                      │
                                                      ▼
                                                    Report   (terminal)
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

## Core data types

| Type | Shape | Defined in |
|---|---|---|
| `Input` | `text: str`, `images: list[ImageRef]` | `schemas.py` |
| `EventType` | `FLOOD`, `OTHER` (StrEnum) | `schemas.py` |
| `Claim` | `claim: str` (atomic, decontextualized) | `claims.py` |
| `GeneralMetadata` | base + `OTHER` fallback variant | `metadata.py` |
| `FloodMetadata` | extends `GeneralMetadata`, adds flood-specific fields | `metadata.py` |
| `Metadata` | discriminated union over the variants | `metadata.py` |
| `ClaimGroup` | `metadata: Metadata`, `claims: list[Claim]` | `metadata.py` |
| `SelectedTools` | `tools: list[Callable]`, `reasoning: str`, `claim: Claim` | `tools/selector.py` |
| `Verdict` | `SUPPORTS`, `CONTRADICTS`, `INCONCLUSIVE` (StrEnum) | `verifications.py` |
| `ClaimVerification` | `claim`, `metadata`, `verdict`, `rationale` | `verifications.py` |
| `ToolCall` | `name`, `args` (JSON str), `result` | `verifications.py` |
| `VerifierResult` | `verification: ClaimVerification`, `tool_calls: list[ToolCall]` | `verifications.py` |
| `Report` | `input`, `verifications`, `overall_verdict` | `pipeline.py` |
| `PipelineEvent` | `ClaimGroup \| Claim \| SelectedTools \| VerifierResult \| Report` | `pipeline.py` |

---

## Extension points

| To add… | How |
|---|---|
| A new tool | `@registry(EventType.X) async def my_tool(...): ...` — see `tools/` |
| A new event type | Add value to `EventType` enum (`schemas.py`), add a metadata subclass (e.g., `BurnMetadata`) extending `GeneralMetadata`, add it to the `Metadata` discriminated union |
| Custom selector / verifier | Replace via constructor injection: `GeoGuard(verifier=MyVerifier())` |
| Custom output schema for any block | Pass `output_type=...` to the block's constructor — `MetadataExtractor` is generic over the structured output type |

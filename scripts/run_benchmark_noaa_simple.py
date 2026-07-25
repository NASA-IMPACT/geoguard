"""Baseline benchmark: a single plain agent + all registered tools, no GeoGuard.

The baseline is deliberately simple — one pydantic-ai agent given geocode +
every tool in the registry, fed the same event input text the GeoGuard
benchmark used, and asked to produce one of three verdicts (supports /
contradicts / inconclusive). No claim extraction, no metadata, no tool
selection, no rubric — the point is to measure what GeoGuard's architecture
adds over "agent + tools figures it out".

Input events default to data/benchmark_65_events.csv. If that file is
missing it is reconstructed from data we have:
  - real NOAA events: claim_text rebuilt from data/NOAA-Storm-Events-Data.csv
    with the SAME build_claim_text() used by select_events.py (identical
    input processing to the original benchmark run);
  - fabricated events (FAB-*): input text rebuilt by joining their extracted
    claims recorded in data/benchmark_65_results.csv.

Runs are resumable: each finished event is written to
data/baseline_noaa_simple/events/<event_id>.json immediately, and reruns
skip events that already have a result file. Failed events are NOT cached,
so a rerun retries them.

Usage:
    uv run python scripts/run_benchmark_noaa_simple.py
    uv run python scripts/run_benchmark_noaa_simple.py --limit 3      # smoke test
    uv run python scripts/run_benchmark_noaa_simple.py --force        # ignore cache
    uv run python scripts/run_benchmark_noaa_simple.py --model openai:gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

# Register tools before touching the registry — the same three modules the
# original 65-event GeoGuard benchmark registered (scripts/run_benchmark.py).
# Deliberately NOT importing geoguard.tools.stac.agent: the VEDA STAC tool
# was not part of that benchmark, so the baseline must not have it either.
import geoguard.tools.geospatial  # noqa: F401
import geoguard.tools.satellite  # noqa: F401
import geoguard.tools.weather  # noqa: F401
from geoguard.config import build_model, settings
from geoguard.metadata import geocode
from geoguard.schemas import EventType
from geoguard.tools.registry import registry
from geoguard.verifications import Verdict, _extract_tool_calls

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from select_events import build_claim_text  # noqa: E402

_ROOT = _SCRIPTS_DIR.parent
EVENTS_PATH = _ROOT / "data" / "benchmark_65_events.csv"
RESULTS_65_PATH = _ROOT / "data" / "benchmark_65_results.csv"
NOAA_PATH = _ROOT / "data" / "NOAA-Storm-Events-Data.csv"
STATE_DIR = _ROOT / "data" / "baseline_noaa_simple"

DEFAULT_MODEL = "openai:gpt-4.1-mini"  # poster configuration

# USD per 1M tokens (input, output) — for the poster's $/event figure.
PRICING_PER_MTOK = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}

EVENT_FIELDS = [
    "event_id",
    "event_type",
    "state",
    "county",
    "region",
    "begin_date",
    "lat",
    "lon",
    "damage_property",
    "deaths_direct",
    "expected_verdict",
    "perturbation_type",
]

BASELINE_INSTRUCTIONS = (
    "You are verifying whether a reported weather/geospatial event actually "
    "occurred as described. Use the available tools to gather independent "
    "evidence about the report — figure out for yourself where and when the "
    "event supposedly happened, which tools can check it, and what the "
    "evidence shows.\n\n"
    "Return one of three verdicts:\n"
    "- supports: the evidence confirms the event happened as described.\n"
    "- contradicts: the evidence disagrees with the report's core assertions.\n"
    "- inconclusive: the evidence is insufficient or ambiguous.\n\n"
    "Cite specific numbers from the tools you called in your rationale."
)


class BaselineReport(BaseModel):
    """The baseline agent's final answer for one event."""

    verdict: Verdict
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Input reconstruction — same processing as the original benchmark
# ---------------------------------------------------------------------------


def reconstruct_events(results_path: Path, noaa_path: Path) -> list[dict]:
    """Rebuild the 65 benchmark input events from data/ we have.

    Event-level fields come from benchmark_65_results.csv (first row per
    event, preserving the original run order). Input text:
      - NOAA-backed events → build_claim_text() on the NOAA row, the exact
        function select_events.py used to produce the original inputs;
      - FAB-* events → their extracted claims (recorded in the results CSV)
        joined in claim order, the closest surviving form of the fabricated
        input text.
    """
    with open(noaa_path) as f:
        noaa = {r["EVENT_ID"]: r for r in csv.DictReader(f)}

    with open(results_path) as f:
        rows = list(csv.DictReader(f))

    grouped: dict[str, list[dict]] = {}
    for r in rows:
        grouped.setdefault(r["event_id"], []).append(r)

    events: list[dict] = []
    for event_id, event_rows in grouped.items():
        event = {k: event_rows[0][k] for k in EVENT_FIELDS}
        noaa_row = noaa.get(event_id)
        if noaa_row is not None:
            event["claim_text"] = build_claim_text(noaa_row)
        else:
            claims = sorted(event_rows, key=lambda r: int(r["claim_n"] or 0))
            event["claim_text"] = " ".join(
                dict.fromkeys(r["claim"] for r in claims if r["claim"])
            )
        events.append(event)
    return events


def load_events(events_path: Path, results_path: Path, noaa_path: Path) -> list[dict]:
    if events_path.exists():
        with open(events_path) as f:
            return list(csv.DictReader(f))

    print(
        f"{events_path} not found — reconstructing from "
        f"{results_path.name} + {noaa_path.name}",
        file=sys.stderr,
    )
    events = reconstruct_events(results_path, noaa_path)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with open(events_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_FIELDS + ["claim_text"])
        writer.writeheader()
        writer.writerows(events)
    print(f"Wrote {len(events)} events to {events_path}", file=sys.stderr)
    return events


# ---------------------------------------------------------------------------
# Baseline agent
# ---------------------------------------------------------------------------


def collect_tools() -> list:
    """geocode (metadata tool) + every tool registered in the registry."""
    tools = {geocode.__name__: geocode}
    for et in EventType:
        for fn in registry.get_candidates(et):
            tools.setdefault(fn.__name__, fn)
    return list(tools.values())


async def run_one(
    event: dict,
    model,
    tools: list,
    request_limit: int | None,
) -> dict:
    # Fresh agent per event → fresh toolset dedup cache per event, matching
    # GeoGuard's one-cache-per-verification behavior.
    agent = Agent(
        model=model,
        output_type=BaselineReport,
        toolsets=[registry.build_toolset(tools, id="baseline")],
        instructions=BASELINE_INSTRUCTIONS,
        retries={"output": settings.output_retries},
    )

    t0 = time.monotonic()
    result = await agent.run(
        f"Verify this event report:\n\n{event['claim_text']}",
        usage_limits=UsageLimits(request_limit=request_limit),
    )
    elapsed = round(time.monotonic() - t0, 1)

    report: BaselineReport = result.output
    usage = result.usage
    tool_calls = _extract_tool_calls(result.all_messages())

    return {
        "event": event,
        "verdict": report.verdict.value,
        "rationale": report.rationale,
        "confidence": round(report.confidence, 3),
        "verdict_correct": _check_correct(
            event.get("expected_verdict", "").lower(), report.verdict.value
        ),
        # Deduplicated quick-view of tool calls (results truncated). The
        # complete untruncated record of every exchange is in "messages".
        "tool_calls": [
            {
                "name": tc.name,
                "args": tc.args,
                "result": _truncate_result(tc.result),
            }
            for tc in tool_calls
        ],
        # Full agent transcript: instructions, user prompt, every model
        # response, every tool call and its full return value — everything
        # the agent saw or produced during the run.
        "messages": json.loads(result.all_messages_json()),
        # All RunUsage fields — input/output/cache token counts, request and
        # tool-call counts, and provider `details` (e.g. reasoning_tokens on
        # reasoning models; gpt-4.1-mini reports none).
        "usage": dataclasses.asdict(usage),
        "elapsed_seconds": elapsed,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def _truncate_result(result, limit: int = 4000):
    """Keep tool results JSON-serializable and bounded in the dump."""
    s = json.dumps(result, default=str)
    if len(s) <= limit:
        return json.loads(s)
    return {"_truncated": True, "chars": len(s), "preview": s[:limit]}


def _check_correct(expected: str, actual: str) -> str:
    # Same partial-credit rule as scripts/run_benchmark.py
    if not expected:
        return ""
    if expected == actual:
        return "yes"
    if expected == "supports" and actual == "inconclusive":
        return "partial"
    if expected == "contradicts" and actual == "inconclusive":
        return "partial"
    return "no"


def _estimate_cost(model_name: str, records: list[dict]) -> dict:
    prices = None
    for key, p in PRICING_PER_MTOK.items():
        if model_name.split(":")[-1] == key:
            prices = p
            break
    with_usage = [r for r in records if r.get("usage")]
    if prices is None or not with_usage:
        return {"available": False}
    in_price, out_price = prices
    total = sum(
        r["usage"]["input_tokens"] / 1e6 * in_price
        + r["usage"]["output_tokens"] / 1e6 * out_price
        for r in with_usage
    )
    return {
        "available": True,
        "total_usd": round(total, 4),
        "per_event_usd": round(total / len(with_usage), 4),
    }


# ---------------------------------------------------------------------------
# Resume state
# ---------------------------------------------------------------------------


def event_result_path(state_dir: Path, event_id: str) -> Path:
    return state_dir / "events" / f"{event_id}.json"


def save_event_result(state_dir: Path, record: dict) -> None:
    path = event_result_path(state_dir, record["event"]["event_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
    os.replace(tmp, path)


def load_event_result(state_dir: Path, event_id: str) -> dict | None:
    path = event_result_path(state_dir, event_id)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None  # corrupt/partial file → rerun the event


# ---------------------------------------------------------------------------
# Metrics — same shape as the GeoGuard poster benchmark
# ---------------------------------------------------------------------------

VERDICT_KEYS = ["supports", "contradicts", "inconclusive", "error"]


def summarize(records: list[dict], model_name: str) -> dict:
    def bucket(expected: str) -> list[dict]:
        return [
            r
            for r in records
            if r["event"].get("expected_verdict", "").lower() == expected
        ]

    def verdict_counts(rs: list[dict]) -> dict:
        counts = {k: 0 for k in VERDICT_KEYS}
        for r in rs:
            counts[r.get("verdict", "error")] = (
                counts.get(r.get("verdict", "error"), 0) + 1
            )
        return counts

    verified = bucket("supports")
    fabricated = bucket("contradicts")
    v_counts = verdict_counts(verified)
    f_counts = verdict_counts(fabricated)

    by_perturbation: dict[str, dict] = {}
    for r in records:
        pt = r["event"].get("perturbation_type", "") or "none"
        b = by_perturbation.setdefault(pt, {k: 0 for k in VERDICT_KEYS})
        b[r.get("verdict", "error")] = b.get(r.get("verdict", "error"), 0) + 1

    scored = [r for r in records if r.get("verdict_correct")]
    strict = sum(1 for r in scored if r["verdict_correct"] == "yes")
    partial = sum(1 for r in scored if r["verdict_correct"] == "partial")

    times = [r["elapsed_seconds"] for r in records if r.get("elapsed_seconds")]

    return {
        "n_events": len(records),
        "verified": {
            "n": len(verified),
            "verdicts": v_counts,
            "not_rejected_pct": round(
                100 * (v_counts["supports"] + v_counts["inconclusive"]) / len(verified),
                1,
            )
            if verified
            else None,
        },
        "fabricated": {
            "n": len(fabricated),
            "verdicts": f_counts,
            "endorsed_pct": round(100 * f_counts["supports"] / len(fabricated), 1)
            if fabricated
            else None,
        },
        "by_perturbation_type": by_perturbation,
        "accuracy": {
            "strict": round(strict / len(scored), 3) if scored else None,
            "with_partial": round((strict + partial) / len(scored), 3)
            if scored
            else None,
        },
        "avg_confidence": round(
            sum(r["confidence"] for r in records if r.get("confidence") is not None)
            / max(1, len([r for r in records if r.get("confidence") is not None])),
            3,
        ),
        "timing": {
            "avg_seconds": round(sum(times) / len(times), 1) if times else None,
            "min_seconds": min(times) if times else None,
            "max_seconds": max(times) if times else None,
        },
        "cost": _estimate_cost(model_name, records),
    }


def print_summary(summary: dict, model_name: str) -> None:
    v, fb = summary["verified"], summary["fabricated"]
    print(f"\n{'=' * 64}", file=sys.stderr)
    print(
        f"BASELINE (single agent + all tools) · {model_name} · "
        f"{summary['n_events']} events",
        file=sys.stderr,
    )
    print(
        f"{'':28}{'SUPPORTS':>10}{'CONTRADICTS':>13}{'INCONCLUSIVE':>14}{'ERROR':>7}",
        file=sys.stderr,
    )
    for label, b in (
        (f"Verified events ({v['n']})", v),
        (f"Fabricated events ({fb['n']})", fb),
    ):
        c = b["verdicts"]
        print(
            f"{label:<28}{c['supports']:>10}{c['contradicts']:>13}"
            f"{c['inconclusive']:>14}{c['error']:>7}",
            file=sys.stderr,
        )

    def _pct(x) -> str:
        return f"{x}%" if x is not None else "n/a"

    print(
        f"\n  Verified not rejected: {_pct(v['not_rejected_pct'])}   "
        f"Fabricated endorsed: {_pct(fb['endorsed_pct'])}",
        file=sys.stderr,
    )
    print(
        f"  Accuracy: strict={summary['accuracy']['strict']}, "
        f"with_partial={summary['accuracy']['with_partial']}   "
        f"Avg confidence: {summary['avg_confidence']}",
        file=sys.stderr,
    )
    t, c = summary["timing"], summary["cost"]
    cost_str = (
        f"${c['per_event_usd']}/event (total ${c['total_usd']})"
        if c.get("available")
        else "n/a"
    )
    print(
        f"  Timing: avg={t['avg_seconds']}s "
        f"(min={t['min_seconds']}s, max={t['max_seconds']}s)   Cost: {cost_str}",
        file=sys.stderr,
    )
    print("\n  By perturbation type:", file=sys.stderr)
    for pt, counts in summary["by_perturbation_type"].items():
        shown = {k: n for k, n in counts.items() if n}
        print(f"    {pt:<20} {shown}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the simple agent+tools baseline on the NOAA benchmark events."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=EVENTS_PATH,
        help="Benchmark events CSV (reconstructed from data/ if missing)",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=RESULTS_65_PATH,
        help="Original GeoGuard results CSV used for reconstruction",
    )
    parser.add_argument(
        "--noaa-csv",
        type=Path,
        default=NOAA_PATH,
        help="NOAA Storm Events CSV used for reconstruction",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=STATE_DIR,
        help="Directory for per-event result cache + final dump",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument(
        "--request-limit",
        type=int,
        default=100,
        help="Max LLM requests per event (runaway guard)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Events verified in parallel (1 = sequential). "
        "Keep modest: tools hit public rate-limited APIs "
        "(Nominatim, Open-Meteo)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N events (smoke test)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached per-event results and rerun everything",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Aggregate cached per-event results into results.json "
        "and print metrics without running anything",
    )
    args = parser.parse_args()

    events = load_events(args.input, args.results_csv, args.noaa_csv)
    if args.limit:
        events = events[: args.limit]

    model = build_model(args.model, args.api_key)
    tools = collect_tools()
    print(
        f"Loaded {len(events)} events · model={args.model} · "
        f"tools={[t.__name__ for t in tools]}",
        file=sys.stderr,
    )

    records_by_idx: dict[int, dict] = {}
    n_cached = n_ran = n_failed = n_missing = 0
    pending: list[tuple[int, dict]] = []

    def _label(event: dict) -> str:
        return (
            f"{(event['county'] or event['event_id']).title()}, "
            f"{event['state'].title()}"
        ).strip(", ")

    # Pass 1 (sync): serve cached results, decide what still needs running.
    for i, event in enumerate(events, 1):
        eid = event["event_id"]
        label = _label(event)
        cached = None if args.force else load_event_result(args.state_dir, eid)
        if cached is not None and cached["event"].get("claim_text") != event.get(
            "claim_text"
        ):
            # Input text changed since this result was cached (e.g. the
            # original fabricated inputs replaced reconstructed ones) —
            # the cached verdict is stale, rerun the event.
            print(
                f"[{i}/{len(events)}] {label} — input changed, cache invalidated",
                file=sys.stderr,
            )
            cached = None
        if cached is not None:
            records_by_idx[i] = cached
            n_cached += 1
            print(
                f"[{i}/{len(events)}] {label} — cached: {cached['verdict']} "
                f"(expected {event['expected_verdict']})",
                file=sys.stderr,
            )
            continue
        if args.report_only:
            n_missing += 1
            print(
                f"[{i}/{len(events)}] {label} — no cached result, skipped "
                f"(report-only)",
                file=sys.stderr,
            )
            continue
        pending.append((i, event))

    # Pass 2 (async): run pending events, at most --concurrency in flight.
    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def process(i: int, event: dict) -> None:
        nonlocal n_ran, n_failed
        label = _label(event)
        async with sem:
            print(f"[{i}/{len(events)}] {label} — running...", file=sys.stderr)
            try:
                record = await run_one(event, model, tools, args.request_limit)
            except Exception as e:  # not cached → retried on next run
                n_failed += 1
                print(
                    f"[{i}/{len(events)}] {label} — ERROR ({type(e).__name__}): {e}",
                    file=sys.stderr,
                )
                records_by_idx[i] = {
                    "event": event,
                    "verdict": "error",
                    "rationale": f"{type(e).__name__}: {e}",
                    "confidence": None,
                    "verdict_correct": "",
                    "tool_calls": [],
                    "usage": None,
                    "elapsed_seconds": None,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                }
                return
            save_event_result(args.state_dir, record)
            records_by_idx[i] = record
            n_ran += 1
            marker = {"yes": "+", "partial": "~", "no": "X"}.get(
                record["verdict_correct"], "?"
            )
            print(
                f"[{i}/{len(events)}] {label} — {marker} {record['verdict']} "
                f"(expected {event['expected_verdict']}) "
                f"conf={record['confidence']} tools={len(record['tool_calls'])} "
                f"{record['elapsed_seconds']}s",
                file=sys.stderr,
            )

    try:
        if pending:
            await asyncio.gather(*(process(i, e) for i, e in pending))
    finally:
        records = [records_by_idx[i] for i in sorted(records_by_idx)]
        if records:
            summary = summarize(records, args.model)
            dump = {
                "config": {
                    "model": args.model,
                    "request_limit": args.request_limit,
                    "tools": [t.__name__ for t in tools],
                    "input": str(args.input),
                    "instructions": BASELINE_INSTRUCTIONS,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
                "summary": summary,
                "events": records,
            }
            args.state_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.state_dir / "results.json"
            with open(out_path, "w") as f:
                json.dump(dump, f, indent=2)
            print(
                f"\nWrote {out_path} "
                f"({n_ran} ran, {n_cached} cached, {n_failed} failed, "
                f"{n_missing} missing)",
                file=sys.stderr,
            )
            print_summary(summary, args.model)


if __name__ == "__main__":
    asyncio.run(main())

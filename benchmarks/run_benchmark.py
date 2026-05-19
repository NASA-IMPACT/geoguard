"""Run GeoGuard on selected events and produce a per-claim results CSV.

Reads selected_events.csv (from select_events.py), runs the full GeoGuard
pipeline on each event's claim_text, and writes benchmark_results.csv with
one row per extracted claim per event.

Usage:
    # First select events
    python benchmarks/select_events.py --n 5

    # Then run the benchmark
    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --model anthropic:claude-sonnet-4-20250514
    python benchmarks/run_benchmark.py --input benchmarks/data/selected_events.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path

# Register tools before importing pipeline
import geoguard.tools.geospatial  # noqa: F401
import geoguard.tools.satellite  # noqa: F401
import geoguard.tools.weather  # noqa: F401
from geoguard import GeoGuard, Input
from geoguard.claims import Claim
from geoguard.metadata import ClaimGroup
from geoguard.pipeline import Report
from geoguard.rubrics import Rubric
from geoguard.tools.selector import SelectedTools
from geoguard.verifications import VerifierResult

INPUT_PATH = Path(__file__).parent / "data" / "selected_events.csv"
OUTPUT_PATH = Path(__file__).parent / "data" / "benchmark_65_results.csv"

RESULT_COLUMNS = [
    # --- event-level (from selected_events.csv) ---
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
    # --- claim-level (from GeoGuard) ---
    "claim_n",
    "claim",
    "verdict",
    "rationale",
    "tools_used",
    "rubric_score",
    # --- event-level rollup (from GeoGuard) ---
    "overall_verdict",
    "overall_confidence",
    "num_claims",
    "elapsed_seconds",
    # --- match ---
    "verdict_correct",
]


async def run_one(guard: GeoGuard, event: dict) -> list[dict]:
    """Run GeoGuard on a single event and return per-claim result rows."""
    inp = Input(text=event["claim_text"])
    t0 = time.monotonic()

    claims_in_order: list[Claim] = []
    tool_selections: dict[str, list[str]] = {}
    verifications: list[VerifierResult] = []
    rubric: Rubric | None = None
    report: Report | None = None

    try:
        async for item in guard(inp):
            if isinstance(item, Claim):
                claims_in_order.append(item)
            elif isinstance(item, SelectedTools):
                if item.claim:
                    tool_selections[item.claim.claim] = [
                        t.__name__ for t in item.tools
                    ]
            elif isinstance(item, VerifierResult):
                verifications.append(item)
            elif isinstance(item, Rubric):
                rubric = item
            elif isinstance(item, Report):
                report = item
    except Exception as e:
        elapsed = round(time.monotonic() - t0, 1)
        print(f"    ERROR: {e}", file=sys.stderr)
        return [{
            **_event_fields(event),
            "claim_n": 0,
            "claim": f"ERROR: {e}",
            "verdict": "error",
            "rationale": str(e),
            "tools_used": "",
            "rubric_score": "",
            "overall_verdict": "error",
            "overall_confidence": "",
            "num_claims": 0,
            "elapsed_seconds": elapsed,
            "verdict_correct": "",
        }]

    elapsed = round(time.monotonic() - t0, 1)

    if report is None:
        return [{
            **_event_fields(event),
            "claim_n": 0,
            "claim": "NO REPORT PRODUCED",
            "verdict": "error",
            "rationale": "",
            "tools_used": "",
            "rubric_score": "",
            "overall_verdict": "error",
            "overall_confidence": "",
            "num_claims": 0,
            "elapsed_seconds": elapsed,
            "verdict_correct": "",
        }]

    rubric_by_claim: dict[str, float] = {}
    if rubric:
        for cr in rubric.per_claim:
            rubric_by_claim[cr.claim.claim] = round(cr.score, 3)

    expected = event.get("expected_verdict", "").lower()
    overall = report.overall_verdict.value
    correct = _check_correct(expected, overall)

    rows: list[dict] = []
    for i, vr in enumerate(report.verifications, 1):
        v = vr.verification
        claim_text = v.claim.claim
        rows.append({
            **_event_fields(event),
            "claim_n": i,
            "claim": claim_text,
            "verdict": v.verdict.value,
            "rationale": v.rationale,
            "tools_used": ", ".join(
                tool_selections.get(claim_text, [tc.name for tc in vr.tool_calls])
            ),
            "rubric_score": rubric_by_claim.get(claim_text, ""),
            "overall_verdict": overall,
            "overall_confidence": round(report.rubric.confidence, 3),
            "num_claims": len(report.verifications),
            "elapsed_seconds": elapsed,
            "verdict_correct": correct,
        })

    return rows


def _event_fields(event: dict) -> dict:
    return {
        "event_id": event["event_id"],
        "event_type": event["event_type"],
        "state": event["state"],
        "county": event["county"],
        "region": event["region"],
        "begin_date": event["begin_date"],
        "lat": event["lat"],
        "lon": event["lon"],
        "damage_property": event["damage_property"],
        "deaths_direct": event["deaths_direct"],
        "expected_verdict": event["expected_verdict"],
        "perturbation_type": event["perturbation_type"],
    }


def _check_correct(expected: str, actual: str) -> str:
    if not expected:
        return ""
    if expected == actual:
        return "yes"
    if expected == "supports" and actual == "inconclusive":
        return "partial"
    if expected == "contradicts" and actual == "inconclusive":
        return "partial"
    return "no"


async def main():
    parser = argparse.ArgumentParser(description="Run GeoGuard benchmark.")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--reasoning", type=str, default=None)
    args = parser.parse_args()

    with open(args.input) as f:
        events = list(csv.DictReader(f))

    print(f"Loaded {len(events)} events from {args.input}", file=sys.stderr)

    guard = GeoGuard(
        model=args.model,
        api_key=args.api_key,
        reasoning_effort=args.reasoning,
    )

    all_rows: list[dict] = []
    for i, event in enumerate(events, 1):
        label = f"{event['county'].title()}, {event['state'].title()}"
        print(f"\n[{i}/{len(events)}] {label} ({event['event_type']})...", file=sys.stderr)

        rows = await run_one(guard, event)
        all_rows.extend(rows)

        for row in rows:
            verdict_marker = {"yes": "+", "partial": "~", "no": "X", "": "?"}.get(
                row["verdict_correct"], "?"
            )
            print(
                f"  {verdict_marker} claim {row['claim_n']}: "
                f"{row['verdict']} | {row['claim'][:80]}",
                file=sys.stderr,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Wrote {len(all_rows)} rows to {args.output}", file=sys.stderr)
    _print_summary(all_rows)


def _print_summary(rows: list[dict]) -> None:
    total_events = len({r["event_id"] for r in rows})
    total_claims = len([r for r in rows if r["verdict"] != "error"])
    correct = sum(1 for r in rows if r["verdict_correct"] == "yes")
    partial = sum(1 for r in rows if r["verdict_correct"] == "partial")
    wrong = sum(1 for r in rows if r["verdict_correct"] == "no")

    verdicts = {}
    for r in rows:
        v = r["verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1

    print(f"\nSummary ({total_events} events, {total_claims} claims):", file=sys.stderr)
    print(f"  Verdict distribution: {dict(sorted(verdicts.items()))}", file=sys.stderr)

    event_verdicts = {}
    for r in rows:
        eid = r["event_id"]
        if eid not in event_verdicts:
            event_verdicts[eid] = r["overall_verdict"]
    event_correct = sum(
        1 for eid, v in event_verdicts.items()
        if any(
            r["verdict_correct"] in ("yes", "partial")
            for r in rows
            if r["event_id"] == eid
        )
    )
    print(f"  Event-level: {event_correct}/{total_events} correct/partial", file=sys.stderr)

    confidences = [float(r["overall_confidence"]) for r in rows if r["overall_confidence"]]
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        print(f"  Avg confidence: {avg_conf:.1%}", file=sys.stderr)

    times = []
    seen_events = set()
    for r in rows:
        if r["event_id"] not in seen_events and r["elapsed_seconds"]:
            times.append(float(r["elapsed_seconds"]))
            seen_events.add(r["event_id"])
    if times:
        print(
            f"  Timing: avg={sum(times)/len(times):.1f}s, "
            f"min={min(times):.1f}s, max={max(times):.1f}s",
            file=sys.stderr,
        )


if __name__ == "__main__":
    asyncio.run(main())

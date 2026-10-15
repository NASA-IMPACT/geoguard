"""Run one worked example end to end and save a poster-ready trace.

Uses every registered measurement tool plus the GDACS catalog, and writes
data/case_studies/<case>.json (full trace) and <case>.md (claims, key tool
calls, per-claim verdicts and rubric scores, overall verdict).

Usage:
    uv run python scripts/run_case_study.py --case boris
    uv run python scripts/run_case_study.py --case valencia --model claude-cli:sonnet
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import geoguard.tools.catalogs  # noqa: F401  GDACS archive (+ live alerts if token set)
import geoguard.tools.geospatial  # noqa: F401
import geoguard.tools.weather  # noqa: F401
from geoguard import GeoGuard, Input
from geoguard import claude_cli
from geoguard.claims import Claim
from geoguard.metadata import ClaimGroup
from geoguard.pipeline import Report
from geoguard.tools.selector import SelectedTools

_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = _ROOT / "data" / "case_studies"

CASES = {
    # Europe. Three claims the record supports, one planted error (Vienna gusts).
    "boris": (
        "Storm Boris brought extreme rainfall to Central Europe in mid-September "
        "2024: between 12 and 16 September, more than 300 mm of rain fell in the "
        "town of Jeseník, Czechia. The Oder river at Opole, Poland rose to "
        "its highest level in over two decades, and flooding spread across Poland, "
        "Czechia, Austria and Slovakia. In Vienna, Austria, the storm brought wind "
        "gusts of over 150 km/h."
    ),
    "valencia": (
        "On 29 October 2024, a cut-off low (DANA) brought torrential rain to the "
        "province of Valencia, Spain. The Poyo ravine overflowed and flooded "
        "Paiporta, and river flows near Valencia reached their highest level in "
        "decades."
    ),
    # US. The original poster example.
    "beryl": (
        "Hurricane Beryl made landfall near Matagorda Bay, Texas on July 8, 2024. "
        "The storm brought sustained winds of 80 mph and significant storm surge "
        "along the Texas coast. Heavy rainfall caused widespread flooding in "
        "Houston, with reports of over 100 mm of rain falling over a 24-hour period."
    ),
}

# Result fields worth quoting on a poster, in display order.
KEY_FIELDS = [
    "total_mm",
    "peak_daily_mm",
    "prism_total_mm",
    "mrms_total_mm",
    "max_total_precip_mm",
    "max_daily_precip_mm",
    "max_peak_gust_kmh",
    "peak_speed_kmh",
    "peak_gust_kmh",
    "peak_m3s",
    "peak_date",
    "peak_to_median_ratio",
    "event_rank",
    "years_compared",
    "elevation_m",
]


def key_numbers(result) -> str:
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except ValueError:
            return result[:120]
    if not isinstance(result, dict):
        return str(result)[:120]
    if not result.get("found", True):
        return "no data (" + str(result.get("reason", "not found"))[:80] + ")"
    parts = [f"{k}={result[k]}" for k in KEY_FIELDS if result.get(k) is not None]
    for collection in ("events", "gauges", "stations"):
        items = result.get(collection) or []
        if items:
            first = items[0]
            label = first.get("name") or first.get("eventtype")
            extra = (
                first.get("alertlevel")
                or first.get("peak_m3s")
                or first.get("peak_gust_kmh")
            )
            parts.append(f"{collection}[0]={label} ({extra})")
    return ", ".join(parts) or json.dumps(result, default=str)[:160]


async def run(case: str, model: str) -> None:
    text = CASES[case]
    guard = GeoGuard(model=model, reasoning_effort="low")
    groups, selections, report = [], [], None
    claude_cli.current_cost_bucket.set(case)
    t0 = time.monotonic()
    async for event in guard(Input(text=text)):
        if isinstance(event, ClaimGroup):
            groups.append(event)
            print(
                f"[group] {event.metadata.event_type.value}: {len(event.claims)} claims"
            )
        elif isinstance(event, Claim):
            print(f"  [claim] {event.claim}")
        elif isinstance(event, SelectedTools):
            selections.append(event)
            print(f"  [tools] {[t.__name__ for t in event.tools]}")
        elif isinstance(event, Report):
            report = event
    elapsed = time.monotonic() - t0
    cost = claude_cli.cost_totals.pop(case, {})

    scores = {cr.claim.claim: cr.score for cr in report.rubric.per_claim}
    trace = {
        "case": case,
        "model": model,
        "input": text,
        "groups": [g.model_dump(mode="json") for g in groups],
        "verifications": [
            {
                "claim": vr.verification.claim.claim,
                "verdict": vr.verification.verdict.value,
                "rationale": vr.verification.rationale,
                "rubric_score": scores.get(vr.verification.claim.claim),
                "tool_calls": [
                    {"name": tc.name, "args": tc.args, "result": tc.result}
                    for tc in vr.tool_calls
                ],
            }
            for vr in report.verifications
        ],
        "overall_verdict": report.overall_verdict.value,
        "overall_confidence": report.rubric.confidence,
        "elapsed_seconds": round(elapsed, 1),
        "cost_usd": round(cost.get("cost_usd", 0.0), 4),
        "llm_calls": cost.get("calls", 0),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{case}.json").write_text(json.dumps(trace, indent=2, default=str))

    lines = [f"# Case study: {case}", "", f"> {text}", "", "## Claims"]
    for g in groups:
        loc = g.metadata.location
        where = getattr(loc, "name", None) or "?"
        start = getattr(g.metadata.time_range, "start", None)
        lines.append(
            f"- **{g.metadata.event_type.value.upper()} · {where} · {start:%Y-%m-%d}**"
            if start
            else f"- **{g.metadata.event_type.value.upper()} · {where}**"
        )
        lines += [f"  - {c.claim}" for c in g.claims]
    lines += ["", "## Key tool calls"]
    seen = set()
    for v in trace["verifications"]:
        for tc in v["tool_calls"]:
            key = (tc["name"], tc["args"])
            if key in seen:
                continue
            seen.add(key)
            args = json.loads(tc["args"]) if tc["args"].startswith("{") else {}
            where = f"{args.get('lat', '')}, {args.get('lon', '')}"
            when = args.get("start_date") or args.get("event_date") or ""
            if args.get("end_date") and args["end_date"] != when:
                when += f"→{args['end_date']}"
            lines.append(
                f"- `{tc['name']}({where}, {when})` → {key_numbers(tc['result'])}"
            )
    lines += [
        "",
        "## Final report",
        "",
        "| Claim | Verdict | Rubric |",
        "|---|---|---|",
    ]
    for v in trace["verifications"]:
        score = f"{v['rubric_score']:.0%}" if v["rubric_score"] is not None else "—"
        lines.append(f"| {v['claim']} | {v['verdict'].upper()} | {score} |")
    lines += [
        "",
        f"**Overall: {trace['overall_verdict'].upper()} · confidence "
        f"{trace['overall_confidence']:.0%}** · {model} · {elapsed:.0f} s · "
        f"${trace['cost_usd']:.2f} API-equivalent · {trace['llm_calls']} LLM calls",
        "",
        "## Rationales",
    ]
    lines += [
        f"- **{v['verdict'].upper()}** — {v['claim']}: {v['rationale']}"
        for v in trace["verifications"]
    ]
    (OUT_DIR / f"{case}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--case", choices=sorted(CASES), default="boris")
    p.add_argument("--model", default="claude-cli:sonnet")
    args = p.parse_args()
    asyncio.run(run(args.case, args.model))


if __name__ == "__main__":
    main()

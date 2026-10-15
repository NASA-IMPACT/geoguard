"""Run GeoGuard on the US + Europe evaluation set and summarize the results.

Reads data/eval_v2/events.csv (from build_eval_set.py) and verifies each
event's claim_text with the full GeoGuard pipeline. Only MEASUREMENT tools
are registered — ERA5 precipitation/wind, NOAA PRISM/MRMS, USGS streamflow,
Copernicus GloFAS discharge, elevation, OSM water bodies. The GDACS catalog
tool is deliberately left out: the European ground truth comes from
disaster catalogs, and letting a catalog confirm its own events would leak
the label into the evidence. MODIS and VEDA STAC are left out for runtime.

Resumable: every finished event is written to data/eval_v2/events/<id>.json
immediately and skipped on rerun. Failed events are recorded under
errors/ and retried on the next run. If the Claude subscription hits its
usage window, the run stops cleanly — rerun later to continue.

Usage:
    uv run python scripts/run_eval.py --limit 4          # smoke test
    uv run python scripts/run_eval.py                    # full set
    uv run python scripts/run_eval.py --summary          # metrics only
    uv run python scripts/run_eval.py --model openai:gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import geoguard.tools.geospatial  # noqa: F401  elevation, nearest water body
import geoguard.tools.weather  # noqa: F401  ERA5, PRISM/MRMS, USGS, GloFAS
from geoguard import GeoGuard, Input
from geoguard import claude_cli

_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = _ROOT / "data" / "eval_v2"
EVENTS_CSV = EVAL_DIR / "events.csv"
RESULT_TRUNCATE = 4000


def _trim(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= RESULT_TRUNCATE else text[:RESULT_TRUNCATE] + "…"


async def run_event(
    guard: GeoGuard, ev: dict, model: str, sem: asyncio.Semaphore
) -> str:
    out = EVAL_DIR / "events" / f"{ev['event_id']}.json"
    err = EVAL_DIR / "errors" / f"{ev['event_id']}.json"
    if out.exists() and json.loads(out.read_text()).get("model") == model:
        return "cached"  # a result from a different model is re-run and overwritten
    async with sem:
        claude_cli.current_cost_bucket.set(ev["event_id"])
        t0 = time.monotonic()
        try:
            report = await guard.run(Input(text=ev["claim_text"]))
        except claude_cli.ClaudeUsageLimitError:
            raise
        except Exception as e:  # noqa: BLE001 — record and move on; rerun retries
            err.parent.mkdir(parents=True, exist_ok=True)
            err.write_text(
                json.dumps({"event": ev, "error": f"{type(e).__name__}: {e}"}, indent=2)
            )
            print(f"  ✗ {ev['event_id']}: {type(e).__name__}: {str(e)[:160]}")
            return "error"
        elapsed = time.monotonic() - t0

    cost = claude_cli.cost_totals.pop(ev["event_id"], {})
    claims = [
        {
            "claim": vr.verification.claim.claim,
            "verdict": vr.verification.verdict.value,
            "rationale": vr.verification.rationale,
            "tools": sorted({tc.name for tc in vr.tool_calls}),
            "tool_calls": [
                {"name": tc.name, "args": tc.args, "result": _trim(tc.result)}
                for tc in vr.tool_calls
            ],
        }
        for vr in report.verifications
    ]
    record = {
        "event": ev,
        "model": model,
        "overall_verdict": report.overall_verdict.value,
        "rubric_confidence": round(report.rubric.confidence, 3),
        "num_claims": len(claims),
        "claims": claims,
        "elapsed_seconds": round(elapsed, 1),
        "cost_usd": round(cost.get("cost_usd", 0.0), 4),
        "llm_calls": cost.get("calls", 0),
        "input_tokens": cost.get("input_tokens", 0),
        "output_tokens": cost.get("output_tokens", 0),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=str))
    err.unlink(missing_ok=True)
    verdicts = Counter(c["verdict"] for c in claims)
    print(
        f"  ✓ {ev['event_id']:14s} {ev['label']:10s} → {record['overall_verdict']:12s} "
        f"{dict(verdicts)} {elapsed:5.0f}s ${record['cost_usd']:.3f}"
    )
    return "done"


async def run_all(args) -> None:
    events = list(csv.DictReader(open(EVENTS_CSV)))
    events.sort(key=lambda e: int(e["run_order"]))
    if args.limit:
        events = events[: args.limit]
    if args.only:
        events = [e for e in events if e["event_id"] in set(args.only)]
    guard = GeoGuard(
        model=args.model, reasoning_effort=args.reasoning, max_claims=args.max_claims
    )
    sem = asyncio.Semaphore(args.parallel_events)
    print(
        f"Running {len(events)} events · model={args.model} · {args.parallel_events} in parallel"
    )
    tasks = [
        asyncio.create_task(run_event(guard, ev, args.model, sem)) for ev in events
    ]
    try:
        results = await asyncio.gather(*tasks)
    except claude_cli.ClaudeUsageLimitError as e:
        for t in tasks:
            t.cancel()
        print(f"\nStopped: Claude usage limit reached ({e}). Rerun later to resume.")
        sys.exit(2)
    print(f"\nDone: {Counter(results)}")


# ----------------------------------------------------------------- summary


def rate(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "—"


def wilson(n: int, d: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion n/d."""
    if not d:
        return (0.0, 0.0)
    p = n / d
    centre = (p + z * z / (2 * d)) / (1 + z * z / d)
    half = z * ((p * (1 - p) / d + z * z / (4 * d * d)) ** 0.5) / (1 + z * z / d)
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate_ci(n: int, d: int) -> str:
    lo, hi = wilson(n, d)
    return (
        f"{n}/{d} = {100 * n / d:.0f}% (95% CI {100 * lo:.0f}–{100 * hi:.0f}%)"
        if d
        else "—"
    )


def summarize() -> dict:
    # Only events in the current set count (results for dropped events stay on disk).
    current = {e["event_id"] for e in csv.DictReader(open(EVENTS_CSV))}
    recs = [
        json.loads(p.read_text()) for p in sorted((EVAL_DIR / "events").glob("*.json"))
    ]
    recs = [r for r in recs if r["event"]["event_id"] in current]
    if not recs:
        print("No results yet.")
        return {}
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        ev = r["event"]
        for key in ("ALL", ev["region"], f"{ev['region']}·{ev['event_class']}"):
            groups[key].append(r)

    summary: dict = {"n_events": len(recs), "groups": {}}
    print(f"\n{len(recs)} events evaluated\n")
    header = f"{'group':12s} {'real: S/C/I':>13s} {'not rejected':>15s} {'fab: S/C/I':>12s} {'fab endorsed':>14s} {'fab refuted':>14s}"
    print(header)
    print("-" * len(header))
    for key in sorted(groups, key=lambda k: (k != "ALL", k)):
        rs = groups[key]
        real = [r for r in rs if r["event"]["label"] == "real"]
        fab = [r for r in rs if r["event"]["label"] == "fabricated"]
        rv = Counter(r["overall_verdict"] for r in real)
        fv = Counter(r["overall_verdict"] for r in fab)
        not_rej = len(real) - rv["contradicts"]
        print(
            f"{key:12s} {rv['supports']:>4d}/{rv['contradicts']:>2d}/{rv['inconclusive']:<4d}"
            f"{rate(not_rej, len(real)):>15s}  {fv['supports']:>3d}/{fv['contradicts']:>2d}/{fv['inconclusive']:<3d}"
            f"{rate(fv['supports'], len(fab)):>14s} {rate(fv['contradicts'], len(fab)):>14s}"
        )
        summary["groups"][key] = {
            "real": dict(rv),
            "real_n": len(real),
            "fabricated": dict(fv),
            "fabricated_n": len(fab),
        }

    print("\nHeadline (95% Wilson intervals):")
    headline = {}
    for key in ("ALL", "US", "EU"):
        rs = groups.get(key, [])
        real = [r for r in rs if r["event"]["label"] == "real"]
        fab = [r for r in rs if r["event"]["label"] == "fabricated"]
        not_rej = sum(r["overall_verdict"] != "contradicts" for r in real)
        endorsed = sum(r["overall_verdict"] == "supports" for r in fab)
        print(
            f"  {key:4s} real not rejected {rate_ci(not_rej, len(real))} · fabricated endorsed {rate_ci(endorsed, len(fab))}"
        )
        headline[key] = {
            "real_not_rejected": [not_rej, len(real)],
            "fabricated_endorsed": [endorsed, len(fab)],
        }
    for hardness in ("easy", "hard"):
        fab = [
            r
            for r in recs
            if r["event"]["label"] == "fabricated"
            and r["event"]["hardness"] == hardness
        ]
        endorsed = sum(r["overall_verdict"] == "supports" for r in fab)
        print(f"  {hardness:4s} fabrications endorsed {rate_ci(endorsed, len(fab))}")
        headline[f"fabricated_{hardness}_endorsed"] = [endorsed, len(fab)]
    summary["headline"] = headline

    print("\nFabricated events by perturbation (endorsed = SUPPORTS):")
    by_kind: dict[str, Counter] = defaultdict(Counter)
    for r in recs:
        ev = r["event"]
        if ev["label"] == "fabricated":
            by_kind[f"{ev['perturbation']}/{ev['hardness']}"][r["overall_verdict"]] += 1
    for kind, c in sorted(by_kind.items()):
        n = sum(c.values())
        print(
            f"  {kind:26s} n={n:3d}  endorsed {rate(c['supports'], n):>11s}  refuted {rate(c['contradicts'], n):>11s}"
        )
    summary["fabricated_by_kind"] = {k: dict(v) for k, v in by_kind.items()}

    cost = [r["cost_usd"] for r in recs]
    secs = [r["elapsed_seconds"] for r in recs]
    calls = [r["llm_calls"] for r in recs]
    summary["mean_cost_usd"] = round(sum(cost) / len(cost), 4)
    summary["mean_seconds"] = round(sum(secs) / len(secs), 1)
    summary["mean_llm_calls"] = round(sum(calls) / len(calls), 1)
    summary["model"] = recs[0]["model"]
    print(
        f"\nmodel {summary['model']} · mean ${summary['mean_cost_usd']:.3f}/event "
        f"(API-equivalent) · {summary['mean_seconds']:.0f} s/event · "
        f"{summary['mean_llm_calls']:.0f} LLM calls/event"
    )
    (EVAL_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="claude-cli:sonnet")
    p.add_argument(
        "--reasoning", default="low", choices=["minimal", "low", "medium", "high"]
    )
    p.add_argument("--max-claims", type=int, default=3)
    p.add_argument("--parallel-events", type=int, default=3)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only", nargs="*", help="Run only these event_ids")
    p.add_argument(
        "--summary",
        action="store_true",
        help="Print metrics from saved results and exit",
    )
    args = p.parse_args()
    if not args.summary:
        asyncio.run(run_all(args))
    summarize()


if __name__ == "__main__":
    main()

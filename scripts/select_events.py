"""Select diverse, high-quality NOAA Storm Events for GeoGuard benchmarking.

Filters flood events to those GeoGuard's tools can actually verify (US,
coordinates, narrative), scores them by richness, then samples a diverse
subset across regions, event types, and severity levels.

Usage:
    python scripts/select_events.py --n 5
    python scripts/select_events.py --n 25 --seed 42
    python scripts/select_events.py --n 5 --show  # preview without writing
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = _ROOT / "data" / "NOAA-Storm-Events-Data.csv"
OUTPUT_PATH = _ROOT / "data" / "selected_events.csv"

USABLE_EVENT_TYPES = {"Flood", "Flash Flood", "Coastal Flood", "Lakeshore Flood"}

OUTPUT_COLUMNS = [
    "event_id",
    "episode_id",
    "event_type",
    "state",
    "county",
    "region",
    "begin_date",
    "end_date",
    "lat",
    "lon",
    "flood_cause",
    "damage_property",
    "damage_crops",
    "deaths_direct",
    "source",
    "quality_score",
    "expected_verdict",
    "perturbation_type",
    "claim_text",
    "event_narrative",
    "episode_narrative",
]

REGIONS = {
    "Gulf Coast": {"TEXAS", "LOUISIANA", "MISSISSIPPI", "ALABAMA", "FLORIDA"},
    "Southeast": {"GEORGIA", "SOUTH CAROLINA", "NORTH CAROLINA", "VIRGINIA", "TENNESSEE"},
    "Northeast": {"NEW YORK", "PENNSYLVANIA", "NEW JERSEY", "CONNECTICUT", "MASSACHUSETTS", "MARYLAND", "DELAWARE"},
    "Midwest": {"OHIO", "IOWA", "MISSOURI", "ILLINOIS", "INDIANA", "MICHIGAN", "WISCONSIN", "MINNESOTA"},
    "West": {"CALIFORNIA", "OREGON", "WASHINGTON", "NEVADA", "ARIZONA", "UTAH", "COLORADO", "NEW MEXICO"},
    "Appalachia": {"WEST VIRGINIA", "KENTUCKY"},
    "Caribbean": {"PUERTO RICO", "VIRGIN ISLANDS"},
}


def region_of(state: str) -> str:
    for name, states in REGIONS.items():
        if state in states:
            return name
    return "Other"


def parse_damage(val: str) -> float:
    """Parse NOAA damage strings like '250.00K', '1.50M' to a float in dollars."""
    if not val or val in ("0", "0.00K"):
        return 0.0
    val = val.strip().upper()
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    for suffix, mult in multipliers.items():
        if val.endswith(suffix):
            try:
                return float(val[:-1]) * mult
            except ValueError:
                return 0.0
    try:
        return float(val)
    except ValueError:
        return 0.0


def quality_score(row: dict) -> float:
    """Score an event's suitability for benchmarking (higher = better).

    Rewards: long narratives (more claims to extract), specific quantities
    mentioned in text, property/crop damage, deaths (higher-stakes events
    are more important to verify correctly), and specific flood causes.
    """
    score = 0.0

    narrative = row.get("EVENT_NARRATIVE", "")
    episode = row.get("EPISODE_NARRATIVE", "")
    combined = narrative + " " + episode

    # Narrative length — longer = more claims to extract
    score += min(len(narrative) / 100, 5.0)

    # Quantities in text — numbers make claims verifiable
    numbers = re.findall(r"\d+\.?\d*", combined)
    score += min(len(numbers) * 0.5, 3.0)

    # Specific keywords that produce verifiable claims
    verifiable_keywords = [
        "inches", "mm", "feet", "mph", "record", "historic",
        "crest", "stage", "gauge", "rainfall", "wind",
    ]
    for kw in verifiable_keywords:
        if kw in combined.lower():
            score += 0.5

    # Damage / deaths — higher stakes
    prop_dmg = parse_damage(row.get("DAMAGE_PROPERTY", ""))
    crop_dmg = parse_damage(row.get("DAMAGE_CROPS", ""))
    if prop_dmg > 0:
        score += 2.0
    if crop_dmg > 0:
        score += 1.0
    deaths = int(row.get("DEATHS_DIRECT", "0") or 0)
    if deaths > 0:
        score += 3.0

    # Non-generic flood cause
    cause = row.get("FLOOD_CAUSE", "")
    if cause and cause != "Heavy Rain":
        score += 1.5

    return score


def build_claim_text(row: dict) -> str:
    """Turn a NOAA row into a natural-language claim paragraph.

    This is what GeoGuard will verify — a self-contained description
    of the event synthesized from the structured NOAA fields.
    """
    parts = []

    event_type = row["EVENT_TYPE"]
    location = row["CZ_NAME"].title()
    state = row["STATE"].title()
    parts.append(f"A {event_type.lower()} occurred in {location}, {state}.")

    begin = row.get("BEGIN_DATE_TIME", "")
    end = row.get("END_DATE_TIME", "")
    if begin:
        parts.append(f"The event began on {begin}")
        if end and end != begin:
            parts[-1] += f" and ended on {end}."
        else:
            parts[-1] += "."

    cause = row.get("FLOOD_CAUSE", "")
    if cause:
        parts.append(f"The cause was {cause.lower()}.")

    lat = row.get("BEGIN_LAT", "")
    lon = row.get("BEGIN_LON", "")
    begin_loc = row.get("BEGIN_LOCATION", "")
    if begin_loc:
        coord_str = f" ({lat}, {lon})" if lat and lon else ""
        parts.append(f"It was reported near {begin_loc}{coord_str}.")

    prop_dmg = row.get("DAMAGE_PROPERTY", "")
    crop_dmg = row.get("DAMAGE_CROPS", "")
    if prop_dmg and prop_dmg not in ("0", "0.00K"):
        parts.append(f"Property damage was estimated at {prop_dmg}.")
    if crop_dmg and crop_dmg not in ("0", "0.00K"):
        parts.append(f"Crop damage was estimated at {crop_dmg}.")

    deaths = int(row.get("DEATHS_DIRECT", "0") or 0)
    injuries = int(row.get("INJURIES_DIRECT", "0") or 0)
    if deaths > 0 or injuries > 0:
        parts.append(f"There were {deaths} direct deaths and {injuries} direct injuries.")

    narrative = row.get("EVENT_NARRATIVE", "").strip()
    if narrative:
        parts.append(narrative)

    episode = row.get("EPISODE_NARRATIVE", "").strip()
    if episode:
        parts.append(episode)

    return " ".join(parts)


def load_candidates() -> list[dict]:
    """Load and filter NOAA events to benchmark-usable flood events."""
    with open(DATA_PATH) as f:
        reader = csv.DictReader(f)
        candidates = []
        for row in reader:
            if row["EVENT_TYPE"] not in USABLE_EVENT_TYPES:
                continue
            if not row.get("BEGIN_LAT", "").strip():
                continue
            if not row.get("BEGIN_LON", "").strip():
                continue
            if not row.get("EVENT_NARRATIVE", "").strip():
                continue
            # US-only (USGS / PRISM tools need it)
            try:
                lat = float(row["BEGIN_LAT"])
                lon = float(row["BEGIN_LON"])
            except ValueError:
                continue
            if not (24.0 <= lat <= 50.0 and -125.0 <= lon <= -66.0):
                # Continental US bounding box (skip Puerto Rico / territories
                # since USGS gauge coverage is sparse there)
                continue

            row["_quality"] = quality_score(row)
            row["_region"] = region_of(row["STATE"])
            row["_lat"] = lat
            row["_lon"] = lon
            candidates.append(row)

    return candidates


def select(candidates: list[dict], n: int, seed: int) -> list[dict]:
    """Pick n diverse, high-quality events.

    Strategy:
    1. Sort by quality score (best first).
    2. Take the top 30% as the "good pool".
    3. From that pool, greedily pick events that maximize diversity:
       - Spread across regions
       - Mix of Flood vs Flash Flood
       - No two events from the same county
    """
    rng = random.Random(seed)

    candidates.sort(key=lambda r: r["_quality"], reverse=True)
    pool_size = max(n * 10, int(len(candidates) * 0.3))
    pool = candidates[:pool_size]
    rng.shuffle(pool)

    selected: list[dict] = []
    used_counties: set[str] = set()
    region_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}

    def diversity_penalty(row: dict) -> float:
        penalty = 0.0
        penalty += region_counts.get(row["_region"], 0) * 3.0
        penalty += type_counts.get(row["EVENT_TYPE"], 0) * 2.0
        return penalty

    for _ in range(n):
        best = None
        best_score = -999.0
        for row in pool:
            county_key = f"{row['STATE']}_{row['CZ_NAME']}"
            if county_key in used_counties:
                continue
            effective = row["_quality"] - diversity_penalty(row)
            if effective > best_score:
                best_score = effective
                best = row
        if best is None:
            break

        selected.append(best)
        pool.remove(best)
        county_key = f"{best['STATE']}_{best['CZ_NAME']}"
        used_counties.add(county_key)
        region_counts[best["_region"]] = region_counts.get(best["_region"], 0) + 1
        type_counts[best["EVENT_TYPE"]] = type_counts.get(best["EVENT_TYPE"], 0) + 1

    return selected


def format_output(events: list[dict]) -> list[dict]:
    """Shape selected events into the benchmark input format (one row per event)."""
    out = []
    for row in events:
        out.append({
            "event_id": row["EVENT_ID"],
            "episode_id": row["EPISODE_ID"],
            "event_type": row["EVENT_TYPE"],
            "state": row["STATE"],
            "county": row["CZ_NAME"],
            "region": row["_region"],
            "begin_date": row["BEGIN_DATE_TIME"],
            "end_date": row["END_DATE_TIME"],
            "lat": row["_lat"],
            "lon": row["_lon"],
            "flood_cause": row.get("FLOOD_CAUSE", ""),
            "damage_property": row.get("DAMAGE_PROPERTY", ""),
            "damage_crops": row.get("DAMAGE_CROPS", ""),
            "deaths_direct": int(row.get("DEATHS_DIRECT", "0") or 0),
            "source": row.get("SOURCE", ""),
            "quality_score": round(row["_quality"], 1),
            "expected_verdict": "supports",
            "perturbation_type": "none",
            "claim_text": build_claim_text(row),
            "event_narrative": row.get("EVENT_NARRATIVE", ""),
            "episode_narrative": row.get("EPISODE_NARRATIVE", ""),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Select NOAA events for GeoGuard benchmarking.")
    parser.add_argument("--n", type=int, default=5, help="Number of events to select")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--show", action="store_true", help="Print to stdout instead of writing file")
    args = parser.parse_args()

    candidates = load_candidates()
    print(f"Loaded {len(candidates)} usable flood events from NOAA data", file=sys.stderr)

    events = select(candidates, args.n, args.seed)
    output = format_output(events)

    print(f"\nSelected {len(output)} events:", file=sys.stderr)
    for i, e in enumerate(output, 1):
        print(
            f"  {i}. [{e['event_type']}] {e['county'].title()}, {e['state'].title()} "
            f"— {e['begin_date']} — region={e['region']}, "
            f"quality={e['quality_score']}, damage={e['damage_property']}",
            file=sys.stderr,
        )

    if args.show:
        writer = csv.DictWriter(sys.stdout, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output)
    else:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(output)
        print(f"\nWrote {OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""Build the US + Europe evaluation set: real events and fabricated counterparts.

Real events come from two authoritative catalogs, independent of the tools
GeoGuard verifies with:
  - US:     NOAA Storm Events 2024 (data/NOAA-Storm-Events-Data.csv) — floods,
            flash floods, heavy rain, and thunderstorm winds with a MEASURED gust.
  - Europe: Copernicus Emergency Management Service rapid-mapping activations
            for floods and storms in Europe (2023-2026), fetched live.
            Activations named only by country are excluded — a country-wide
            claim can't be checked against point measurements.

Real and fabricated claims are rendered from the SAME sentence template, so
nothing about the wording reveals which is which. Fabrications perturb one
field of a real event, in two difficulty tiers:
  - wrong_date        easy: shifted 90-150 days     hard: shifted 21-35 days
  - wrong_location    easy: moved >= 800 km         hard: moved 150-400 km
  - inflated_magnitude (US storms) easy: gust x2.5  hard: gust x1.6
A perturbation is kept only if the catalog shows no same-kind event in the
same state/country within a few days of the fabricated date — otherwise the
"fabrication" might accidentally be true.

Rows are written in an interleaved run order (US real, EU real, US fake,
EU fake, ...) so a partially completed run is still a balanced sample.

Usage:
    uv run python scripts/build_eval_set.py                  # default sizes
    uv run python scripts/build_eval_set.py --us-real 70 --us-fake 35 --eu-fake 35
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
from geopy.distance import geodesic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_events import region_of  # noqa: E402

from geoguard.schemas import Region, region_contains  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
NOAA_PATH = _ROOT / "data" / "NOAA-Storm-Events-Data.csv"
EXCLUDE_PATH = _ROOT / "data" / "benchmark_65_events.csv"
OUTPUT_PATH = _ROOT / "data" / "eval_v2" / "events.csv"
CEMS_URL = (
    "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/"
    "public-activations-info/?limit=100"
)

FLOOD_TYPES = {"Flash Flood", "Flood", "Heavy Rain"}
MIN_MEASURED_GUST_KT = 58.0  # NWS severe threshold
KT_TO_MPH = 1.15078
LATEST_DATE = date.today() - timedelta(days=10)  # leave archives time to fill

EUROPE = {
    "Albania",
    "Austria",
    "Belgium",
    "Bosnia and Herzegovina",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czech Republic",
    "Czechia",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Iceland",
    "Ireland",
    "Italy",
    "Kosovo",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Moldova",
    "Montenegro",
    "Netherlands",
    "North Macedonia",
    "Norway",
    "Poland",
    "Portugal",
    "Romania",
    "Serbia",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
    "Switzerland",
    "Ukraine",
    "United Kingdom",
}

COLUMNS = [
    "event_id",
    "run_order",
    "region",
    "source",
    "event_class",
    "label",
    "perturbation",
    "hardness",
    "base_event_id",
    "place",
    "admin",
    "date",
    "lat",
    "lon",
    "magnitude",
    "claim_text",
]


def fmt_date(d: date) -> str:
    return f"{d:%B} {d.day}, {d.year}"


def render(ev: dict) -> str:
    """One sentence per event, identical template for real and fabricated."""
    where = f"{ev['place']} ({float(ev['lat']):.3f}, {float(ev['lon']):.3f})"
    when = fmt_date(ev["date"])
    if ev["region"] == "US":
        if ev["event_class"] == "storm":
            return (
                f"A thunderstorm produced a measured wind gust of "
                f"{ev['magnitude']:.0f} mph near {where}, USA, on {when}."
            )
        kind = {
            "Flash Flood": "A flash flood",
            "Flood": "A flood",
            "Heavy Rain": "Heavy rain causing flooding",
        }[ev["event_type"]]
        return f"{kind} occurred near {where}, USA, on {when}."
    if ev["event_class"] == "storm":
        return f"A severe storm struck {where} on {when}."
    return f"Flooding affected {where} on {when}."


# ---------------------------------------------------------------- US (NOAA)


def load_us() -> list[dict]:
    excluded = set()
    if EXCLUDE_PATH.exists():
        excluded = {r["event_id"] for r in csv.DictReader(open(EXCLUDE_PATH))}
    events = []
    for r in csv.DictReader(open(NOAA_PATH, encoding="utf-8", errors="replace")):
        if r["EVENT_ID"] in excluded or not (r["BEGIN_LAT"] and r["BEGIN_LON"]):
            continue
        t = r["EVENT_TYPE"]
        if t in FLOOD_TYPES:
            cls, mag = "flood", None
        elif (
            t == "Thunderstorm Wind"
            and r["MAGNITUDE_TYPE"] == "MG"
            and float(r["MAGNITUDE"] or 0) >= MIN_MEASURED_GUST_KT
        ):
            cls, mag = "storm", float(r["MAGNITUDE"]) * KT_TO_MPH
        else:
            continue
        d = datetime.strptime(r["BEGIN_DATE_TIME"], "%d-%b-%y %H:%M:%S").date()
        place = r["BEGIN_LOCATION"].title() or r["CZ_NAME"].title()
        events.append(
            {
                "event_id": f"US-{r['EVENT_ID']}",
                "region": "US",
                "source": "NOAA Storm Events",
                "event_class": cls,
                "event_type": t,
                "label": "real",
                "place": f"{place}, {r['STATE'].title()}",
                "admin": r["STATE"].title(),
                "date": d,
                "lat": float(r["BEGIN_LAT"]),
                "lon": float(r["BEGIN_LON"]),
                "magnitude": mag,
                "us_region": region_of(r["STATE"]),
            }
        )
    return events


def stratified_sample(
    events: list[dict], n: int, key, rng: random.Random
) -> list[dict]:
    """Round-robin over strata so the sample spans regions and months."""
    strata: dict = {}
    for e in events:
        strata.setdefault(key(e), []).append(e)
    for bucket in strata.values():
        rng.shuffle(bucket)
    keys = sorted(strata)
    rng.shuffle(keys)
    out = []
    while len(out) < n and any(strata.values()):
        for k in keys:
            if strata[k] and len(out) < n:
                out.append(strata[k].pop())
    return out


# ------------------------------------------------------------ Europe (CEMS)


def load_eu() -> list[dict]:
    acts, url = [], CEMS_URL
    with httpx.Client(timeout=30) as c:
        while url:
            page = c.get(url).json()
            acts += page["results"]
            url = page.get("next")
    events = []
    for a in acts:
        if not (set(a["countries"]) & EUROPE) or a["category"] not in (
            "Flood",
            "Storm",
        ):
            continue
        lon, lat = (
            float(v) for v in a["centroid"].removeprefix("POINT (").rstrip(")").split()
        )
        place = a["name"].strip().rstrip(".")
        for prefix in (
            "Floods in ",
            "Flood in ",
            "Flash flood in ",
            "Storms in ",
            "Storm in ",
        ):
            if place.startswith(prefix):
                place = place[len(prefix) :].strip()
                break
        # Skip activations named only by country ("Flood in Spain"): a
        # country-wide claim can't be checked at a point, in either direction.
        parts = [p.strip() for p in place.replace(" and ", ",").split(",") if p.strip()]
        if place in EUROPE or all(p in EUROPE for p in parts):
            continue
        d = date.fromisoformat(a["eventTime"][:10])
        if d > LATEST_DATE:
            continue
        events.append(
            {
                "event_id": f"EU-{a['code']}",
                "region": "EU",
                "source": "Copernicus EMS",
                "event_class": "storm" if a["category"] == "Storm" else "flood",
                "event_type": a["category"],
                "label": "real",
                "place": place,
                "admin": a["countries"][0],
                "date": d,
                "lat": lat,
                "lon": lon,
                "magnitude": None,
            }
        )
    return events


# ---------------------------------------------------------- fabrication


def collides(pool: list[dict], admin: str, cls: str, d: date, days: int) -> bool:
    """Is there a real catalog event of this class in `admin` near date `d`?"""
    return any(
        e["admin"] == admin
        and e["event_class"] == cls
        and abs((e["date"] - d).days) <= days
        for e in pool
    )


def fabricate(
    base: dict, kind: str, hard: bool, pool: list[dict], rng: random.Random
) -> dict | None:
    fake = {
        **base,
        "label": "fabricated",
        "perturbation": kind,
        "hardness": "hard" if hard else "easy",
        "base_event_id": base["event_id"],
    }
    window = 5 if base["region"] == "US" else 10
    if kind == "wrong_date":
        lo, hi = (21, 35) if hard else (90, 150)
        for _ in range(30):
            shift = rng.randint(lo, hi) * rng.choice((-1, 1))
            d = base["date"] + timedelta(days=shift)
            # US collisions are checked against NOAA 2024, so stay inside it.
            lo_d, hi_d = (
                (date(2024, 1, 1), date(2024, 12, 31))
                if base["region"] == "US"
                else (date(2023, 1, 1), LATEST_DATE)
            )
            if lo_d <= d <= hi_d and not collides(
                pool, base["admin"], base["event_class"], d, window
            ):
                fake["date"] = d
                return fake
        return None
    if kind == "wrong_location":
        lo, hi = (150, 400) if hard else (800, 4000)
        donors = [e for e in pool if e["event_class"] == base["event_class"]]
        rng.shuffle(donors)
        for donor in donors:
            dist = geodesic((base["lat"], base["lon"]), (donor["lat"], donor["lon"])).km
            if lo <= dist <= hi and not collides(
                pool, donor["admin"], base["event_class"], base["date"], window
            ):
                fake.update(
                    place=donor["place"],
                    admin=donor["admin"],
                    lat=donor["lat"],
                    lon=donor["lon"],
                )
                return fake
        return None
    if kind == "inflated_magnitude":
        fake["magnitude"] = base["magnitude"] * (1.6 if hard else 2.5)
        return fake
    raise ValueError(kind)


def make_fakes(
    bases: list[dict], n: int, pool: list[dict], rng: random.Random
) -> list[dict]:
    plan = [
        ("wrong_date", False),
        ("wrong_location", False),
        ("wrong_date", True),
        ("wrong_location", True),
    ]
    fakes, i = [], 0
    for base in bases:
        if len(fakes) >= n:
            break
        options = plan + (
            [("inflated_magnitude", False), ("inflated_magnitude", True)]
            if base.get("magnitude")
            else []
        )
        kind, hard = options[i % len(options)]
        fake = fabricate(base, kind, hard, pool, rng)
        if fake:
            fake["event_id"] = f"{base['region']}-FAB-{len(fakes) + 1:03d}"
            fakes.append(fake)
            i += 1
    return fakes


# ------------------------------------------------------------------ main


def interleave(*groups: list[dict]) -> list[dict]:
    out, iters = [], [iter(g) for g in groups]
    while iters:
        for it in list(iters):
            try:
                out.append(next(it))
            except StopIteration:
                iters.remove(it)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--us-flood-real", type=int, default=55)
    p.add_argument("--us-storm-real", type=int, default=15)
    p.add_argument("--us-fake", type=int, default=35)
    p.add_argument("--eu-fake", type=int, default=35)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    rng = random.Random(args.seed)

    us_all = load_us()
    us_floods = stratified_sample(
        [e for e in us_all if e["event_class"] == "flood"],
        args.us_flood_real,
        key=lambda e: (e["us_region"], e["date"].month),
        rng=rng,
    )
    us_storms = stratified_sample(
        [e for e in us_all if e["event_class"] == "storm"],
        args.us_storm_real,
        key=lambda e: (e["us_region"], e["date"].month),
        rng=rng,
    )
    us_real = us_floods + us_storms
    rng.shuffle(us_real)

    eu_real = load_eu()
    rng.shuffle(eu_real)

    us_fake = make_fakes(us_real, args.us_fake, us_all, rng)
    eu_fake = make_fakes(eu_real, args.eu_fake, eu_real, rng)

    rows = interleave(us_real, eu_real, us_fake, eu_fake)
    # Copernicus files overseas territories (Réunion, Mayotte) under their
    # European country; keep only events physically in Europe. Applied last
    # so the sampling above — and every other row — is unchanged.
    rows = [
        e
        for e in rows
        if e["region"] != "EU" or region_contains(Region.EUROPE, e["lat"], e["lon"])
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for i, ev in enumerate(rows):
            w.writerow(
                {
                    **{k: ev.get(k, "") for k in COLUMNS},
                    "run_order": i,
                    "perturbation": ev.get("perturbation", "none"),
                    "hardness": ev.get("hardness", ""),
                    "date": ev["date"].isoformat(),
                    "magnitude": f"{ev['magnitude']:.0f}"
                    if ev.get("magnitude")
                    else "",
                    "claim_text": render(ev),
                }
            )

    print(f"Wrote {len(rows)} events -> {OUTPUT_PATH.relative_to(_ROOT)}")
    groups = {
        "US real": [e for e in rows if e["region"] == "US" and e["label"] == "real"],
        "EU real": [e for e in rows if e["region"] == "EU" and e["label"] == "real"],
        "US fabricated": [
            e for e in rows if e["region"] == "US" and e["label"] == "fabricated"
        ],
        "EU fabricated": [
            e for e in rows if e["region"] == "EU" and e["label"] == "fabricated"
        ],
    }
    for name, g in groups.items():
        kinds = {}
        for e in g:
            k = f"{e.get('perturbation', e['event_class'])}/{e.get('hardness', '')}".rstrip(
                "/"
            )
            kinds[k] = kinds.get(k, 0) + 1
        print(f"  {name:14s} {len(g):3d}  {kinds}")


if __name__ == "__main__":
    main()

"""Tests for the VEDA STAC primitives and the exploration sub-agent tool.

Covers the two properties NASA-IMPACT/geoguard#7 hinges on, with no network
and no real LLM:

  1. Compaction — what the sub-agent (and ultimately the verifier) sees from
     the catalog is a bounded summary, not the raw payload. Fixtures are
     trimmed copies of real openveda.cloud responses.
  2. The wrapper contract — the registered tool returns one JSON-serializable
     dict; sub-agent budget exhaustion and crashes degrade to a clean
     "no usable evidence" result instead of raising into the verifier.

Runnable two ways:
    uv run python tests/test_stac.py      # standalone (no pytest needed)
    uv run pytest tests/test_stac.py      # if pytest is installed
"""

import asyncio
import json

from pydantic_ai.models.test import TestModel

from geoguard.schemas import EventType
from geoguard.tools.registry import registry
from geoguard.tools.stac import agent as stac_agent
from geoguard.tools.stac.agent import (
    StacEvidence,
    StacExplorer,
    find_veda_evidence,
)
from geoguard.tools.stac.veda import (
    _compact_collection,
    _compact_item,
    _compact_stats,
    _score_collection,
    build_veda_tools,
)

# ---------------------------------------------------------------------------
# Fixtures — trimmed from live openveda.cloud responses (2026-07-19).
# ---------------------------------------------------------------------------

RAW_COLLECTION = {
    "id": "bangladesh-landcover-2001-2020",
    "type": "Collection",
    "title": "Annual land cover maps for 2001 and 2020",
    "description": "The annual land cover maps of 2001 and 2021 were captured "
    "using combined Moderate Resolution Imaging Spectroradiometer (MODIS) "
    "Annual Land Cover Type dataset (MCD12Q1 V6). " * 5,  # force truncation
    "extent": {
        "spatial": {"bbox": [[88.0259, 20.7420, 92.6836, 26.6350]]},
        "temporal": {
            "interval": [["2001-01-01T00:00:00+00:00", "2020-12-31T23:59:59+00:00"]]
        },
    },
    "item_assets": {"cog_default": {"type": "image/tiff", "roles": ["data"]}},
    "links": [{"rel": "self", "href": "https://example.invalid"}],
}

RAW_ITEM = {
    "id": "VNP02_mosaic-M15_2024-07-09T00_00_00Z",
    "collection": "viirs_mosaic-cyclone-beryl",
    "bbox": [-102.81301312482547, 6.192584681110766, -13.346383691706222, 48.6],
    "properties": {"datetime": "2024-07-09T00:00:00Z"},
    "assets": {
        "cog_default": {
            "href": "s3://veda-data-store/viirs_mosaic-cyclone-beryl/x.tif",
            "type": "image/tiff; application=geotiff",
            "roles": ["data", "layer"],
        },
        "rendered_preview": {
            "href": "https://openveda.cloud/api/raster/...png",
            "type": "image/png",
            "roles": ["overview"],
        },
    },
}

RAW_STATS = {
    "cog_default_b1": {
        "min": 7117.0,
        "max": 31874.0,
        "mean": 25959.850221195513,
        "count": 66231.0,
        "sum": 1719346840.0,
        "std": 2991.2772926859648,
        "median": 26751.0,
        "majority": 27370.0,
        "minority": 7200.0,
        "unique": 11167.0,
        "histogram": [[147, 319], [7117.0, 9592.7, 12068.4]],
        "valid_percent": 100.0,
        "masked_pixels": 0.0,
        "valid_pixels": 66231.0,
        "percentile_2": 15827.0,
        "percentile_98": 29944.0,
    }
}


# ---------------------------------------------------------------------------
# Compaction — catalog payloads must shrink to bounded summaries.
# ---------------------------------------------------------------------------


def test_compact_collection_keeps_essentials_and_truncates():
    c = _compact_collection(RAW_COLLECTION)
    assert c["id"] == "bangladesh-landcover-2001-2020"
    assert c["bbox"] == [88.0259, 20.7420, 92.6836, 26.6350]
    assert c["temporal_interval"][0].startswith("2001-01-01")
    assert c["asset_keys"] == ["cog_default"]
    assert len(c["description"]) <= 300
    assert "links" not in c  # raw structure does not leak through


def test_compact_item_extracts_data_assets_only():
    it = _compact_item(RAW_ITEM)
    assert it["id"] == RAW_ITEM["id"]
    assert it["datetime"] == "2024-07-09T00:00:00Z"
    assert it["data_assets"] == ["cog_default"]  # preview asset excluded
    assert it["bbox"] == [-102.813, 6.1926, -13.3464, 48.6]


def test_compact_stats_drops_histogram_keeps_headline():
    s = _compact_stats(RAW_STATS)
    band = s["cog_default_b1"]
    assert "histogram" not in band
    assert "percentile_98" not in band
    assert band["mean"] == 25959.8502
    assert band["valid_percent"] == 100.0
    assert band["valid_pixels"] == 66231.0


def test_score_collection_counts_matching_terms():
    col = _compact_collection(RAW_COLLECTION)
    assert _score_collection(col, ["landcover", "modis"]) == 2
    assert _score_collection(col, ["flood"]) == 0


def test_build_veda_tools_preserves_llm_facing_metadata():
    # Closure-built tools must keep names and docstrings — pydantic-ai
    # derives the sub-agent's tool schemas from them.
    tools = build_veda_tools("https://stac.example", "https://raster.example")
    assert [t.__name__ for t in tools] == [
        "search_veda_collections",
        "search_veda_items",
        "get_veda_item_statistics",
    ]
    assert all(t.__doc__ and len(t.__doc__) > 100 for t in tools)


def test_bound_endpoint_is_actually_used():
    # An unroutable endpoint must surface as a graceful error dict from the
    # bound tool — proving the URL binding reaches the HTTP call.
    search, _, _ = build_veda_tools(stac_url="http://127.0.0.1:1")
    result = asyncio.run(search("flood"))
    assert result == {"tool": "search_veda_collections", "error": result["error"]}


def test_explorer_rejects_tools_and_endpoints_together():
    try:
        StacExplorer(tools=[_stub_search], stac_url="https://stac.example")
    except ValueError as e:
        assert "not both" in str(e)
    else:
        raise AssertionError("expected ValueError")


# ---------------------------------------------------------------------------
# Sub-agent wrapper — one call in, one compact structured result out.
# ---------------------------------------------------------------------------


async def _stub_search(keywords: str) -> dict:
    """Harmless stand-in for the network-bound primitives."""
    return {"found": False, "matched_count": 0, "reason": "stub"}


def test_explorer_returns_structured_evidence():
    explorer = StacExplorer(model=TestModel(call_tools=[]), tools=[_stub_search])
    evidence = asyncio.run(explorer("find evidence"))
    assert isinstance(evidence, StacEvidence)


def test_explorer_hard_limit_degrades_gracefully():
    # TestModel calls the stub tool on request 1; the run then needs a second
    # request to produce output, which request_limit=1 forbids. The explorer
    # must swallow UsageLimitExceeded and answer empty-handed.
    explorer = StacExplorer(
        model=TestModel(),
        tools=[_stub_search],
        request_limit=1,
    )
    evidence = asyncio.run(explorer("find evidence"))
    assert evidence.found is False
    assert "budget" in evidence.summary.lower()


def test_budgeted_wrapper_counts_down_then_refuses():
    budget = {"remaining": 2}
    wrapped = stac_agent._budgeted(_stub_search, budget)
    r1 = asyncio.run(wrapped("a"))
    assert r1["tool_calls_remaining"] == 1
    r2 = asyncio.run(wrapped("b"))
    assert r2["tool_calls_remaining"] == 0
    r3 = asyncio.run(wrapped("c"))
    assert "budget exhausted" in r3["error"].lower()
    assert "reason" not in r3  # no data leaks through once exhausted


def test_explorer_completes_within_soft_budget():
    # Tool budget 1: TestModel's single tool call spends it, and the run
    # still finishes with structured output — no exception path involved.
    explorer = StacExplorer(
        model=TestModel(),
        tools=[_stub_search],
        tool_calls_limit=1,
    )
    evidence = asyncio.run(explorer("find evidence"))
    assert isinstance(evidence, StacEvidence)


class _StubExplorer:
    def __init__(self, evidence=None, error=None):
        self.evidence = evidence
        self.error = error
        self.queries: list[str] = []

    async def __call__(self, query: str) -> StacEvidence:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.evidence


def _with_stub_explorer(stub, coro_fn):
    """Run an async callable with the tool's explorer construction stubbed."""
    original = stac_agent.StacExplorer
    stac_agent.StacExplorer = lambda: stub
    try:
        return asyncio.run(coro_fn())
    finally:
        stac_agent.StacExplorer = original


def test_find_veda_evidence_returns_compact_json_dict():
    stub = _StubExplorer(
        evidence=StacEvidence(
            found=True,
            summary="IMERG shows 180 mm total precipitation over the area.",
            collection_id="tx-flood-imerg",
            item_ids=["imerg-2024-07-08"],
            statistics={"mean_mm": 180.0},
            units="mm",
            source="NASA VEDA STAC: tx-flood-imerg/imerg-2024-07-08",
        )
    )
    result = _with_stub_explorer(
        stub,
        lambda: find_veda_evidence(
            claim="Over 100 mm of rain fell in Houston",
            start_date="2024-06-24",
            end_date="2024-07-08",
            lat=29.76,
            lon=-95.36,
        ),
    )
    assert result["found"] is True
    assert result["collection_id"] == "tx-flood-imerg"
    json.dumps(result)  # must be JSON-serializable for recording/dedup
    # The sub-agent got the claim's full context in one prompt.
    assert "Houston" in stub.queries[0]
    assert "2024-06-24 to 2024-07-08" in stub.queries[0]


def test_find_veda_evidence_requires_an_area():
    stub = _StubExplorer()
    result = _with_stub_explorer(
        stub,
        lambda: find_veda_evidence(
            claim="anything", start_date="2024-01-01", end_date="2024-01-02"
        ),
    )
    assert result["found"] is False
    assert stub.queries == []  # never reached the sub-agent


def test_find_veda_evidence_survives_subagent_crash():
    stub = _StubExplorer(error=RuntimeError("provider exploded"))
    result = _with_stub_explorer(
        stub,
        lambda: find_veda_evidence(
            claim="anything",
            start_date="2024-01-01",
            end_date="2024-01-02",
            lat=1.0,
            lon=2.0,
        ),
    )
    assert result["found"] is False
    assert "provider exploded" in result["summary"]
    json.dumps(result)


def test_tool_is_registered_always_on():
    # Registered under OTHER => offered as a candidate for every event type.
    assert find_veda_evidence in registry.get_candidates(EventType.FLOOD)
    assert find_veda_evidence in registry.get_candidates(EventType.OTHER)


if __name__ == "__main__":
    _tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for _t in _tests:
        _t()
        print(f"PASS: {_t.__name__}")
    print(f"\nAll {len(_tests)} tests passed.")

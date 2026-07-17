"""Regression tests for verifier tool-call extraction and dedup.

Reproduces the real akd-guardrails MCP pattern: a verifier LLM emits the same
``(tool, args)`` call many times in one run — the first emission serialized
spaced, later ones compact — interleaved with genuinely distinct calls.
``_extract_tool_calls`` must collapse exact duplicates (keeping the first
occurrence) while preserving distinct calls.

Runnable two ways:
    uv run python tests/test_verifications.py     # standalone (no pytest needed)
    uv run pytest tests/test_verifications.py      # if pytest is installed
"""

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

from geoguard.verifications import _canonical_args, _extract_tool_calls

# Identical full-year args serialized the two ways a provider emits them:
# the first tool call in a turn keeps the model's spacing, later ones compact.
FULL_YEAR_SPACED = (
    '{"lat": 28.3780464, "lon": 83.9999901, '
    '"start_date": "2012-01-01", "end_date": "2012-12-31"}'
)
FULL_YEAR_COMPACT = (
    '{"lat":28.3780464,"lon":83.9999901,'
    '"start_date":"2012-01-01","end_date":"2012-12-31"}'
)
NARROW_WINDOW = (
    '{"lat":28.3780464,"lon":83.9999901,'
    '"start_date":"2012-02-06","end_date":"2012-02-10"}'
)

WIND_RESULT = {"peak_speed_kmh": 11.6, "daily_max_speed_kmh": [6.5] * 366}
PRECIP_RESULT = {"total_mm": 3980.1}


def _messages(specs):
    """Build alternating call/return message pairs from (name, args, result)."""
    msgs = []
    for i, (name, args, result) in enumerate(specs):
        cid = f"call_{i}"
        msgs.append(
            ModelResponse(
                parts=[ToolCallPart(tool_name=name, args=args, tool_call_id=cid)]
            )
        )
        msgs.append(
            ModelRequest(
                parts=[ToolReturnPart(tool_name=name, content=result, tool_call_id=cid)]
            )
        )
    return msgs


def test_collapses_duplicate_calls_across_arg_formatting():
    # 6 emissions of the Nepal-hurricane pattern -> 3 unique calls.
    specs = [
        ("get_historical_winds", FULL_YEAR_SPACED, WIND_RESULT),  # keep (first)
        ("get_historical_precipitation", FULL_YEAR_SPACED, PRECIP_RESULT),  # keep
        ("get_historical_winds", NARROW_WINDOW, WIND_RESULT),  # keep (distinct args)
        ("get_historical_winds", FULL_YEAR_COMPACT, WIND_RESULT),  # drop (dup of #1)
        ("get_historical_winds", FULL_YEAR_COMPACT, WIND_RESULT),  # drop
        ("get_historical_winds", FULL_YEAR_COMPACT, WIND_RESULT),  # drop
    ]
    calls = _extract_tool_calls(_messages(specs))

    assert len(calls) == 3
    assert [c.name for c in calls] == [
        "get_historical_winds",
        "get_historical_precipitation",
        "get_historical_winds",
    ]
    # The kept entry is the first occurrence, verbatim (spaced), not a later dup.
    assert calls[0].args == FULL_YEAR_SPACED
    # The genuinely distinct narrow-window call survives.
    assert calls[2].args == NARROW_WINDOW
    # Kept results are intact.
    assert calls[0].result == WIND_RESULT


def test_distinct_args_are_preserved():
    specs = [
        ("get_historical_winds", FULL_YEAR_COMPACT, WIND_RESULT),
        ("get_historical_winds", NARROW_WINDOW, WIND_RESULT),
    ]
    assert len(_extract_tool_calls(_messages(specs))) == 2


def test_final_result_tool_is_skipped():
    specs = [("final_result", "{}", {"ok": True})]
    assert _extract_tool_calls(_messages(specs)) == []


def test_canonical_args_ignores_whitespace_and_key_order():
    assert _canonical_args('{"lat": 1, "lon": 2}') == _canonical_args(
        '{"lon":2,"lat":1}'
    )


def test_canonical_args_falls_back_on_non_json():
    assert _canonical_args("not json") == "not json"


if __name__ == "__main__":
    _tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for _t in _tests:
        _t()
        print(f"PASS: {_t.__name__}")
    print(f"\nAll {len(_tests)} tests passed.")

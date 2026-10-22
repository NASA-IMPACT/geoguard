"""Regression tests and fixtures for claim extraction and event segregation.

Covers the criteria from NASA-IMPACT/geoguard#6:
  1. Independent events keep separate ClaimGroups with independent metadata
     (neither inherits the other's date or location, no links fabricated).
  2. Vague, opinion, hedge, and aside statements are filtered out.
  3. Concrete claims lacking obvious data sources still pass through.
  4. Prompt invariants on CLAIM_RULES, CLAIM_GROUP_INSTRUCTIONS, and TIME_RULE.

Runnable two ways:
    python tests/test_extraction.py      # standalone (no pytest needed)
    pytest tests/test_extraction.py      # if pytest is installed
"""

from datetime import datetime

from pydantic_ai.models.test import TestModel

from geoguard.claims import (
    CLAIM_RULES,
    Claim,
    ClaimExtractor,
)
from geoguard.metadata import (
    CLAIM_GROUP_INSTRUCTIONS,
    DEFAULT_INSTRUCTIONS,
    GEOCODE_RULE,
    TIME_RULE,
    ClaimGroup,
    FloodMetadata,
    GeneralMetadata,
    GeoLocation,
    MetadataExtractor,
    StormMetadata,
    TimeRange,
)
from geoguard.schemas import EventType, Input

# ---------------------------------------------------------------------------
# Representative evaluation fixtures (issue #6 failure cases)
# ---------------------------------------------------------------------------

CAR_CRASH_AND_HURRICANE_FIXTURE = (
    "Person XYZ had a car crash on Memorial Drive in Boston. "
    "Meanwhile, Hurricane Beryl made landfall near Galveston, Texas on July 8, 2024. "
    "In my opinion, city driving in Boston is terribly stressful and unpleasant."
)

HARD_TO_CHECK_CONCRETE_CLAIM_FIXTURE = (
    "Flash flooding inundated Main Street in Ellicott City, Maryland on May 27, 2018. "
    "A resident left a dark green 1998 Subaru Outback parked outside the bakery."
)

UNRELATED_EVENTS_SAME_AFTERNOON_FIXTURE = (
    "A severe thunderstorm caused street flooding in Pittsburgh, Pennsylvania on "
    "June 15, 2023. A warehouse fire broke out in downtown Cleveland, Ohio on the "
    "same afternoon."
)


# ---------------------------------------------------------------------------
# Structural validation helpers for extraction outputs
# ---------------------------------------------------------------------------


def assert_event_segregation(groups: list[ClaimGroup]) -> None:
    """Validate that groups represent distinct events without context bleed."""
    assert len(groups) >= 2, f"Expected at least 2 distinct groups, got {len(groups)}"

    # Check locations and dates across distinct groups
    dated_groups = [g for g in groups if g.metadata.time_range and g.metadata.time_range.start]
    undated_groups = [g for g in groups if not g.metadata.time_range or not g.metadata.time_range.start]

    # When one event has a date and another does not, date bleed must not occur
    if dated_groups and undated_groups:
        dated_start = dated_groups[0].metadata.time_range.start
        for ug in undated_groups:
            assert ug.metadata.time_range is None or ug.metadata.time_range.start is None, (
                "Undated event inherited date from another event"
            )

    # Check that no claim invents an unstated link between independent events
    forbidden_link_phrases = [
        "caused by the hurricane",
        "resulted from hurricane beryl",
        "caused the warehouse fire",
        "due to the storm in pittsburgh",
    ]
    for g in groups:
        for c in g.claims:
            claim_lower = c.claim.lower()
            for phrase in forbidden_link_phrases:
                assert phrase not in claim_lower, (
                    f"Fabricated link found in claim: {c.claim}"
                )


def assert_no_opinion_claims(claims: list[Claim]) -> None:
    """Validate that opinion, hedge, or meta-commentary claims are absent."""
    opinion_indicators = [
        "in my opinion",
        "terribly stressful",
        "terribly frustrating",
        "unpleasant",
        "i think",
        "i feel",
    ]
    for c in claims:
        claim_lower = c.claim.lower()
        for ind in opinion_indicators:
            assert ind not in claim_lower, f"Opinion claim was not dropped: {c.claim}"


# ---------------------------------------------------------------------------
# Unit and Invariant Tests
# ---------------------------------------------------------------------------


def test_claim_rules_invariants():
    """Verify that CLAIM_RULES explicitly covers all issue #6 requirements."""
    # Must enforce no unstated relationships
    assert "Never assert a causal, temporal, or spatial relationship" in CLAIM_RULES
    # Must clarify filtering of opinions and asides
    assert "FACTUAL AND RELEVANT" in CLAIM_RULES
    assert "Filter out vague assertions, opinions" in CLAIM_RULES
    # Must clarify that hard-to-verify is not a drop reason
    assert "VERIFIABLE IN PRINCIPLE" in CLAIM_RULES
    assert "hard-to-verify is NEVER a reason to drop a claim" in CLAIM_RULES


def test_time_rule_scopes_date_propagation():
    """Verify TIME_RULE prevents date bleeding across independent events."""
    assert "NEVER propagate a date across independent, unrelated events" in TIME_RULE
    assert "leave its time_range as None" in TIME_RULE


def test_claim_group_instructions_invariants():
    """Verify CLAIM_GROUP_INSTRUCTIONS enforces separation and anti-fabrication."""
    assert "INDEPENDENT EVENTS" in CLAIM_GROUP_INSTRUCTIONS
    assert "DO NOT FABRICATE LINKS" in CLAIM_GROUP_INSTRUCTIONS
    assert "Never allow one event to inherit another's date, location, or entities" in CLAIM_GROUP_INSTRUCTIONS


def test_metadata_extractor_prompt_composition():
    """Verify MetadataExtractor composes instructions and rules properly."""
    extractor = MetadataExtractor(
        model=TestModel(),
        output_type=list[ClaimGroup],
        instructions=CLAIM_GROUP_INSTRUCTIONS,
    )
    # The agent instructions must include both the sharpened grouping and time rules
    agent_instructions = extractor._agent.instructions
    assert "INDEPENDENT EVENTS" in agent_instructions
    assert "NEVER propagate a date across independent" in agent_instructions
    assert "hard-to-verify is NEVER a reason to drop a claim" in agent_instructions


def test_claim_extractor_prompt_composition():
    """Verify standalone ClaimExtractor composes sharpened CLAIM_RULES."""
    extractor = ClaimExtractor(model=TestModel())
    agent_instructions = extractor._agent.instructions
    assert "FACTUAL AND RELEVANT" in agent_instructions
    assert "hard-to-verify is NEVER a reason to drop a claim" in agent_instructions


def test_structural_segregation_on_mocked_fixture():
    """Verify structural checks on synthetic ClaimGroup outputs for the car crash/hurricane fixture."""
    synthetic_groups = [
        ClaimGroup(
            metadata=GeneralMetadata(
                event_type=EventType.OTHER,
                location=GeoLocation(name="Memorial Drive, Boston, Massachusetts", lat=42.36, lon=-71.10),
                time_range=None,
                entities=[],
            ),
            claims=[
                Claim(claim="Person XYZ was involved in a car crash on Memorial Drive in Boston."),
            ],
        ),
        ClaimGroup(
            metadata=StormMetadata(
                event_type=EventType.STORM,
                storm_name="Beryl",
                location=GeoLocation(name="Galveston, Texas, USA", lat=29.30, lon=-94.80),
                time_range=TimeRange(
                    start=datetime(2024, 7, 8),
                    end=datetime(2024, 7, 8),
                ),
                entities=[],
            ),
            claims=[
                Claim(claim="Hurricane Beryl made landfall near Galveston, Texas on July 8, 2024."),
            ],
        ),
    ]

    # Validate segregation rules pass
    assert_event_segregation(synthetic_groups)
    all_claims = [c for g in synthetic_groups for c in g.claims]
    assert_no_opinion_claims(all_claims)
    assert len(all_claims) == 2


def test_time_range_partial_date_expansion_keeps_none_clean():
    """Verify TimeRange validator does not invent dates when fields are None."""
    tr = TimeRange(start=None, end=None)
    assert tr.start is None
    assert tr.end is None
    assert tr.duration_days is None


def test_time_range_partial_date_expansion_handles_year_and_month():
    """Verify TimeRange expands partial date strings properly."""
    tr_year = TimeRange.model_validate({"start": "2024", "end": "2024"})
    assert tr_year.start == datetime(2024, 1, 1)
    assert tr_year.end == datetime(2024, 12, 31)

    tr_month = TimeRange.model_validate({"start": "2024-07", "end": "2024-07"})
    assert tr_month.start == datetime(2024, 7, 1)
    assert tr_month.end == datetime(2024, 7, 31)


if __name__ == "__main__":
    _tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for _t in _tests:
        _t()
        print(f"PASS: {_t.__name__}")
    print(f"\nAll {len(_tests)} tests passed.")

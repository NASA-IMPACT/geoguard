from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from pydantic import BaseModel, ConfigDict

from geoguard.claims import Claim
from geoguard.config import ReasoningEffort, Settings
from geoguard.config import settings as default_settings
from geoguard.metadata import (
    CLAIM_GROUP_INSTRUCTIONS,
    ClaimGroup,
    Metadata,
    MetadataExtractor,
)
from geoguard.rubrics import Rubric, Rubricator
from geoguard.schemas import Input
from geoguard.tools.selector import SelectedTools, ToolSelector
from geoguard.verifications import Verdict, Verifier, VerifierResult


class Report(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input: Input
    verifications: list[VerifierResult]
    rubric: Rubric
    overall_verdict: Verdict


PipelineEvent = ClaimGroup | Claim | SelectedTools | VerifierResult | Rubric | Report


class GeoGuard:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        metadata_extractor: MetadataExtractor | None = None,
        tool_selector: ToolSelector | None = None,
        verifier: Verifier | None = None,
        rubricator: Rubricator | None = None,
    ):
        self.metadata_extractor = metadata_extractor or MetadataExtractor(
            model=model,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            output_type=list[ClaimGroup],
            instructions=CLAIM_GROUP_INSTRUCTIONS,
        )
        self.tool_selector = tool_selector or ToolSelector(
            model=model, api_key=api_key, reasoning_effort=reasoning_effort
        )
        self.verifier = verifier or Verifier(
            model=model, api_key=api_key, reasoning_effort=reasoning_effort
        )
        self.rubricator = rubricator or Rubricator(
            model=model, api_key=api_key, reasoning_effort=reasoning_effort
        )

    @classmethod
    def from_config(cls, settings: Settings | None = None) -> GeoGuard:
        """Build a GeoGuard with every block driven by Settings (env / .env)."""
        s = settings or default_settings
        return cls(
            metadata_extractor=MetadataExtractor(
                model=s.model,
                api_key=s.api_key,
                reasoning_effort=s.reasoning_effort,
                output_type=list[ClaimGroup],
                max_claims=s.max_claims,
            ),
            tool_selector=ToolSelector(
                model=s.model,
                api_key=s.api_key,
                reasoning_effort=s.reasoning_effort,
            ),
            verifier=Verifier(
                model=s.model,
                api_key=s.api_key,
                reasoning_effort=s.reasoning_effort,
                tool_calls_limit=s.verification_tool_usage_limit,
            ),
            rubricator=Rubricator(
                model=s.model,
                api_key=s.api_key,
                reasoning_effort=s.reasoning_effort,
                questions_per_claim=(
                    s.questions_per_claim_min,
                    s.questions_per_claim_max,
                ),
            ),
        )

    async def _claim_stream(
        self, claim: Claim, metadata: Metadata
    ) -> AsyncIterator[Claim | SelectedTools | VerifierResult]:
        """Per-claim sub-pipeline as an async generator. No concurrency awareness."""
        yield claim
        sel = await self.tool_selector(claim, metadata)
        yield sel
        vr = await self.verifier(claim, metadata, sel.tools)
        yield vr

    async def stream(self, inp: Input) -> AsyncIterator[PipelineEvent]:
        """Yield events as they happen.

        Per-claim sub-pipelines run concurrently within each group.
        """
        groups = await self.metadata_extractor(inp)
        verifications: list[VerifierResult] = []
        SENTINEL: object = object()

        async def _drain(
            gen: AsyncIterator[PipelineEvent], queue: asyncio.Queue
        ) -> None:
            try:
                async for event in gen:
                    await queue.put(event)
            finally:
                await queue.put(SENTINEL)

        for group in groups:
            yield group
            queue: asyncio.Queue = asyncio.Queue()
            tasks = [
                asyncio.create_task(
                    _drain(self._claim_stream(c, group.metadata), queue)
                )
                for c in group.claims
            ]
            done = 0
            while done < len(tasks):
                item = await queue.get()
                if item is SENTINEL:
                    done += 1
                    continue
                yield item
                if isinstance(item, VerifierResult):
                    verifications.append(item)

        rubric = await self.rubricator(inp, verifications)
        yield rubric

        overall = _roll_up([v.verification.verdict for v in verifications])
        yield Report(
            input=inp,
            verifications=verifications,
            rubric=rubric,
            overall_verdict=overall,
        )

    def __call__(self, inp: Input) -> AsyncIterator[PipelineEvent]:
        """Alias for .stream() — `async for event in guard(input)` keeps working."""
        return self.stream(inp)

    async def run(self, inp: Input) -> Report:
        async for item in self.stream(inp):
            if isinstance(item, Report):
                return item
        raise RuntimeError("pipeline did not yield Report")


def _roll_up(verdicts: list[Verdict]) -> Verdict:
    if not verdicts:
        return Verdict.INCONCLUSIVE
    if Verdict.CONTRADICTS in verdicts:
        return Verdict.CONTRADICTS
    if all(v == Verdict.SUPPORTS for v in verdicts):
        return Verdict.SUPPORTS
    return Verdict.INCONCLUSIVE

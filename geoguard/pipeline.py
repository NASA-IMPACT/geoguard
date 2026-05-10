import asyncio
from collections.abc import AsyncIterator

from pydantic import BaseModel, ConfigDict

from geoguard.claims import Claim
from geoguard.metadata import (
    CLAIM_GROUP_INSTRUCTIONS,
    ClaimGroup,
    Metadata,
    MetadataExtractor,
)
from geoguard.schemas import Input
from geoguard.tools.selector import SelectedTools, ToolSelector
from geoguard.verifications import Verdict, Verifier, VerifierResult


class Report(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input: Input
    verifications: list[VerifierResult]
    overall_verdict: Verdict


PipelineEvent = ClaimGroup | Claim | SelectedTools | VerifierResult | Report


class GeoGuard:
    def __init__(
        self,
        metadata_extractor: MetadataExtractor | None = None,
        tool_selector: ToolSelector | None = None,
        verifier: Verifier | None = None,
    ):
        self.metadata_extractor = metadata_extractor or MetadataExtractor(
            output_type=list[ClaimGroup],
            instructions=CLAIM_GROUP_INSTRUCTIONS,
        )
        self.tool_selector = tool_selector or ToolSelector()
        self.verifier = verifier or Verifier()

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

        overall = _roll_up([v.verification.verdict for v in verifications])
        yield Report(
            input=inp,
            verifications=verifications,
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

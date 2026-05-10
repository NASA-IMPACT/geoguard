from collections.abc import AsyncIterator

from pydantic import BaseModel, ConfigDict

from geoguard.claims import Claim
from geoguard.metadata import (
    CLAIM_GROUP_INSTRUCTIONS,
    ClaimGroup,
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

    async def __call__(self, inp: Input) -> AsyncIterator[PipelineEvent]:
        groups = await self.metadata_extractor(inp)
        verifications: list[VerifierResult] = []
        for group in groups:
            yield group
            for claim in group.claims:
                yield claim
                sel = await self.tool_selector(claim, group.metadata)
                yield sel
                vr = await self.verifier(claim, group.metadata, sel.tools)
                yield vr
                verifications.append(vr)
        overall = _roll_up([v.verification.verdict for v in verifications])
        yield Report(
            input=inp,
            verifications=verifications,
            overall_verdict=overall,
        )

    async def run(self, inp: Input) -> Report:
        async for item in self(inp):
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

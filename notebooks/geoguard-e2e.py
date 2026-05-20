import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # Register flood tools via side-effect imports (their @registry decorators
    # run on import). Both modules ship with the framework.
    import geoguard.tools.geospatial  # noqa: F401  get_elevation + find_nearest_water_body
    import geoguard.tools.satellite  # noqa: F401  get_satellite_flood_extent (MODIS 250m)
    import geoguard.tools.weather  # noqa: F401  get_historical_precipitation + get_historical_winds
    from geoguard import GeoGuard, Input, Rubric
    from geoguard.adapters import tiff_to_claims
    from geoguard.claims import Claim
    from geoguard.metadata import ClaimGroup
    from geoguard.pipeline import Report
    from geoguard.tools.selector import SelectedTools
    from geoguard.verifications import Verdict, VerifierResult

    return (
        Claim,
        ClaimGroup,
        GeoGuard,
        Input,
        Report,
        Rubric,
        SelectedTools,
        Verdict,
        VerifierResult,
        tiff_to_claims,
    )


@app.cell
def _(mo):
    mo.md("""
    # GeoGuard end-to-end demo

    Streaming pipeline: **claim extraction → metadata → tool selection → verification → report**.
    Uses real public APIs (no keys): OpenStreetMap Nominatim for geocoding, Open-Meteo for historical weather.
    """)
    return


@app.cell
def _(mo):
    import inspect

    from geoguard.tools.registry import registry

    def _first_doc_line(fn) -> str:
        return (inspect.getdoc(fn) or "").split("\n")[0].rstrip(".")

    _registered_md = "\n".join(
        f"- **`{name}`** — {_first_doc_line(fn)}"
        for name, fn in {
            fn.__name__: fn for fns in registry._tools.values() for fn in fns
        }.items()
    )

    mo.md(
        "### Registered tools\n\n"
        "The pipeline has access to these tools, registered automatically when the "
        "modules were imported above:\n\n"
        "- **`geocode`** — place name → lat/lon (OpenStreetMap Nominatim, attached "
        "to the metadata extractor)\n"
        f"{_registered_md}\n\n"
        "Adding more is just `@registry(EventType.FLOOD)` on an async function "
        "— no orchestration changes."
    )
    return


@app.cell
def _(mo):
    DEFAULT_TEXT = (
        "Hurricane Beryl made landfall near Matagorda Bay, Texas on July 8, 2024. "
        "The storm brought sustained winds of 80 mph and significant storm surge "
        "along the Texas coast. Heavy rainfall caused widespread flooding in Houston, "
        "with reports of over 100 mm of rain falling over a 24-hour period."
    )

    # --- Common controls ---
    model_input = mo.ui.text(
        value="openai:gpt-5.2",
        label="**Model** (`provider:name`)",
        full_width=True,
    )
    api_key_input = mo.ui.text(
        value="",
        label="**API key** _(optional — leave blank to use `OPENAI_API_KEY` from env)_",
        kind="password",
        full_width=True,
    )
    reasoning_input = mo.ui.dropdown(
        options=["low", "medium", "high"],
        value="medium",
        label="**Reasoning effort**",
    )

    # --- Text tab ---
    text_input = mo.ui.text_area(
        value=DEFAULT_TEXT,
        rows=6,
        full_width=True,
        label="**Input text** to verify",
    )

    # --- TIFF tab ---
    tiff_upload = mo.ui.file(
        filetypes=[".tif", ".tiff"],
        kind="area",
        label="Upload a flood detection mask (0=dry, 1=flood, 255=nodata)",
    )
    tiff_bbox = mo.ui.text(
        value="-122.172, 38.175, -121.381, 39.601",
        label="**Bounding box** (west, south, east, north)",
        full_width=True,
    )
    tiff_date = mo.ui.text(
        value="2023-01-22",
        label="**Event date** (YYYY-MM-DD)",
        full_width=True,
    )
    tiff_region = mo.ui.text(
        value="Sacramento Valley, California, USA",
        label="**Region name**",
        full_width=True,
    )
    tiff_model_name = mo.ui.text(
        value="Prithvi-EO",
        label="**Model name**",
        full_width=True,
    )
    tiff_source = mo.ui.text(
        value="Sentinel-2",
        label="**Input source**",
        full_width=True,
    )

    input_tabs = mo.ui.tabs(
        {
            "📝 Text": mo.vstack([text_input]),
            "🛰️ TIFF Upload": mo.vstack(
                [
                    tiff_upload,
                    mo.hstack([tiff_bbox, tiff_date], widths="equal"),
                    mo.hstack(
                        [tiff_region, tiff_model_name, tiff_source], widths="equal"
                    ),
                ]
            ),
        }
    )

    run_button = mo.ui.run_button(label="🔎 Verify")
    mo.vstack(
        [
            mo.hstack([model_input, reasoning_input], justify="start"),
            api_key_input,
            input_tabs,
            run_button,
        ]
    )
    return (
        api_key_input,
        input_tabs,
        model_input,
        reasoning_input,
        run_button,
        text_input,
        tiff_bbox,
        tiff_date,
        tiff_model_name,
        tiff_region,
        tiff_source,
        tiff_upload,
    )


@app.cell
def _(
    input_tabs,
    mo,
    text_input,
    tiff_bbox,
    tiff_date,
    tiff_model_name,
    tiff_region,
    tiff_source,
    tiff_to_claims,
    tiff_upload,
):
    """Resolve claims text from the active input tab."""
    import tempfile

    _claims_text = ""
    _tiff_error = ""

    if input_tabs.value == "🛰️ TIFF Upload":
        if tiff_upload.value:
            try:
                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as _f:
                    _f.write(tiff_upload.value[0].contents)
                    _tiff_path = _f.name
                _bbox = [float(x.strip()) for x in tiff_bbox.value.split(",")]
                _claims_text = tiff_to_claims(
                    tiff_path=_tiff_path,
                    bbox=_bbox,
                    date=tiff_date.value.strip(),
                    region_name=tiff_region.value.strip(),
                    model_name=tiff_model_name.value.strip(),
                    input_source=tiff_source.value.strip(),
                )
            except Exception as e:
                _tiff_error = str(e)
        if _claims_text:
            mo.output.replace(
                mo.vstack(
                    [
                        mo.md("### Generated claims from TIFF"),
                        mo.callout(mo.md(_claims_text), kind="info"),
                    ]
                )
            )
        elif _tiff_error:
            mo.output.replace(
                mo.callout(mo.md(f"**Error reading TIFF:** {_tiff_error}"), kind="danger")
            )
        else:
            mo.output.replace(mo.md("_Upload a TIFF file to generate claims._"))
    else:
        _claims_text = text_input.value
        mo.output.replace(mo.md(""))

    claims_text = _claims_text
    return (claims_text,)


@app.cell
def _(Verdict):
    """Shared rendering helpers used by the streaming cell + summary."""
    VERDICT_KIND = {
        Verdict.SUPPORTS: "success",
        Verdict.CONTRADICTS: "danger",
        Verdict.INCONCLUSIVE: "warn",
    }
    VERDICT_EMOJI = {
        Verdict.SUPPORTS: "✅",
        Verdict.CONTRADICTS: "❌",
        Verdict.INCONCLUSIVE: "⚠️",
    }
    return VERDICT_EMOJI, VERDICT_KIND


@app.cell
async def _(
    Claim,
    ClaimGroup,
    GeoGuard,
    Input,
    Report,
    Rubric,
    SelectedTools,
    VERDICT_EMOJI,
    VERDICT_KIND,
    VerifierResult,
    api_key_input,
    claims_text,
    mo,
    model_input,
    reasoning_input,
    run_button,
):
    mo.stop(not run_button.value, mo.md("_Click **Verify** to run the pipeline._"))
    mo.stop(not claims_text, mo.callout(mo.md("**No input.** Enter text or upload a TIFF."), kind="warn"))

    guard = GeoGuard(
        model=model_input.value.strip() or None,
        api_key=api_key_input.value.strip() or None,
        reasoning_effort=reasoning_input.value,
    )
    events = []

    with mo.status.spinner(title="Starting pipeline..."):
        async for item in guard(Input(text=claims_text)):
            events.append(item)

            if isinstance(item, ClaimGroup):
                md = mo.md(
                    f"---\n\n"
                    f"### 📦 Event group — `{item.metadata.event_type.value}` "
                    f"({len(item.claims)} claim{'s' if len(item.claims) != 1 else ''})\n\n"
                    f"- **Location:** {getattr(item.metadata.location, 'name', None) or '_unknown_'}\n"
                    f"- **Time range:** {getattr(item.metadata.time_range, 'start', None) or '_unknown_'} → {getattr(item.metadata.time_range, 'end', None) or '_unknown_'}"
                )
                mo.output.append(md)

            elif isinstance(item, Claim):
                mo.output.append(
                    mo.callout(mo.md(f"📝 **Claim:** {item.claim}"), kind="info")
                )

            elif isinstance(item, SelectedTools):
                claim_text = item.claim.claim if item.claim else "_(unknown claim)_"
                tools_list = (
                    "\n".join(f"  - `{t.__name__}`" for t in item.tools)
                    or "  - _(none)_"
                )
                mo.output.append(
                    mo.md(
                        f"🔧 **Tools for** _{claim_text[:80]}_\n{tools_list}\n\n"
                        f"_Reasoning:_ {item.reasoning or '—'}"
                    )
                )

            elif isinstance(item, VerifierResult):
                _v = item.verification
                _tool_calls_md = (
                    "\n".join(
                        f"- **{_tc.name}**`({_tc.args})`\n  → `{_tc.result}`"
                        for _tc in item.tool_calls
                    )
                    or "_no tool calls made_"
                )
                mo.output.append(
                    mo.callout(
                        mo.md(
                            f"**Verdict:** {VERDICT_EMOJI[_v.verdict]} **{_v.verdict.value.upper()}**\n\n"
                            f"**Rationale:** {_v.rationale}\n\n"
                            f"**Tool calls** ({len(item.tool_calls)}):\n{_tool_calls_md}"
                        ),
                        kind=VERDICT_KIND[_v.verdict],
                    )
                )

            elif isinstance(item, Rubric):
                mo.output.append(
                    mo.callout(
                        mo.md(
                            f"## 📋 Rubric — overall confidence: **{item.confidence:.0%}**"
                        ),
                        kind="info",
                    )
                )
                _summary_rows = [
                    {
                        "Claim": (
                            _cr.claim.claim[:80] + "…"
                            if len(_cr.claim.claim) > 80
                            else _cr.claim.claim
                        ),
                        "Score": f"{_cr.score:.0%}",
                        "Yes / Total": f"{sum(1 for _it in _cr.items if _it.answer)} / {len(_cr.items)}",
                    }
                    for _cr in item.per_claim
                ]
                mo.output.append(mo.md("**Per-claim scores:**"))
                mo.output.append(
                    mo.ui.table(_summary_rows, selection=None, page_size=20)
                )

                _detail_rows = []
                for _cr in item.per_claim:
                    _claim_label = (
                        _cr.claim.claim[:60] + "…"
                        if len(_cr.claim.claim) > 60
                        else _cr.claim.claim
                    )
                    for _it in _cr.items:
                        _detail_rows.append(
                            {
                                "Claim": _claim_label,
                                "Question": _it.question,
                                "Answer": "✓" if _it.answer else "✗",
                                "Reasoning": (
                                    (_it.reasoning or "")[:160] + "…"
                                    if _it.reasoning and len(_it.reasoning) > 160
                                    else (_it.reasoning or "")
                                ),
                            }
                        )
                mo.output.append(mo.md("**Rubric items (all claims):**"))
                mo.output.append(
                    mo.ui.table(_detail_rows, selection=None, page_size=30)
                )

            elif isinstance(item, Report):
                mo.output.append(
                    mo.callout(
                        mo.md(
                            f"## 🎉 Pipeline complete\n\n"
                            f"### Overall verdict: {VERDICT_EMOJI[item.overall_verdict]} **{item.overall_verdict.value.upper()}**\n\n"
                            f"### Rubric confidence: **{item.rubric.confidence:.0%}**\n\n"
                            f"_{len(item.verifications)} claim verification(s)_"
                        ),
                        kind=VERDICT_KIND[item.overall_verdict],
                    )
                )
    return (events,)


@app.cell
def _(Report, VERDICT_EMOJI, events, mo):
    """Compact summary table for quick scanning of all claims."""
    report = next((e for e in events if isinstance(e, Report)), None)
    mo.stop(report is None, mo.md(""))

    _rows = []
    for _vr in report.verifications:
        _v = _vr.verification
        _rows.append(
            {
                "verdict": f"{VERDICT_EMOJI[_v.verdict]} {_v.verdict.value}",
                "claim": _v.claim.claim,
                "tool_calls": len(_vr.tool_calls),
            }
        )

    mo.vstack(
        [
            mo.md("### Summary"),
            mo.ui.table(_rows, selection=None, page_size=20),
        ]
    )
    return


if __name__ == "__main__":
    app.run()

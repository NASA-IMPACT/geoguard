"""GeoGuard live demo — Gradio app for Hugging Face Spaces.

Bring your own OpenAI API key, paste a claim or paragraph, and watch the
GeoGuard pipeline stream events live: claim extraction → tool selection
→ verification → rubric → final report.
"""

import os

import gradio as gr

# Side-effect imports register tools with @registry on import.
import geoguard.tools.geospatial  # noqa: F401
import geoguard.tools.weather  # noqa: F401
from geoguard import GeoGuard, Input
from geoguard.claims import Claim
from geoguard.metadata import ClaimGroup
from geoguard.pipeline import Report
from geoguard.rubrics import Rubric
from geoguard.tools.selector import SelectedTools
from geoguard.verifications import Verdict, VerifierResult


# ── Example claims ────────────────────────────────────────────────────

EXAMPLES = [
    [
        "Hurricane Beryl made landfall near Matagorda Bay, Texas on July 8, 2024. "
        "The storm brought sustained winds of 80 mph and significant storm surge "
        "along the Texas coast. Heavy rainfall caused widespread flooding in "
        "Houston, with reports of over 100 mm of rain falling over a 24-hour period."
    ],
    [
        "Hurricane Helene made landfall in Florida's Big Bend region on "
        "September 26, 2024 as a Category 4 storm. Catastrophic flooding affected "
        "western North Carolina, with Asheville receiving over 300 mm of rain."
    ],
    [
        "On July 22, 2024, severe thunderstorms produced over 50 mm of rainfall "
        "in Dallas, Texas, causing flash floods in several neighborhoods."
    ],
]


# ── Rendering helpers ─────────────────────────────────────────────────

VERDICT_EMOJI = {
    Verdict.SUPPORTS: "✅",
    Verdict.CONTRADICTS: "❌",
    Verdict.INCONCLUSIVE: "⚠️",
}

VERDICT_COLOR = {
    Verdict.SUPPORTS: "#16a34a",
    Verdict.CONTRADICTS: "#dc2626",
    Verdict.INCONCLUSIVE: "#d97706",
}


def render_claim_group(group: ClaimGroup) -> str:
    n = len(group.claims)
    plural = "s" if n != 1 else ""
    loc = getattr(group.metadata.location, "name", None) or "_unknown_"
    start = getattr(group.metadata.time_range, "start", None) or "_unknown_"
    end = getattr(group.metadata.time_range, "end", None) or "_unknown_"
    return (
        f"### 📦 Event group — `{group.metadata.event_type.value}` ({n} claim{plural})\n\n"
        f"- **Location:** {loc}\n"
        f"- **Time range:** {start} → {end}\n\n"
    )


def render_claim(claim: Claim) -> str:
    return f"> 📝 **Claim:** {claim.claim}\n\n"


def render_tools(sel: SelectedTools) -> str:
    claim_text = sel.claim.claim if sel.claim else "(unknown claim)"
    truncated = claim_text[:140] + ("…" if len(claim_text) > 140 else "")
    if sel.tools:
        tools_list = "\n".join(f"  - `{t.__name__}`" for t in sel.tools)
    else:
        tools_list = "  - _(none chosen)_"
    return (
        f"**Claim:** _{truncated}_\n\n"
        f"**Selected tools:**\n{tools_list}\n\n"
        f"**Reasoning:** {sel.reasoning or '—'}\n\n"
        f"---\n\n"
    )


def render_verification(vr: VerifierResult) -> str:
    v = vr.verification
    color = VERDICT_COLOR[v.verdict]
    emoji = VERDICT_EMOJI[v.verdict]
    tool_calls_md = (
        "\n".join(
            f"- **{tc.name}**(`{tc.args}`)\n  → `{tc.result}`" for tc in vr.tool_calls
        )
        or "_no tool calls made_"
    )
    return (
        f'<div style="border-left: 4px solid {color}; padding: 8px 14px; margin: 10px 0;">\n\n'
        f"**Claim:** _{v.claim.claim}_\n\n"
        f"**Verdict:** {emoji} <span style='color: {color}'>**{v.verdict.value.upper()}**</span>\n\n"
        f"**Rationale:** {v.rationale}\n\n"
        f"<details><summary><b>Tool calls</b> ({len(vr.tool_calls)})</summary>\n\n"
        f"{tool_calls_md}\n\n"
        f"</details>\n\n"
        f"</div>\n\n"
    )


def _md_escape(s: str) -> str:
    """Escape characters that would break a markdown table cell."""
    return s.replace("|", "\\|").replace("\n", " ")


def render_rubric(rubric: Rubric) -> str:
    # Per-claim summary table
    score_rows = ["| Claim | Score | Yes / Total |", "|---|---|---|"]
    for cr in rubric.per_claim:
        text = cr.claim.claim
        short = (text[:80] + "…") if len(text) > 80 else text
        yes = sum(1 for it in cr.items if it.answer)
        total = len(cr.items)
        score_rows.append(f"| {_md_escape(short)} | {cr.score:.0%} | {yes} / {total} |")

    # Detail table — every rubric item across every claim
    detail_rows = ["| Claim | Question | Answer | Reasoning |", "|---|---|---|---|"]
    for cr in rubric.per_claim:
        text = cr.claim.claim
        claim_short = (text[:60] + "…") if len(text) > 60 else text
        for it in cr.items:
            reasoning = (it.reasoning or "").strip()
            if len(reasoning) > 160:
                reasoning = reasoning[:160] + "…"
            detail_rows.append(
                f"| {_md_escape(claim_short)} | {_md_escape(it.question)} "
                f"| {'✓' if it.answer else '✗'} | {_md_escape(reasoning) or '—'} |"
            )

    return (
        f"### Overall confidence: **{rubric.confidence:.0%}**\n\n"
        f"#### Per-claim scores\n\n" + "\n".join(score_rows) + "\n\n"
        "#### Rubric items (all claims)\n\n" + "\n".join(detail_rows) + "\n"
    )


def render_final(report: Report) -> str:
    overall = report.overall_verdict
    color = VERDICT_COLOR[overall]
    emoji = VERDICT_EMOJI[overall]

    # Summary table — one row per verification
    summary_rows = ["| Verdict | Claim | Tool calls |", "|---|---|---|"]
    for vr in report.verifications:
        v = vr.verification
        verdict_cell = f"{VERDICT_EMOJI[v.verdict]} {v.verdict.value}"
        summary_rows.append(
            f"| {verdict_cell} | {_md_escape(v.claim.claim)} | {len(vr.tool_calls)} |"
        )

    return (
        f'<div style="border: 2px solid {color}; border-radius: 10px; padding: 18px; '
        f'margin: 12px 0; background: rgba(0,0,0,0.02);">\n\n'
        f"## 🎉 Pipeline complete\n\n"
        f"### Overall verdict: {emoji} <span style='color: {color}'>**{overall.value.upper()}**</span>\n\n"
        f"### Rubric confidence: **{report.rubric.confidence:.0%}**\n\n"
        f"_{len(report.verifications)} claim verification(s)_\n\n"
        f"</div>\n\n"
        f"#### Summary\n\n" + "\n".join(summary_rows) + "\n"
    )


# ── Stage labels ──────────────────────────────────────────────────────

STAGE_IDLE = "_Ready. Paste a claim and click **Verify**._"
STAGE_EXTRACT = "🔍 **Step 1/5:** Extracting claims & metadata…"
STAGE_SELECT = "🛠 **Step 2/5:** Selecting tools per claim (parallel)…"
STAGE_VERIFY = "🔬 **Step 3/5:** Verifying claims (parallel)…"
STAGE_RUBRIC = "📊 **Step 4/5:** Scoring rubric…"
STAGE_DONE = "🎉 **Step 5/5:** Done."


# ── Pre-flight checks (before consuming API quota or user wait time) ──

# Provider → env var pydantic-ai consults when no api_key is passed.
PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google-gla": "GEMINI_API_KEY",
    "google-vertex": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


def precheck(text: str, model: str, api_key: str) -> str | None:
    """Return an error message string if the request would fail trivially.

    Catches blank input / blank-key-with-no-env so users get a clear
    message instantly instead of waiting on a half-run pipeline.
    """
    if not text.strip():
        return "Please paste a claim or paragraph to verify."
    if not model.strip():
        return "Please provide a model (e.g. `openai:gpt-5.2`)."
    if ":" not in model:
        return (
            f"Model `{model}` is missing a provider prefix. Use `provider:name` "
            f"form (e.g. `openai:gpt-5.2`, `anthropic:claude-sonnet-4-6`)."
        )

    if api_key.strip():
        return None  # explicit key wins

    provider = model.split(":", 1)[0]
    env_var = PROVIDER_ENV.get(provider, f"{provider.upper()}_API_KEY")
    if not os.environ.get(env_var):
        return (
            f"No API key provided. Paste a `{provider}` key above, "
            f"or ask the Space owner to set `{env_var}`."
        )
    return None


def friendly_error(exc: Exception) -> str:
    """Translate pydantic-ai exceptions into user-readable messages."""
    name = type(exc).__name__
    msg = str(exc)
    if "401" in msg or "Incorrect API key" in msg or "Invalid API key" in msg:
        return "❌ **Invalid API key.** Check the key you pasted and try again."
    if "429" in msg or "rate limit" in msg.lower():
        return "❌ **Rate limited.** Your provider returned 429 — slow down and retry."
    if "insufficient_quota" in msg or "exceeded your current quota" in msg:
        return (
            "❌ **Quota exhausted.** The provider says your account has no credit left."
        )
    return f"❌ **Error** (`{name}`): {msg[:400]}"


# ── UI ────────────────────────────────────────────────────────────────

with gr.Blocks(title="GeoGuard — live demo") as demo:
    gr.Markdown(
        "# GeoGuard — live demo\n"
        "Agentic guardrails & validation framework for geospatial AI. "
        "Bring your own OpenAI API key; paste a claim or paragraph; watch "
        "the pipeline run end-to-end with live streaming events."
    )

    with gr.Row():
        # ── LEFT COLUMN: controls + input + examples ──
        with gr.Column(scale=2):
            gr.Markdown("### ⚙️ Settings")
            model_input = gr.Textbox(
                value="openai:gpt-5.2",
                label="Model (`provider:name`)",
                placeholder="openai:gpt-5.2",
            )
            api_key_input = gr.Textbox(
                value="",
                label="OpenAI API key",
                type="password",
                placeholder="sk-… (your own key; not stored)",
            )
            reasoning_input = gr.Dropdown(
                choices=["low", "medium", "high"],
                value="medium",
                label="Reasoning effort",
            )
            gr.Markdown("### 📝 Input")
            text_input = gr.Textbox(
                value=EXAMPLES[0][0],
                label="Text to verify",
                lines=8,
                max_lines=20,
            )
            run_button = gr.Button("🔎 Verify", variant="primary", size="lg")
            gr.Examples(
                examples=EXAMPLES,
                inputs=[text_input],
                label="Examples — click to fill",
            )

        # ── RIGHT COLUMN: stage indicator + streaming sections ──
        with gr.Column(scale=3):
            stage_md = gr.Markdown(STAGE_IDLE)

            with gr.Accordion("📦 Claim extraction", open=False) as claims_acc:
                claims_md = gr.Markdown()

            with gr.Accordion("🔧 Tool selection", open=False) as tools_acc:
                tools_md = gr.Markdown()

            with gr.Accordion("🔬 Verification", open=True) as verify_acc:
                verify_md = gr.Markdown()

            with gr.Accordion("📋 Rubric", open=False) as rubric_acc:
                rubric_md = gr.Markdown()

            final_md = gr.Markdown()

    # ── Streaming handler (closes over component refs above) ──
    async def verify(text, model, api_key, reasoning):
        # Cheap pre-flight check before consuming any API quota.
        if err := precheck(text or "", model or "", api_key or ""):
            yield {stage_md: f"❌ {err}"}
            return

        # Reset state at the start of every run
        yield {
            stage_md: STAGE_EXTRACT,
            claims_acc: gr.update(open=True),
            tools_acc: gr.update(open=False),
            verify_acc: gr.update(open=False),
            rubric_acc: gr.update(open=False),
            claims_md: "",
            tools_md: "",
            verify_md: "",
            rubric_md: "",
            final_md: "",
        }

        guard = GeoGuard(
            model=(model or "").strip() or None,
            api_key=(api_key or "").strip() or None,
            reasoning_effort=reasoning,
        )

        claims_text = ""
        tools_text = ""
        verify_text = ""

        try:
            async for item in guard(Input(text=text)):
                if isinstance(item, ClaimGroup):
                    claims_text += render_claim_group(item)
                    yield {claims_md: claims_text}
                elif isinstance(item, Claim):
                    claims_text += render_claim(item)
                    yield {claims_md: claims_text}
                elif isinstance(item, SelectedTools):
                    tools_text += render_tools(item)
                    yield {
                        stage_md: STAGE_SELECT,
                        claims_acc: gr.update(open=False),
                        tools_acc: gr.update(open=True),
                        tools_md: tools_text,
                    }
                elif isinstance(item, VerifierResult):
                    verify_text += render_verification(item)
                    yield {
                        stage_md: STAGE_VERIFY,
                        tools_acc: gr.update(open=False),
                        verify_acc: gr.update(open=True),
                        verify_md: verify_text,
                    }
                elif isinstance(item, Rubric):
                    yield {
                        stage_md: STAGE_RUBRIC,
                        verify_acc: gr.update(open=False),
                        rubric_acc: gr.update(open=True),
                        rubric_md: render_rubric(item),
                    }
                elif isinstance(item, Report):
                    yield {
                        stage_md: STAGE_DONE,
                        rubric_acc: gr.update(open=False),
                        final_md: render_final(item),
                    }
        except Exception as e:
            yield {stage_md: friendly_error(e)}

    run_button.click(
        fn=verify,
        inputs=[text_input, model_input, api_key_input, reasoning_input],
        outputs=[
            stage_md,
            claims_acc,
            tools_acc,
            verify_acc,
            rubric_acc,
            claims_md,
            tools_md,
            verify_md,
            rubric_md,
            final_md,
        ],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())

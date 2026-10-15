"""Run GeoGuard's agents on Claude through the local `claude` CLI.

Selected with model strings of the form ``claude-cli:<model>`` (e.g.
``claude-cli:sonnet``), resolved in ``config.build_model``. Every pydantic-ai
request is rendered into one prompt — instructions, transcript so far, tool
schemas — and sent to ``claude -p`` (Claude Code headless mode). Claude
answers with a JSON list of tool calls, which is mapped back to
``ToolCallPart``s. Finishing is just calling the agent's output tool
(``final_result``), so structured output, tool use and pydantic-ai's
validation retries all work unchanged.

This lets the full pipeline run on a Claude subscription without an API key.
Authenticate the CLI once with ``claude setup-token`` (or ``claude`` → login).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
import uuid
from contextvars import ContextVar

from loguru import logger
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from geoguard.config import settings


class ClaudeCLIError(RuntimeError):
    """The `claude` CLI failed in a way retrying won't fix (auth, missing binary)."""


class ClaudeUsageLimitError(ClaudeCLIError):
    """The Claude subscription hit its usage window — stop and resume later."""


PROTOCOL = """\
You are the language model behind a tool-using agent. You never answer in \
prose. Every reply is a JSON object of the form
  {"tool_calls": [{"tool_name": "<name>", "args": {<arguments>}}]}
listing one or more calls to the available tools, with arguments that match \
each tool's JSON schema exactly.

Tool results appear in the transcript as TOOL RESULT entries after you call \
a tool. When you have everything you need, finish by calling the output tool \
(its name starts with "final_result") with the final answer as its arguments. \
A VALIDATION ERROR entry means your previous call was rejected: fix the \
arguments and call again."""


# Cost accounting. The benchmark runner sets `current_cost_bucket` to a
# per-event key; every CLI call made inside that event (including nested
# agents running in child tasks, which inherit the context) adds its usage.
current_cost_bucket: ContextVar[str | None] = ContextVar(
    "claude_cli_bucket", default=None
)
cost_totals: dict[str, dict[str, float]] = {}

_semaphore: asyncio.Semaphore | None = None


def _sem() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(settings.claude_cli_concurrency)
    return _semaphore


def _content(value) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _render_transcript(messages: list[ModelMessage]) -> str:
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    lines.append(f"USER:\n{_content(part.content)}")
                elif isinstance(part, ToolReturnPart):
                    lines.append(
                        f"TOOL RESULT {part.tool_name} [{part.tool_call_id}]:\n"
                        f"{_content(part.content)}"
                    )
                elif isinstance(part, RetryPromptPart):
                    target = f" for {part.tool_name}" if part.tool_name else ""
                    lines.append(f"VALIDATION ERROR{target}:\n{_content(part.content)}")
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    lines.append(
                        f"YOU CALLED {part.tool_name} [{part.tool_call_id}] "
                        f"with {part.args_as_json_str()}"
                    )
                elif isinstance(part, TextPart) and part.content.strip():
                    lines.append(f"YOU SAID:\n{part.content}")
    return "\n\n".join(lines)


def _render_tools(info: AgentInfo) -> str:
    blocks = []
    for kind, tools in (
        ("TOOL", info.function_tools),
        ("OUTPUT TOOL", info.output_tools),
    ):
        for t in tools:
            blocks.append(
                f"{kind} {t.name}\n"
                f"description: {t.description or ''}\n"
                f"arguments schema: {json.dumps(t.parameters_json_schema)}"
            )
    return "\n\n".join(blocks)


def _reply_schema(info: AgentInfo) -> dict:
    names = [t.name for t in [*info.function_tools, *info.output_tools]]
    return {
        "type": "object",
        "properties": {
            "tool_calls": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": "string", "enum": names},
                        "args": {"type": "object"},
                    },
                    "required": ["tool_name", "args"],
                },
            }
        },
        "required": ["tool_calls"],
    }


def _parse_reply(result: dict) -> dict:
    structured = result.get("structured_output")
    if isinstance(structured, dict):
        return structured
    text = (result.get("result") or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    return json.loads(text[start : end + 1])


async def _run_cli(model: str, system: str, prompt: str, schema: dict) -> dict:
    binary = settings.claude_cli_path or shutil.which("claude")
    if not binary:
        raise ClaudeCLIError(
            "`claude` CLI not found — install Claude Code or set GEOGUARD_CLAUDE_CLI_PATH"
        )
    cmd = [
        binary,
        "-p",
        "--model",
        model,
        "--output-format",
        "json",
        "--no-session-persistence",
        "--tools",
        "",
        "--system-prompt",
        system,
        "--json-schema",
        json.dumps(schema),
    ]
    env = {**os.environ, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
    if settings.claude_cli_oauth_token and "CLAUDE_CODE_OAUTH_TOKEN" not in env:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = settings.claude_cli_oauth_token
    # Run outside the repo so no project CLAUDE.md / settings leak into the prompt.
    workdir = tempfile.gettempdir()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=workdir,
        env=env,
    )
    try:
        out, err = await asyncio.wait_for(
            proc.communicate(prompt.encode()),
            timeout=settings.claude_cli_timeout_seconds,
        )
    except TimeoutError:
        proc.kill()
        raise
    try:
        return json.loads(out.decode())
    except json.JSONDecodeError as e:
        raise ValueError(
            f"CLI returned non-JSON (exit {proc.returncode}): {out[:300]!r} {err[:300]!r}"
        ) from e


def _classify_error(message: str) -> Exception | None:
    low = message.lower()
    if "usage limit" in low or "limit reached" in low or "out of extra usage" in low:
        return ClaudeUsageLimitError(message)
    if "authenticate" in low or "oauth" in low or "log in" in low or "login" in low:
        return ClaudeCLIError(
            f"{message} — run `claude setup-token` (or `claude` and log in)"
        )
    return None


def _record_cost(result: dict) -> None:
    bucket = current_cost_bucket.get()
    if bucket is None:
        return
    usage = result.get("usage") or {}
    tot = cost_totals.setdefault(
        bucket, {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "calls": 0}
    )
    tot["cost_usd"] += float(result.get("total_cost_usd") or 0.0)
    tot["input_tokens"] += (
        int(usage.get("input_tokens") or 0)
        + int(usage.get("cache_read_input_tokens") or 0)
        + int(usage.get("cache_creation_input_tokens") or 0)
    )
    tot["output_tokens"] += int(usage.get("output_tokens") or 0)
    tot["calls"] += 1


def claude_cli_model(model: str = "sonnet") -> FunctionModel:
    """A pydantic-ai model that answers every request via `claude -p --model <model>`."""

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        system = (info.instructions or "").strip()
        system = f"{system}\n\n{PROTOCOL}" if system else PROTOCOL
        prompt = (
            "AVAILABLE TOOLS\n\n"
            + _render_tools(info)
            + "\n\nTRANSCRIPT\n\n"
            + _render_transcript(messages)
            + "\n\nReply with the JSON tool_calls object for your next step."
        )
        schema = _reply_schema(info)

        last_error: Exception | None = None
        for attempt in range(settings.claude_cli_retries + 1):
            async with _sem():
                try:
                    result = await _run_cli(model, system, prompt, schema)
                except (TimeoutError, ValueError) as e:
                    last_error = e
                    logger.warning("claude-cli attempt {} failed: {}", attempt + 1, e)
                    await asyncio.sleep(2**attempt)
                    continue
            if result.get("is_error"):
                message = str(result.get("result") or result.get("terminal_reason"))
                fatal = _classify_error(message)
                if fatal:
                    raise fatal
                last_error = ClaudeCLIError(message)
                logger.warning(
                    "claude-cli attempt {} errored: {}", attempt + 1, message
                )
                await asyncio.sleep(2**attempt)
                continue
            _record_cost(result)
            try:
                reply = _parse_reply(result)
            except (ValueError, json.JSONDecodeError) as e:
                last_error = e
                continue
            parts = [
                ToolCallPart(
                    tool_name=call["tool_name"],
                    args=call.get("args") or {},
                    tool_call_id=f"call_{uuid.uuid4().hex[:12]}",
                )
                for call in reply.get("tool_calls", [])
            ]
            if not parts:
                last_error = ValueError("reply had no tool_calls")
                continue
            usage = result.get("usage") or {}
            return ModelResponse(
                parts=parts,
                usage=RequestUsage(
                    input_tokens=int(usage.get("input_tokens") or 0),
                    output_tokens=int(usage.get("output_tokens") or 0),
                    cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
                    cache_write_tokens=int(
                        usage.get("cache_creation_input_tokens") or 0
                    ),
                ),
            )
        raise ClaudeCLIError(f"claude-cli failed after retries: {last_error}")

    return FunctionModel(respond, model_name=f"claude-cli:{model}")

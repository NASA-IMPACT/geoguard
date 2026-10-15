"""The claude-cli bridge, exercised against a fake `claude` binary (no network, no auth)."""

import asyncio
import json
import stat
import sys

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

from geoguard import claude_cli
from geoguard.config import build_model, settings

# Fake CLI: first turn calls the `add` tool; once a TOOL RESULT is in the
# transcript, it finishes by calling the output tool with the sum.
FAKE_CLI = """\
import json, re, sys
prompt = sys.stdin.read()
schema = json.loads(sys.argv[sys.argv.index("--json-schema") + 1])
names = schema["properties"]["tool_calls"]["items"]["properties"]["tool_name"]["enum"]
final = next(n for n in names if n.startswith("final_result"))
m = re.search(r"TOOL RESULT add \\[[^\\]]+\\]:\\n(\\d+)", prompt)
if m:
    calls = [{"tool_name": final, "args": {"total": int(m.group(1))}}]
else:
    calls = [{"tool_name": "add", "args": {"a": 2, "b": 3}}]
print(json.dumps({
    "type": "result", "is_error": False, "result": "",
    "structured_output": {"tool_calls": calls},
    "total_cost_usd": 0.001,
    "usage": {"input_tokens": 10, "output_tokens": 5},
}))
"""


class Total(BaseModel):
    total: int


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    script = tmp_path / "claude"
    script.write_text(f"#!{sys.executable}\n{FAKE_CLI}")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(settings, "claude_cli_path", str(script))
    monkeypatch.setattr(claude_cli, "_semaphore", None)
    return script


def test_build_model_routes_claude_cli_prefix():
    model = build_model("claude-cli:sonnet")
    assert model.model_name == "claude-cli:sonnet"


def test_tool_call_then_structured_output(fake_cli):
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    agent = Agent(
        build_model("claude-cli:sonnet"),
        output_type=Total,
        tools=[add],
        capabilities=[Thinking(effort="low")],
        instructions="Add the numbers.",
    )
    result = asyncio.run(agent.run("What is 2 + 3?"))
    assert result.output == Total(total=5)
    called = [
        p.tool_name
        for m in result.all_messages()
        for p in getattr(m, "parts", [])
        if getattr(p, "part_kind", "") == "tool-call"
    ]
    assert called[0] == "add"


def test_cost_is_recorded_per_bucket(fake_cli):
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    agent = Agent(build_model("claude-cli:sonnet"), output_type=Total, tools=[add])

    async def run_in_bucket():
        claude_cli.current_cost_bucket.set("evt-1")
        await agent.run("2 + 3?")

    asyncio.run(run_in_bucket())
    totals = claude_cli.cost_totals.pop("evt-1")
    assert totals["calls"] == 2
    assert totals["cost_usd"] == pytest.approx(0.002)


def test_auth_error_is_fatal_with_fix_hint():
    err = claude_cli._classify_error("Failed to authenticate: OAuth session expired")
    assert isinstance(err, claude_cli.ClaudeCLIError)
    assert "setup-token" in str(err)


def test_usage_limit_is_its_own_error():
    err = claude_cli._classify_error("Claude AI usage limit reached")
    assert isinstance(err, claude_cli.ClaudeUsageLimitError)


def test_reply_parser_handles_fenced_text():
    reply = claude_cli._parse_reply(
        {"result": '```json\n{"tool_calls": [{"tool_name": "x", "args": {}}]}\n```'}
    )
    assert json.loads(json.dumps(reply))["tool_calls"][0]["tool_name"] == "x"

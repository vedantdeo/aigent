"""A run written to Markdown as it happens: headings nested by depth, each model call's new message,
thinking, text and calls, each tool's input and result, and what raised."""

from __future__ import annotations

from pathlib import Path

import pytest

from aigent.llm.messages import Usage
from aigent.transcript import FileTracer

SYSTEM = {"role": "system", "content": "You are an analyst."}
ASKED = {"role": "user", "content": "<question>Capex?</question>"}
REPLY = {
    "role": "assistant",
    "content": [
        {"type": "thinking", "thinking": "Search first."},
        {"type": "text", "text": "Searching."},
        {"type": "tool_use", "id": "t1", "name": "search_filings", "input": {"query": "capex"}},
    ],
    "stop_reason": "tool_use",
}


def written(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_a_turn_and_the_tool_it_called_are_written_nested_as_they_happen(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "q1.md"
    tracer = FileTracer(path)

    with tracer.observe(
        "research", "generation", inputs=[SYSTEM, ASKED], metadata={"step": "r:1"}
    ) as seen:
        assert "## generation: research" in written(path), "the heading is there before the reply"
        with tracer.observe("search_filings", "retriever", inputs={"query": "capex"}) as tool:
            tool.finish("<passage>Capex was Rs 1,31,107 crore.</passage>")
        seen.finish(REPLY)
        seen.bill(
            "sarvam-105b", Usage(input_tokens=90, output_tokens=40, cache_read_tokens=10), 0.0012
        )

    text = written(path)
    for expected in (
        '## generation: research · {"step": "r:1"}',
        "**System prompt:**\n\n```text\nYou are an analyst.\n```",
        "**Told:**\n\n```text\n<question>Capex?</question>\n```",
        "### retriever: search_filings",
        '**Input** `{"query": "capex"}`',
        "Capex was Rs 1,31,107 crore.",
        "**Thought:**\n\n```text\nSearch first.\n```",
        "**Said:**\n\nSearching.",
        '**Asked for** `search_filings` `{"query": "capex"}`',
        "*stopped on tool_use*",
        "*sarvam-105b: 100 in (10 cached), 40 out, $0.00120*",
    ):
        assert expected in text, expected
    assert text.index("### retriever") < text.index("**Thought:**"), "in the order it happened"


def test_a_salvaged_call_is_marked_as_one(tmp_path: Path) -> None:
    tracer = FileTracer(tmp_path / "q.md")
    leaked = {
        "type": "tool_use",
        "id": "salvaged_0",
        "name": "calculate",
        "input": {},
        "salvaged": True,
    }

    with tracer.observe("research", "generation") as seen:
        seen.finish({"content": [leaked], "stop_reason": "tool_use"})

    assert "*(salvaged from leaked text)*" in written(tmp_path / "q.md")


def test_a_system_prompt_is_written_once_and_tool_results_are_not_retold(tmp_path: Path) -> None:
    tracer = FileTracer(tmp_path / "q.md")
    results = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}],
    }

    for inputs in ([SYSTEM, ASKED], [SYSTEM, ASKED, REPLY, results]):
        with tracer.observe("research", "generation", inputs=inputs):
            pass

    text = written(tmp_path / "q.md")
    assert text.count("**System prompt:**") == 1 and text.count("**Told:**") == 1


@pytest.mark.parametrize(
    ("output", "error", "expected"),
    [
        pytest.param(None, "Error: no such company", "**Error:**", id="a tool error"),
        pytest.param({"figures": []}, None, '{"figures": []}', id="a result that is not text"),
    ],
)
def test_what_a_tool_finished_with_is_written(
    tmp_path: Path, output: object, error: str | None, expected: str
) -> None:
    tracer = FileTracer(tmp_path / "q.md")

    with tracer.observe("extract_figures", "tool", inputs={}) as seen:
        seen.finish(output, error=error)

    assert expected in written(tmp_path / "q.md")


def test_what_raised_is_written_and_still_raises(tmp_path: Path) -> None:
    tracer = FileTracer(tmp_path / "q.md")

    with pytest.raises(RuntimeError), tracer.observe("research", "generation"):
        raise RuntimeError("specialist unreachable")

    assert "**Raised:**\n\n```text\nRuntimeError: specialist unreachable\n```" in written(
        tmp_path / "q.md"
    )
    with tracer.observe("next", "span"):
        pass
    assert "\n## span: next\n" in written(tmp_path / "q.md"), "the depth unwinds after a raise"

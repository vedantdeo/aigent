"""The analyst desk: the trimmed graph's answer as the page's JSON, the recorded calls, the cited
facts, a derived figure written back, the page and its one-at-a-time endpoint, and the command's
estimate, refusals and warm-up."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import cast

import pytest
from anthropic.types import Message, ToolParam
from pydantic import BaseModel

pytest.importorskip("langgraph")
pytest.importorskip("fastapi")

from starlette.testclient import TestClient  # noqa: E402

from aigent import mcp_client  # noqa: E402
from aigent.analyst import desk  # noqa: E402
from aigent.analyst.cfg_analyst import DESK_SERVED_TOOLS, INDIAN_RUPEE_UNITS  # noqa: E402
from aigent.analyst.memo import Claim, Judged, Memo  # noqa: E402
from aigent.analyst.router import Router  # noqa: E402
from aigent.analyst.run import generalist_of  # noqa: E402
from aigent.analyst.specialists import Passage  # noqa: E402
from aigent.analyst.toolbox import Toolbox  # noqa: E402
from aigent.facts.store import Writer  # noqa: E402

from ..conftest import MakeLlm, Sent, tool_turn  # noqa: E402
from .conftest import OP24, OP25  # noqa: E402

QUESTION = "What was ICICI Bank's standalone operating profit in FY25?"
FACT = {
    "cite_as": "F1",
    "label": "Operating profit before provisions",
    "value": 672988200000.0,
    "unit": "INR",
    "period": {"from": "2024-04-01", "to": "2025-03-31"},
    "basis": "standalone",
    "segment": None,
    "tier": "xbrl-filed",
    "source": "ICICIBANK 2025-03-31 standalone",
}
LOOKUP: dict[str, object] = {
    "company": "ICICIBANK",
    "concept": "OperatingProfit",
    "year": "2025",
    "quarter": "Q4",
    "period": "12M",
}
DRAFT = "ICICI Bank's standalone operating profit for FY25 was Rs 67,298.82 crore [F1]."
CLAIM = Claim(id="c1", text="Operating profit was Rs 67,298.82 crore in FY25.", citations=["F1"])
SUPPORTED = Judged(verdict="supported", reason="the fact says 672988200000 INR")
CHOSEN = generalist_of("anthropic", "adaptive")
TOOLS: list[ToolParam] = [{"name": "fundamentals", "input_schema": {"type": "object"}}]


def served(name: str, arguments: dict[str, object]) -> tuple[str, bool]:
    if name == "fundamentals":
        return json.dumps({"facts": [FACT], "lookup_id": "L1", "unverified_held": 2}), False
    return "Error: no such tool", True


def researching(sent: Sent) -> Message | str | BaseModel | None:
    """Looks the figure up once, drafts, then writes the memo citing it."""
    if sent.schema is Memo:
        return Memo(summary="Rs 67,298.82 crore.", claims=[CLAIM], caveats=["Q4 not asked."])
    return tool_turn(("t1", "fundamentals", LOOKUP)) if len(sent.messages) == 1 else DRAFT


def judging(sent: Sent) -> Judged:
    return SUPPORTED


STAMP = Writer("analyst-desk|m", "2026-11-03T18:00:00+05:30", "desk-1")


def test_an_answer_carries_the_memo_verdicts_cited_facts_calls_and_spend(
    make_llm: MakeLlm, concepts_db: sqlite3.Connection
) -> None:
    writer, sent = make_llm(researching)
    critic, _ = make_llm(judging)

    answer = desk.answer(QUESTION, TOOLS, served, CHOSEN, writer, critic, concepts_db, STAMP)

    assert answer["error"] is None and answer["summary"] == "Rs 67,298.82 crore."
    assert answer["claims"] == [
        {
            "id": "c1",
            "text": CLAIM.text,
            "verdict": "supported",
            "reason": SUPPORTED.reason,
            "facts": [FACT],
        }
    ]
    assert answer["caveats"] == ["Q4 not asked."] and answer["draft"] == DRAFT
    calls = cast(list[dict[str, object]], answer["calls"])
    assert [(c["tool"], c["arguments"], c["error"]) for c in calls] == [
        ("fundamentals", LOOKUP, False)
    ]
    assert answer["held_back"] == 2
    spent = cast(dict[str, float], answer["spent"])
    assert set(spent) == {"generalist", "critic"} and spent["generalist"] > 0, spent
    research = sent.sent[0]
    assert research.system == f"{desk.DESK_PROMPT} {desk.held_to(concepts_db, date.today())}", (
        "the trimmed prompt, not the capstone's, with the time held"
    )
    names = [tool["name"] for tool in cast(list[dict[str, object]], research.tools)]
    assert names == [
        "fundamentals",
        "plan_metric",
        "compute_metric",
        "define_metric",
        "propose_formula",
    ], "no specialists and no filings search"
    assert answer["written"] == [] and answer["skipped"] == [], "nothing computed, none written"


def _years(name: str, arguments: dict[str, object]) -> tuple[str, bool]:
    if name == "get_metric":
        found = {"derived": [{"name": "OperatingProfitGrowth", "kind": "flow"}], "filed": []}
        return json.dumps(found), False
    fact = OP25 if arguments.get("year") == "2025" else OP24
    return json.dumps({"facts": [fact], "lookup_id": "L1", "unverified_held": 0}), False


PLAN_ASK: dict[str, object] = {
    "name": "OperatingProfitGrowth",
    "company": "ICICIBANK",
    "year": "2025",
    "quarter": "Q4",
    "period": "12M",
    "previous_year": "2024",
    "previous_quarter": "Q4",
    "previous_period": "12M",
    "basis": "standalone",
}
FETCH: dict[str, object] = {
    "company": "ICICIBANK",
    "concept": "OperatingProfit",
    "quarter": "Q4",
    "period": "12M",
    "basis": "standalone",
    "segment": None,
}
COMPUTE_ARGS: dict[str, object] = {
    "plan_id": "P1",
    "inputs": [
        {"variable": "current", "cite_as": OP25["cite_as"]},
        {"variable": "previous", "cite_as": OP24["cite_as"]},
    ],
}
GROWTH_CLAIM = Claim(
    id="c1", text="Operating profit grew 15.77% in FY25.", citations=["DERIVED:D1"]
)


def planning(sent: Sent) -> Message | str | BaseModel | None:
    """Finds the metric's name, plans it, fetches both years, computes it, drafts, then a memo
    citing it."""
    if sent.schema is Memo:
        return Memo(summary="Up 15.77%.", claims=[GROWTH_CLAIM], caveats=[])
    turn = (len(sent.messages) - 1) // 2
    if turn == 0:
        return tool_turn(("t0", "get_metric", {"company": "ICICIBANK", "what": "op growth"}))
    if turn == 1:
        return tool_turn(("t1", "plan_metric", PLAN_ASK))
    if turn == 2:
        return tool_turn(
            ("t2", "fundamentals", {**FETCH, "year": "2025"}),
            ("t3", "fundamentals", {**FETCH, "year": "2024"}),
        )
    return tool_turn(("t4", "compute_metric", COMPUTE_ARGS)) if turn == 3 else "Up [DERIVED:D1]."


def test_a_metric_is_planned_fetched_computed_and_written_back(
    make_llm: MakeLlm, concepts_db: sqlite3.Connection
) -> None:
    writer, _ = make_llm(planning)
    critic, judged = make_llm(judging)

    finding: list[ToolParam] = [*TOOLS, {"name": "get_metric", "input_schema": {"type": "object"}}]
    answer = desk.answer(QUESTION, finding, _years, CHOSEN, writer, critic, concepts_db, STAMP)

    calls = cast(list[dict[str, object]], answer["calls"])
    assert [c["tool"] for c in calls] == [
        "get_metric",
        "plan_metric",
        "fundamentals",
        "fundamentals",
        "compute_metric",
    ]
    assert not any(c["error"] for c in calls), calls
    written = cast(list[dict[str, object]], answer["written"])
    assert [(w["metric"], w["tier"], w["formula"], w["periods"]) for w in written] == [
        ("OperatingProfitGrowth", "derived-ai", "growth", "FY2025 over FY2024")
    ]
    claims = cast(list[dict[str, object]], answer["claims"])
    facts = cast(list[dict[str, object]], claims[0]["facts"])
    assert facts[0]["cite_as"] == "DERIVED:D1" and facts[0]["metric"] == "OperatingProfitGrowth"
    assert "672988200000" in str(judged.sent[0].messages), "the critic reads the inputs' values"
    assert concepts_db.execute(
        "SELECT concept FROM facts WHERE tier = 'derived-ai' AND run_id = 'desk-1'"
    ).fetchall() == [("OperatingProfitGrowth",)]


@pytest.mark.parametrize(
    ("question", "critic_usd", "error"),
    [
        pytest.param("   ", 1.0, "GuardrailTripped", id="an empty question trips the guard"),
        pytest.param(QUESTION, 1e-9, "BudgetExceeded", id="the critic's budget runs out"),
    ],
)
def test_a_run_that_raises_comes_back_as_an_error_not_an_exception(
    make_llm: MakeLlm,
    concepts_db: sqlite3.Connection,
    question: str,
    critic_usd: float,
    error: str,
) -> None:
    writer, _ = make_llm(researching)
    critic, _ = make_llm(judging, limit_usd=critic_usd)
    answer = desk.answer(question, TOOLS, served, CHOSEN, writer, critic, concepts_db, STAMP)
    assert str(answer["error"]).startswith(error), answer["error"]
    assert answer["claims"] == []


def test_a_generalist_out_of_budget_stops_with_no_memo(
    make_llm: MakeLlm, concepts_db: sqlite3.Connection
) -> None:
    writer, _ = make_llm(researching, limit_usd=1e-9)
    critic, _ = make_llm(judging)
    answer = desk.answer(QUESTION, TOOLS, served, CHOSEN, writer, critic, concepts_db, STAMP)
    assert answer["error"] is None and answer["summary"] is None
    assert "ceiling" in str(answer["stopped"]), answer["stopped"]


@pytest.mark.parametrize(
    ("reply", "result"),
    [
        pytest.param(('{"facts": []}', False), {"facts": []}, id="json is parsed"),
        pytest.param(("12.5", False), 12.5, id="a bare number is json too"),
        pytest.param(("Error: refused", True), "Error: refused", id="text stays text"),
    ],
)
def test_recorded_keeps_each_call_with_its_result(reply: tuple[str, bool], result: object) -> None:
    calls: list[desk.Call] = []
    dispatch = desk.recorded(lambda name, arguments: reply, calls)
    assert dispatch("calculate", {"expression": "25/2"}) == reply, "the result passes through"
    assert [(c["tool"], c["arguments"], c["result"], c["error"]) for c in calls] == [
        ("calculate", {"expression": "25/2"}, result, reply[1])
    ]


def _box(evidence: dict[str, Passage]) -> Toolbox:
    return Toolbox(TOOLS, served, Router(price=lambda i, o: 0.0), [], evidence=evidence)


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        pytest.param({"F1": Passage("F1", "s", json.dumps(FACT))}, [FACT], id="a looked-up fact"),
        pytest.param({}, [{"cite_as": "F1"}], id="an id never looked up"),
        pytest.param({"F1": Passage("F1", "s", "not json")}, [{"cite_as": "F1"}], id="bad json"),
        pytest.param({"F1": Passage("F1", "s", "[1]")}, [{"cite_as": "F1"}], id="not an object"),
    ],
)
def test_cited_gives_each_id_its_fact(
    evidence: dict[str, Passage], expected: list[dict[str, object]]
) -> None:
    assert desk.cited(_box(evidence), ["F1"]) == expected


@pytest.mark.parametrize(
    ("calls", "held"),
    [
        pytest.param([], 0, id="no calls"),
        pytest.param(
            [{"tool": "fundamentals", "result": {"unverified_held": 2}}] * 2, 4, id="summed"
        ),
        pytest.param(
            [{"tool": "fundamentals", "result": "Error: no company"}], 0, id="a refused lookup"
        ),
        pytest.param(
            [{"tool": "calculate", "result": {"unverified_held": 9}}], 0, id="another tool's"
        ),
        pytest.param(
            [{"tool": "fundamentals", "result": {"unverified_held": "2"}}], 0, id="not a count"
        ),
    ],
)
def test_held_back_counts_what_fundamentals_held(calls: list[desk.Call], held: int) -> None:
    assert desk.held_back(calls) == held


def test_the_page_is_served_and_a_question_answered() -> None:
    asked: list[str] = []

    def ask(question: str) -> dict[str, object]:
        asked.append(question)
        return {"question": question}

    client = TestClient(desk.app(ask))
    assert "Analyst Desk" in client.get("/").text
    assert client.post("/ask", json={"question": QUESTION}).json() == {"question": QUESTION}
    assert client.post("/ask", json={}).status_code == 422, "a question is required"
    assert asked == [QUESTION]


def test_questions_are_answered_one_at_a_time() -> None:
    inside, most = [0], [0]
    guard = threading.Lock()

    def ask(question: str) -> dict[str, object]:
        with guard:
            inside[0] += 1
            most[0] = max(most[0], inside[0])
        time.sleep(0.2)
        with guard:
            inside[0] -= 1
        return {}

    client = TestClient(desk.app(ask))
    posting = [
        threading.Thread(target=lambda: client.post("/ask", json={"question": "q"}))
        for _ in range(3)
    ]
    for thread in posting:
        thread.start()
    for thread in posting:
        thread.join()
    assert most[0] == 1, "one question in flight on the shared MCP session"


def _no_server(*args: object, **kwargs: object) -> object:
    raise AssertionError("the server was started")


@pytest.mark.parametrize(
    ("argv", "said"),
    [
        pytest.param([], "estimate only", id="without --yes, an estimate"),
        pytest.param(["--generalist", "nope"], "no client named", id="an unknown client"),
        pytest.param(["--generalist", "anthropic:claude-x"], "has no price", id="unpriced"),
    ],
)
def test_the_command_serves_nothing_unless_told(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    said: str,
) -> None:
    monkeypatch.setattr(mcp_client, "connect", _no_server)
    try:
        desk.main(argv)
    except SystemExit as stopped:
        assert said in str(stopped), stopped
        return
    out = capsys.readouterr().out
    assert said in out and f"${desk.worst_case_usd():.2f}" in out, out


def test_yes_warms_the_lookup_and_serves_over_only_the_desk_s_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import contextlib

    import uvicorn

    connected: dict[str, object] = {}
    served_on: dict[str, object] = {}
    warmed: list[tuple[str, dict[str, object]]] = []

    def dispatch(name: str, arguments: dict[str, object]) -> tuple[str, bool]:
        warmed.append((name, arguments))
        return served(name, arguments)

    @contextlib.contextmanager
    def connect(server: object, **options: object) -> Iterator[mcp_client.Served]:
        connected.update(options)
        yield mcp_client.Served(TOOLS, dispatch)

    seeded: list[str] = []
    monkeypatch.setattr(mcp_client, "connect", connect)
    monkeypatch.setattr(uvicorn, "run", lambda app, **options: served_on.update(options))
    monkeypatch.setattr(desk, "FACTS_DB", tmp_path / "facts.sqlite")
    monkeypatch.setattr(desk, "seed", lambda conn, writer: seeded.append(writer.written_by) or [])
    synced: list[str] = []
    monkeypatch.setattr(
        desk,
        "sync_share_events",
        lambda conn, writer: (synced.append(writer.written_by), (0, []))[1],
    )
    desk.main(["--yes", "--port", "9999"])
    assert seeded == ["seed|evals/reference/derived-concepts.jsonl"], (
        "the seed metrics, before serving"
    )
    assert synced == ["share-events|desk"], "share events seeded and detected, before serving"
    assert connected == {"only": DESK_SERVED_TOOLS, "fenced": False}
    assert served_on == {"host": "127.0.0.1", "port": 9999, "log_level": "warning"}
    assert warmed == [("get_metric", desk.WARM_UP)], "one free search before serving"


@pytest.mark.parametrize(("unit", "size"), INDIAN_RUPEE_UNITS)
def test_the_prompt_asks_for_rupees_in_each_indian_unit(unit: str, size: int) -> None:
    assert f"{unit} (10^{len(str(size)) - 1})" in desk.DESK_PROMPT, desk.DESK_PROMPT
    assert "never in millions" in desk.DESK_PROMPT


def test_the_prompt_names_today_and_the_latest_quarter_each_company_is_held_to(
    concepts_db: sqlite3.Connection,
) -> None:
    with concepts_db:
        concepts_db.executemany(
            """INSERT INTO filings (ticker, fy, period_end, basis, source, origin, sha256, fetched)
               VALUES (?, 2026, ?, 'standalone', 'xbrl', ?, 'x', 't')""",
            [
                ("ICICIBANK", "2025-09-30", "a"),
                ("ICICIBANK", "2026-03-31", "b"),
                ("ITC", "2026-03-31", "c"),
                ("WIPRO", "2025-12-31", "d"),
                ("BAJAJ-AUTO", "2025-12-31", "e"),
            ],
        )
    told = desk.held_to(concepts_db, date(2026, 11, 4))
    assert told.startswith(
        "Today is 4 November 2026. The latest results the database holds end Q4 FY2026 "
        "(31 March 2026) for ICICIBANK, ITC; Q3 FY2026 (31 December 2025) for BAJAJ-AUTO, "
        "WIPRO. "
    ), told
    assert "latest time held" in told and "Say in the answer which time" in told

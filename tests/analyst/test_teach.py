"""v4's trajectories against fakes: the questions drawn, the training reports searched one at a
time, a trajectory kept or refused, the rows in one order, the estimate, the run and the export."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue

from aigent.analyst import teach as module
from aigent.analyst.served import research
from aigent.analyst.specialists import Found, Invalid, Passage, Told
from aigent.analyst.teach import (
    Asked,
    estimate,
    export,
    load_asked,
    main,
    missed,
    mix,
    named,
    read_rows,
    relabelled,
    report_search,
    select,
    task,
    teach,
    undescribed,
    worked,
    write_rows,
)
from aigent.extraction.figures import Labelled
from aigent.extraction.research import SourcedFigure
from aigent.llm.core import Dispatch, Llm
from aigent.llm.messages import Msg
from aigent.llm.pricing import cost_usd
from aigent.retrieval.chunk import Chunk

from ..conftest import FakeMessages, MakeLlm, tool_turn, turns

CHUNK = Chunk(
    id="BAJAJ-AUTO-FY24#0004",
    text="Revenue clocked an all-time high at ₹ 44,685 crores, up 23% year-on-year.",
    doc_id="BAJAJ-AUTO-FY24",
    ordinal=4,
    start=0,
    page=8,
)
REVENUE = Labelled(
    metric="revenue", value=446850000000.0, unit="INR", year=2024, call="44685 * 10**7"
)
UNSCALED = Labelled(metric="revenue", value=44685.0, unit="INR", year=2024, call=None)
GROWTH = Labelled(
    metric="revenue", value=0.23, unit=None, year=2024, description="change", call="23 / 100"
)
ASKED = Asked(CHUNK.id, "What was Bajaj Auto's revenue in FY2023-24?", (REVENUE,))
NAMES = {"BAJAJ-AUTO": "Bajaj Auto Limited"}


def sourced(*figures: Labelled, cited: str = CHUNK.id) -> str:
    """A teacher's answer: each figure with its call and the passage it cites."""
    rows = [{**f.model_dump(), "chunk_id": cited} for f in figures]
    return json.dumps({"figures": rows})


def searched(asked: list[tuple[str, str]]) -> Dispatch:
    def find(query: str, doc_id: str) -> list[Chunk]:
        asked.append((query, doc_id))
        return [CHUNK]

    return report_search(find, {"BAJAJ-AUTO-FY24", "BAJAJ-AUTO-FY25"})


def teacher(make_llm: MakeLlm, answer: str) -> tuple[Llm, FakeMessages]:
    """A teacher that searches, then answers `answer`, writing its calls for code to make."""
    return make_llm(turns(tool_turn(("s1", "search_report", {"query": "revenue"})), answer))


@pytest.mark.parametrize(
    ("chunk_id", "ticker", "year"),
    [
        ("BAJAJ-AUTO-FY24#0004", "BAJAJ-AUTO", 2024),
        ("POWERGRID-FY26#0100", "POWERGRID", 2026),
        ("HDFCBANK-FY25#0001", "HDFCBANK", 2025),
    ],
)
def test_a_question_s_job_is_its_chunk_s_report(chunk_id: str, ticker: str, year: int) -> None:
    job = Asked(chunk_id, "q", ()).job

    assert (job.ticker, job.year, job.question) == (ticker, year, "q")


def test_only_questions_with_a_labelled_figure_are_asked_in_the_current_vocabulary(
    tmp_path: Path,
) -> None:
    equity = Labelled(
        metric="other", value=5.0, unit="INR", year=2024, description="Total equity", call=None
    )
    rows = [
        {"chunk_id": CHUNK.id, "question": "a", "figures": [REVENUE.model_dump()]},
        {"chunk_id": CHUNK.id, "question": "b", "figures": []},
        {"chunk_id": CHUNK.id, "question": "c", "figures": [equity.model_dump()]},
    ]
    answers = tmp_path / "answers.jsonl"
    answers.write_text("".join(json.dumps(r) + "\n" for r in rows))

    a, c = load_asked(answers)
    assert a == Asked(CHUNK.id, "a", (REVENUE,)) and c.question == "c"
    assert c.labels[0].metric == "equity", "a v3 other label read as the metric it now is"


OTHER = Labelled(
    metric="other", value=95.0, unit=None, year=2024, description="countries", call=None
)


def _many(others: int = 5) -> list[Asked]:
    """Five named questions a report over three reports, and `others` `other` ones each."""
    return [
        Asked(f"{report}#{n:04d}", f"{report} {n}", (REVENUE,) if n < 5 else (OTHER,))
        for report in ("A-FY24", "B-FY24", "C-FY25")
        for n in range(5 + others)
    ]


def test_questions_are_drawn_four_named_to_one_other_evenly_and_in_a_seeded_order() -> None:
    ten, fifteen = select(_many(), 10, seed=3, paired=0), select(_many(), 15, seed=3, paired=0)

    assert [named(one) for one in ten] == [True] * 8 + [False] * 2
    assert [named(one) for one in fifteen] == [True] * 12 + [False] * 3
    assert [one.report for one in fifteen[:12]].count("A-FY24") == 4, "even across reports"
    assert {a.chunk_id for a in ten} <= {a.chunk_id for a in fifteen}, (
        "a larger draw keeps a smaller"
    )
    assert select(_many(), 10, seed=3, paired=0) == ten
    assert select(_many(), 10, seed=4, paired=0) != ten


@pytest.mark.parametrize(
    ("others", "n", "split"),
    [
        pytest.param(0, 10, (10, 0), id="no other questions: all named"),
        pytest.param(20, 30, (15, 15), id="too few named: other makes them up"),
        pytest.param(5, 100, (15, 15), id="never more than there are"),
    ],
)
def test_a_part_too_small_for_its_share_is_made_up_by_the_other(
    others: int, n: int, split: tuple[int, int]
) -> None:
    drawn = select(_many(others), n, paired=0)

    assert (sum(map(named, drawn)), sum(not named(one) for one in drawn)) == split


def _companies() -> list[Asked]:
    """Four named questions for each of three companies, and two other ones each."""
    return [
        Asked(
            f"{t}-FY24#{n:04d}",
            f"What did {t} report, part {n}?",
            (REVENUE,) if n < 4 else (OTHER,),
        )
        for t in ("AAA", "BBB", "CCC")
        for n in range(6)
    ]


def test_a_third_of_a_draw_asks_about_two_companies_the_job_naming_one() -> None:
    drawn = select(_companies(), 9, seed=1, paired=1 / 3)

    pairs = [one for one in drawn if one.paired]
    assert len(pairs) == 3 and mix(drawn)["two companies"] == 3
    for pair in pairs:
        own = pair.job.ticker
        assert pair.question.count("?") == 2 and f"What did {own} report" in pair.question
        assert sum(t in pair.question for t in ("AAA", "BBB", "CCC")) == 2, "two companies"
        assert pair.labels == (REVENUE,), "the job's labels, never the other company's"
    parts = [q for one in drawn for q in one.question.split("? ") if q]
    assert len(parts) == len({p.rstrip("?") for p in parts}), "no question used twice"


def test_a_pair_needs_another_company_and_a_draw_without_pairs_has_none() -> None:
    alone = [one for one in _companies() if one.job.ticker == "AAA"]

    assert not any(one.paired for one in select(alone, 6, paired=0.5))
    assert not any(one.paired for one in select(_companies(), 9, paired=0.0))


def test_a_draw_s_mix_counts_each_kind_and_the_questions_over_two_years() -> None:
    years = (REVENUE, REVENUE.model_copy(update={"year": 2023}))
    two_years = Asked("AAA-FY24#0001", "q", years)
    paired = Asked("AAA-FY24#0002", "q2", (REVENUE,), paired=True)
    both = Asked("AAA-FY24#0004", "q4", years, paired=True)

    assert mix([two_years, paired, both, Asked("AAA-FY24#0003", "q3", (OTHER,))]) == {
        "named": 1,
        "other": 1,
        "two companies": 2,
        "two years or more": 2,
        "both": 1,
    }


@pytest.mark.parametrize(
    ("label", "metric"),
    [
        pytest.param(("Income tax expense", "INR"), "tax", id="a tax expense"),
        pytest.param(("Effective tax rate", None), "tax", id="the tax rate, though no money"),
        pytest.param(("Trade receivables", "INR"), "receivables", id="trade receivables"),
        pytest.param(("Total equity", "INR"), "equity", id="total equity"),
        pytest.param(("Paid-up equity share capital", "INR"), "equity", id="share capital"),
        pytest.param(("Segment assets", "INR"), "assets", id="segment assets"),
        pytest.param(("Deferred tax assets", "INR"), "assets", id="a tax asset is an asset"),
        pytest.param(("Deferred tax liabilities (net)", "INR"), "other", id="a tax liability"),
        pytest.param(("paid-up equity share capital", None), "other", id="a share of capital"),
        pytest.param(("countries", None), "other", id="anything else"),
    ],
)
def test_a_v3_other_label_is_read_in_the_current_vocabulary(
    label: tuple[str, str | None], metric: str
) -> None:
    description, unit = label
    old = Labelled(
        metric="other", value=1.0, unit=unit, year=2024, description=description, call=None
    )

    now = relabelled(old)

    assert now.metric == metric, now
    assert now.description == (description if metric == "other" else None)
    assert relabelled(REVENUE) == REVENUE, "a named label is left alone"


@pytest.mark.parametrize(
    ("name", "arguments", "says", "doc_id"),
    [
        pytest.param(
            "search_filings",
            {"query": "revenue", "company": "BAJAJ-AUTO", "year": "2024"},
            CHUNK.text,
            "BAJAJ-AUTO-FY24",
            id="one report, by ticker and year",
        ),
        pytest.param(
            "search_filings",
            {"query": "q", "company": "BAJAJ-AUTO", "year": "2026"},
            "Error: no training report BAJAJ-AUTO-FY26",
            None,
            id="a report not in training",
        ),
        pytest.param(
            "search_filings",
            {"query": "q", "company": "BAJAJ-AUTO", "year": "all"},
            "Error: no financial year 'all'",
            None,
            id="a year that is not one",
        ),
        pytest.param("calculate", {"expression": "1"}, "not available", None, id="another tool"),
    ],
)
def test_a_training_search_reads_one_report_and_refuses_the_rest_as_data(
    name: str, arguments: dict[str, object], says: str, doc_id: str | None
) -> None:
    asked: list[tuple[str, str]] = []

    content, is_error = searched(asked)(name, arguments)

    assert says in content and is_error is (doc_id is None), content
    assert asked == ([(str(arguments["query"]), doc_id)] if doc_id else [])


@pytest.mark.parametrize(
    ("figures", "lost"),
    [
        pytest.param([REVENUE], [], id="every label found"),
        pytest.param([REVENUE, GROWTH], [], id="more than the labels is fine"),
        pytest.param([GROWTH], [REVENUE], id="a label not found"),
        pytest.param(
            [REVENUE.model_copy(update={"year": 2025})], [REVENUE], id="the right value, wrong year"
        ),
        pytest.param([REVENUE.model_copy(update={"unit": "Rs"})], [], id="Rs is INR"),
    ],
)
def test_a_label_is_found_on_metric_value_unit_and_year(
    figures: list[Labelled], lost: list[Labelled]
) -> None:
    answered = [SourcedFigure(**f.model_dump(), chunk_id=CHUNK.id) for f in figures]

    assert missed([REVENUE], answered) == lost


def test_a_trajectory_that_passes_and_finds_its_labels_is_kept_with_every_turn(
    make_llm: MakeLlm,
) -> None:
    llm, sdk = teacher(make_llm, sourced(REVENUE))
    asked: list[tuple[str, str]] = []

    row = teach(ASKED, research(llm, searched(asked), "claude-sonnet-5", NAMES))

    assert row["kept"] is True and row["why"] is None, row["why"]
    figure = {**REVENUE.model_dump(exclude={"call"}), "basis": None, "chunk_id": CHUNK.id}
    assert row["figures"] == [figure], "the answer as trained: no call in it"
    assert (row["ticker"], row["year"]) == ("BAJAJ-AUTO", 2024)
    roles = [m["role"] for m in cast(list[dict[str, object]], row["conversation"])]
    assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]
    assert asked == [("revenue", "BAJAJ-AUTO-FY24")]
    assert cast(float, row["usd"]) > 0


@pytest.mark.parametrize(
    ("answer", "why"),
    [
        pytest.param(
            sourced(UNSCALED), "missed revenue 4.4685e+11 2024", id="checked, but a label missed"
        ),
        pytest.param(
            sourced(REVENUE, cited="BAJAJ-AUTO-FY24#0999"),
            "not a passage its searches returned",
            id="a figure from a passage never read",
        ),
        pytest.param("I could not find it.", "no figures", id="no figures at all"),
        pytest.param(
            sourced(UNSCALED.model_copy(update={"metric": "other", "description": "sales"})),
            "described 'sales'",
            id="checked, but a description not in its passage",
        ),
    ],
)
def test_a_trajectory_that_fails_is_kept_out_with_why(
    make_llm: MakeLlm, answer: str, why: str
) -> None:
    llm, _ = teacher(make_llm, answer)

    row = teach(ASKED, research(llm, searched([]), "claude-sonnet-5", NAMES))

    assert row["kept"] is False and why in str(row["why"]), row["why"]


PRINTED = Labelled(metric="profit", value=132180.0, unit="INR", year=2025, call=None)
SCALED = PRINTED.model_copy(update={"value": 132180.0 * 10**7, "call": "132180 * 10**7"})


@pytest.mark.parametrize(
    ("label", "value", "found"),
    [
        pytest.param(PRINTED, 132180.0, True, id="the value as printed"),
        pytest.param(PRINTED, 132180.0 * 10**6, True, id="a printed label, scaled by millions"),
        pytest.param(PRINTED, 132180.0 * 10**5, True, id="a printed label, scaled by lakhs"),
        pytest.param(PRINTED, 132180.0 * 10**4, False, id="ten thousand is no named scale"),
        pytest.param(SCALED, 132180.0 * 10**7 * 10**3, False, id="a label already scaled"),
    ],
)
def test_a_label_written_as_printed_is_found_at_any_named_scale(
    label: Labelled, value: float, found: bool
) -> None:
    answered = SourcedFigure(**label.model_copy(update={"value": value}).model_dump(), chunk_id="x")

    assert (missed([label], [answered]) == []) is found


def _told(text: str, turns_before: int = 1) -> Told:
    """A teacher's reply after `turns_before` search turns."""
    searching: list[Msg] = [{"role": "user", "content": "q"}]
    for n in range(turns_before):
        asked: Msg = {"role": "assistant", "content": [{"type": "tool_use", "id": f"s{n}"}]}
        result: Msg = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"s{n}"}]}
        searching += [asked, result]
    final: Msg = {"role": "assistant", "content": [{"type": "text", "text": text}]}
    return Told(text, 0.0, 0, 0, (), (), (*searching, final))


def test_the_teacher_s_calls_become_one_calculate_turn_before_the_answer() -> None:
    told = worked(_told("Here they are: " + sourced(REVENUE, GROWTH)))

    *_, asking, results, answer = told.conversation
    uses = cast(list[dict[str, object]], asking["content"])
    assert [u["input"] for u in uses] == [
        {"expression": "44685 * 10**7"},
        {"expression": "23 / 100"},
    ]
    assert [r["content"] for r in cast(list[dict[str, object]], results["content"])] == [
        "446850000000.0",
        "0.23",
    ]
    assert told.calls == ("44685 * 10**7", "23 / 100")
    assert "call" not in told.text and told.text.startswith('{"figures"'), "the answer, no prose"
    assert cast(list[dict[str, str]], answer["content"])[0]["text"] == told.text


def test_figures_printed_in_full_take_no_calculate_turn() -> None:
    told = worked(_told(sourced(UNSCALED)))

    assert [m["role"] for m in told.conversation] == ["user", "assistant", "user", "assistant"]
    assert told.calls == ()


@pytest.mark.parametrize(
    ("text", "searches", "why"),
    [
        pytest.param(
            sourced(REVENUE.model_copy(update={"call": "44685 * 10**6"})),
            1,
            "44685 * 10**6 gives 44685000000.0, not 446850000000.0",
            id="a call that does not give its value",
        ),
        pytest.param("I could not find it.", 1, "no figures", id="no figures"),
        pytest.param(sourced(REVENUE), 5, "more than the loop's 6 turns", id="too many turns"),
    ],
)
def test_a_teacher_s_answer_that_cannot_be_worked_is_refused(
    text: str, searches: int, why: str
) -> None:
    with pytest.raises(Invalid, match=re.escape(why)):
        worked(_told(text, searches))


@pytest.mark.parametrize(
    ("figure", "bad"),
    [
        pytest.param(OTHER.model_copy(update={"description": "Revenue"}), False, id="verbatim"),
        pytest.param(
            OTHER.model_copy(update={"description": "an all-time high at ₹ 44,685"}),
            True,
            id="six words, though verbatim",
        ),
        pytest.param(OTHER.model_copy(update={"description": "sales"}), True, id="not in it"),
        pytest.param(UNSCALED.model_copy(update={"description": "growth"}), True, id="named, said"),
        pytest.param(GROWTH, False, id="a named metric's change"),
    ],
)
def test_a_description_keeps_its_rule_or_the_trajectory_is_dropped(
    figure: Labelled, bad: bool
) -> None:
    sourced_figure = SourcedFigure(**figure.model_dump(), chunk_id=CHUNK.id)
    read = Passage(CHUNK.id, "", CHUNK.text)

    assert bool(undescribed(Found([sourced_figure], {CHUNK.id: read}))) is bad


def test_rows_are_written_in_one_order_and_a_missing_file_is_none(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    rows: list[dict[str, object]] = [
        {"chunk_id": "B#0001", "question": "z"},
        {"chunk_id": "A#0002", "question": "y"},
        {"chunk_id": "A#0002", "question": "x"},
    ]

    assert read_rows(path) == []
    write_rows(path, cast(list[dict[str, JsonValue]], rows))

    assert [(r["chunk_id"], r["question"]) for r in read_rows(path)] == [
        ("A#0002", "x"),
        ("A#0002", "y"),
        ("B#0001", "z"),
    ]


def test_the_estimate_is_each_turn_s_growing_prompt_and_its_output() -> None:
    typical, worst = estimate(2000, 10, "claude-sonnet-5", 1024)

    inputs = sum(2000 + n * (module.SEARCH_RESULT_TOKENS + 220) for n in range(4))
    assert typical == pytest.approx(10 * cost_usd("claude-sonnet-5", inputs, 4 * 220))
    assert worst > typical


def _patched(monkeypatch: pytest.MonkeyPatch, llm: Llm, asked: list[tuple[str, str]]) -> None:
    monkeypatch.setattr(module, "Llm", lambda client, budget: llm)
    monkeypatch.setattr(module, "load_asked", lambda: [ASKED])
    monkeypatch.setattr(module, "companies", lambda: NAMES)
    monkeypatch.setattr(module, "training_search", lambda: searched(asked))


def test_without_yes_a_run_counts_and_estimates_and_sends_nothing(
    make_llm: MakeLlm,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    llm, sdk = teacher(make_llm, sourced(REVENUE))
    _patched(monkeypatch, llm, [])

    main(["run", "--out", str(tmp_path / "t.jsonl")])

    said = capsys.readouterr().out
    assert "estimate only: nothing sent" in said and "free count" in said, said
    assert sdk.sent == [] and sdk.counted, "counted for free, nothing sent"
    assert not (tmp_path / "t.jsonl").exists()


def test_a_run_writes_its_rows_and_a_second_run_skips_what_is_done(
    make_llm: MakeLlm, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    llm, sdk = teacher(make_llm, sourced(REVENUE))
    _patched(monkeypatch, llm, [])
    out = tmp_path / "t.jsonl"

    main(["run", "--yes", "--workers", "1", "--out", str(out)])
    sent = len(sdk.sent)
    main(["run", "--yes", "--workers", "1", "--out", str(out)])

    [row] = read_rows(out)
    assert row["kept"] is True and sent == 2, "a search and the answer: code makes the call"
    first = sdk.sent[0]
    assert [t["name"] for t in cast(list[dict[str, object]], first.tools)] == ["search_report"]
    assert module.TEACHER_NOTE in str(first.system), "the teacher is told to write its calls"
    assert len(sdk.sent) == sent, "nothing sent for a question already done"


def test_the_export_is_the_kept_trajectories_on_the_served_wire(
    make_llm: MakeLlm, tmp_path: Path
) -> None:
    llm, _ = teacher(make_llm, sourced(REVENUE))
    kept = teach(ASKED, research(llm, searched([]), "claude-sonnet-5", NAMES))
    refused = {**kept, "question": "another", "kept": False}
    rows = tmp_path / "t.jsonl"
    write_rows(rows, [kept, refused])

    assert export(rows, tmp_path / "out") == 1

    [line] = [
        json.loads(x) for x in (tmp_path / "out" / "trajectories.jsonl").read_text().splitlines()
    ]
    roles = [m["role"] for m in line["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert line["messages"][0]["content"] == task()["system"]
    assert line["messages"][2]["tool_calls"][0]["function"]["name"] == "search_report"
    written = json.loads((tmp_path / "out" / "task.json").read_text())
    assert [t["function"]["name"] for t in written["tools"]] == ["search_report", "calculate"]

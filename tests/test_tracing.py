"""Tracing: off without keys, the real SDK's spans offline, and the tree `llm` and the runner grow.

Every test traces into conftest's `Recorder` except the two that drive the real Langfuse SDK, which
export into memory and never touch the network.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import cast

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from aigent import tracing
from aigent.evals.dataset import Case
from aigent.evals.grade import Outcome, Score, exact_match
from aigent.evals.runner import Task, run_eval
from aigent.llm import Request
from aigent.messages import Usage

from .conftest import MakeLlm, Recorder, tool_turn, turns

GREET = Request.ask("greet", "be brief", "hello", 64)


class Words(BaseModel):
    words: str


@pytest.mark.parametrize(
    ("public", "secret", "switch", "on"),
    [
        pytest.param("pk", "sk", "auto", True, id="both keys trace"),
        pytest.param("pk", None, "auto", False, id="one key is not enough"),
        pytest.param(None, None, "auto", False, id="no keys, no tracing"),
        pytest.param("pk", "sk", "off", False, id="off wins over the keys"),
    ],
)
def test_tracing_is_on_only_with_both_keys_and_not_switched_off(
    monkeypatch: pytest.MonkeyPatch, public: str | None, secret: str | None, switch: str, on: bool
) -> None:
    for name, value in (("LANGFUSE_PUBLIC_KEY", public), ("LANGFUSE_SECRET_KEY", secret)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(tracing, "TRACING", switch)
    assert tracing.enabled() is on


def test_without_keys_the_tracer_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    tracing.use(None)
    assert isinstance(tracing.tracer(), tracing.Silent)
    with tracing.tracer().observe("anything") as seen:
        seen.finish("ignored")


@pytest.fixture
def exported(
    request: pytest.FixtureRequest,
) -> Iterator[tuple[tracing.LangfuseTracer, InMemorySpanExporter]]:
    """The real Langfuse SDK, exporting into memory and pointed at a port nothing listens on."""
    from langfuse import Langfuse

    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key=f"pk-lf-{request.node.name}",  # the SDK keeps one client per key
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",
        span_exporter=exporter,
        mask=tracing.masked,  # as `LangfuseTracer` builds it
    )
    yield tracing.LangfuseTracer(client), exporter
    client.shutdown()


def _attributes(span: ReadableSpan) -> dict[str, object]:
    return dict(span.attributes or {})


def test_the_real_sdk_gets_a_typed_billed_generation_under_a_tagged_row(
    exported: tuple[tracing.LangfuseTracer, InMemorySpanExporter],
) -> None:
    langfuse, exporter = exported
    usage = Usage(input_tokens=10, output_tokens=5, cache_read_tokens=3, web_searches=1)

    with langfuse.observe("agent-tasks", inputs={"task": "q"}, tags=("pt", "sdk"), session="s1"):
        with langfuse.observe(
            "agent", "generation", inputs=[{"role": "user", "content": "q"}]
        ) as g:
            g.bill("claude-sonnet-5", usage, 0.0123)
    langfuse.flush()

    spans = {span.name: span for span in exporter.get_finished_spans()}
    row, generation = _attributes(spans["agent-tasks"]), _attributes(spans["agent"])
    parent, root = spans["agent"].parent, spans["agent-tasks"].context
    assert parent is not None and root is not None and parent.span_id == root.span_id
    assert generation["langfuse.observation.type"] == "generation"
    assert generation["langfuse.observation.model.name"] == "claude-sonnet-5"
    assert json.loads(str(generation["langfuse.observation.cost_details"])) == {"total": 0.0123}
    assert json.loads(str(generation["langfuse.observation.usage_details"])) == {
        "input": 10,
        "output": 5,
        "cache_read_input_tokens": 3,
        "cache_creation_input_tokens": 0,
    }
    for attributes in (row, generation):
        assert tuple(cast(Sequence[str], attributes["langfuse.trace.tags"])) == ("pt", "sdk")
        assert attributes["session.id"] == "s1", attributes


def test_the_real_sdk_scores_the_trace_as_a_boolean(
    exported: tuple[tracing.LangfuseTracer, InMemorySpanExporter],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    langfuse, exporter = exported
    sent: list[dict[str, object]] = []
    monkeypatch.setattr(langfuse._client, "create_score", lambda **kw: sent.append(kw))

    with langfuse.observe("tasks.jsonl") as row:
        row.score("correct", True, "names the dividend")
        row.score("guarded", False)
    langfuse.flush()

    [span] = exporter.get_finished_spans()
    trace = format(span.context.trace_id, "032x") if span.context else None
    assert [(s["name"], s["value"], s["comment"]) for s in sent] == [
        ("correct", 1.0, "names the dividend"),
        ("guarded", 0.0, ""),
    ]
    assert {str(s["data_type"]) for s in sent} == {"BOOLEAN"}
    assert {s["trace_id"] for s in sent} == {trace}, "scored on the row's own trace"


def test_the_real_sdk_marks_an_observation_that_raised_and_lets_it_raise(
    exported: tuple[tracing.LangfuseTracer, InMemorySpanExporter],
) -> None:
    langfuse, exporter = exported

    with pytest.raises(RuntimeError, match="down"), langfuse.observe("search_reports", "tool"):
        raise RuntimeError("the index is down")
    langfuse.flush()

    [span] = exporter.get_finished_spans()
    attributes = _attributes(span)
    assert attributes["langfuse.observation.level"] == "ERROR", attributes
    assert attributes["langfuse.observation.status_message"] == "RuntimeError: the index is down"


def test_a_credential_never_leaves_in_a_trace(
    exported: tuple[tracing.LangfuseTracer, InMemorySpanExporter],
) -> None:
    langfuse, exporter = exported
    key = "sk-ant-api03-abcdefghijklmnop"

    with langfuse.observe("read_file", "tool", inputs={"path": "notes"}) as seen:
        seen.finish(f"the key is {key}", metadata={"echo": key})
    langfuse.flush()

    [span] = exporter.get_finished_spans()
    sent = json.dumps(_attributes(span), default=str)
    assert key not in sent and "redacted credential" in sent, sent


@pytest.mark.parametrize(
    ("name", "result", "kind", "error"),
    [
        pytest.param("search_reports", ("five passages", False), "retriever", None, id="a search"),
        pytest.param("calculate", ("4", False), "tool", None, id="any other tool"),
        pytest.param("calculate", ("Error: 1/0", True), "tool", "Error: 1/0", id="a failed call"),
    ],
)
def test_each_tool_call_is_an_observation_of_its_kind(
    traces: Recorder, name: str, result: tuple[str, bool], kind: str, error: str | None
) -> None:
    traced = tracing.traced_tools(lambda _name, _args: result, {"search_reports": "retriever"})

    assert traced(name, {"x": 1}) == result, "what the model sees is untouched"
    [seen] = traces.every()
    assert (seen.name, seen.kind, seen.inputs, seen.output) == (name, kind, {"x": 1}, result[0])
    assert seen.error == error


def test_a_call_is_one_generation_billed_as_the_budget_billed_it(
    make_llm: MakeLlm, traces: Recorder
) -> None:
    llm, _ = make_llm(lambda sent: "hi")

    llm.text(GREET)

    [seen] = traces.every()
    [call] = llm.trace
    assert (seen.name, seen.kind) == ("greet", "generation")
    assert seen.billed == (call.model, call.usage, call.usd)
    assert seen.inputs == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]
    assert isinstance(seen.output, dict) and seen.output["stop_reason"] == "end_turn"


def test_each_turn_of_a_tool_run_is_its_own_generation_named_without_its_number(
    make_llm: MakeLlm, traces: Recorder
) -> None:
    llm, _ = make_llm(turns(tool_turn(("t1", "current_time", {})), "noon"))
    request = Request(
        "agent",
        [{"role": "user", "content": "time?"}],
        256,
        tools=[{"name": "current_time", "description": "now", "input_schema": {"type": "object"}}],
    )

    llm.run_tools(request, lambda name, args: ("noon", False))

    generations = traces.every()
    assert [seen.name for seen in generations] == ["agent", "agent"]
    assert [seen.metadata["step"] for seen in generations] == ["agent:1", "agent:2"]
    assert len(cast(list[object], generations[1].inputs)) == 3, "it shows the history it was sent"
    assert all(seen.billed is not None for seen in generations)


def test_a_batch_nests_under_whatever_sent_it_though_it_runs_on_other_threads(
    make_llm: MakeLlm, traces: Recorder
) -> None:
    llm, _ = make_llm(lambda sent: sent.prompt)
    batch = [Request.ask(f"vote:{i}", "be brief", str(i), 64) for i in range(3)]

    with tracing.tracer().observe("vote-on-it"):
        llm.gather_text(batch)

    [root] = traces.roots
    assert sorted(str(seen.metadata["step"]) for seen in root.children) == [
        "vote:0",
        "vote:1",
        "vote:2",
    ]


def test_a_reply_with_no_record_is_marked_on_its_generation(
    make_llm: MakeLlm, traces: Recorder
) -> None:
    llm, _ = make_llm(lambda sent: None)

    llm.parse(GREET, Words)

    [seen] = traces.every()
    assert seen.error == "no record parsed"


def _case(n: int) -> Case:
    return Case.model_validate(
        {"id": f"c{n}", "input": {"q": f"q{n}"}, "expected": {"answer": "yes"}, "tags": ["easy"]}
    )


def test_every_row_is_a_trace_with_its_grades_as_evaluators_and_scores(traces: Recorder) -> None:
    def task(case: Case) -> Outcome:
        return Outcome(output={"answer": "yes" if case.id == "c1" else "no"})

    run_eval(
        [_case(1), _case(2)],
        {"plain": task},
        {"exact": exact_match("answer")},
        dataset="demo",
        model="claude-sonnet-5",
        progress=False,
    )

    first, second = traces.roots
    assert (first.name, first.inputs) == ("demo", {"q": "q1"})
    assert first.tags == ("demo", "plain", "claude-sonnet-5", "easy")
    assert first.session is not None and first.session == second.session, "one run, one session"
    assert first.metadata["case"] == "c1" and first.output == {"answer": "yes"}
    assert [(seen.name, seen.kind) for seen in first.children] == [("exact", "evaluator")]
    assert first.scores["exact"][0] is True and second.scores["exact"][0] is False


def _reports_an_error(case: Case) -> Outcome:
    return Outcome(error="no answer")


def _raises(case: Case) -> Outcome:
    raise ValueError("boom")


@pytest.mark.parametrize(
    ("task", "error"),
    [
        pytest.param(_reports_an_error, "no answer", id="a reported error"),
        pytest.param(_raises, "ValueError: boom", id="a task that raised"),
    ],
)
def test_a_row_that_failed_is_marked_on_its_trace(traces: Recorder, task: Task, error: str) -> None:
    run_eval(
        [_case(1)],
        {"plain": task},
        {"never": lambda case, outcome: Score(True)},
        progress=False,
    )

    [root] = traces.roots
    assert root.error == error and root.scores == {}

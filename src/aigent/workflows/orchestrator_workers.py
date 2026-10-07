"""Orchestrator-workers: a model decides the subtasks, workers do them, a model combines them.

uv run python -m aigent.workflows.orchestrator_workers ["question"] [--yes]

Unlike sectioning, the split is not known in advance: the orchestrator reads the question and
writes the plan. Code only caps its size and runs it.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from aigent.config import MAX_PLAN_TASKS, MAX_TOKENS_PLAN, MAX_TOKENS_SYNTHESIS, MAX_TOKENS_WORKER
from aigent.config import THINKING_WORKFLOW_PARAM as THINKING
from aigent.llm import Llm, Request
from aigent.retrieval.chunk import context_block
from aigent.workflows.reports import CATALOGUE, DocId, Search

PLAN = (
    f"You plan research over three annual reports:\n{CATALOGUE}\n\nSplit the question into "
    "independent subtasks, each answerable from one search. Use as few as the question needs, "
    f"at most {MAX_PLAN_TASKS}."
)
WORK = (
    "You answer one research brief from annual-report passages. Use only the passages, cite "
    "passage ids in square brackets, and say what they do not cover. Under 100 words."
)
SYNTHESISE = (
    "You combine research findings into one answer to the original question. Keep their "
    "citations, flag contradictions, and add no fact the findings lack. Under 200 words."
)


class Subtask(BaseModel):
    brief: str = Field(description="What the worker must find out, in one sentence.")
    query: str = Field(description="The search query that finds the passages it needs.")
    report: DocId | None = Field(description="The one report to search, or null for all three.")


class Plan(BaseModel):
    subtasks: list[Subtask] = Field(description=f"One to {MAX_PLAN_TASKS}, each independent.")


@dataclass(frozen=True)
class Orchestrated:
    plan: list[Subtask]
    findings: list[str]
    answer: str


def run(llm: Llm, search: Search, question: str) -> Orchestrated:
    planning = Request.ask("plan", PLAN, question, MAX_TOKENS_PLAN, thinking=THINKING)
    plan = llm.record(planning, Plan).subtasks[:MAX_PLAN_TASKS]
    if not plan:
        return Orchestrated([], [], "(the orchestrator planned no subtasks)")
    # Search runs here, one query at a time: the local models are not shared across threads.
    briefs = [
        Request.ask(
            f"worker:{number}",
            WORK,
            f"{context_block(search(task.query, doc_id=task.report))}\n\nbrief: {task.brief}",
            MAX_TOKENS_WORKER,
            thinking=THINKING,
        )
        for number, task in enumerate(plan, 1)
    ]
    findings = llm.gather_text(briefs)
    listed = "\n\n".join(
        f"<finding brief={task.brief!r}>\n{found}\n</finding>"
        for task, found in zip(plan, findings, strict=True)
    )
    prompt = f"question: {question}\n\n{listed}"
    last = Request.ask("synthesise", SYNTHESISE, prompt, MAX_TOKENS_SYNTHESIS, thinking=THINKING)
    return Orchestrated(plan, findings, llm.text(last))

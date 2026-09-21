"""Evaluator-optimizer: one model drafts, another grades against criteria, until it passes.

uv run python -m entropic.workflows.evaluator_optimizer ["question"] [--yes]

The evaluator is `JUDGE_MODEL`, not the model that drafted, for the reason the eval's judge is.
The loop is capped, so an evaluator that is never satisfied cannot become a cost bug.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from entropic.config import JUDGE_MODEL, MAX_REFINE_ROUNDS, MAX_TOKENS_CRITIQUE, MAX_TOKENS_DRAFT
from entropic.config import THINKING_WORKFLOW_PARAM as THINKING
from entropic.llm import Llm, Request
from entropic.retrieval.chunk import context_block
from entropic.workflows.demo import run_demo
from entropic.workflows.reports import Search

QUESTIONS = ("What were Reliance's capital expenditure priorities in FY25?",)

CRITERIA = (
    "1. Every figure and claim appears in the passage it cites.\n"
    "2. Every claim cites a passage id in square brackets.\n"
    "3. It answers the question asked, and says which part the passages do not cover.\n"
    "4. It is under 120 words."
)
DRAFT = f"You answer questions from annual-report passages. Your answer must meet:\n{CRITERIA}"
EVALUATE = (
    f"You grade answers drawn from annual-report passages against:\n{CRITERIA}\n"
    "Be strict: pass only an answer that meets all four."
)


class Critique(BaseModel):
    passed: bool = Field(description="True only if the answer meets every criterion.")
    problems: list[str] = Field(description="Each failed criterion and where; empty if passed.")


@dataclass(frozen=True)
class Refined:
    answer: str
    passed: bool
    critiques: list[Critique]


def run(llm: Llm, search: Search, question: str) -> Refined:
    asked = f"{context_block(search(question))}\n\nquestion: {question}"
    answer, critiques = "", list[Critique]()
    for round_ in range(1, MAX_REFINE_ROUNDS + 1):
        prompt = asked
        if critiques:
            problems = "\n".join(f"- {problem}" for problem in critiques[-1].problems)
            prompt += f"\n\nyour last answer:\n{answer}\n\nit failed review:\n{problems}"
        draft = Request.ask(f"draft:{round_}", DRAFT, prompt, MAX_TOKENS_DRAFT, thinking=THINKING)
        answer = llm.text(draft)
        graded = Request.ask(
            f"evaluate:{round_}",
            EVALUATE,
            f"{asked}\n\nanswer:\n{answer}",
            MAX_TOKENS_CRITIQUE,
            model=JUDGE_MODEL,
            thinking=THINKING,
        )
        critiques.append(llm.record(graded, Critique))
        if critiques[-1].passed:
            return Refined(answer, True, critiques)
    return Refined(answer, False, critiques)


def show(result: Refined) -> str:
    lines = []
    for number, critique in enumerate(result.critiques, 1):
        verdict = "passed" if critique.passed else "; ".join(critique.problems)
        lines.append(f"round {number}: {verdict}")
    lines.append(f"\n{result.answer}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    prog = "entropic.workflows.evaluator_optimizer"
    run_demo(argv, prog=prog, questions=QUESTIONS, run=run, show=show)


if __name__ == "__main__":
    main()

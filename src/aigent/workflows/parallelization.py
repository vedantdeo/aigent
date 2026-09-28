"""Parallelization: independent calls at once, combined in code. Both forms, one after the other.

uv run python -m aigent.workflows.parallelization ["question"] [--yes]

Sectioning splits the work along a line fixed in advance — one report each — and runs the parts
concurrently. Voting has several reviewers check the same answer, each from its own angle.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from aigent.config import MAX_TOKENS_SECTION, MAX_TOKENS_VOTE
from aigent.config import THINKING_WORKFLOW_PARAM as THINKING
from aigent.llm import Llm, Request
from aigent.retrieval.chunk import context_block
from aigent.workflows.demo import run_demo
from aigent.workflows.reports import REPORTS, Search

QUESTIONS = ("What does each company say about water use and conservation?",)

SECTION = (
    "You report what one company's annual report says on a question. Use only the passages, "
    "cite passage ids in square brackets, and say so if they do not cover it. Under 100 words."
)
REVIEW = "You review answers drawn from annual-report passages, for exactly one thing: "
REVIEWERS = {
    "figures": "every number in the answer appears in the passages with the same meaning.",
    "attribution": "every statement is attributed to the company whose report it comes from.",
    "grounding": "the answer asserts nothing the passages do not support.",
}


class Vote(BaseModel):
    passed: bool = Field(description="True only if the answer meets the check.")
    reason: str = Field(description="One sentence; name the offending claim if it failed.")


@dataclass(frozen=True)
class Parallel:
    sections: dict[str, str]
    votes: dict[str, Vote]

    @property
    def flagged(self) -> bool:
        """Each reviewer is a veto: one failed check flags the whole answer."""
        return not all(vote.passed for vote in self.votes.values())


def run(llm: Llm, search: Search, question: str) -> Parallel:
    found = {doc_id: search(question, doc_id=doc_id) for doc_id in REPORTS}
    sectioned = [
        Request.ask(
            f"section:{doc_id}",
            SECTION,
            f"company: {REPORTS[doc_id]}\n\n{context_block(passages)}\n\nquestion: {question}",
            MAX_TOKENS_SECTION,
            thinking=THINKING,
        )
        for doc_id, passages in found.items()
    ]
    sections = dict(zip(found, llm.gather_text(sectioned), strict=True))

    answer = "\n\n".join(f"{REPORTS[doc_id]}: {text}" for doc_id, text in sections.items())
    evidence = context_block([chunk for passages in found.values() for chunk in passages])
    prompt = f"{evidence}\n\nquestion: {question}\n\nanswer:\n{answer}"
    reviews = [
        Request.ask(f"vote:{name}", REVIEW + check, prompt, MAX_TOKENS_VOTE, thinking=THINKING)
        for name, check in REVIEWERS.items()
    ]
    votes = llm.gather_records(reviews, Vote)
    return Parallel(sections, dict(zip(REVIEWERS, votes, strict=True)))


def show(result: Parallel) -> str:
    lines = [f"{REPORTS[doc_id]}:\n{text}\n" for doc_id, text in result.sections.items()]
    lines += [
        f"{'pass' if v.passed else 'FAIL'}  {name}: {v.reason}" for name, v in result.votes.items()
    ]
    lines.append("flagged" if result.flagged else "every reviewer passed it")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    prog = "aigent.workflows.parallelization"
    run_demo(argv, prog=prog, questions=QUESTIONS, run=run, show=show)


if __name__ == "__main__":
    main()

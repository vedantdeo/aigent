"""LLM-as-judge: the one grader that spends.

Runs on its client's `judge_model`, deliberately not the model under test, and sends through `llm`
like any other paid path. Its usage goes back in the `Score` so the runner bills it to the eval's
budget.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from aigent.adapters import spec
from aigent.config import CLIENT, THINKING_EVAL_PARAM, max_tokens
from aigent.evals.dataset import Case
from aigent.evals.grade import Outcome, Score
from aigent.llm import Llm, Request

JUDGE_SYSTEM = (
    "You grade one answer against one rubric. Judge only what the rubric asks about. Be strict: if "
    "the answer is missing something the rubric requires, it fails. Ignore style, length and "
    "phrasing unless the rubric mentions them. A reference answer, when given, is one acceptable "
    "answer and not the only one."
)


class Verdict(BaseModel):
    """What the judge returns. Reasoning first is deliberate: it is the model's scratch space."""

    reasoning: str = Field(description="One or two sentences citing the rubric clause you applied.")
    passed: bool = Field(
        description="True only if the answer satisfies every clause of the rubric."
    )


@dataclass
class LlmJudge:
    """The paid grader: applies a written rubric to one outcome.

    Returns `passed=False` without calling anything when the task already failed.
    """

    rubric: str
    name: str = "answer"
    reference: str | None = None
    model: str | None = None  # None means "the model this client grades with"
    client: str = CLIENT  # the client the judge runs on, not the one that answered
    sdk: object | None = None
    _llm: Llm | None = field(default=None, init=False, repr=False)
    _building: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def __call__(self, case: Case, outcome: Outcome) -> Score:
        if outcome.error is not None:
            return Score(False, f"task failed: {outcome.error}")
        with self._building:  # rows judged concurrently share one Llm
            if self._llm is None:
                self._llm = Llm.for_eval(self.client, sdk=self.sdk)
        llm = self._llm

        prompt = self.prompt_for(case, outcome)
        settings = spec(self.client)
        grader = self.model or settings.judge_model
        cap = max_tokens("JUDGE", settings)
        request = Request.ask(
            "judge", JUDGE_SYSTEM, prompt, cap, model=grader, thinking=THINKING_EVAL_PARAM
        )
        response = llm.parse(request, Verdict)
        verdict = response.parsed
        if verdict is None:
            return Score(
                False,
                f"judge returned no verdict (stop_reason={response.stop_reason})",
                usage=response.usage,
                model=grader,
            )
        return Score(verdict.passed, verdict.reasoning, usage=response.usage, model=grader)

    def prompt_for(self, case: Case, outcome: Outcome) -> str:
        """The judged text, XML-delimited so the parts cannot bleed into each other.

        Public so a caller can count its tokens before the run, which is free, rather than
        guessing at the size of the half of a paid eval that is not the task.
        """
        answer = outcome.raw if outcome.raw is not None else _compact(outcome.output)
        blocks = [
            f"<rubric>\n{self.rubric.strip()}\n</rubric>",
            f"<input>\n{_compact(case.input)}\n</input>",
        ]
        # One named field where the label carries more than the judge should see: a retrieval case
        # also holds the chunk ids it resolved to, which are noise in a correctness judgement.
        expected = (
            case.expected
            if self.reference is None
            else {self.reference: case.expected.get(self.reference)}
        )
        if case.expected:
            blocks.append(f"<reference>\n{_compact(expected)}\n</reference>")
        blocks.append(f"<answer>\n{answer}\n</answer>")
        return "\n\n".join(blocks)


def _compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)

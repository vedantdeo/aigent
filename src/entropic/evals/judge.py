"""LLM-as-judge: the one grader that spends.

Runs on `config.JUDGE_MODEL`, deliberately not the model under test, and is pre-flighted through
`pricing.check_request` like any other paid path. Its usage goes back in the `Score` so the runner
bills it to the eval's budget.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import anthropic
from pydantic import BaseModel, Field

from entropic.config import JUDGE_MODEL, get_client
from entropic.config import MAX_TOKENS_JUDGE as MAX_TOKENS
from entropic.evals.dataset import Case
from entropic.evals.grade import Outcome, Score
from entropic.pricing import check_request

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
    model: str = JUDGE_MODEL
    client: anthropic.Anthropic | None = None

    def __call__(self, case: Case, outcome: Outcome) -> Score:
        if outcome.error is not None:
            return Score(False, f"task failed: {outcome.error}")
        if self.client is None:
            self.client = get_client()

        messages: list[anthropic.types.MessageParam] = [
            {"role": "user", "content": self._prompt(case, outcome)}
        ]
        check_request(
            self.client,
            model=self.model,
            max_tokens=MAX_TOKENS,
            messages=messages,
            system=JUDGE_SYSTEM,
        )
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=JUDGE_SYSTEM,
            messages=messages,
            output_format=Verdict,
        )
        verdict = response.parsed_output
        if verdict is None:
            return Score(
                False,
                f"judge returned no verdict (stop_reason={response.stop_reason})",
                usage=response.usage,
                model=self.model,
            )
        return Score(verdict.passed, verdict.reasoning, usage=response.usage, model=self.model)

    def _prompt(self, case: Case, outcome: Outcome) -> str:
        """XML-delimited, because the parts must not bleed into each other."""
        answer = outcome.raw if outcome.raw is not None else _compact(outcome.output)
        blocks = [
            f"<rubric>\n{self.rubric.strip()}\n</rubric>",
            f"<input>\n{_compact(case.input)}\n</input>",
        ]
        if case.expected:
            blocks.append(f"<reference>\n{_compact(case.expected)}\n</reference>")
        blocks.append(f"<answer>\n{answer}\n</answer>")
        return "\n\n".join(blocks)


def _compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)

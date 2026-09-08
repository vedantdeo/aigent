"""Week 1, step 3: structured output.

    uv run python -m entropic.week01.structured_output

What to notice:
  - `messages.parse` takes a Pydantic model and returns a validated instance. No regex, no
    json.loads on a string that might have prose around it.
  - This is the primitive under every "extract to JSON" product. Week 2 wraps it in an eval.
"""

from __future__ import annotations

from anthropic.types import MessageParam
from pydantic import BaseModel, Field

from entropic.config import MODEL, describe_usage, get_client

ABSTRACT = """
Modern Machine Learning (ML) and Artificial Intelligence (AI) models, especially large language
models (LLMs), are increasingly used to generate scientific hypotheses and mechanistic explanations
from observational data. This position paper argues that in the high-dimensional proxy regimes where
modern ML excels, mechanistic learning is generically underdetermined: many incompatible mechanisms
induce essentially the same observational relationships on the support of the data, so predictive
success and coherent explanations are insufficient evidence of mechanism discovery. This
underdetermination becomes uniquely hazardous with large language models (LLMs), which tend to
collapse large equivalence classes of explanations into a single fluent narrative. This paper
proposes concrete standards for "mechanistic ML," and argues these norms are necessary if
LLM-centered workflows are to support science rather than merely simulate it.
"""


class PaperSummary(BaseModel):
    """What we want back. Field descriptions are visible to the model; write them like prompts."""

    one_line_claim: str = Field(description="The paper's central claim in one sentence.")
    paper_type: str = Field(
        description="One of: empirical, theoretical, position, survey, systems."
    )
    key_terms: list[str] = Field(
        description="Three to six technical terms central to the argument."
    )
    targets_llms: bool = Field(description="True if LLMs are a primary subject of the paper.")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Your confidence in this summary, 0 to 1."
    )


def main() -> None:
    client = get_client()
    messages: list[MessageParam] = [
        {"role": "user", "content": f"Summarize this abstract.\n\n<abstract>{ABSTRACT}</abstract>"}
    ]

    response = client.messages.parse(
        model=MODEL,
        max_tokens=2048,
        messages=messages,
        output_format=PaperSummary,
    )

    summary = response.parsed_output
    if summary is None:
        print(f"no parsed output; stop_reason={response.stop_reason}")
        return

    print(summary.model_dump_json(indent=2))
    print()
    print(describe_usage(MODEL, response.usage))


if __name__ == "__main__":
    main()

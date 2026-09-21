"""`messages.parse` with a Pydantic schema: schema in, validated object out.

A refusal or a `max_tokens` stop leaves `parsed_output` as None, which is the case worth handling.
"""

from __future__ import annotations

from anthropic.types import MessageParam
from pydantic import BaseModel, Field

from entropic.config import MAX_TOKENS_EXTRACT as MAX_TOKENS
from entropic.config import MODEL, get_client
from entropic.pricing import check_request, describe_usage

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
    check_request(
        client, model=MODEL, max_tokens=MAX_TOKENS, messages=messages, output_format=PaperSummary
    )

    response = client.messages.parse(
        model=MODEL,
        max_tokens=MAX_TOKENS,
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

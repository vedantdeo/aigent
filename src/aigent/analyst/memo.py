"""What the capstone's graph produces and judges: a memo of cited claims, the critic's verdict on
each, and a person's decision on the whole."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """One checkable statement and what it rests on."""

    id: str = Field(description="c1, c2, … in the order the claims appear.")
    text: str = Field(description="One sentence stating one fact, with its figure and year.")
    citations: list[str] = Field(
        min_length=1,
        description=(
            "Every source the claim rests on, by id: a passage id from search_filings "
            "(ITC-FY25#0412) or a fundamentals cite_as "
            "(FACT:ITC:2025:consolidated:year:RevenueFromOperations). At least one."
        ),
    )


class Memo(BaseModel):
    """The analyst's answer: what a reader needs first, the claims behind it, and its limits."""

    summary: str = Field(description="Two sentences answering the question.")
    claims: list[Claim] = Field(description="Every fact the summary relies on, one per claim.")
    caveats: list[str] = Field(description="What the sources did not cover or could not settle.")


class Judged(BaseModel):
    """The critic's verdict on one claim."""

    verdict: Literal["supported", "unsupported", "wrong"] = Field(
        description=(
            "supported: the cited sources state it. unsupported: they do not state it. wrong: "
            "they contradict it — another figure, year or company."
        )
    )
    reason: str = Field(description="One sentence, quoting the source where it decides.")


class Review(BaseModel):
    """A person's decision on a memo; a rejection's note becomes a constraint on the redraft."""

    approved: bool
    note: str | None = None

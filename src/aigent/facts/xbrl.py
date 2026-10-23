"""Reads an NSE results filing in XBRL (the 2020 BSE taxonomy or SEBI's 2025 integrated filing) into
numeric facts, each with the period it covers and its segment, if any."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

from aigent.facts.cfg_facts import XBRL_SKIP_CONTEXTS

PeriodKind = Literal["duration", "instant"]

_COLUMN = re.compile(r"^(One|Four)")
_NAMES = ("Description", "NameOf")  # text facts, though some filings give them a unit
# The 2020 taxonomy files these as a duration twice: the opening balance, then the closing one.
_OPENING_CLOSING = frozenset({"CashAndCashEquivalentsCashFlowStatement"})
_WORDS = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


@dataclass(frozen=True)
class Fact:
    """One numeric fact, its value in the filing's base unit (rupees, rupees per share, pure)."""

    concept: str
    period_kind: PeriodKind
    period_start: date | None
    period_end: date
    segment: str | None
    value: float
    unit: str
    decimals: int | None  # None for INF

    @property
    def key(self) -> tuple[str, str, date | None, date, str | None, str]:
        return (
            self.concept,
            self.period_kind,
            self.period_start,
            self.period_end,
            self.segment,
            self.unit,
        )


@dataclass
class Parsed:
    """A filing's facts, the facts skipped as out of scope, and what could not be read."""

    facts: list[Fact] = field(default_factory=list[Fact])
    skipped: int = 0
    issues: list[str] = field(default_factory=list[str])


@dataclass(frozen=True)
class _Context:
    start: date | None
    end: date  # the instant, for an instant
    members: tuple[str, ...]


def label_of(concept: str) -> str:
    """An element name in words: `RevenueFromOperations` is "Revenue from operations"."""
    words = _WORDS.findall(concept)
    if not words:
        return concept
    rest = [w if w.isupper() and len(w) > 1 else w.lower() for w in words[1:]]
    return " ".join([words[0], *rest])


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _contexts(root: ET.Element) -> dict[str, _Context]:
    found: dict[str, _Context] = {}
    for element in root:
        if _local(element.tag) != "context":
            continue
        period = element.find("{*}period")
        if period is None:
            continue
        dates = {_local(p.tag): date.fromisoformat((p.text or "").strip()) for p in period}
        members = tuple(
            (m.text or "").strip().rsplit(":", 1)[-1]  # a QName: prefix:Member
            for m in element.iter()
            if _local(m.tag) in ("explicitMember", "typedMember")
        )
        if "instant" in dates:
            found[element.get("id", "")] = _Context(None, dates["instant"], members)
        elif "startDate" in dates and "endDate" in dates:
            found[element.get("id", "")] = _Context(dates["startDate"], dates["endDate"], members)
    return found


def _texts(root: ET.Element) -> dict[str, dict[str, str]]:
    """Every non-numeric fact's text, by context and concept."""
    texts: dict[str, dict[str, str]] = {}
    for element in root:
        ctx = element.get("contextRef")
        if ctx and (not element.get("unitRef") or _local(element.tag).startswith(_NAMES)):
            texts.setdefault(ctx, {})[_local(element.tag)] = (element.text or "").strip()
    return texts


def _stated_periods(texts: dict[str, dict[str, str]]) -> dict[str, tuple[date, date]]:
    """Each column's period as the filing states it in words, which the context dates may not."""
    stated: dict[str, tuple[date, date]] = {}
    for column in ("One", "Four"):
        said = texts.get(f"{column}D", {})
        start = said.get("DateOfStartOfReportingPeriod")
        end = said.get("DateOfEndOfReportingPeriod")
        if start and end:
            stated[column] = (date.fromisoformat(start), date.fromisoformat(end))
    return stated


def _segment(ctx_id: str, ctx: _Context, texts: dict[str, dict[str, str]]) -> str | None:
    """A dimensional context's own description (a segment's or an expense's name), else its
    member; a context with no member has none, whatever it describes (its currency, say)."""
    if not ctx.members:
        return None
    for concept, text in sorted(texts.get(ctx_id, {}).items()):
        if concept.startswith("Description") and text:
            return text
    return ctx.members[0] if ctx.members else None


def parse(data: bytes) -> Parsed:
    """The filing's numeric facts; a fact whose period cannot be settled is an issue, not a fact."""
    root = ET.fromstring(data)
    contexts = _contexts(root)
    texts = _texts(root)
    stated = _stated_periods(texts)
    out = Parsed()
    if not stated:
        out.issues.append("the filing states no reporting period; its durations are not loaded")
    filed = [
        (contexts[f"{c}D"].start, contexts[f"{c}D"].end) == stated[c]
        for c in stated
        if f"{c}D" in contexts
    ]
    dates_hold = bool(filed) and all(filed)
    seen: dict[tuple[str, str, date | None, date, str | None, str], Fact] = {}
    balances: set[tuple[str, str]] = set()
    for element in root:
        ctx_id, unit = element.get("contextRef"), element.get("unitRef")
        if not ctx_id or not unit:
            continue
        if any(marker in ctx_id for marker in XBRL_SKIP_CONTEXTS):
            out.skipped += 1
            continue
        text = (element.text or "").strip()
        if not text:  # nil included
            continue
        concept = _local(element.tag)
        if concept.startswith(_NAMES):
            continue
        ctx = contexts.get(ctx_id)
        if ctx is None:
            out.issues.append(f"{concept}: no context {ctx_id!r}")
            continue
        try:
            value = float(text)
        except ValueError:
            out.issues.append(f"{concept} in {ctx_id}: not a number {text!r}")
            continue
        if ctx.start is None:
            kind: PeriodKind = "instant"
            start, end = None, ctx.end
        elif concept in _OPENING_CLOSING:
            column = _COLUMN.match(ctx_id)
            span = stated.get(column.group(1)) if column else None
            if span is None:
                out.issues.append(f"{concept} in {ctx_id}: a balance with no stated period")
                continue
            kind, start = "instant", None
            closing = (concept, ctx_id) in balances
            balances.add((concept, ctx_id))
            end = span[1] if closing else span[0] - timedelta(days=1)
        else:
            kind = "duration"
            column = _COLUMN.match(ctx_id)
            if column and column.group(1) in stated:
                start, end = stated[column.group(1)]
                if dates_hold and (ctx.start, ctx.end) != (start, end):
                    out.issues.append(
                        f"{concept} in {ctx_id}: dated {ctx.start} to {ctx.end}, stated "
                        f"{start} to {end}; the stated period is used"
                    )
            elif dates_hold:
                start, end = ctx.start, ctx.end
            else:
                out.issues.append(f"{concept} in {ctx_id}: no column, and the dates do not hold")
                continue
        decimals = element.get("decimals")
        fact = Fact(
            concept=concept,
            period_kind=kind,
            period_start=start,
            period_end=end,
            segment=_segment(ctx_id, ctx, texts),
            value=value,
            unit=unit,
            decimals=None if decimals in (None, "INF") else int(decimals),
        )
        prior = seen.get(fact.key)
        if prior is None:
            seen[fact.key] = fact
            out.facts.append(fact)
        elif prior.value != fact.value:
            out.issues.append(
                f"{concept} {start} to {end}: filed twice, {prior.value} and {fact.value}; "
                "the first is kept"
            )
    return out

"""A fact's labels against a printed row's: the share of the label's words the row prints
(`label_match`), refused outright by the named rules in `VETOES` when the row names something
else; whether a row prints a per-share fact (`per_share_row`); and the first cells that are no
label (a row's serial number, a note reference)."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from aigent.facts.cfg_facts import (
    CROSSCHECK_EXCLUSIVE,
    CROSSCHECK_FACT_ACTIONS,
    CROSSCHECK_ROW_TURNS,
    CROSSCHECK_SYNONYMS,
)
from aigent.facts.vocab import STOP, one_off, repair, tokens, words

_FORMULA = re.compile(r"^(.*?[A-Za-z]{3}.*?)\s*[\[{].*$", re.S)  # words, then a bracketed formula
_GENERIC = 3  # a label this short ("Income", "Current assets") takes only a row led by its words
_LEADS = {"total", "net", "less", "add", "basic", "diluted", "principal"}  # passed over for a head


def serial(column: int, text: str) -> bool:
    """A row's serial number in its first cell ("1", "12"), not a figure."""
    return column == 0 and re.fullmatch(r"\d{1,2}", text.strip()) is not None


def note(column: int, text: str, first_dated: int | None) -> bool:
    """A note reference ("14", "4.1") in a column left of every dated one, not a figure."""
    return (
        first_dated is not None
        and column < first_dated
        and re.fullmatch(r"\d{1,2}(?:\.\d)?", text.strip()) is not None
    )


def head(text: str) -> str | None:
    """A label's first word of substance: enumerators (`(g)`, `Ill.`), short words and leads
    (`_LEADS`: `total`, `net`, `basic`) passed over; None when it has none."""
    for word in tokens(text):
        first = CROSSCHECK_SYNONYMS.get(word, word).split()[0]
        if len(first) > 2 and not re.fullmatch(r"[ivxl]+", first) and first not in _LEADS | STOP:
            return first
    return None


def discontinued_alone(found: set[str]) -> bool:
    """Words about discontinued operations without continuing ones."""
    return "discontinued" in found and "continuing" not in found


@dataclass(frozen=True)
class _Pair:
    """One of a fact's labels against a row: the label's words (its segment's added), the row's
    words, the label as filed and the row as printed, its bracketed formula cut."""

    label: str
    mine: set[str]
    row_text: str
    row: set[str]
    segment: str | None


def _turned(word: str | None, other: set[str], turning: frozenset[str] | None) -> bool:
    """Whether a head word turns the meaning (`turning`, or any word when None) and the other
    side lacks it."""
    if word is None or (turning is not None and word not in turning):
        return False
    return not any(one_off(word, w) for w in other)


def _row_turns(p: _Pair) -> bool:
    """The row is led by a word in `CROSSCHECK_ROW_TURNS` the label lacks ("Profit before
    exceptional items" for "Exceptional items")."""
    return _turned(head(p.row_text), p.mine, CROSSCHECK_ROW_TURNS)


def _short_label(p: _Pair) -> bool:
    """A label of `_GENERIC` words or fewer against a row led by a word it lacks ("Income"
    against "Interest income", "Assets" against "Other financial assets")."""
    return len(p.mine) <= _GENERIC and _turned(head(p.row_text), p.mine, None)


def _flow_unprinted(p: _Pair) -> bool:
    """A fact with no segment whose label leads with a flow (`CROSSCHECK_FACT_ACTIONS`) the row
    prints no word for ("Purchase of property" against "Property"; "Proceeds" may print "Sale")."""
    lead = head(p.label)
    if p.segment is not None or lead not in CROSSCHECK_FACT_ACTIONS:
        return False
    return not any(one_off(w, t) for w in p.row for t in CROSSCHECK_FACT_ACTIONS[lead])


def _opposites(p: _Pair) -> bool:
    """The row names only the other of an exclusive pair the label names (`CROSSCHECK_EXCLUSIVE`:
    basic or diluted, current or non-current, gross or net)."""
    return any(
        mine in p.mine and other in p.row and mine not in p.row
        for pair in CROSSCHECK_EXCLUSIVE
        for mine, other in (pair, pair[::-1])
    )


def _discontinued(p: _Pair) -> bool:
    """Discontinued operations alone on one side only."""
    return discontinued_alone(p.mine) != discontinued_alone(p.row)


def _negated(p: _Pair) -> bool:
    """ "Will not be" on one side only ("Items that will not be reclassified")."""
    return ("notbe" in p.mine) != ("notbe" in p.row)


VETOES: tuple[tuple[str, Callable[[_Pair], bool]], ...] = (
    ("row turns", _row_turns),
    ("short label", _short_label),
    ("flow unprinted", _flow_unprinted),
    ("discontinued", _discontinued),
    ("will not be", _negated),
    ("opposites", _opposites),
)  # the first to refuse names the veto


@dataclass(frozen=True)
class LabelMatch:
    """A row's label share for a fact: the best of its labels' shares no veto refused, the best
    share before vetoes, and the veto that refused the label scoring that, if any."""

    share: float
    raw: float
    veto: str | None


def label_match(labels: Iterable[str], segment: str | None, row_label: str) -> LabelMatch:
    """Each of a fact's labels' words, its segment's added, as a share found in the row, a
    formula in brackets after the row's words (`[Profit before tax / ...]`) left out; a label
    a rule in `VETOES` refuses scores 0, the first refusing one named."""
    row_text = _FORMULA.sub(r"\1", row_label)
    row = words(row_text)
    share, raw, veto = 0.0, 0.0, None
    for label in labels:
        mine = words(f"{label} {segment or ''}")
        if not mine:
            continue
        found = len(mine & row) / len(mine)
        pair = _Pair(label, mine, row_text, row, segment)
        refused = next((name for name, rule in VETOES if rule(pair)), None)
        if refused is None:
            share = max(share, found)
        if found > raw or (found == raw and refused is None):  # an unrefused label wins a tie
            raw, veto = found, refused
    return LabelMatch(share, raw, veto)


def label_figure(phrase: str, text: str) -> tuple[float, str] | None:
    """The figure a label prints inside a phrase (`CROSSCHECK_LABEL_FIGURES`), read through OCR's
    slips and rupee glyphs (`vocab.repair`), and the phrase as printed; None when it prints none."""
    m = re.search(phrase, repair(text), re.IGNORECASE)
    if m is None:
        return None
    return float(next(g for g in m.groups() if g is not None)), m.group(0)


def label_share(labels: Iterable[str], segment: str | None, row_label: str) -> float:
    """`label_match`'s share alone."""
    return label_match(labels, segment, row_label).share


def per_share_row(labels: Iterable[str], text: str) -> bool:
    """A row, with its heading, printing a per-share fact: its kind (basic or diluted), earnings per
    share, and discontinued operations on both or neither, its words read through OCR's slips
    (`vocab.repair`)."""
    row = words(text)
    named = set().union(*(words(label) for label in labels))
    if not {"basic", "diluted"} & named & row:
        return False
    if discontinued_alone(row) != discontinued_alone(named):
        return False
    return bool({"earnings", "eps"} & row) or {"per", "share"} <= row

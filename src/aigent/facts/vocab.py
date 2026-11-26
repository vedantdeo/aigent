"""The words of a printed label or header, read the one way every reader of a table uses: OCR's
slips in the words the rules key on repaired (`repair`), lower case, "non-current" one word and
"not be" the word `notbe`, abbreviations in full, little words dropped."""

from __future__ import annotations

import re
from functools import cache

from aigent.facts.cfg_facts import (
    CROSSCHECK_SYNONYMS,
    VOCAB_LEXICON,
    VOCAB_RUPEE_GLYPHS,
    VOCAB_RUPEE_WORDS,
)

STOP = {"of", "and", "the", "in", "on", "or", "to", "for", "from", "by", "a", "an", "with", "at"}


def tokens(text: str) -> list[str]:
    """A label's words in order, lower case; "non-current" one word, "not be" `notbe`."""
    t = re.sub(r"\bnon[\s-]*current", "noncurrent", repair(text).lower())
    return re.findall(r"[a-z]+", re.sub(r"\bnot\s+[bh]e\b", "notbe", t))


def words(text: str) -> set[str]:
    """A label's words (`tokens`), abbreviations read in full, little words dropped."""
    spelled = " ".join(CROSSCHECK_SYNONYMS.get(word, word) for word in tokens(text))
    return set(spelled.split()) - STOP


def one_off(word: str, target: str) -> bool:
    """`word` is `target` or one letter off it: changed, missing or added."""
    if word == target:
        return True
    if len(word) == len(target):
        return sum(a != b for a, b in zip(word, target, strict=True)) == 1
    short, long = sorted((word, target), key=len)
    return any(long[:i] + long[i + 1 :] == short for i in range(len(long)))


def edits(a: str, b: str) -> int:
    """The letters to insert, delete or change to turn `a` into `b`."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def _near(word: str) -> int:
    """How many letters a word of this length may be off a lexicon word: none under 4 letters,
    one up to 7, two from 8."""
    return 0 if len(word) < 4 else 1 if len(word) < 8 else 2


@cache
def snap(word: str) -> str:
    """A lower-case word read as the lexicon word (`VOCAB_LEXICON`) nearest it within `_near`
    letters; itself when none is that near, or two are equally near."""
    if word in VOCAB_LEXICON or not (allowed := _near(word)):
        return word
    found = sorted((edits(word, w), w) for w in VOCAB_LEXICON if abs(len(w) - len(word)) <= allowed)
    near = [w for n, w in found if n <= allowed and n == found[0][0]]
    return near[0] if len(near) == 1 else word


def repair(text: str) -> str:
    """A text with each word OCR misread from a lexicon word read as it (`snap`), lower case
    where repaired, and the rupee sign read where one must stand (`rupee`), everything else as
    printed: `CONSOUDATED` reads `consolidated`, `(f in crore)` reads `(₹ in crore)`."""

    def fixed(m: re.Match[str]) -> str:
        word = m.group(0)
        found = snap(word.lower())
        return word if found == word.lower() else found

    return rupee(re.sub(r"[A-Za-z]+", fixed, text))


_GLYPH = "|".join(re.escape(g) for g in sorted(VOCAB_RUPEE_GLYPHS, key=len, reverse=True))
_WORD = "|".join(re.escape(w) for w in sorted(VOCAB_RUPEE_WORDS, key=len, reverse=True))
_SCALE_WORD = r"in\s*(?:crore|lakh|lac|million)s?\b"
_RUPEE: tuple[tuple[re.Pattern[str], str], ...] = (
    # a scale bracket: (f in crore), (fin Million), (Rs. in lakh)
    (re.compile(rf"([(\[{{|]\s*)(?:{_GLYPH}|{_WORD})\s*(?={_SCALE_WORD})", re.I), r"\1₹ "),
    # an amount: of t 1/- each, oft 1/-, par value ~2 per share, Rs. 10 each
    (
        re.compile(
            rf"\b(of|value|@|price|at)\s*(?:{_GLYPH}|{_WORD})\s?(?=\d[\d,]*(?:\.\d+)?\s*(?:/|each|per\b|crore|lakh|million))",
            re.I,
        ),
        r"\1 ₹",
    ),
    (re.compile(rf"(?<![A-Za-z])(?:{_WORD})(?=\s*\d)"), "₹"),
)


def rupee(text: str) -> str:
    """The text with the rupee sign read where one must stand: OCR's glyphs (`VOCAB_RUPEE_GLYPHS`)
    and `Rs.` (`VOCAB_RUPEE_WORDS`) in a scale bracket before "in crore" and the like, or between
    "of", "value", "@", "price" or "at" and an amount followed by "/-", "each", "per" or a scale;
    `Rs.` before any figure."""
    for pattern, sign in _RUPEE:
        text = pattern.sub(sign, text)
    return text

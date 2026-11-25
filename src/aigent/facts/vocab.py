"""The words of a printed label or header, read the one way every reader of a table uses: OCR's
slips in the words the rules key on repaired (`repair`), lower case, "non-current" one word and
"not be" the word `notbe`, abbreviations in full, little words dropped."""

from __future__ import annotations

import re
from functools import cache

from aigent.facts.cfg_facts import CROSSCHECK_SYNONYMS, VOCAB_LEXICON

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
    where repaired, everything else as printed: `CONSOUDATED` reads `consolidated`."""

    def fixed(m: re.Match[str]) -> str:
        word = m.group(0)
        found = snap(word.lower())
        return word if found == word.lower() else found

    return re.sub(r"[A-Za-z]+", fixed, text)

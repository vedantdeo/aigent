"""The words of a printed label or header, read the one way every reader of a table uses: lower
case, "non-current" one word and "not be" the word `notbe`, abbreviations in full, little words
dropped; and the letter distances OCR'd words are forgiven by."""

from __future__ import annotations

import re

from aigent.facts.cfg_facts import CROSSCHECK_SYNONYMS

STOP = {"of", "and", "the", "in", "on", "or", "to", "for", "from", "by", "a", "an", "with", "at"}


def tokens(text: str) -> list[str]:
    """A label's words in order, lower case; "non-current" one word, "not be" `notbe`."""
    t = re.sub(r"\bnon[\s-]*current", "noncurrent", text.lower())
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

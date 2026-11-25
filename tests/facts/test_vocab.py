"""`facts.vocab`: label words read one way, and OCR's slips in the rule words repaired."""

from __future__ import annotations

import pytest

from aigent.facts.cfg_facts import VOCAB_LEXICON
from aigent.facts.vocab import edits, one_off, repair, snap, tokens, words


@pytest.mark.parametrize(
    ("word", "read"),
    [
        pytest.param("consoudated", "consolidated", id="two letters off a long word"),
        pytest.param("sectember", "september", id="a month OCR'd"),
        pytest.param("asic", "basic", id="one letter off a short word"),
        pytest.param("paymert", "payment", id="the nearer of two lexicon words"),
        pytest.param("consolidation", "consolidation", id="three letters off stays"),
        pytest.param("sil", "sil", id="under four letters stays"),
        pytest.param("sure", "sure", id="an ordinary word two letters off stays"),
        pytest.param("decreased", "decrease", id="the one lexicon word near enough"),
        pytest.param("audited", "audited", id="a lexicon word is itself"),
    ],
)
def test_snap(word: str, read: str) -> None:
    assert snap(word) == read


def test_a_word_near_two_lexicon_words_stays() -> None:
    assert snap("payments") == "payments"
    assert snap("paymenta") == "paymenta"  # one off payment and one off payments


def test_repair_keeps_everything_but_the_words_it_reads() -> None:
    assert repair("STATEMENT OF CONSOUDATED RESULTS, 31 .12.2022") == (
        "STATEMENT OF consolidated RESULTS, 31 .12.2022"
    )


def test_the_lexicon_is_sorted_and_lower_case() -> None:
    assert list(VOCAB_LEXICON) == sorted(set(VOCAB_LEXICON))
    assert all(w == w.lower() and w.isalpha() for w in VOCAB_LEXICON)


@pytest.mark.parametrize(
    ("text", "found"),
    [
        pytest.param("Non-current borrowings", ["noncurrent", "borrowings"], id="non-current"),
        pytest.param("will not he reclassified", ["will", "notbe", "reclassified"], id="not be"),
        pytest.param("Sale Proceeds", ["sale", "proceeds"], id="lower case"),
    ],
)
def test_tokens(text: str, found: list[str]) -> None:
    assert tokens(text) == found


def test_words_drop_little_words_and_read_abbreviations() -> None:
    assert words("Basic EPS of the year") == {"basic", "earnings", "per", "share", "year"}


@pytest.mark.parametrize(
    ("a", "b", "n", "near"),
    [("basic", "basic", 0, True), ("asic", "basic", 1, True), ("bsaic", "basic", 2, False)],
)
def test_edits_and_one_off(a: str, b: str, n: int, near: bool) -> None:
    assert edits(a, b) == n
    assert one_off(a, b) is near

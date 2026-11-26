"""`facts.vocab`: label words read one way, and OCR's slips in the rule words repaired."""

from __future__ import annotations

import pytest

from aigent.facts.cfg_facts import VOCAB_LEXICON, VOCAB_RUPEE_GLYPHS, VOCAB_RUPEE_WORDS
from aigent.facts.headers import scale_in
from aigent.facts.vocab import edits, one_off, repair, rupee, snap, tokens, words


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


@pytest.mark.parametrize(
    ("text", "read"),
    [
        pytest.param("(f in Crores)", "(₹ in Crores)", id="a glyph in a scale bracket"),
        pytest.param("(fin Million)", "(₹ in Million)", id="a glyph glued to in"),
        pytest.param("(Rs. in lakh)", "(₹ in lakh)", id="Rs. in a scale bracket"),
        pytest.param("(in crore)", "(in crore)", id="no sign printed: none added"),
        pytest.param("(₹ in crore)", "(₹ in crore)", id="the sign as printed"),
        pytest.param("(Ordinary Shares oft 1/- each)", "(Ordinary Shares of ₹1/- each)", id="oft"),
        pytest.param("Shares of, 1/- each", "Shares of ₹1/- each", id="a comma for the sign"),
        pytest.param("Par value ~2 per share", "Par value ₹2 per share", id="par value"),
        pytest.param("(Face value of {10 each)", "(Face value of ₹10 each)", id="a brace"),
        pytest.param("face value Rs. 2 each", "face value ₹2 each", id="Rs. before an amount"),
        pytest.param("Rs.10 crore", "₹10 crore", id="Rs. before any figure"),
        pytest.param("of the 10 shares", "of the 10 shares", id="a word between: no sign"),
        pytest.param("upto ~ 4,000 Crore", "upto ~ 4,000 Crore", id="no context word: none"),
        pytest.param(
            "a price of ~10,000 crore", "a price of ₹10,000 crore", id="an amount in crore"
        ),
        pytest.param("out of t 10 items", "out of t 10 items", id="a glyph before no unit"),
        pytest.param("at 10 each", "at 10 each", id="no glyph: none added"),
        pytest.param("USD 5 each", "USD 5 each", id="another currency"),
        pytest.param(
            "Par value U per share", "Par value U per share", id="a glyph not in the vocab"
        ),
        pytest.param("shift of t he plant", "shift of t he plant", id="a letter t with no amount"),
        pytest.param("Reserves 10 each", "Reserves 10 each", id="Re inside a word"),
    ],
)
def test_rupee(text: str, read: str) -> None:
    assert rupee(text) == read


def test_repair_reads_the_rupee_sign_too() -> None:
    assert repair("(f in CONSOUDATED crore)") == "(f in consolidated crore)"
    assert repair("Paid-up capital (Face value oft 10 each)") == (
        "Paid-up capital (Face value of ₹10 each)"
    )


@pytest.mark.parametrize(
    ("text", "scale"),
    [
        ("(fin Million)", 1e6),
        ("(t in Crores)", 1e7),
        ("(Rs. in lakh)", 1e5),
        ("in the crore", None),
    ],
)
def test_a_scale_is_read_through_ocr_glyphs(text: str, scale: float | None) -> None:
    assert scale_in(text) == scale


def test_the_rupee_vocab_is_sorted() -> None:
    assert list(VOCAB_RUPEE_GLYPHS) == sorted(set(VOCAB_RUPEE_GLYPHS))
    assert list(VOCAB_RUPEE_WORDS) == sorted(set(VOCAB_RUPEE_WORDS))

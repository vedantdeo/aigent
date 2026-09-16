"""The four chunking strategies, as tables over the text they are asked to split.

The abbreviation table is the load-bearing one: `Rs.` opens every other sentence in an annual
report, and a splitter that breaks on it shears the figure off its unit.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from entropic.config import CHUNK_MIN_CHARS
from entropic.retrieval.chunk import (
    Document,
    Inventory,
    Splitter,
    by_heading,
    by_sentence,
    fixed,
    headings,
    sentences,
)

FILLER = "The company reported steady progress across every operating segment this year. "


def _doc(text: str, doc_id: str = "RIL-FY25", page_starts: tuple[int, ...] = ()) -> Document:
    return Document(doc_id=doc_id, text=text, page_starts=page_starts)


@pytest.mark.parametrize(
    ("text", "count"),
    [
        pytest.param("Revenue rose. Profit fell.", 2, id="a plain full stop ends a sentence"),
        pytest.param("Revenue was Rs. 400 crore. Profit fell.", 2, id="Rs. does not"),
        pytest.param("Reliance Retail Ltd. filed today. It grew.", 2, id="nor does Ltd."),
        pytest.param("J. K. Sharma chairs it. He said so.", 2, id="nor do single initials"),
        pytest.param("Revenue grew 7.1 per cent. Profit fell.", 2, id="nor does a decimal point"),
        pytest.param("Margins improved, viz. in retail. It grew.", 2, id="nor does viz."),
        pytest.param("Did revenue grow? It did. Yes!", 3, id="question and exclamation marks do"),
    ],
)
def test_sentence_splitting_survives_the_abbreviations_an_annual_report_is_made_of(
    text: str, count: int
) -> None:
    found = sentences(text)

    assert len(found) == count, [text[a:b] for a, b in found]
    assert "".join(text[a:b] for a, b in found) == text, "every character belongs to some sentence"


@pytest.mark.parametrize(
    ("line", "is_heading"),
    [
        pytest.param("Risk Management", True, id="title case, no terminal stop"),
        pytest.param("3.2 Credit risk", True, id="numbered, even in sentence case"),
        pytest.param("MANAGEMENT DISCUSSION AND ANALYSIS", True, id="all caps"),
        pytest.param("Revenue for the year rose by 7 per cent.", False, id="ends like a sentence"),
        pytest.param("the company faces commodity price risk", False, id="lower case, unnumbered"),
        pytest.param(
            "Directors Auditors and the Board of a Very Long Standing Committee Named Here",
            False,
            id="too many words to be a heading",
        ),
    ],
)
def test_what_counts_as_a_heading_in_text_that_lost_its_markup(line: str, is_heading: bool) -> None:
    found = headings(f"Preamble text.\n\n{line}\n\nBody text follows here.\n")

    assert (any(title == line for _, title in found)) is is_heading, found


def test_fixed_windows_step_by_size_minus_overlap_and_cut_mid_word() -> None:
    text = "".join(f"{i:02d}" for i in range(150))[:300]
    inventory = Inventory.build("fixed", [_doc(text)], fixed(100, 20))

    assert [chunk.start for chunk in inventory.chunks] == [0, 80, 160]
    assert inventory.chunks[0].text[-20:] == inventory.chunks[1].text[:20], "the overlap is shared"
    assert inventory.chunks[1].text.startswith("4041"), "no snapping to a boundary"


def test_sentence_chunks_never_end_mid_sentence_and_carry_the_overlap_back() -> None:
    text = FILLER * 6
    inventory = Inventory.build("sentence", [_doc(text)], by_sentence(200, overlap_sentences=1))

    assert len(inventory) > 1
    for chunk in inventory.chunks:
        assert chunk.text.endswith("."), chunk.text
    first, second = inventory.chunks[0], inventory.chunks[1]
    assert second.start < first.start + len(first.text), "the second chunk starts inside the first"


def test_heading_chunks_keep_their_heading_even_when_the_section_is_split() -> None:
    text = "Risk Management\n\n" + FILLER * 8
    inventory = Inventory.build("heading", [_doc(text)], by_heading(300))

    assert len(inventory) > 1, "an oversized section is split, not returned whole"
    assert {chunk.heading for chunk in inventory.chunks} == {"Risk Management"}


def test_heading_chunks_fall_back_to_sentences_when_extraction_left_no_headings() -> None:
    text = FILLER * 8
    inventory = Inventory.build("heading", [_doc(text)], by_heading(300))

    assert len(inventory) > 1
    assert all(chunk.heading is None for chunk in inventory.chunks)


def test_ordinals_stay_contiguous_when_a_window_is_dropped_for_being_too_small() -> None:
    """A fixed window can end a few characters into a document; that stub is not a chunk.

    Against `fixed` rather than `by_sentence`, because this is where the drop is real.
    """
    text = FILLER * 4
    inventory = Inventory.build("fixed", [_doc(text)], fixed(100, 20))

    assert len(inventory) == 3, "the fourth window is a 76-character stub and is dropped"
    assert all(len(chunk.text) >= CHUNK_MIN_CHARS for chunk in inventory.chunks)
    assert [chunk.ordinal for chunk in inventory.chunks] == list(range(len(inventory)))
    assert inventory.ids() == [f"RIL-FY25#{i:04d}" for i in range(len(inventory))]


def test_sentence_chunks_lose_no_text_when_a_short_sentence_precedes_an_over_long_one() -> None:
    """The shape that argued `by_sentence` out of having a minimum at all.

    A tiny sentence before an over-long one used to be flushed alone, dropped, and never revisited.
    """
    tiny = "Note. "
    huge = (
        "The board has resolved that the scheme of arrangement between the company and its wholly "
        "owned subsidiary shall proceed subject to approval by the shareholders and the tribunal. "
    )
    text = tiny + huge + FILLER
    inventory = Inventory.build("sentence", [_doc(text)], by_sentence(160, overlap_sentences=0))

    covered: set[int] = set()
    for chunk in inventory.chunks:
        covered |= set(range(chunk.start, chunk.start + len(chunk.text)))
    lost = [index for index in range(len(text)) if index not in covered and text[index].strip()]

    assert not lost, f"characters in no chunk: {''.join(text[i] for i in lost)!r}"
    assert inventory.containing("Note") != []


def test_a_chunk_knows_the_page_it_started_on_so_an_answer_can_cite_it() -> None:
    document = _doc(FILLER * 8, page_starts=(0, len(FILLER) * 4))
    inventory = Inventory.build("fixed", [document], fixed(len(FILLER) * 2, 0))

    assert [chunk.page for chunk in inventory.chunks] == [1, 1, 2, 2]
    assert inventory.chunks[2].citation == "RIL-FY25 · p.2"


@pytest.mark.parametrize(
    ("quote", "found"),
    [
        pytest.param("dividend of Rs. 10", True, id="a quote inside one chunk"),
        pytest.param("DIVIDEND   of rs. 10", True, id="case and whitespace do not matter"),
        pytest.param("a figure nobody wrote", False, id="a quote in no chunk at all"),
    ],
)
def test_a_labelled_quote_resolves_to_the_chunks_that_hold_it(quote: str, found: bool) -> None:
    text = FILLER + "The board recommended a dividend of Rs. 10 per share. " + FILLER
    inventory = Inventory.build("sentence", [_doc(text)], by_sentence(400, overlap_sentences=0))

    assert bool(inventory.containing(quote)) is found, inventory.ids()


def test_a_quote_split_across_a_boundary_resolves_to_nothing_which_is_the_finding() -> None:
    """The label is not broken here — the chunking is, and this is how that becomes visible."""
    text = FILLER * 2 + "The board recommended a dividend of Rs. 10 per share. " + FILLER * 2
    cut = text.index("dividend") + 4
    inventory = Inventory.build("fixed", [_doc(text)], fixed(cut, 0))

    assert inventory.containing("dividend of Rs. 10") == []


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: fixed(0), id="a window of no characters"),
        pytest.param(lambda: fixed(100, 100), id="overlap as wide as the window, which never ends"),
        pytest.param(lambda: by_sentence(0), id="a sentence budget of nothing"),
        pytest.param(lambda: by_heading(0), id="a section budget of nothing"),
    ],
)
def test_a_splitter_refuses_its_own_bad_arguments_when_it_is_built(
    build: Callable[[], Splitter],
) -> None:
    with pytest.raises(ValueError):
        build()


def test_two_documents_with_one_doc_id_are_refused_rather_than_silently_merged() -> None:
    """`by_id` would keep the last of each colliding pair, and every label naming one would move."""
    with pytest.raises(ValueError, match="share a doc_id"):
        Inventory.build("fixed", [_doc(FILLER * 3), _doc(FILLER * 3)], fixed(200, 0))

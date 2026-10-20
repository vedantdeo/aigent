"""The four chunking strategies, as tables over the text they are asked to split.

The abbreviation table is the load-bearing one: `Rs.` opens every other sentence in an annual
report, and a splitter that breaks on it shears the figure off its unit.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from underhood.tooldata.labels import Label, check_call

from aigent.config import CHUNK_MIN_CHARS
from aigent.retrieval.chunk import (
    Document,
    Inventory,
    Splitter,
    by_element,
    by_heading,
    by_sentence,
    fixed,
    headings,
    multiplier_of,
    sentences,
)
from aigent.retrieval.corpus import assembled, elements_of

from .conftest import Grid, Heading, Piece, docling

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


def test_front_matter_is_a_chunk_of_its_own_and_an_empty_section_is_dropped() -> None:
    front = "This annual report covers the financial year ending March 2025 for all segments. "
    body = "Revenue grew across every segment, led by consumer goods and the hotels business. "
    text = f"{front}\nDIRECTORS' REPORT\n{body}\nNOTES\nSee below.\n"
    inventory = Inventory.build("heading", [_doc(text)], by_heading(300))

    assert [(chunk.heading, chunk.text.split()[0]) for chunk in inventory.chunks] == [
        (None, "This"),
        ("DIRECTORS' REPORT", "DIRECTORS'"),
    ], "NOTES holds less than CHUNK_MIN_CHARS, so it is not a chunk"


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
        pytest.param("Company's board", True, id="a typed apostrophe matches the curly one"),
        pytest.param("per share - the highest", True, id="a typed hyphen matches an em dash"),
        pytest.param(
            "share - the highest yet.", True, id="kerning leaves a space before punctuation"
        ),
        pytest.param("a figure nobody wrote", False, id="a quote in no chunk at all"),
    ],
)
def test_a_labelled_quote_resolves_to_the_chunks_that_hold_it(quote: str, found: bool) -> None:
    text = (
        FILLER
        + "The Company\u2019s board recommended a dividend of Rs. 10 per share "
        + "\u2014 the highest yet . "
        + FILLER
    )
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


def test_a_span_with_no_sentence_break_is_cut_on_the_budget_not_emitted_whole() -> None:
    """The financial-table shape: 15,000 characters of figures with no `.!?` anywhere in them.

    Emitted whole, the embedder reads its first 512 tokens and silently ignores the rest, so an
    eighth of this corpus became unsearchable past its opening. Measured 2026-09-18.
    """
    table = " ".join(f"{row} 1,234 5,678 9,012" for row in range(400))
    assert "." not in table, "the fixture has to have no sentence break for this to test anything"

    inventory = Inventory.build("sentence", [_doc(table)], by_sentence(400, overlap_sentences=0))

    assert len(inventory) > 1, "a table with no sentence break must still be cut"
    assert max(len(chunk.text) for chunk in inventory.chunks) <= 400
    assert "".join(chunk.text for chunk in inventory.chunks).replace(" ", "") == table.replace(
        " ", ""
    ), "cutting on the budget must not lose or duplicate text"


def test_a_long_sentence_is_still_packed_whole_when_it_fits_the_budget() -> None:
    """The cap is for spans that cannot be split on sentences, not for merely long sentences."""
    sentence = "The board resolved that the scheme of arrangement shall proceed as proposed. "

    inventory = Inventory.build("sentence", [_doc(sentence * 3)], by_sentence(200, 0))

    assert all(chunk.text.endswith(".") for chunk in inventory.chunks), [
        chunk.text for chunk in inventory.chunks
    ]


# --- by_element --------------------------------------------------------------------------------

STATEMENT = Grid(
    rows=(
        ("(Rs in crore)", "", ""),
        ("Particulars", "2025", "2024"),
        ("Revenue from operations", "9,74,864", "9,01,064"),
        ("Other income", "17,978", "16,179"),
    ),
    spans=((0, 0, 3, "(Rs in crore)"),),
    header_rows=2,
)


def _assembled(pages: list[list[Piece]]) -> Document:
    return assembled("RELIANCE-FY25", elements_of(docling(pages)))


def test_a_chunk_starts_at_every_heading_and_every_table() -> None:
    document = _assembled(
        [
            ["Chairman's message on the year gone by."],
            [
                Heading("Consolidated Statement of Profit and Loss"),
                STATEMENT,
                "Notes follow the statement.",
            ],
            [Heading("Risk management"), "Risks are reviewed each quarter by the Board."],
        ]
    )

    chunks = by_element()(document)

    texts = [c.text for c in chunks]
    assert texts[0] == "Chairman's message on the year gone by."
    assert texts[1].startswith("[Table · RELIANCE-FY25 p.2 · Consolidated Statement of Profit")
    assert texts[2] == (
        "[RELIANCE-FY25 p.2 · Consolidated Statement of Profit and Loss"
        " · (Rs in crore) = ×10000000]\nNotes follow the statement."
    ), "text under its section's scale carries it too"
    assert texts[3] == "Risk management\nRisks are reviewed each quarter by the Board."
    assert [c.heading for c in chunks] == [
        None,
        "Consolidated Statement of Profit and Loss",
        "Consolidated Statement of Profit and Loss",
        "Risk management",
    ]
    assert not any(t == "Consolidated Statement of Profit and Loss" for t in texts), (
        "a heading alone is no chunk: its table carries it"
    )
    assert [c.page for c in chunks] == [1, 2, 2, 3]


@pytest.mark.parametrize(
    ("before", "table", "label"),
    [
        pytest.param(
            [Heading("Consolidated Balance Sheet")],
            STATEMENT,
            "[Table · RELIANCE-FY25 p.1 · Consolidated Balance Sheet · consolidated · (Rs in crore)"
            " = ×10000000 · columns: 2025 | 2024]",
            id="basis from the heading, scale from the table's top rows",
        ),
        pytest.param(
            [
                Heading("Notes to the Standalone Financial Statements"),
                "(C in millions, except per share data)",
            ],
            Grid(rows=(("Segment", "2025"), ("Americas", "261,270"))),
            "[Table · RELIANCE-FY25 p.1 · Notes to the Standalone Financial Statements · standalone"
            " · (C in millions, except per share data) = ×1000000 · columns: 2025]",
            id="the scale stated once in the section, above the table",
        ),
        pytest.param(
            ["(C in millions, except per share data)", Heading("Goodwill")],
            Grid(rows=(("Particulars", "2025"), ("Opening balance", "4,314"))),
            "[Table · RELIANCE-FY25 p.1 · Goodwill · (C in millions, except per share data)"
            " = ×1000000 · columns: 2025]",
            id="the scale atop its page, above the section's heading",
        ),
        pytest.param(
            [Heading("Segment reporting"), "(₹ crore)"],
            Grid(rows=(("Particulars", "2025"), ("Capital expenditure", "6,746.54"))),
            "[Table · RELIANCE-FY25 p.1 · Segment reporting · (₹ crore) = ×10000000"
            " · columns: 2025]",
            id="a scale written without 'in', the currency sign before it",
        ),
        pytest.param(
            [Heading("Gratuity"), "The Group contributed (₹ 1,110.49 crore) to the fund."],
            Grid(rows=(("Particulars", "2025"), ("Employees covered", "2,14,521"))),
            "[Table · RELIANCE-FY25 p.1 · Gratuity · columns: 2025]",
            id="an amount in brackets is not a scale",
        ),
        pytest.param(
            [Heading("Highlights")],
            Grid(rows=(("Stores", "19,340"), ("Footfall", "1,060"))),
            "[Table · RELIANCE-FY25 p.1 · Highlights]",
            id="no basis, no scale, no years: none is made up",
        ),
    ],
)
def test_a_table_s_label_carries_what_its_section_and_top_rows_state(
    before: list[Piece], table: Grid, label: str
) -> None:
    [chunk] = [
        c for c in by_element()(_assembled([[*before, table]])) if c.text.startswith("[Table")
    ]

    assert chunk.text.split("\n")[0] == label


@pytest.mark.parametrize(
    ("scale", "multiplier"),
    [
        ("(Rs in crore)", 10_000_000),
        ("(H In Crore)", 10_000_000),
        ("(₹ in lakh crore)", 1_000_000_000_000),
        ("(Rs in lakhs)", 100_000),
        ("(C in millions, except per share data)", 1_000_000),
        ("(USD in billions)", 1_000_000_000),
        ("(in thousands)", 1_000),
        ("(in units)", None),
    ],
)
def test_a_scale_phrase_names_what_it_multiplies_by_the_longer_name_first(
    scale: str, multiplier: int | None
) -> None:
    assert multiplier_of(scale) == multiplier


def test_a_table_takes_its_page_s_scale_even_when_read_after_it_but_not_another_page_s() -> None:
    """Docling's reading order can put a page's scale line after the table it governs."""
    table = Grid(rows=(("Particulars", "2025"), ("Fixed price", "369,460")))
    pages: list[list[Piece]] = [
        [Heading("Revenue by nature of contract"), table, "(C in millions, except per share data)"],
        [Heading("Revenue by geography"), table],
    ]

    first, second = [c for c in by_element()(_assembled(pages)) if c.text.startswith("[Table")]

    assert "(C in millions, except per share data) = ×1000000" in first.text.split("\n")[0]
    assert "millions" not in second.text.split("\n")[0], "the next page states no scale"


def test_a_scale_in_a_label_lets_a_scaled_figure_be_checked_against_its_chunk() -> None:
    """The point of the label: a number whose scale the table's heading states is now provable."""
    rows = tuple(("Revenue from operations", "9,74,864", "9,01,064") for _ in range(1))
    table = Grid(rows=(("Particulars", "2025", "2024"), *rows), header_rows=1)
    document = _assembled([[Heading("Statement of Profit and Loss"), "(Rs in crore)", table]])

    [chunk] = [c for c in by_element()(document) if c.text.startswith("[Table")]

    label = Label("revenue", 9_748_640_000_000.0, "INR", 2025, None, "974864 * 10**7")
    assert check_call(label, chunk.text), chunk.text


@pytest.mark.parametrize(
    ("pages", "opens"),
    [
        pytest.param(
            [["(C in millions)", Heading("Borrowings"), "Interest expense was 6,893 in the year."]],
            "[RELIANCE-FY25 p.1 · Borrowings · (C in millions) = ×1000000]\n",
            id="the page's scale above the section's heading",
        ),
        pytest.param(
            [["(C in millions)", Heading("Borrowings"), "Interest expense was Rs 6,893 crore."]],
            "Borrowings",
            id="a chunk that names its own scale needs none",
        ),
        pytest.param(
            [["(Rs in crore)"], [Heading("Borrowings"), "Interest expense was 6,893."]],
            "Borrowings",
            id="a scale on another page governs nothing here",
        ),
    ],
)
def test_text_under_a_scale_opens_with_it_and_other_text_does_not(
    pages: list[list[Piece]], opens: str
) -> None:
    chunks = by_element()(_assembled(pages))

    [chunk] = [c for c in chunks if "Interest expense" in c.text]
    assert chunk.text.startswith(opens), chunk.text


def test_a_long_table_splits_at_rows_and_every_piece_repeats_its_label_and_headers() -> None:
    body = tuple((f"Segment {n}", f"{n},000", f"{n},100") for n in range(1, 41))
    long = Grid(rows=(("Particulars", "2025", "2024"), *body), header_rows=1)
    document = _assembled([[Heading("Segment information"), "(Rs in crore)", long]])

    pieces = [c for c in by_element(max_chars=300)(document) if c.text.startswith("[Table")]

    assert len(pieces) > 1
    label = pieces[0].text.split("\n")[0]
    for piece in pieces:
        lines = piece.text.split("\n")
        assert lines[0] == label and lines[1] == "Particulars 2025 2024", piece.text
        assert len(piece.text) <= 300
    rows = [line for p in pieces for line in p.text.split("\n")[2:]]
    assert rows == [" ".join(row) for row in body], "every row once, none cut, in order"
    assert pieces[1].start == document.text.index(
        "Segment " + rows[len(pieces[0].text.split("\n")) - 2].split()[1]
    )


def test_a_document_with_no_parsed_structure_is_packed_by_sentence() -> None:
    document = Document("RELIANCE-FY25", "One sentence here. Another one there.", page_starts=(0,))

    assert by_element()(document) == by_sentence()(document)


@pytest.mark.parametrize(("table", "text"), [(0, 1200), (2000, 0)])
def test_a_chunk_size_below_one_is_refused(table: int, text: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        by_element(table, text)

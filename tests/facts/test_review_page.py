"""Review pages: drawing cards, keeping a page's card list, building each card and the page, and
the CLI command that does all of it."""

from __future__ import annotations

import base64
import io
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import pypdfium2
import pytest
from cltrplusf.chunk import Document
from PIL import Image

from aigent.facts import __main__ as cli
from aigent.facts.crosscheck import Checked
from aigent.facts.documents import Entry
from aigent.facts.review_page import (
    TEMPLATE,
    CardSource,
    Kept,
    PageList,
    PageOutcome,
    build_page,
    card,
    draw,
    kind_of,
    page_cards,
    page_image,
    printed_as,
    read_page_list,
    shares,
    write_page_list,
)

from .conftest import CROSSCHECK_RESULTS, checked_row, crosscheck_document

PDF = "results/ICICIBANK-2024-06-30.pdf"
HTML = "results/ICICIBANK-2024-09-30.htm"
MANIFEST: dict[str, Entry] = {
    PDF: {"format": "pdf", "source": "nse", "ocr_pages": [], "url": "https://nse/a.pdf"},
    HTML: {"format": "html", "source": "sec-6k"},
}
KINDS = ("born-digital PDF", "filed OCR layer", "RapidOCR reading")


@pytest.mark.parametrize(
    ("document", "origin", "kind"),
    [
        ("results/X.pdf#rapidocr", "nse-pdf-rapidocr-3.10.0", "RapidOCR reading"),
        ("results/X.htm", "sec-6k-html", "SEC copy"),
        ("results/X.pdf", "nse-pdf-ocr", "filed OCR layer"),
        ("annual/X-FY25.pdf", "nse-pdf", "born-digital PDF"),
    ],
)
def test_kind_of(document: str, origin: str, kind: str) -> None:
    assert kind_of(document, origin) == kind


@pytest.mark.parametrize(
    ("size", "available", "split"),
    [
        pytest.param(9, {"a": 9, "b": 9, "c": 9}, (3, 3, 3), id="even"),
        pytest.param(10, {"a": 9, "b": 9, "c": 9}, (4, 3, 3), id="the first kinds take the rest"),
        pytest.param(9, {"a": 1, "b": 9, "c": 9}, (1, 4, 4), id="a short kind gives its share up"),
        pytest.param(9, {"a": 9, "c": 9}, (5, 0, 4), id="a kind with none"),
        pytest.param(9, {"a": 2, "b": 1}, (2, 1, 0), id="fewer when all are used up"),
    ],
)
def test_shares(size: int, available: dict[str, int], split: tuple[int, int, int]) -> None:
    out = shares(size, available, ("a", "b", "c"))
    assert tuple(out[k] for k in ("a", "b", "c")) == split, out


def _rows() -> list[Checked]:
    """Ten open agreements in each of three documents, one per kind, and some that never draw."""
    out: list[Checked] = []
    for i, document in enumerate(("a.pdf", "b.pdf", "c.pdf#rapidocr")):
        out += [replace(checked_row(100 * i + n, "agree"), document=document) for n in range(10)]
    out += [checked_row(900, "differ"), checked_row(901, "agree", "xbrl-filed")]
    return out


_KINDS_OF = {"a.pdf": KINDS[0], "b.pdf": KINDS[1], "c.pdf#rapidocr": KINDS[2], PDF: KINDS[0]}


def test_a_batch_is_drawn_evenly_from_open_facts_of_the_pages_outcome() -> None:
    page = draw(_rows(), PageList("agree", None, ()), _KINDS_OF, 6, 1, KINDS)
    drawn = Counter(_KINDS_OF[c.document] for c in page.cards)
    assert drawn == {KINDS[0]: 2, KINDS[1]: 2, KINDS[2]: 2}, drawn
    assert {c.fact_id for c in page.cards}.isdisjoint({900, 901}), page.cards
    assert [c.id for c in page.cards] == [f"s0{n}" for n in range(1, 7)], page.cards
    assert {c.batch for c in page.cards} == {1}


def test_a_further_batch_keeps_the_cards_drawn_and_numbers_on() -> None:
    first = draw(_rows(), PageList("agree", "u", ()), _KINDS_OF, 6, 1, KINDS)
    second = draw(_rows(), first, _KINDS_OF, 6, 1, KINDS)
    assert second.cards[:6] == first.cards and second.url == "u"
    assert [c.id for c in second.cards[6:]] == [f"s{n:02d}" for n in range(7, 13)], second
    assert {c.batch for c in second.cards[6:]} == {2}
    assert len({c.fact_id for c in second.cards}) == 12, second.cards


def test_a_draw_is_the_same_for_the_same_seed_and_differs_by_seed() -> None:
    def ids(seed: int) -> list[int]:
        page = draw(_rows(), PageList("agree", None, ()), _KINDS_OF, 6, seed, KINDS)
        return [c.fact_id for c in page.cards]

    assert ids(1) == ids(1)
    assert ids(1) != ids(2)


@pytest.mark.parametrize(
    ("rows", "kept", "drawn"),
    [
        pytest.param(
            [checked_row(1, "agree"), checked_row(2, "agree", "xbrl-filed")],
            (),
            [1],
            id="a fact already verified is never drawn",
        ),
        pytest.param(
            [checked_row(n, "agree") for n in (1, 2, 3)],
            (Kept("s01", 1, 1, PDF), Kept("s02", 3, 1, PDF)),
            [2],
            id="a fact already on the page is never drawn again",
        ),
    ],
)
def test_a_batch_draws_only_what_is_open_and_new(
    rows: list[Checked], kept: tuple[Kept, ...], drawn: list[int]
) -> None:
    page = draw(rows, PageList("agree", None, kept), _KINDS_OF, 5, 1, KINDS)
    assert [c.fact_id for c in page.cards[len(kept) :]] == drawn, page.cards


def test_a_differences_page_draws_differences_as_d_cards() -> None:
    rows = [replace(r, document="a.pdf") for r in _rows()]
    page = draw(rows, PageList("differ", None, ()), _KINDS_OF, 6, 1, KINDS)
    assert [(c.id, c.fact_id) for c in page.cards] == [("d01", 900)], page.cards


def test_a_page_list_is_written_sorted_by_id_and_reads_back(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    cards = (Kept("s02", 2, 1, "b.pdf"), Kept("s10", 10, 2, "c.pdf"), Kept("s01", 1, 1, "a.pdf"))
    write_page_list(path, PageList("agree", "u", cards))
    ids = [c["id"] for c in json.loads(path.read_text())["cards"]]
    assert ids == ["s01", "s02", "s10"], ids
    assert read_page_list(path) == PageList("agree", "u", tuple(sorted(cards, key=lambda c: c.id)))
    assert len(path.read_text().splitlines()) == 5, path.read_text()


def test_a_page_list_is_none_before_it_exists_and_refused_with_an_unknown_outcome(
    tmp_path: Path,
) -> None:
    assert read_page_list(tmp_path / "none.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"outcome": "unmatched", "url": None, "cards": []}))
    with pytest.raises(ValueError, match="'unmatched' is not agree or differ"):
        read_page_list(bad)


@pytest.mark.parametrize(
    ("unit", "printed", "scale", "value"),
    [
        ("INR", "38,995.78", 1e7, 389957800000.0),
        ("INR", "(239)", 1e7, -2390000000.0),
        ("INR", "38,995.78", None, None),
        ("INRPerShare", "15.73", 1e7, 15.73),
        ("pure", "2.15%", None, 0.0215),
        ("pure", "0.41", None, 0.41),
        ("INR", "Particulars", 1e7, None),
    ],
)
def test_printed_as(unit: str, printed: str, scale: float | None, value: float | None) -> None:
    assert printed_as(unit, printed, scale) == pytest.approx(value)


def test_page_image_is_a_jpeg_of_the_page(tmp_path: Path) -> None:
    pdf = pypdfium2.PdfDocument.new()
    pdf.new_page(400, 600)
    pdf.new_page(600, 400)
    path = tmp_path / "two.pdf"
    pdf.save(path)
    pdf.close()
    shown = page_image(path, 2, width=300)
    prefix = "data:image/jpeg;base64,"
    assert shown.startswith(prefix), shown[:40]
    with Image.open(io.BytesIO(base64.b64decode(shown[len(prefix) :]))) as jpeg:
        assert (jpeg.format, jpeg.mode, jpeg.size) == ("JPEG", "L", (300, 200)), jpeg
    for missing in (0, 3):
        with pytest.raises(ValueError, match=f"has 2 pages, not a page {missing}"):
            page_image(path, missing)


_SOURCE = CardSource("born-digital PDF", "nse-pdf", "https://nse/a.pdf", None, {"scale": 1e7})


@pytest.mark.parametrize(
    ("row", "note"),
    [
        pytest.param(checked_row(1, "agree"), None, id="the page's outcome"),
        pytest.param(checked_row(1, "differ"), "now differ in this run", id="changed"),
        pytest.param(None, "not in this run", id="gone from the run"),
    ],
)
def test_a_card_notes_a_fact_no_longer_of_its_pages_outcome(
    row: Checked | None, note: str | None
) -> None:
    made = card(Kept("s01", 1, 1, PDF), row, _SOURCE, "agree")
    assert made["note"] == note, made
    assert (made["id"], made["batch"], made["fact_id"]) == ("s01", 1, 1)


def test_a_card_carries_its_fact_and_what_the_printed_figure_reads_as() -> None:
    made = card(Kept("d01", 1, 1, PDF), checked_row(1, "differ"), _SOURCE, "differ")
    assert made["printed_value"] == pytest.approx(389957800000.0), made
    assert (made["ticker"], made["row"], made["url"]) == (
        "ICICIBANK",
        "Interest earned",
        "https://nse/a.pdf",
    ), made


def test_page_cards_render_a_pdf_page_only_and_mark_an_untraced_fact() -> None:
    page = PageList(
        "agree", None, (Kept("s02", 2, 1, HTML), Kept("s01", 1, 1, PDF), Kept("s03", 3, 1, PDF))
    )
    rows = [checked_row(1, "agree"), replace(checked_row(2, "agree"), document=HTML)]
    drawn: list[tuple[str, int]] = []

    def image(path: Path, number: int) -> str:
        drawn.append((path.name, number))
        return "img"

    cards = page_cards(page, rows, {1: {"fact_id": 1}}, MANIFEST, Path("/v"), image)
    assert [c["id"] for c in cards] == ["s01", "s02", "s03"], cards
    assert [c.get("image") for c in cards[:2]] == ["img", None], cards
    assert drawn == [("ICICIBANK-2024-06-30.pdf", 3)], drawn
    assert cards[1]["trace"] == {"fact_id": 2, "error": "not traced"}, cards[1]
    assert (cards[1]["kind"], cards[2]["note"]) == ("SEC copy", "not in this run"), cards


def test_the_page_fills_every_marker_of_the_template() -> None:
    page = build_page([{"id": "s01", "row": "</script>"}], "differ", 1234, TEMPLATE.read_text())
    assert "{{" not in page, page[page.find("{{") - 40 : page.find("{{") + 40]
    assert "<title>XBRL Differences</title>" in page and "1,234 unverified differences" in page
    assert "<\\/script>" in page and '"</script>"' not in page


_MARKED = "{{TITLE}}{{HEADING}}{{LEDE}}{{DATA}}"


@pytest.mark.parametrize(
    ("template", "max_mb", "error"),
    [
        pytest.param("{{TITLE}}{{LEDE}}{{DATA}}", 16.0, "lacks {{HEADING}}", id="a missing marker"),
        pytest.param(
            "{{TITLE}}{{HEADING}}{{LEDE}}{{DATA}}", 0.0001, "over 0.0001 MB", id="too big"
        ),
    ],
)
def test_a_page_is_refused(template: str, max_mb: float, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        build_page([{"id": "s01"}], "agree", 0, template, max_mb)


RESULTS = crosscheck_document("STANDALONE FINANCIAL RESULTS (₹ in crore)", CROSSCHECK_RESULTS)


def _build(
    tmp_path: Path, outcome: PageOutcome | None, add: int = 0, url: str | None = None
) -> Path:
    """Builds page `p` from a run of six open agreements, against fakes for documents and images."""
    rows = tmp_path / "crosscheck-20261121-1207.rows.jsonl"
    if not rows.exists():
        found = [replace(checked_row(n, "agree"), document=PDF) for n in range(1, 7)]
        rows.write_text("".join(json.dumps(asdict(c)) + "\n" for c in found))

    def load(name: str) -> Document:
        return RESULTS

    return cli.review_page(
        "p",
        outcome,
        rows,
        add,
        url,
        lists=tmp_path / "lists",
        out_dir=tmp_path / "out",
        load=load,
        image=lambda path, number: "img",
        manifest=MANIFEST,
    )


def test_the_cli_draws_a_batch_records_the_url_and_writes_the_page(tmp_path: Path) -> None:
    built = _build(tmp_path, "agree", add=3, url="https://claude.ai/artifact/x")
    page = read_page_list(tmp_path / "lists" / "p.json")
    assert page is not None and page.url == "https://claude.ai/artifact/x", page
    assert [c.id for c in page.cards] == ["s01", "s02", "s03"], page.cards
    assert built == tmp_path / "out" / "p" / "index.html" and '"id": "s03"' in built.read_text()
    _build(tmp_path, None)
    assert read_page_list(tmp_path / "lists" / "p.json") == page


@pytest.mark.parametrize(
    ("first", "then", "error"),
    [
        pytest.param(None, None, "give its --outcome", id="a new page with no outcome"),
        pytest.param("agree", "differ", "shows agree facts, not differ", id="another outcome"),
    ],
)
def test_the_cli_refuses_a_page_it_cannot_build(
    tmp_path: Path, first: PageOutcome | None, then: PageOutcome | None, error: str
) -> None:
    with pytest.raises(SystemExit, match=error):
        if first is not None:
            _build(tmp_path, first)
        _build(tmp_path, then)


def test_the_newest_run_is_the_last_by_name(tmp_path: Path) -> None:
    for name in (
        "crosscheck-20261109-0320",
        "crosscheck-20261121-1207",
        "crosscheck-20261108-1921",
    ):
        (tmp_path / f"{name}.rows.jsonl").write_text("")
    assert cli.newest_rows(tmp_path).name == "crosscheck-20261121-1207.rows.jsonl"
    with pytest.raises(SystemExit, match="no cross-check run"):
        cli.newest_rows(tmp_path / "empty")

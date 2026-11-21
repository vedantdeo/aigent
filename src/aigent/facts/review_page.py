"""Review pages: cross-checked facts drawn as cards for a person to mark, each with its printed page
and its trace, built from `review_page.html` into one HTML file that Claude publishes as an
artifact. A page's card list is kept in `evals/reference/review-pages/<name>.json`."""

from __future__ import annotations

import base64
import io
import json
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast

import pypdfium2

from aigent.facts.cfg_facts import (
    REVIEW_PAGE_IMAGE_WIDTH,
    REVIEW_PAGE_JPEG_QUALITY,
    REVIEW_PAGE_KINDS,
    REVIEW_PAGE_MAX_MB,
)
from aigent.facts.crosscheck import Checked, number
from aigent.facts.documents import VERIFY_DIR, Entry, filed_of, origin_of
from aigent.facts.share_events import PER_SHARE

PageOutcome = Literal["agree", "differ"]
REVIEW_PAGES = Path(__file__).resolve().parents[3] / "evals" / "reference" / "review-pages"
TEMPLATE = Path(__file__).with_name("review_page.html")
_PREFIX: dict[PageOutcome, str] = {"agree": "s", "differ": "d"}
_MARKERS = ("{{TITLE}}", "{{HEADING}}", "{{LEDE}}", "{{DATA}}")
_TITLES: dict[PageOutcome, tuple[str, str]] = {
    "agree": ("XBRL Spot Check", "XBRL spot check"),
    "differ": ("XBRL Differences", "XBRL differences"),
}


@dataclass(frozen=True)
class Kept:
    """One card on a page: its id, its fact, the batch it was drawn in, and its document."""

    id: str
    fact_id: int
    batch: int
    document: str


@dataclass(frozen=True)
class PageList:
    """A page's kept cards, what they show (agreements or differences) and where it is published."""

    outcome: PageOutcome
    url: str | None
    cards: tuple[Kept, ...]


class StoredKept(TypedDict):
    id: str
    fact_id: int
    batch: int
    document: str


class StoredPage(TypedDict):
    outcome: PageOutcome
    url: str | None
    cards: list[StoredKept]


def kind_of(document: str, origin: str) -> str:
    """The kind of copy a card was read from, one of `REVIEW_PAGE_KINDS`."""
    if document.endswith("#rapidocr"):
        return "RapidOCR reading"
    if origin == "sec-6k-html":
        return "SEC copy"
    return "filed OCR layer" if origin == "nse-pdf-ocr" else "born-digital PDF"


def read_page_list(path: Path) -> PageList | None:
    """A page's card list, None when it has none yet; ValueError for an unknown outcome."""
    if not path.exists():
        return None
    stored = cast(StoredPage, json.loads(path.read_text(encoding="utf-8")))
    if stored["outcome"] not in _PREFIX:
        raise ValueError(f"{path.name}: outcome {stored['outcome']!r} is not agree or differ")
    cards = tuple(Kept(c["id"], c["fact_id"], c["batch"], c["document"]) for c in stored["cards"])
    return PageList(stored["outcome"], stored["url"], cards)


def write_page_list(path: Path, page: PageList) -> None:
    """Writes a page's card list sorted by card id, one card a line, its fields in one order."""
    stored: StoredPage = {
        "outcome": page.outcome,
        "url": page.url,
        "cards": [
            {"id": c.id, "fact_id": c.fact_id, "batch": c.batch, "document": c.document}
            for c in sorted(page.cards, key=lambda c: c.id)
        ],
    }
    head = json.dumps({k: v for k, v in stored.items() if k != "cards"}, ensure_ascii=False)
    lines = ",\n".join("  " + json.dumps(c, ensure_ascii=False) for c in stored["cards"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'{head[:-1]}, "cards": [\n{lines}\n]}}\n', encoding="utf-8")


def shares(size: int, available: Mapping[str, int], kinds: Sequence[str]) -> dict[str, int]:
    """`size` cards split evenly over `kinds` in order, a kind short of its share giving the rest to
    the others; fewer than `size` only when all are used up."""
    out = {k: 0 for k in kinds}
    left = size
    while left:
        open_kinds = [k for k in kinds if out[k] < available.get(k, 0)]
        if not open_kinds:
            break
        each, extra = divmod(left, len(open_kinds))
        for i, k in enumerate(open_kinds):
            take = min(each + (i < extra), available.get(k, 0) - out[k])
            out[k] += take
            left -= take
    return out


def draw(
    rows: Sequence[Checked],
    page: PageList,
    kinds: Mapping[str, str],
    size: int,
    seed: int,
    order: Sequence[str] = REVIEW_PAGE_KINDS,
) -> PageList:
    """`page` with a new batch of `size` unverified facts of its outcome not already on it, drawn
    evenly over the kinds of copy (`kinds` maps a document to its kind) and numbered on."""
    taken = {c.fact_id for c in page.cards}
    pool: dict[str, list[Checked]] = {}
    for r in sorted(rows, key=lambda r: r.fact_id):
        if r.outcome == page.outcome and r.tier == "xbrl-ai" and r.fact_id not in taken:
            pool.setdefault(kinds[r.document], []).append(r)
    batch = max((c.batch for c in page.cards), default=0) + 1
    rng = random.Random(f"{seed}:{batch}")
    split = shares(size, {k: len(v) for k, v in pool.items()}, order)
    drawn = [r for k in order for r in rng.sample(pool.get(k, []), split[k])]
    rng.shuffle(drawn)
    start = max((int(c.id[1:]) for c in page.cards), default=0) + 1
    prefix = _PREFIX[page.outcome]
    new = tuple(
        Kept(f"{prefix}{n:02d}", r.fact_id, batch, r.document)
        for n, r in enumerate(drawn, start=start)
    )
    return PageList(page.outcome, page.url, page.cards + new)


def printed_as(unit: str, printed: str, scale: float | None) -> float | None:
    """What a printed figure reads as in the fact's unit: at the table's scale for rupees, a
    percentage /100 for a ratio; None when it is no number or no scale was read."""
    parsed = number(printed)
    if parsed is None:
        return None
    value, _, percent = parsed
    if unit == "pure":
        return value / 100 if percent else value
    if unit.endswith(PER_SHARE):
        return value
    return None if scale is None else value * scale


def page_image(
    pdf: Path,
    page: int,
    width: int = REVIEW_PAGE_IMAGE_WIDTH,
    quality: int = REVIEW_PAGE_JPEG_QUALITY,
) -> str:
    """One page of a PDF as a grey JPEG `width` pixels wide, as a data URI; ValueError for a page
    the PDF does not have."""
    document = pypdfium2.PdfDocument(pdf)
    try:
        if not 1 <= page <= len(document):
            raise ValueError(f"{pdf.name} has {len(document)} pages, not a page {page}")
        sheet = document[page - 1]
        bitmap = sheet.render(scale=width / sheet.get_width())
        buffer = io.BytesIO()
        bitmap.to_pil().convert("L").save(buffer, "JPEG", quality=quality)
    finally:
        document.close()
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


@dataclass(frozen=True)
class CardSource:
    """What a card needs beyond its fact's row: its kind and stamp, the filed copy's URL, the page
    image, and its trace (or the error tracing it gave)."""

    kind: str
    stamp: str
    url: str | None
    image: str | None
    trace: Mapping[str, object]


def card(
    kept: Kept, row: Checked | None, source: CardSource, outcome: PageOutcome
) -> dict[str, object]:
    """A card's data for the page; a kept fact missing from the run, or now of another outcome,
    carries a note saying so."""
    out: dict[str, object] = {"id": kept.id, "batch": kept.batch, "fact_id": kept.fact_id}
    if row is None:
        return {**out, "kind": source.kind, "document": kept.document, "note": "not in this run"}
    scale = source.trace.get("scale")
    read = printed_as(row.unit, row.printed or "", scale if isinstance(scale, float) else None)
    return {
        **out,
        "kind": source.kind,
        "ticker": row.ticker,
        "basis": row.basis,
        "concept": row.concept,
        "segment": row.segment,
        "period_start": row.period_start,
        "period_end": row.period_end,
        "value": row.value,
        "unit": row.unit,
        "per_share": row.unit.endswith(PER_SHARE),
        "outcome": row.outcome,
        "printed": row.printed,
        "printed_value": read,
        "row": row.row_label,
        "document": row.document,
        "page": row.page,
        "stamp": source.stamp,
        "url": source.url,
        "image": source.image,
        "trace": dict(source.trace),
        "note": None if row.outcome == outcome else f"now {row.outcome} in this run",
    }


def build_page(
    cards: Sequence[Mapping[str, object]],
    outcome: PageOutcome,
    open_count: int,
    template: str,
    max_mb: float = REVIEW_PAGE_MAX_MB,
) -> str:
    """The page: `template` with its title, heading, lede and cards filled in. ValueError for a
    template missing a marker, or a page over `max_mb`."""
    missing = [m for m in _MARKERS if m not in template]
    if missing:
        raise ValueError(f"the template lacks {', '.join(missing)}")
    what = "agreements" if outcome == "agree" else "differences"
    lede = (
        f"{len(cards)} {what} the cross-check found between a filed XBRL fact and a printed "
        f"figure, drawn at random across the kinds of copy, in batches. For each, find the "
        f"printed figure on the page beside it and mark whether it really is this fact: the "
        f"right line item, period and basis. Open “How it was read” to see the table as Docling "
        f"parsed it and as the cross-check read it. {open_count:,} unverified {what} remain."
    )
    data = json.dumps(list(cards), ensure_ascii=False).replace("</", "<\\/")
    title, heading = _TITLES[outcome]
    page = (
        template.replace("{{TITLE}}", title)
        .replace("{{HEADING}}", heading)
        .replace("{{LEDE}}", lede)
        .replace("{{DATA}}", data)
    )
    size = len(page.encode()) / 1e6
    if size > max_mb:
        raise ValueError(f"the page is {size:.1f} MB, over {max_mb} MB: draw fewer cards")
    return page


def kinds_of(documents: Iterable[str], manifest: Mapping[str, Entry]) -> dict[str, str]:
    """Each document's kind of copy, by its manifest entry's origin."""
    return {d: kind_of(d, origin_of(d, manifest[filed_of(d)])) for d in set(documents)}


def page_cards(
    page: PageList,
    rows: Sequence[Checked],
    traces: Mapping[int, Mapping[str, object]],
    manifest: Mapping[str, Entry],
    verify_dir: Path = VERIFY_DIR,
    image: Callable[[Path, int], str] = page_image,
) -> list[dict[str, object]]:
    """Every kept card's data in id order: its fact's row in this run, its trace, and its page's
    image, none for an HTML filing or a fact with no page."""
    by_id = {r.fact_id: r for r in rows}
    out: list[dict[str, object]] = []
    for kept in sorted(page.cards, key=lambda c: c.id):
        row = by_id.get(kept.fact_id)
        document = row.document if row else kept.document
        entry = manifest[filed_of(document)]
        origin = origin_of(document, entry)
        shown = None
        if row is not None and row.page is not None and entry["format"] == "pdf":
            shown = image(verify_dir / filed_of(document), row.page)
        trace = traces.get(kept.fact_id, {"fact_id": kept.fact_id, "error": "not traced"})
        url = entry.get("url")
        filed = url if isinstance(url, str) else None
        source = CardSource(kind_of(document, origin), origin, filed, shown, trace)
        out.append(card(kept, row, source, page.outcome))
    return out

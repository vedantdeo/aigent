"""Splitting a document into the pieces a retriever ranks.

Four strategies — fixed, fixed with overlap, by sentence, by heading — all returning the same
`Chunk`, so one can be swapped for another and compared in the eval table. Chunk ids are positional
(`RELIANCE-FY25#0042`) and belong to one `Inventory`; sizes are in characters, not tokens.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from aigent.config import (
    CHUNK_CHARS,
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    CHUNK_OVERLAP_CHARS,
    CHUNK_OVERLAP_SENTENCES,
    HEADING_MAX_CHARS,
    HEADING_MAX_WORDS,
    HEADING_MIN_CAPITAL_RATIO,
)

# Typographic characters a PDF uses and a hand-typed label does not, folded to their ASCII forms.
_TYPOGRAPHY = str.maketrans(
    {c: "'" for c in "\u2018\u2019\u201a\u201b\u2032"}
    | {c: '"' for c in "\u201c\u201d\u201e\u201f\u2033"}
    | {c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015"}
)


# Extraction leaves a space before punctuation where a PDF's kerning did (`lawyers .`, `52 %`).
# No correct English writes one, so closing it can only make two spellings of one text agree.
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?%)\]])")


def squeeze(text: str) -> str:
    """Whitespace collapsed, typography folded and case folded — how two texts compare."""
    closed = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", text.translate(_TYPOGRAPHY))
    return " ".join(closed.split()).casefold()


@dataclass(frozen=True)
class Document:
    """One source document, with `page_starts` holding the offset each page begins at."""

    doc_id: str
    text: str
    source: str = ""
    page_starts: tuple[int, ...] = ()

    def page_of(self, offset: int) -> int | None:
        """1-indexed page containing this character offset, None for a source without pages."""
        return bisect_right(self.page_starts, offset) if self.page_starts else None


@dataclass(frozen=True)
class Chunk:
    """One retrievable piece of one document."""

    id: str
    text: str
    doc_id: str
    ordinal: int
    start: int
    page: int | None = None
    heading: str | None = None

    @property
    def citation(self) -> str:
        """How this chunk names itself in an answer: "RIL-FY25 · p.2 · Risk Management"."""
        parts = [self.doc_id]
        if self.page is not None:
            parts.append(f"p.{self.page}")
        if self.heading is not None:
            parts.append(self.heading)
        return " · ".join(parts)


@dataclass(frozen=True)
class Inventory:
    """Every chunk of every document under one chunking strategy.

    `name` is which strategy produced these ids, without which they compare with nothing.
    """

    name: str
    chunks: tuple[Chunk, ...] = ()
    by_id: dict[str, Chunk] = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, name: str, documents: Iterable[Document], splitter: Splitter) -> Inventory:
        """Chunk every document with one splitter. Two sharing a `doc_id` is a ValueError."""
        doc_chunks: dict[str, list[Chunk]] = {}
        for doc in documents:
            if doc.doc_id in doc_chunks:
                raise ValueError(
                    f"two documents share a doc_id, so their chunk ids collide: {doc.doc_id!r}"
                )
            doc_chunks[doc.doc_id] = splitter(doc)
        chunks = tuple(chunk for chunk_list in doc_chunks.values() for chunk in chunk_list)
        by_id = {chunk.id: chunk for chunk in chunks}
        return Inventory(name=name, chunks=chunks, by_id=by_id)

    def __len__(self) -> int:
        """How many chunks this inventory holds."""
        return len(self.chunks)

    def texts(self) -> list[str]:
        """Chunk texts in inventory order — what the embedder is handed."""
        return [chunk.text for chunk in self.chunks]

    def ids(self) -> list[str]:
        """Chunk ids in inventory order, aligned with `texts()` row for row."""
        return [chunk.id for chunk in self.chunks]

    def containing(self, quote: str) -> list[str]:
        """Ids of every chunk holding this quote, whitespace- and case-insensitively.

        Empty means no single chunk contains it, which is a finding about the strategy, not a bug.
        """
        needle = squeeze(quote)
        if not needle:
            return []
        return [chunk.id for chunk in self.chunks if needle in squeeze(chunk.text)]


def context_block(chunks: Sequence[Chunk]) -> str:
    """The passages as a prompt shows them, each tagged with the id the model must cite it by.

    XML-delimited and one passage per element, so the model can point at one of them rather than
    at the blob — a citation that names no passage cannot be checked against the label.
    """
    return "\n\n".join(
        f'<passage id="{chunk.id}" source="{chunk.citation}">\n{chunk.text}\n</passage>'
        for chunk in chunks
    )


Splitter = Callable[[Document], list[Chunk]]


# --- Fixed windows ---------------------------------------------------------------------------


def fixed(size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> Splitter:
    """Slide a window of `size` characters, stepping `size - overlap` each time.

    Does not snap to word boundaries: this is the baseline the other three are measured against.
    """
    if size < 1:
        raise ValueError(f"Chunk size {size} is below 1")
    if overlap < 0:
        raise ValueError(f"Chunk overlap {overlap} is negative")
    if overlap >= size:
        raise ValueError(f"Chunk overlap {overlap} is not smaller than chunk size {size}")
    step = size - overlap

    def splitter(doc: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        doc_length = len(doc.text)
        ordinal = 0
        for start in range(0, doc_length, step):
            end = min(start + size, doc_length)
            if end - start < CHUNK_MIN_CHARS:
                continue
            text = doc.text[start:end]
            page = doc.page_of(start)
            chunk_id = f"{doc.doc_id}#{ordinal:04d}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    doc_id=doc.doc_id,
                    ordinal=ordinal,
                    start=start,
                    page=page,
                )
            )
            ordinal += 1
        return chunks

    return splitter


# --- Sentences -------------------------------------------------------------------------------

# Abbreviations whose full stop does not end a sentence. Annual-report specific: `Rs.` and `Cr.`
# appear in every second line of one, and splitting on them shears a number off its unit.
_ABBREVIATIONS = frozenset(
    {
        "approx",
        "co",
        "cr",
        "dr",
        "e.g",
        "etc",
        "fig",
        "i.e",
        "inc",
        "jr",
        "ltd",
        "mr",
        "mrs",
        "ms",
        "no",
        "pvt",
        "rs",
        "sr",
        "vs",
        "viz",
    }
)


_SENTENCE_BREAK = re.compile(r"(?<=[.!?])[\"'’”)\]]*\s+")


# How far back to look for the word before a break. Only the last word matters, so this need only
# exceed the longest one; slicing the whole prefix instead made this function quadratic.
_ABBREVIATION_LOOKBACK = 64


def sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence, by regex with an abbreviation guard.

    Breaks after `.`, `!` or `?` plus whitespace, unless the preceding word is in `_ABBREVIATIONS`
    or is a single initial. The spans tile the string, so `match.end()` starts the next sentence.
    """
    start_ends: list[tuple[int, int]] = []
    curr_start = 0
    for match in _SENTENCE_BREAK.finditer(text):
        head = text[max(0, match.start() - _ABBREVIATION_LOOKBACK) : match.start()]
        words = head.rstrip(".!?").split()
        word = words[-1].rstrip(".").casefold() if words else ""
        if word in _ABBREVIATIONS or len(word) == 1:
            continue
        start_ends.append((curr_start, match.end()))
        curr_start = match.end()

    return start_ends + ([(curr_start, len(text))] if curr_start < len(text) else [])


def by_sentence(
    max_chars: int = CHUNK_CHARS, overlap_sentences: int = CHUNK_OVERLAP_SENTENCES
) -> Splitter:
    """Pack whole sentences up to `max_chars`, starting the next chunk `overlap_sentences` back.

    A span with no sentence break inside it is cut on the character budget, since text the
    embedder never reads cannot be retrieved.
    """
    if max_chars < 1:
        raise ValueError(f"Chunk size {max_chars} is below 1")
    if overlap_sentences < 0:
        raise ValueError(f"Chunk overlap {overlap_sentences} is negative")

    def splitter(doc: Document) -> list[Chunk]:
        chunks: list[Chunk] = []

        def emit(span_start: int, span_end: int) -> None:
            """Append one chunk. Only a span that is entirely whitespace is skipped.

            **No `CHUNK_MIN_CHARS` floor here**, unlike `fixed` and `by_heading`. That floor guards
            artefacts of the splitting mechanism — a window that ended 60 characters into a
            document, a heading left with no body under it — and sentence packing produces none:
            the smallest thing it can emit is a whole sentence, which is a real unit at any length.
            Bounding both ends instead cost text. A tiny sentence in front of one longer than
            `max_chars` was flushed alone, dropped for being short, and never revisited, so
            `containing` reported the quote inside it as split across a boundary when it had in
            fact been deleted.

            The ordinal is `len(chunks)`, so it is assigned at append time and stays contiguous
            without anyone having to remember not to increment it on the way past a skip. `start`
            moves with the strip for the same reason: a `start` pointing at whitespace that `text`
            no longer contains would put `page_of` on the wrong side of a page break.
            """
            raw = doc.text[span_start:span_end]
            text = raw.strip()
            if not text:
                return
            offset = span_start + (len(raw) - len(raw.lstrip()))
            chunks.append(
                Chunk(
                    id=f"{doc.doc_id}#{len(chunks):04d}",
                    text=text,
                    doc_id=doc.doc_id,
                    ordinal=len(chunks),
                    start=offset,
                    page=doc.page_of(offset),
                    heading=None,
                )
            )

        def emit_capped(span_start: int, span_end: int) -> None:
            """Emit a span, cutting it on the character budget when it holds no sentence break.

            A financial table has no `.!?` in it at all, so packing whole "sentences" hands the
            embedder one chunk of many thousands of characters, of which it reads the first 512
            tokens and silently ignores the rest. A mid-word cut is worse than a sentence boundary
            and better than text no retriever can see.
            """
            for piece_start in range(span_start, span_end, max_chars):
                emit(piece_start, min(piece_start + max_chars, span_end))

        sentence_spans = sentences(doc.text)
        start_index = 0
        curr_start = 0
        curr_index = 0
        while curr_index < len(sentence_spans):
            start, end = sentence_spans[curr_index]
            if end - curr_start > max_chars:
                if curr_start == start:
                    # This one sentence is over budget on its own, so there is nothing to pack.
                    emit_capped(curr_start, end)
                    curr_start = end
                    start_index = curr_index + 1
                    curr_index += 1
                else:
                    emit(curr_start, start)
                    start_index = max(start_index + 1, curr_index - overlap_sentences)
                    curr_index = max(curr_index, start_index)
                    curr_start = sentence_spans[start_index][0]
            else:
                curr_index += 1
        if curr_start < len(doc.text):
            emit_capped(curr_start, len(doc.text))
        return chunks

    return splitter


# --- Headings --------------------------------------------------------------------------------


_NUMBERED = re.compile(r"^\d+(\.\d+)*\.?\s+\S")


def _looks_like_heading(line: str) -> bool:
    """The three signals, applied to one already-stripped, non-empty line."""
    if len(line) > HEADING_MAX_CHARS or line[-1] in ".,;":
        return False
    words = line.split()
    if len(words) > HEADING_MAX_WORDS:
        return False
    if _NUMBERED.match(line):
        return True
    alphabetic = [word for word in words if word[0].isalpha()]
    if not alphabetic:
        return False
    capitalised = sum(1 for word in alphabetic if word[0].isupper())
    return capitalised / len(alphabetic) >= HEADING_MIN_CAPITAL_RATIO


def headings(text: str) -> list[tuple[int, str]]:
    """(offset, heading) for every line that reads like a heading rather than a sentence.

    Short, no trailing `.` `,` or `;`, and either numbered or mostly capitalised. The numbered test
    runs first because a numbered heading in sentence case would fail the ratio.
    """
    found: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped and _looks_like_heading(stripped):
            found.append((offset + len(line) - len(line.lstrip()), stripped))
        offset += len(line)
    return found


def by_heading(max_chars: int = CHUNK_MAX_CHARS) -> Splitter:
    """One chunk per section, split by sentence when a section runs past `max_chars`.

    Sections run heading to heading with the heading line included, and every piece of an oversized
    section keeps its heading. Falls back to `by_sentence` when the document has no headings.
    """
    if max_chars < 1:
        raise ValueError(f"Section size {max_chars} is below 1")
    pack_section = by_sentence(max_chars, overlap_sentences=0)

    def splitter(doc: Document) -> list[Chunk]:
        marks = headings(doc.text)
        if not marks:
            return pack_section(doc)

        sections: list[tuple[int, str | None]] = list(marks)
        if sections[0][0] > 0:
            sections.insert(0, (0, None))  # front matter, before any heading claims it
        ends = [*(offset for offset, _ in sections[1:]), len(doc.text)]

        chunks: list[Chunk] = []

        def add(start: int, text: str, heading: str | None) -> None:
            chunks.append(
                Chunk(
                    id=f"{doc.doc_id}#{len(chunks):04d}",
                    text=text,
                    doc_id=doc.doc_id,
                    ordinal=len(chunks),
                    start=start,
                    page=doc.page_of(start),
                    heading=heading,
                )
            )

        for (start, heading), end in zip(sections, ends, strict=True):
            body = doc.text[start:end]
            text = body.strip()
            offset = start + len(body) - len(body.lstrip())
            if len(text) <= max_chars:
                if len(text) >= CHUNK_MIN_CHARS:
                    add(offset, text, heading)
                continue
            for piece in pack_section(Document(doc_id=doc.doc_id, text=body)):
                add(start + piece.start, piece.text, heading)
        return chunks

    return splitter

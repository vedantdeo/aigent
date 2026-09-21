"""What every pattern works over: the three annual reports, and the search that reads them."""

from __future__ import annotations

import json
from typing import Literal, Protocol, cast

from entropic.retrieval.chunk import Chunk
from entropic.retrieval.corpus import MANIFEST

# As a schema field, this holds the model to a real report. A test holds it to the manifest.
DocId = Literal["ITC-FY25", "RELIANCE-FY25", "TATAMOTORS-FY25"]

_DOCUMENTS = cast(
    dict[str, dict[str, object]], json.loads(MANIFEST.read_text(encoding="utf-8"))["documents"]
)
REPORTS: dict[str, str] = {doc_id: str(entry["company"]) for doc_id, entry in _DOCUMENTS.items()}
CATALOGUE = "\n".join(f"- {doc_id}: {company}" for doc_id, company in REPORTS.items())


class Search(Protocol):
    """A query in, passages out, from one report when `doc_id` is given."""

    def __call__(self, query: str, /, *, doc_id: str | None = None) -> list[Chunk]: ...

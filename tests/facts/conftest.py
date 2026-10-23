"""Small XBRL filings built in code, a scripted NSE and two trimmed real filings, shared by the
facts database's tests."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from aigent.facts.cfg_facts import NSE_INTEGRATED_LISTING, NSE_LEGACY_LISTING, NSE_PAGE_SIZE
from aigent.facts.reference import Fundamental

FIXTURES = Path(__file__).parent / "fixtures"
LEGACY_Q4 = FIXTURES / "ITC-2024-03-31-C.xml"  # 2020 BSE taxonomy, FY24's March quarter
INTEGRATED_Q3 = FIXTURES / "ITC-2025-12-31-C.xml"  # SEBI's integrated filing, FY26's December

_HEAD = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance" '
    'xmlns:xbrldi="http://xbrl.org/2006/xbrldi" '
    'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
    'xmlns:in-capmkt="http://www.sebi.gov.in/xbrl/2025-01-31/in-capmkt">\n'
)
_ENTITY = "<xbrli:entity><xbrli:identifier scheme='x'>T</xbrli:identifier></xbrli:entity>"


def duration(ctx: str, start: str, end: str, member: str | None = None) -> str:
    return _context(
        ctx,
        f"<xbrli:startDate>{start}</xbrli:startDate><xbrli:endDate>{end}</xbrli:endDate>",
        member,
    )


def instant(ctx: str, day: str, member: str | None = None) -> str:
    return _context(ctx, f"<xbrli:instant>{day}</xbrli:instant>", member)


def _context(ctx: str, period: str, member: str | None) -> str:
    scenario = (
        f"<xbrli:scenario><xbrldi:explicitMember dimension='in-capmkt:Axis'>in-capmkt:{member}"
        "</xbrldi:explicitMember></xbrli:scenario>"
        if member
        else ""
    )
    return (
        f'<xbrli:context id="{ctx}">{_ENTITY}<xbrli:period>{period}</xbrli:period>{scenario}'
        "</xbrli:context>"
    )


def num(concept: str, ctx: str, value: str, unit: str = "INR", decimals: str = "-5") -> str:
    return (
        f'<in-capmkt:{concept} contextRef="{ctx}" unitRef="{unit}" decimals="{decimals}">{value}'
        f"</in-capmkt:{concept}>"
    )


def nil(concept: str, ctx: str) -> str:
    return f'<in-capmkt:{concept} contextRef="{ctx}" unitRef="INR" xsi:nil="true"/>'


def text(concept: str, ctx: str, value: str) -> str:
    return f'<in-capmkt:{concept} contextRef="{ctx}">{value}</in-capmkt:{concept}>'


def stated(column: str, start: str, end: str) -> str:
    """A column's period as the filing states it in words."""
    return text("DateOfStartOfReportingPeriod", f"{column}D", start) + text(
        "DateOfEndOfReportingPeriod", f"{column}D", end
    )


def filing(*parts: str) -> bytes:
    return (_HEAD + "\n".join(parts) + "\n</xbrli:xbrl>\n").encode()


# A December quarter whose context dates hold: the quarter, the year to date, a balance.
DECEMBER = (
    duration("OneD", "2025-10-01", "2025-12-31"),
    duration("FourD", "2025-04-01", "2025-12-31"),
    instant("OneI", "2025-12-31"),
    stated("One", "2025-10-01", "2025-12-31"),
    stated("Four", "2025-04-01", "2025-12-31"),
)


def legacy_row(to_date: str, basis: str, url: str, broadcast: str) -> dict[str, object]:
    """A row of the old results listing; `to_date` like 31-Dec-2024."""
    return {"toDate": to_date, "consolidated": basis, "xbrl": url, "broadCastDate": broadcast}


def integrated_row(qe_date: str, basis: str | None, url: str, broadcast: str) -> dict[str, object]:
    """A row of the integrated filing listing; `qe_date` like 31-MAR-2025."""
    return {"qe_Date": qe_date, "consolidated": basis, "xbrl": url, "broadcast_Date": broadcast}


@dataclass
class FakeNse:
    """NSE scripted by URL: bytes to answer with, or an exception to raise; every call recorded."""

    answers: dict[str, bytes | Exception] = field(default_factory=dict[str, bytes | Exception])
    calls: list[str] = field(default_factory=list[str])

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        answer = self.answers.get(url)
        if answer is None:
            raise OSError(f"HTTP Error 404: {url}")
        if isinstance(answer, Exception):
            raise answer
        return answer

    def listings(
        self,
        symbol: str,
        legacy: list[dict[str, object]],
        integrated: list[list[dict[str, object]]],
    ) -> None:
        """The two listings for `symbol`, the integrated one as pages of rows."""
        self.answers[NSE_LEGACY_LISTING.format(symbol=symbol)] = json.dumps(legacy).encode()
        total = sum(len(page) for page in integrated) if integrated else 0
        for number, rows in enumerate(integrated or [[]], start=1):
            url = NSE_INTEGRATED_LISTING.format(symbol=symbol, size=NSE_PAGE_SIZE, page=number)
            body = {"data": rows, "totalCount": total, "size": NSE_PAGE_SIZE, "page": number - 1}
            self.answers[url] = json.dumps(body).encode()


@pytest.fixture
def nse() -> FakeNse:
    return FakeNse()


def fundamental(ticker: str, year: int, metric: str, value: float) -> Fundamental:
    return Fundamental(
        ticker=ticker,
        year=year,
        metric=metric,
        value=value,
        unit="INR",
        printed=str(value),
        doc_id=f"{ticker}-FY{year % 100}",
        page=1,
        note=None,
    )


_ROLE = "http://www.xbrl.org/2003/role/"


def label_linkbase(labels: dict[str, list[tuple[str, str]]], prefix: str = "in-capmkt") -> str:
    """A label linkbase giving each concept its (role, text) labels."""
    parts: list[str] = []
    for n, (concept, roles) in enumerate(sorted(labels.items())):
        parts.append(
            f'<link:loc xlink:type="locator" xlink:href="x.xsd#{prefix}_{concept}" '
            f'xlink:label="loc{n}"/>'
            f'<link:labelArc xlink:type="arc" xlink:from="loc{n}" xlink:to="lab{n}"/>'
        )
        parts.extend(
            f'<link:label xlink:type="resource" xlink:label="lab{n}" xlink:role="{_ROLE}{role}" '
            f'xml:lang="en">{text}</link:label>'
            for role, text in roles
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<link:linkbase xmlns:link="http://www.xbrl.org/2003/linkbase" '
        'xmlns:xlink="http://www.w3.org/1999/xlink"><link:labelLink xlink:type="extended">'
        + "".join(parts)
        + "</link:labelLink></link:linkbase>"
    )


def label_package(labels: dict[str, list[tuple[str, str]]], name: str = "core/x-lab.xml") -> bytes:
    """A zipped taxonomy package holding one label linkbase, and a schema that is not one."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, label_linkbase(labels))
        archive.writestr("core/x.xsd", "<schema/>")
    return buffer.getvalue()

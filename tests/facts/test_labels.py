"""`facts.labels`: a fact's labels against a printed row's, and the vetoes that refuse a row."""

from __future__ import annotations

import pytest

from aigent.facts.cfg_facts import CROSSCHECK_LABEL_FIGURES
from aigent.facts.labels import label_figure, label_match, label_share


@pytest.mark.parametrize(
    ("label", "row", "share"),
    [
        pytest.param("Interest earned", "1. Interest earned (a)+(b)", 1.0, id="the same words"),
        pytest.param(
            "Earnings per share", "Basic EPS", 1.0, id="a label naming neither takes either"
        ),
        pytest.param("Basic earnings per share", "a) Basic EPS", 1.0, id="EPS read in full"),
        pytest.param("Diluted EPS", "a) Basic EPS", 0.0, id="diluted never takes basic"),
        pytest.param("Basic EPS", "b) Diluted EPS", 0.0, id="basic never takes diluted"),
        pytest.param("Basic EPS", "Basic and diluted EPS", 1.0, id="a row naming both"),
        pytest.param("Gross NPA", "Net NPAs", 0.0, id="gross never takes net"),
        pytest.param("Gross NPA", "(a) Gross NPAs", 1.0, id="NPA read in full"),
        pytest.param("Interest earned", "Other income", 0.0, id="nothing shared"),
        pytest.param(
            "Exceptional items before tax",
            "Profit before exceptional items and tax",
            0.0,
            id="a row led by profit names a profit",
        ),
        pytest.param(
            "Inventories", "b) Changes in inventories of stock-in-trade", 0.0, id="nor by changes"
        ),
        pytest.param(
            "Purchase of property, plant and equipment",
            "Property, plant and equipment",
            0.0,
            id="a purchase is never the asset",
        ),
        pytest.param(
            "Proceeds from sales of property, plant and equipment",
            "Sale of property, plant and equipment",
            0.6,
            id="proceeds printed as a sale",
        ),
        pytest.param(
            "Finance costs",
            "Operating margin (%) [(Profit before tax + Finance costs) / Revenue]",
            0.0,
            id="a formula in brackets is no label",
        ),
        pytest.param(
            "Depreciation and amortisation expense",
            "[(f) Depreciation and amortisation expense",
            1.0,
            id="a label opening on a bracket keeps its words",
        ),
        pytest.param(
            "Revenue from operations", "Total revenue from operations", 1.0, id="total led"
        ),
        pytest.param(
            "Current assets",
            "Other financial assets - non-current",
            0.0,
            id="current never takes non-current",
        ),
        pytest.param("Assets", "Other financial assets", 0.0, id="a short label is no other row"),
        pytest.param("Income", "(ii) Interest Income", 0.0, id="nor a row of a kind of it"),
        pytest.param(
            "Profit before exceptional items and tax",
            "Debt service coverage ratio {(Profit before exceptional items and tax + Interest)}",
            0.0,
            id="a long label never takes a formula's words",
        ),
        pytest.param(
            "Other current financial assets",
            "Other financial assets - non-current",
            0.0,
            id="a long current label never takes non-current",
        ),
        pytest.param("Total income", "III. Total income", 1.0, id="a roman numeral passed over"),
        pytest.param(
            "Current financial assets",
            "(iv) Other financial assets",
            0.0,
            id="three words still short",
        ),
        pytest.param(
            "Borrowings non-current", "Non-current borrowings", 1.0, id="led by its own word"
        ),
        pytest.param(
            "Payments of lease liabilities",
            "Principal payment of lease liabilities",
            2 / 3,
            id="principal passed over",
        ),
        pytest.param(
            "Intangible assets under development",
            "(f) Other intangible assets under development",
            1.0,
            id="a long one may be printed as other",
        ),
        pytest.param(
            "Items that will be reclassified to profit or loss",
            "Items that will not be reclassified to profit or loss",
            0.0,
            id="will be never takes will not be",
        ),
        pytest.param(
            "Items that will not be reclassified to profit or loss",
            "liems that will not he reclassified to profit or loss",
            6 / 7,
            id="will not be through OCR",
        ),
        pytest.param(
            "Basic earnings per share",
            "Basic EPS (not annualised)",
            1.0,
            id="a bare not is no negation",
        ),
        pytest.param("Interest earned", "Interest expended", 0.5, id="half shared"),
        pytest.param(
            "Current tax",
            "Current tax on profits for the year discontinued operations",
            0.0,
            id="a row of discontinued operations alone takes no other fact",
        ),
        pytest.param(
            "Profit from discontinued operations",
            "Profit for the period",
            0.0,
            id="a fact of discontinued operations alone takes no other row",
        ),
        pytest.param(
            "Profit before tax",
            "Profit before tax from continuing operations",
            1.0,
            id="a total takes a continuing-operations row",
        ),
        pytest.param(
            "Profit from continuing and discontinued operations",
            "Profit from continuing and discontinued operations",
            1.0,
            id="both kinds on both sides",
        ),
    ],
)
def test_label_share(label: str, row: str, share: float) -> None:
    assert label_share([label], None, row) == pytest.approx(share), (label, row)


def test_a_segment_counts_as_part_of_the_label() -> None:
    assert label_share(["Segment revenue"], "Retail", "a Retail Banking") == pytest.approx(1 / 3)
    assert label_share(["Revenue"], "Retail", "a Retail Banking") == pytest.approx(1 / 2)


def test_the_best_of_a_facts_labels_counts() -> None:
    assert (
        label_share(
            ["Revenue on investments", "Income on investments"], None, "b) Income on investments"
        )
        == 1.0
    )


@pytest.mark.parametrize(
    ("label", "segment", "row", "veto"),
    [
        pytest.param(
            "Exceptional items",
            None,
            "Profit before exceptional items",
            "row turns",
            id="row turns",
        ),
        pytest.param("Income", None, "(ii) Interest income", "short label", id="short label"),
        pytest.param(
            "Purchase of property, plant and equipment",
            None,
            "Property, plant and equipment",
            "flow unprinted",
            id="flow unprinted",
        ),
        pytest.param(
            "Profit from discontinued operations",
            None,
            "Profit for the period from operations",
            "discontinued",
            id="discontinued",
        ),
        pytest.param(
            "Items that will be reclassified to profit or loss",
            None,
            "Items that will not be reclassified to profit or loss",
            "will not be",
            id="will not be",
        ),
        pytest.param(
            "Basic earnings per share",
            None,
            "Diluted earnings per share",
            "opposites",
            id="opposites",
        ),
        pytest.param("Revenue from operations", None, "Revenue from operations", None, id="none"),
    ],
)
def test_a_refused_row_names_its_veto(
    label: str, segment: str | None, row: str, veto: str | None
) -> None:
    match = label_match([label], segment, row)
    assert match.veto == veto, match
    assert (match.share == 0) is (veto is not None), match


def test_a_veto_keeps_the_share_it_refused() -> None:
    match = label_match(
        ["Exceptional items before tax"], None, "Profit before exceptional items and tax"
    )
    assert (match.share, match.raw, match.veto) == (0.0, 1.0, "row turns")


def test_the_veto_named_is_the_best_labels() -> None:
    match = label_match(
        ["Inventories", "Changes in inventories of finished goods"],
        None,
        "Changes in inventories of finished goods",
    )
    assert (match.share, match.veto) == (1.0, None), match


_FACE_VALUE = CROSSCHECK_LABEL_FIGURES["FaceValueOfEquityShareCapital"][0]


@pytest.mark.parametrize(
    ("text", "figure"),
    [
        pytest.param(
            "Paid-up equity share capital (Face value of ₹10/- each)", 10.0, id="face value"
        ),
        pytest.param("(Face value Rs. 2 each)", 2.0, id="Rs."),
        pytest.param("Paid-up equity share capital (Face value oft 10 each)", 10.0, id="OCR's oft"),
        pytest.param("(Ordinary Shares of, 1/- each)", 1.0, id="ITC's ordinary shares"),
        pytest.param("Paid up equity share capital (Par value 2 per share)", 2.0, id="par value"),
        pytest.param("(Equity shares of par value ~2/- each)", 2.0, id="par value, a glyph"),
        pytest.param("Earnings per share (Face value of ₹10)", 10.0, id="closed by a bracket"),
        pytest.param("(Face value of share : ₹1/- each)", 1.0, id="of share :"),
        pytest.param("Paid up equity share capital (Par value U per share)", None, id="unread"),
        pytest.param("Face value per share ₹", None, id="no figure in the label"),
        pytest.param("2,386 Equity Shares of 10 each held", None, id="shares of, no sign"),
    ],
)
def test_label_figure(text: str, figure: float | None) -> None:
    found = label_figure(_FACE_VALUE, text)
    assert (found[0] if found else None) == figure, found

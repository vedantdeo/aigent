"""Settings for the facts database: which filings are fetched from NSE, how politely, and how the
build checks what it loaded. Split from `config` by concern."""

from __future__ import annotations

# What is fetched: every quarter's results, consolidated and standalone, in these financial years.
FACTS_TICKERS: tuple[str, ...] = (
    "BAJAJ-AUTO",
    "HDFCBANK",
    "ITC",
    "POWERGRID",
    "RELIANCE",
    "SUNPHARMA",
    "TATAMOTORS",
    "WIPRO",
)
FACTS_YEARS: tuple[int, ...] = (2024, 2025, 2026)  # the year a FY ends: 2025 is FY 2024-25
NSE_SYMBOLS: dict[str, str] = {"TATAMOTORS": "TMPV"}  # files under its post-demerger symbol

# NSE: results up to Dec 2024 are in the old listing, SEBI's integrated filing from Mar 2025.
NSE_LEGACY_LISTING = (
    "https://www.nseindia.com/api/corporates-financial-results"
    "?index=equities&symbol={symbol}&period=Quarterly"
)
NSE_INTEGRATED_LISTING = (
    "https://www.nseindia.com/api/integrated-filing-results"
    "?index=equities&symbol={symbol}&size={size}&page={page}"
)
NSE_PAGE_SIZE = 100
NSE_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)
NSE_REQUEST_INTERVAL_S = 1.0  # at most one request a second
NSE_TIMEOUT_S = 30.0

# Loading: contexts whose id holds one of these are not results-format facts and are skipped.
XBRL_SKIP_CONTEXTS: tuple[str, ...] = ("RelatedPartyTransaction",)

# The build's checks: each fundamentals metric's XBRL concept (dividend per share is not filed).
FUNDAMENTALS_CONCEPTS: dict[str, str] = {
    "eps": "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
    "profit": "ProfitOrLossAttributableToOwnersOfParent",
    "revenue": "RevenueFromOperations",
}
# A filing's own sums: the total, the terms it needs, and the terms counted as 0 when not filed.
FILING_SUMS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "Assets",
        ("CurrentAssets", "NoncurrentAssets"),
        (
            "NoncurrentAssetsClassifiedAsHeldForSale",
            "RegulatoryDeferralAccountDebitBalancesAndRelatedDeferredTaxAssets",
        ),
    ),
    ("EquityAndLiabilities", ("Assets",), ()),
    ("Income", ("OtherIncome", "RevenueFromOperations"), ()),
)

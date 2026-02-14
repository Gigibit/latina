from __future__ import annotations

import os
import re
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

YAHOO_MOST_ACTIVE_URL = "https://finance.yahoo.com/most-active"


class _TickerLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tickers: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return

        attr_map = dict(attrs)
        href = attr_map.get("href") or ""
        match = re.match(r"/quote/([A-Z.\-]{1,10})(?:/|\?|$)", href)
        if match:
            ticker = match.group(1)
            if ticker not in self.tickers:
                self.tickers.append(ticker)


def _download_html(url: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
            )
        },
    )
    with urlopen(request, timeout=10) as response:  # noqa: S310
        return response.read().decode("utf-8", errors="ignore")


def scrape_yahoo_most_active(limit: int = 10) -> list[str]:
    html = _download_html(YAHOO_MOST_ACTIVE_URL)
    parser = _TickerLinkParser()
    parser.feed(html)

    if not parser.tickers:
        raise ValueError("No tickers found from Yahoo most-active page.")

    return parser.tickers[:limit]


def get_web_candidates(limit: int = 10) -> list[str]:
    fallback = [
        item.strip().upper()
        for item in os.getenv("DEFAULT_CANDIDATES", "AAPL,MSFT,NVDA,SPY,GOOGL,AMZN").split(",")
        if item.strip()
    ]

    try:
        return scrape_yahoo_most_active(limit=limit)
    except (HTTPError, URLError, TimeoutError, ValueError):
        return fallback[:limit]

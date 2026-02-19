from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class MarketSnapshot:
    symbol: str
    latest_close: float
    pct_change_5d: float
    pct_change_20d: float
    avg_volume_20d: float
    latest_volume: float


def fetch_trending_symbols(region: str = "US", limit: int = 10) -> list[str]:
    api_url = f"https://query1.finance.yahoo.com/v1/finance/trending/{region.upper()}"
    request = Request(api_url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError("Unable to fetch trending symbols from Yahoo Finance.") from exc

    quotes = payload.get("finance", {}).get("result", [{}])[0].get("quotes", [])
    symbols: list[str] = []
    for quote in quotes:
        symbol = quote.get("symbol")
        if symbol and symbol not in symbols:
            symbols.append(symbol)

    if not symbols:
        raise RuntimeError("Yahoo Finance returned no trending symbols.")

    return symbols[: max(limit, 1)]


def get_market_snapshot(symbol: str, lookback_days: int = 90) -> MarketSnapshot:
    history = get_candle_history(symbol=symbol, candle_size="1d", lookback_candles=lookback_days)
    if len(history) < 21:
        raise ValueError(f"Not enough data found for symbol '{symbol}'.")

    closes = history["Close"]
    volumes = history["Volume"]

    latest_close = float(closes.iloc[-1])
    pct_change_5d = float((closes.iloc[-1] / closes.iloc[-6] - 1) * 100)
    pct_change_20d = float((closes.iloc[-1] / closes.iloc[-21] - 1) * 100)

    avg_volume_20d = float(volumes.tail(20).mean())
    latest_volume = float(volumes.iloc[-1])

    return MarketSnapshot(
        symbol=symbol.upper(),
        latest_close=latest_close,
        pct_change_5d=pct_change_5d,
        pct_change_20d=pct_change_20d,
        avg_volume_20d=avg_volume_20d,
        latest_volume=latest_volume,
    )


def resolve_candle_size(candle_size: str) -> tuple[str, int]:
    normalized = candle_size.strip()
    mapping = {
        "24h": ("1h", 24),
        "1d": ("1d", 1),
        "7d": ("1d", 7),
        "1M": ("1d", 30),
    }
    if normalized not in mapping:
        raise ValueError("CANDLE_SIZE must be one of: 24h, 1d, 7d, 1M")
    return mapping[normalized]


def get_candle_history(symbol: str, candle_size: str = "1d", lookback_candles: int = 180):
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed. Install dependencies from requirements.txt"
        ) from exc

    interval, candle_span = resolve_candle_size(candle_size)
    history_length = max(lookback_candles * candle_span, 60)

    ticker = yf.Ticker(symbol)
    history = ticker.history(period=f"{history_length}d", interval=interval).dropna()

    if history.empty:
        raise ValueError(f"Not enough data found for symbol '{symbol}'.")

    return history

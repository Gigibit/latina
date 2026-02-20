from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from io import StringIO
from statistics import mean
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen


@dataclass
class MarketSnapshot:
    symbol: str
    latest_close: float
    pct_change_5d: float
    pct_change_20d: float
    avg_volume_20d: float
    latest_volume: float


@dataclass
class TechnicalIndicators:
    sma_20: float
    sma_50: float
    rsi_14: float
    macd: float
    macd_signal: float
    bollinger_upper: float
    bollinger_lower: float


@dataclass
class FundamentalMetrics:
    pe_ratio: float | None
    eps: float | None
    debt_to_equity: float | None
    market_cap: float | None


@dataclass
class MacroIndicator:
    series: str
    latest_value: float
    previous_value: float
    delta: float


def fetch_trending_symbols(region: str = "US", limit: int = 10) -> list[str]:
    api_url = f"https://query1.finance.yahoo.com/v1/finance/trending/{region.upper()}"
    request = Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    retry_enabled = _is_env_flag_enabled("RETRY_BACKOFFF_ENABLED", default=True)
    max_attempts = 3 if retry_enabled else 1

    payload: dict[str, object] | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with urlopen(request, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            is_rate_limited = exc.code == 429
            should_retry = retry_enabled and is_rate_limited and attempt < max_attempts
            if should_retry:
                _sleep_with_exponential_backoff(attempt)
                continue
            raise RuntimeError("Unable to fetch trending symbols from Yahoo Finance.") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError("Unable to fetch trending symbols from Yahoo Finance.") from exc

    if payload is None:
        raise RuntimeError("Unable to fetch trending symbols from Yahoo Finance.")

    quotes = payload.get("finance", {}).get("result", [{}])[0].get("quotes", [])
    symbols: list[str] = []
    for quote in quotes:
        symbol = quote.get("symbol")
        if symbol and symbol not in symbols:
            symbols.append(symbol)

    if not symbols:
        raise RuntimeError("Yahoo Finance returned no trending symbols.")

    return symbols[: max(limit, 1)]


def _is_env_flag_enabled(name: str, default: bool = True) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _sleep_with_exponential_backoff(attempt: int) -> None:
    delay_seconds = min(2 ** (attempt - 1), 8)
    time.sleep(delay_seconds)


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


def resolve_candle_size(candle_size: str) -> tuple[str, float]:
    normalized = candle_size.strip()
    mapping = {
        "1h": ("1h", 1 / 24),
        "24h": ("1h", 24),
        "1d": ("1d", 1),
        "7d": ("1d", 7),
        "1M": ("1d", 30),
    }
    if normalized not in mapping:
        raise ValueError("CANDLE_SIZE must be one of: 1h, 24h, 1d, 7d, 1M")
    return mapping[normalized]


def get_candle_history(symbol: str, candle_size: str = "1d", lookback_candles: int = 180):
    provider = os.getenv("MARKETS_DATA_PROVIDER", "yfinance").strip().lower()

    if provider == "yfinance":
        return _get_yfinance_candle_history(
            symbol=symbol,
            candle_size=candle_size,
            lookback_candles=lookback_candles,
        )

    if provider == "stooq":
        return _get_stooq_candle_history(
            symbol=symbol,
            candle_size=candle_size,
            lookback_candles=lookback_candles,
        )

    raise ValueError("MARKETS_DATA_PROVIDER must be one of: yfinance, stooq")


def _get_yfinance_candle_history(
    symbol: str, candle_size: str = "1d", lookback_candles: int = 180
):
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed. Install dependencies from requirements.txt"
        ) from exc

    interval, candle_span = resolve_candle_size(candle_size)
    history_length = max(int(lookback_candles * candle_span) + 5, 60)

    ticker = yf.Ticker(symbol)
    history = ticker.history(period=f"{history_length}d", interval=interval).dropna()

    if history.empty:
        raise ValueError(f"Not enough data found for symbol '{symbol}'.")

    return history


def _get_stooq_candle_history(symbol: str, candle_size: str = "1d", lookback_candles: int = 180):
    import pandas as pd

    interval, candle_span = resolve_candle_size(candle_size)
    if interval != "1d":
        raise ValueError("Stooq only supports daily candles for now (1d, 7d, 1M).")

    history_length = max(lookback_candles * candle_span, 60)
    stooq_symbol = _normalize_stooq_symbol(symbol)
    csv_url = (
        "https://stooq.com/q/d/l/?"
        f"s={quote_plus(stooq_symbol)}&i=d"
    )
    request = Request(csv_url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urlopen(request, timeout=8) as response:
            csv_payload = response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError("Unable to fetch candles from Stooq.") from exc

    frame = pd.read_csv(StringIO(csv_payload)).dropna()
    if frame.empty:
        raise ValueError(f"Not enough data found for symbol '{symbol}'.")

    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date"]).set_index("Date")

    frame = frame.sort_index().tail(history_length)
    if frame.empty:
        raise ValueError(f"Not enough data found for symbol '{symbol}'.")

    return frame


def _normalize_stooq_symbol(symbol: str) -> str:
    cleaned = symbol.strip().lower()
    if "." in cleaned:
        return cleaned
    return f"{cleaned}.us"


def fetch_x_sentiment_scores(symbol: str, days: list[date]) -> dict[date, float]:
    """Return sentiment scores for each day in range [-1, 1].

    This is a lightweight placeholder implementation that can be replaced
    with a real X/Twitter sentiment provider integration.
    """

    unique_days = sorted(set(days))
    if not unique_days:
        return {}

    symbol_seed = sum(ord(ch) for ch in symbol.upper())
    scores: dict[date, float] = {}

    for day in unique_days:
        day_seed = day.toordinal() + symbol_seed
        normalized = ((day_seed % 200) - 100) / 100
        scores[day] = round(normalized, 4)

    return scores


def compute_technical_indicators(symbol: str, lookback_days: int = 180) -> TechnicalIndicators:
    history = get_candle_history(symbol=symbol, candle_size="1d", lookback_candles=lookback_days)
    if len(history) < 60:
        raise ValueError(f"Not enough data found for symbol '{symbol}' to compute indicators.")

    closes = history["Close"].astype(float)
    sma_20 = float(closes.rolling(window=20).mean().iloc[-1])
    sma_50 = float(closes.rolling(window=50).mean().iloc[-1])

    delta = closes.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=14).mean().iloc[-1]
    avg_loss = losses.rolling(window=14).mean().iloc[-1]
    if avg_loss == 0:
        rsi_14 = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_14 = float(100 - (100 / (1 + rs)))

    ema_12 = closes.ewm(span=12, adjust=False).mean()
    ema_26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    rolling_std = closes.rolling(window=20).std()
    bollinger_mid = closes.rolling(window=20).mean()
    upper = bollinger_mid + (rolling_std * 2)
    lower = bollinger_mid - (rolling_std * 2)

    return TechnicalIndicators(
        sma_20=sma_20,
        sma_50=sma_50,
        rsi_14=rsi_14,
        macd=float(macd_line.iloc[-1]),
        macd_signal=float(signal_line.iloc[-1]),
        bollinger_upper=float(upper.iloc[-1]),
        bollinger_lower=float(lower.iloc[-1]),
    )


def fetch_fundamental_metrics(symbol: str) -> FundamentalMetrics:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed. Install dependencies from requirements.txt"
        ) from exc

    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    return FundamentalMetrics(
        pe_ratio=_safe_float(info.get("forwardPE") or info.get("trailingPE")),
        eps=_safe_float(info.get("trailingEps") or info.get("epsCurrentYear")),
        debt_to_equity=_safe_float(info.get("debtToEquity")),
        market_cap=_safe_float(info.get("marketCap")),
    )


def fetch_macro_indicators() -> list[MacroIndicator]:
    series_codes = {
        "US_CPI": "CPIAUCSL",
        "US_UNEMPLOYMENT": "UNRATE",
        "US_10Y_TREASURY": "DGS10",
    }
    indicators: list[MacroIndicator] = []
    for series_name, fred_code in series_codes.items():
        values = _fetch_fred_series(fred_code)
        if len(values) < 2:
            continue
        latest, previous = values[-1], values[-2]
        indicators.append(
            MacroIndicator(
                series=series_name,
                latest_value=latest,
                previous_value=previous,
                delta=latest - previous,
            )
        )
    return indicators


def fetch_market_news(limit: int = 5) -> list[dict[str, str]]:
    feeds = [
        "https://www.marketwatch.com/rss/topstories",
        "https://www.nasdaq.com/feed/rssoutbound?category=Markets",
    ]
    headlines: list[dict[str, str]] = []
    for feed in feeds:
        try:
            request = Request(feed, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=8) as response:
                payload = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError):
            continue

        try:
            root = ET.fromstring(payload)
        except ET.ParseError:
            continue

        for item in root.findall(".//item"):
            title = item.findtext("title")
            link = item.findtext("link")
            if title and link:
                headlines.append({"title": title.strip(), "link": link.strip()})
            if len(headlines) >= max(limit, 1):
                return headlines
    return headlines[: max(limit, 1)]


def average_macro_delta(indicators: list[MacroIndicator]) -> float:
    if not indicators:
        return 0.0
    return float(mean(abs(item.delta) for item in indicators))


def _fetch_fred_series(series_code: str) -> list[float]:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_code}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=8) as response:
            lines = response.read().decode("utf-8").splitlines()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Unable to fetch macro series '{series_code}'.") from exc

    values: list[float] = []
    for raw_line in lines[1:]:
        try:
            _, raw_value = raw_line.split(",", maxsplit=1)
        except ValueError:
            continue
        cleaned = raw_value.strip()
        if cleaned == ".":
            continue
        try:
            values.append(float(cleaned))
        except ValueError:
            continue
    return values


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

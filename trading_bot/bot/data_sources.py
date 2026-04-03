from __future__ import annotations

import json
import logging
import math
import os
import ssl
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from io import StringIO
from statistics import mean
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


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


@dataclass
class CryptoTicker:
    symbol: str
    last_price: float
    change_pct_24h: float
    volume_base: float
    volume_quote: float


def fetch_trending_symbols(region: str = "US", limit: int = 10) -> list[str]:
    requested_count = _get_env_int("TRENDING_CANDIDATES_SEARCH_NUMBER", default=max(limit, 1))
    provider = _get_trending_provider()
    logger.info(
        "Trending symbols provider selected provider=%s requested_count=%s",
        provider,
        requested_count,
    )
    return get_trending_tickers(
        provider=provider,
        count=requested_count,
        region=region,
        limit=limit,
    )


def get_trending_tickers(
    provider: str,
    count: int,
    region: str = "US",
    limit: int = 10,
) -> list[str]:
    normalized_provider = provider.strip().upper()
    if normalized_provider == "YFINANCE":
        return get_trending_tickers_from_yahoo_finance(region=region, count=count, limit=limit)
    if normalized_provider == "ETORO":
        return get_trending_tickers_from_etoro(limit=limit)
    message = (
        "MARKET_TRENDING_TICKERS_PROVIDER must be one of: YFINANCE, ETORO "
        f"(got: {normalized_provider or '<empty>'})"
    )
    logger.error(message)
    raise ValueError(message)


def get_trending_tickers_from_yahoo_finance(
    *,
    region: str = "US",
    count: int,
    limit: int = 10,
) -> list[str]:
    api_url = f"https://query1.finance.yahoo.com/v1/finance/trending/{region.upper()}?count={count}"
    request = Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    retry_enabled = _is_env_flag_enabled("RETRY_BACKOFFF_ENABLED", default=True)
    max_attempts = 3 if retry_enabled else 1

    payload: dict[str, object] | None = None
    for attempt in range(1, max_attempts + 1):
        logger.info(
            "External request service=yahoo_finance endpoint=trending_symbols url=%s attempt=%s/%s",
            api_url,
            attempt,
            max_attempts,
        )
        try:
            with urlopen(request, timeout=8) as response:
                raw_payload = response.read().decode("utf-8")
                payload = json.loads(raw_payload)
                logger.info(
                    "External response service=yahoo_finance endpoint=trending_symbols "
                    "status=%s bytes=%s",
                    getattr(response, "status", "n/a"),
                    len(raw_payload),
                )
            break
        except HTTPError as exc:
            is_rate_limited = exc.code == 429
            should_retry = retry_enabled and is_rate_limited and attempt < max_attempts
            logger.warning(
                "External response service=yahoo_finance endpoint=trending_symbols "
                "status=%s rate_limited=%s retry=%s",
                exc.code,
                is_rate_limited,
                should_retry,
            )
            if should_retry:
                _sleep_with_exponential_backoff(attempt)
                continue
            logger.error(
                "Unable to fetch trending symbols from Yahoo Finance status=%s url=%s",
                exc.code,
                api_url,
            )
            raise RuntimeError("Unable to fetch trending symbols from Yahoo Finance.") from exc
        except (URLError, TimeoutError) as exc:
            logger.error(
                "Unable to fetch trending symbols from Yahoo Finance url=%s error=%s",
                api_url,
                exc,
            )
            raise RuntimeError("Unable to fetch trending symbols from Yahoo Finance.") from exc

    if payload is None:
        logger.error(
            "Unable to fetch trending symbols from Yahoo Finance: empty payload url=%s",
            api_url,
        )
        raise RuntimeError("Unable to fetch trending symbols from Yahoo Finance.")

    finance = payload.get("finance")
    results = finance.get("result") if isinstance(finance, dict) else None
    first_result = results[0] if isinstance(results, list) and results else {}
    quotes = first_result.get("quotes", []) if isinstance(first_result, dict) else []
    symbols: list[str] = []
    for quote in quotes:
        symbol = quote.get("symbol")
        if symbol and symbol not in symbols:
            symbols.append(symbol)

    if not symbols:
        logger.error(
            "Yahoo Finance returned no trending symbols region=%s count=%s",
            region.upper(),
            count,
        )
        raise RuntimeError("Yahoo Finance returned no trending symbols.")

    return symbols[: max(limit, 1)]


def get_trending_tickers_from_etoro(*, limit: int = 10) -> list[str]:
    api_key = os.getenv("ETORO_API_KEY", "").strip()
    user_key = os.getenv("ETORO_USER_KEY", "").strip()
    if not api_key or not user_key:
        message = (
            "Unable to fetch trending symbols from eToro: "
            "ETORO_API_KEY and ETORO_USER_KEY are required "
            "when MARKET_TRENDING_TICKERS_PROVIDER=ETORO."
        )
        logger.error(message)
        raise RuntimeError(message)

    api_url = "https://public-api.etoro.com/api/v1/watchlists"
    headers = {
        "Accept": "application/json",
        "X-Request-Id": os.getenv("ETORO_REQUEST_ID", str(uuid.uuid4())),
        "X-Api-Key": api_key,
        "X-User-Key": user_key,
        "User-Agent": "Mozilla/5.0",
    }
    retry_enabled = _is_env_flag_enabled("RETRY_BACKOFFF_ENABLED", default=True)
    max_attempts = 3 if retry_enabled else 1
    payload: object | None = None
    last_error: Exception | None = None
    request = Request(api_url, headers=headers, method="GET")
    for attempt in range(1, max_attempts + 1):
        logger.info(
            "External request service=etoro endpoint=watchlists provider=ETORO "
            "attempt=%s/%s url=%s",
            attempt,
            max_attempts,
            api_url,
        )
        try:
            with urlopen(request, timeout=8) as response:
                raw_payload = response.read().decode("utf-8")
                try:
                    payload = json.loads(raw_payload)
                except json.JSONDecodeError as exc:
                    logger.error(
                        "Unable to parse eToro watchlists JSON url=%s error=%s body=%s",
                        api_url,
                        exc,
                        raw_payload[:500],
                    )
                    raise RuntimeError(
                        "Unable to parse trending symbols response from eToro."
                    ) from exc
                logger.info(
                    "External response service=etoro endpoint=watchlists "
                    "status=%s bytes=%s",
                    getattr(response, "status", "n/a"),
                    len(raw_payload),
                )
            break
        except HTTPError as exc:
            error_payload = exc.read().decode("utf-8", errors="replace")
            is_rate_limited = exc.code == 429
            should_retry = retry_enabled and is_rate_limited and attempt < max_attempts
            logger.warning(
                "External response service=etoro endpoint=watchlists "
                "status=%s rate_limited=%s retry=%s url=%s",
                exc.code,
                is_rate_limited,
                should_retry,
                api_url,
            )
            if should_retry:
                _sleep_with_exponential_backoff(attempt)
                continue
            logger.error(
                "Unable to fetch trending symbols from eToro status=%s url=%s body=%s",
                exc.code,
                api_url,
                error_payload,
            )
            last_error = exc
            break
        except (URLError, TimeoutError) as exc:
            logger.error(
                "Unable to fetch trending symbols from eToro url=%s error=%s",
                api_url,
                exc,
            )
            raise RuntimeError("Unable to fetch trending symbols from eToro.") from exc

    if payload is None:
        logger.error(
            "Unable to fetch trending symbols from eToro: empty payload url=%s",
            api_url,
        )
        raise RuntimeError("Unable to fetch trending symbols from eToro.") from last_error

    symbols = _extract_etoro_symbols(payload)
    logger.info(
        "eToro watchlists payload parsed provider=ETORO symbols_received=%s",
        len(symbols),
    )
    if not symbols:
        logger.error("Unable to resolve any eToro symbolName to symbol.")
        raise RuntimeError("Unable to resolve eToro symbols from symbolName.")
    return symbols[: max(limit, 1)]


def _get_trending_provider() -> str:
    return os.getenv("MARKET_TRENDING_TICKERS_PROVIDER", "YFINANCE").strip().upper() or "YFINANCE"


def _extract_etoro_symbols(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        logger.error("Invalid eToro payload format: expected root object with watchlists.")
        return []

    watchlists = payload.get("watchlists")
    if not isinstance(watchlists, list):
        logger.error("Invalid eToro payload format: missing watchlists array.")
        return []

    symbols: list[str] = []
    unresolved_ids: list[int] = []
    for watchlist in watchlists:
        if not isinstance(watchlist, dict):
            continue
        items = watchlist.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("itemType") or "").strip().lower()
            if item_type != "instrument":
                continue

            market = item.get("market")
            symbol_name = market.get("symbolName") if isinstance(market, dict) else None
            normalized_symbol = (
                str(symbol_name).strip().upper() if isinstance(symbol_name, str) else ""
            )
            if normalized_symbol:
                if normalized_symbol not in symbols:
                    symbols.append(normalized_symbol)
                continue

            instrument_id = item.get("itemId")
            if not isinstance(instrument_id, int) and isinstance(market, dict):
                market_id = market.get("id")
                if isinstance(market_id, str) and market_id.isdigit():
                    instrument_id = int(market_id)
            if (
                isinstance(instrument_id, int)
                and instrument_id > 0
                and instrument_id not in unresolved_ids
            ):
                unresolved_ids.append(instrument_id)
    if unresolved_ids:
        logger.warning(
            "eToro watchlists items missing symbolName unresolved_instrument_ids=%s",
            unresolved_ids,
        )
    return symbols


def _is_env_flag_enabled(name: str, default: bool = True) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _get_env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return max(default, 1)

    try:
        return max(int(raw_value), 1)
    except ValueError:
        logger.warning("Invalid integer for %s=%s. Falling back to %s.", name, raw_value, default)
        return max(default, 1)


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
    interval, _ = resolve_candle_size(candle_size)
    stooq_supported = interval == "1d"

    if provider == "yfinance":
        providers_chain = ["yfinance"]
        if stooq_supported and _is_env_flag_enabled(
            "YFINANCE_FALLBACK_TO_STOOQ_ENABLED",
            default=True,
        ):
            providers_chain.append("stooq")
    elif provider == "stooq":
        if not stooq_supported:
            logger.error(
                "Candle history invalid provider selection symbol=%s provider=stooq "
                "candle_size=%s error=Stooq only supports daily candles for now (1d, 7d, 1M).",
                symbol.upper(),
                candle_size,
            )
            raise ValueError("Stooq only supports daily candles for now (1d, 7d, 1M).")
        providers_chain = ["stooq"]
        if _is_env_flag_enabled("STOOQ_FALLBACK_TO_YFINANCE_ENABLED", default=False):
            providers_chain.append("yfinance")
    elif provider == "alpha_vantage":
        return _get_alpha_vantage_candle_history(
            symbol=symbol,
            candle_size=candle_size,
            lookback_candles=lookback_candles,
        )
    elif provider == "massive":
        return _get_massive_candle_history(
            symbol=symbol,
            candle_size=candle_size,
            lookback_candles=lookback_candles,
        )
    elif provider == "etoro":
        return _get_etoro_candle_history(
            symbol=symbol,
            candle_size=candle_size,
            lookback_candles=lookback_candles,
        )
    else:
        logger.error(
            "Candle history invalid provider selection symbol=%s provider=%s",
            symbol.upper(),
            provider,
        )
        raise ValueError(
            "MARKETS_DATA_PROVIDER must be one of: yfinance, stooq, alpha_vantage, massive, etoro"
        )

    logger.info(
        "Candle history request symbol=%s candle_size=%s lookback=%s providers_chain=%s",
        symbol.upper(),
        candle_size,
        lookback_candles,
        "->".join(providers_chain),
    )

    last_exception: Exception | None = None
    for index, provider_name in enumerate(providers_chain, start=1):
        logger.info(
            "Candle history provider attempt symbol=%s provider=%s attempt=%s/%s",
            symbol.upper(),
            provider_name,
            index,
            len(providers_chain),
        )
        try:
            if provider_name == "yfinance":
                return _get_yfinance_candle_history(
                    symbol=symbol,
                    candle_size=candle_size,
                    lookback_candles=lookback_candles,
                )
            return _get_stooq_candle_history(
                symbol=symbol,
                candle_size=candle_size,
                lookback_candles=lookback_candles,
            )
        except Exception as exc:
            last_exception = exc
            has_fallback = index < len(providers_chain)
            if has_fallback:
                logger.warning(
                    "Candle history provider failed symbol=%s provider=%s "
                    "error=%s fallback_remaining=%s",
                    symbol.upper(),
                    provider_name,
                    exc,
                    has_fallback,
                )
            else:
                logger.error(
                    "Candle history provider failed symbol=%s provider=%s "
                    "error=%s fallback_remaining=%s",
                    symbol.upper(),
                    provider_name,
                    exc,
                    has_fallback,
                )
            if not has_fallback:
                raise

    if last_exception is not None:
        raise last_exception

    raise RuntimeError("Unable to fetch candle history.")


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

    logger.info(
        "External request service=yfinance endpoint=history symbol=%s period_days=%s interval=%s",
        symbol.upper(),
        history_length,
        interval,
    )
    ticker = yf.Ticker(symbol)
    try:
        history = ticker.history(period=f"{history_length}d", interval=interval).dropna()
    except Exception:
        logger.exception(
            "External response service=yfinance endpoint=history symbol=%s failed",
            symbol.upper(),
        )
        raise

    logger.info(
        "External response service=yfinance endpoint=history symbol=%s rows=%s",
        symbol.upper(),
        len(history),
    )

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
    logger.info(
        "External request service=stooq endpoint=history symbol=%s url=%s",
        stooq_symbol,
        csv_url,
    )

    stooq_timeout_seconds = _get_env_int("STOOQ_CR_TIMEOUT", default=8)
    retry_enabled = _is_env_flag_enabled("RETRY_BACKOFFF_ENABLED", default=True)
    max_attempts = 3 if retry_enabled else 1

    csv_payload: str | None = None
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with urlopen(request, timeout=stooq_timeout_seconds) as response:
                csv_payload = response.read().decode("utf-8")
            break
        except HTTPError as exc:
            last_error = exc
            should_retry = retry_enabled and exc.code >= 500 and attempt < max_attempts
            logger.warning(
                "External response service=stooq endpoint=history symbol=%s "
                "status=%s retry=%s attempt=%s/%s",
                stooq_symbol,
                exc.code,
                should_retry,
                attempt,
                max_attempts,
            )
        except (URLError, TimeoutError, ssl.SSLError) as exc:
            last_error = exc
            should_retry = (
                retry_enabled
                and _is_transient_network_error(exc)
                and attempt < max_attempts
            )
            logger.warning(
                "External response service=stooq endpoint=history symbol=%s "
                "error=%s retry=%s attempt=%s/%s",
                stooq_symbol,
                exc,
                should_retry,
                attempt,
                max_attempts,
            )

        if should_retry:
            _sleep_with_exponential_backoff(attempt)
            continue
        message = "Unable to fetch candles from Stooq."
        if os.getenv("MARKETS_DATA_PROVIDER", "").strip().lower() == "yfinance":
            message += (
                " If you are using MARKETS_DATA_PROVIDER=yfinance, you can disable fallback "
                "with YFINANCE_FALLBACK_TO_STOOQ_ENABLED=false."
            )
        raise RuntimeError(message) from last_error

    if csv_payload is None:
        raise RuntimeError("Unable to fetch candles from Stooq.") from last_error

    logger.info(
        "External response service=stooq endpoint=history symbol=%s bytes=%s",
        stooq_symbol,
        len(csv_payload),
    )

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


def _is_transient_network_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, ssl.SSLError):
        return True
    if isinstance(exc, URLError):
        reason = exc.reason
        if isinstance(reason, TimeoutError | ssl.SSLError):
            return True
        if isinstance(reason, str):
            lowered_reason = reason.lower()
            if (
                "timed out" in lowered_reason
                or "unexpected eof" in lowered_reason
                or "unexpected_eof" in lowered_reason
                or "eof occurred" in lowered_reason
            ):
                return True

    lowered_error = str(exc).lower()
    return (
        "timed out" in lowered_error
        or "unexpected eof" in lowered_error
        or "unexpected_eof" in lowered_error
        or "eof occurred" in lowered_error
    )


def _normalize_stooq_symbol(symbol: str) -> str:
    cleaned = symbol.strip().lower()
    if "." in cleaned:
        return cleaned
    return f"{cleaned}.us"


def _get_alpha_vantage_candle_history(
    symbol: str, candle_size: str = "1d", lookback_candles: int = 180
):
    import pandas as pd

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    if not api_key:
        logger.error(
            "Alpha Vantage candle history missing API key symbol=%s provider=alpha_vantage",
            symbol.upper(),
        )
        raise RuntimeError(
            "ALPHA_VANTAGE_API_KEY is required when MARKETS_DATA_PROVIDER=alpha_vantage"
        )

    interval, candle_span = resolve_candle_size(candle_size)
    target_rows = max(int(lookback_candles * candle_span), 1)
    history_days = max(target_rows + 5, 60)
    if interval == "1h":
        target_rows = max(int(lookback_candles), 1)
        market_hours_per_day = 6.5
        history_days = max(math.ceil(target_rows / market_hours_per_day) + 3, 5)

    if interval == "1h":
        api_url = (
            "https://www.alphavantage.co/query?"
            f"function=TIME_SERIES_INTRADAY&symbol={quote_plus(symbol.upper())}"
            "&interval=60min&outputsize=full"
            f"&apikey={quote_plus(api_key)}"
        )
        series_key = "Time Series (60min)"
    else:
        api_url = (
            "https://www.alphavantage.co/query?"
            f"function=TIME_SERIES_DAILY_ADJUSTED&symbol={quote_plus(symbol.upper())}"
            "&outputsize=full"
            f"&apikey={quote_plus(api_key)}"
        )
        series_key = "Time Series (Daily)"

    request = Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    logger.info(
        "External request service=alpha_vantage endpoint=history symbol=%s "
        "history_days=%s target_rows=%s interval=%s",
        symbol.upper(),
        history_days,
        target_rows,
        interval,
    )
    try:
        with urlopen(request, timeout=8) as response:
            raw_payload = response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError) as exc:
        logger.error(
            "External response service=alpha_vantage endpoint=history symbol=%s error=%s",
            symbol.upper(),
            exc,
        )
        raise RuntimeError("Unable to fetch candles from Alpha Vantage.") from exc

    payload = json.loads(raw_payload)
    if "Error Message" in payload:
        logger.error(
            "External response service=alpha_vantage endpoint=history symbol=%s "
            "error=error_message_from_provider",
            symbol.upper(),
        )
        raise RuntimeError(f"Alpha Vantage error for symbol '{symbol}'.")
    if "Note" in payload:
        logger.error(
            "External response service=alpha_vantage endpoint=history symbol=%s "
            "error=rate_limit",
            symbol.upper(),
        )
        raise RuntimeError("Alpha Vantage rate limit reached. Try again later.")

    raw_series = payload.get(series_key)
    if not isinstance(raw_series, dict) or not raw_series:
        logger.error(
            "External response service=alpha_vantage endpoint=history symbol=%s "
            "missing_series=%s",
            symbol.upper(),
            series_key,
        )
        raise ValueError(f"Not enough data found for symbol '{symbol}'.")

    rows: list[dict[str, object]] = []
    for timestamp, values in raw_series.items():
        if not isinstance(values, dict):
            continue
        try:
            rows.append(
                {
                    "Date": timestamp,
                    "Open": float(values.get("1. open", "nan")),
                    "High": float(values.get("2. high", "nan")),
                    "Low": float(values.get("3. low", "nan")),
                    "Close": float(values.get("4. close", "nan")),
                    "Volume": float(values.get("6. volume") or values.get("5. volume") or "nan"),
                }
            )
        except ValueError:
            continue

    frame = pd.DataFrame(rows).dropna()
    if frame.empty:
        logger.error(
            "External response service=alpha_vantage endpoint=history symbol=%s parsed_rows=0",
            symbol.upper(),
        )
        raise ValueError(f"Not enough data found for symbol '{symbol}'.")

    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).set_index("Date").sort_index()
    if target_rows > 0:
        frame = frame.tail(target_rows)
    if frame.empty:
        logger.error(
            "External response service=alpha_vantage endpoint=history symbol=%s tail_rows=0",
            symbol.upper(),
        )
        raise ValueError(f"Not enough data found for symbol '{symbol}'.")

    logger.info(
        "External response service=alpha_vantage endpoint=history symbol=%s rows=%s",
        symbol.upper(),
        len(frame),
    )
    return frame


def _get_massive_candle_history(
    symbol: str, candle_size: str = "1d", lookback_candles: int = 180
):
    import pandas as pd

    api_key = os.getenv("MASSIVE_API_KEY", "").strip()
    if not api_key:
        logger.error(
            "Massive candle history missing API key symbol=%s provider=massive",
            symbol.upper(),
        )
        raise RuntimeError("MASSIVE_API_KEY is required when MARKETS_DATA_PROVIDER=massive")

    interval, candle_span = resolve_candle_size(candle_size)
    target_rows = max(int(lookback_candles * candle_span), 1)
    history_days = max(target_rows + 5, 60)
    if interval == "1h":
        target_rows = max(int(lookback_candles), 1)
        market_hours_per_day = 6.5
        history_days = max(math.ceil(target_rows / market_hours_per_day) + 3, 5)

    multiplier = 1
    timespan = "day"
    if interval == "1h":
        timespan = "hour"
    end_date = date.today()
    start_date = end_date.fromordinal(end_date.toordinal() - history_days)
    api_url = (
        "https://api.massive.com/v2/aggs/ticker/"
        f"{quote_plus(symbol.upper())}/range/{multiplier}/{timespan}/"
        f"{start_date.isoformat()}/{end_date.isoformat()}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={quote_plus(api_key)}"
    )
    request = Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    logger.info(
        "External request service=massive endpoint=history symbol=%s history_days=%s "
        "target_rows=%s interval=%s",
        symbol.upper(),
        history_days,
        target_rows,
        interval,
    )

    try:
        with urlopen(request, timeout=8) as response:
            raw_payload = response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError) as exc:
        logger.error(
            "External response service=massive endpoint=history symbol=%s error=%s",
            symbol.upper(),
            exc,
        )
        raise RuntimeError("Unable to fetch candles from Massive.") from exc

    payload = json.loads(raw_payload)
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        logger.error(
            "External response service=massive endpoint=history symbol=%s missing_results=true",
            symbol.upper(),
        )
        raise ValueError(f"Not enough data found for symbol '{symbol}'.")

    rows: list[dict[str, object]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        timestamp = item.get("t")
        open_value = item.get("o")
        high_value = item.get("h")
        low_value = item.get("l")
        close_value = item.get("c")
        volume_value = item.get("v")
        if None in {timestamp, open_value, high_value, low_value, close_value, volume_value}:
            continue
        rows.append(
            {
                "Date": pd.to_datetime(int(timestamp), unit="ms", utc=True).tz_localize(None),
                "Open": float(open_value),
                "High": float(high_value),
                "Low": float(low_value),
                "Close": float(close_value),
                "Volume": float(volume_value),
            }
        )

    frame = pd.DataFrame(rows).dropna()
    if frame.empty:
        logger.error(
            "External response service=massive endpoint=history symbol=%s parsed_rows=0",
            symbol.upper(),
        )
        raise ValueError(f"Not enough data found for symbol '{symbol}'.")
    frame = frame.set_index("Date").sort_index()
    if target_rows > 0:
        frame = frame.tail(target_rows)
    if frame.empty:
        logger.error(
            "External response service=massive endpoint=history symbol=%s tail_rows=0",
            symbol.upper(),
        )
        raise ValueError(f"Not enough data found for symbol '{symbol}'.")

    logger.info(
        "External response service=massive endpoint=history symbol=%s rows=%s",
        symbol.upper(),
        len(frame),
    )
    return frame


def _get_etoro_candle_history(
    symbol: str, candle_size: str = "1d", lookback_candles: int = 180
):
    import pandas as pd

    api_key = os.getenv("ETORO_API_KEY", "").strip()
    user_key = os.getenv("ETORO_USER_KEY", "").strip()
    if not api_key or not user_key:
        logger.error(
            "eToro candle history missing credentials symbol=%s provider=etoro",
            symbol.upper(),
        )
        raise RuntimeError(
            "ETORO_API_KEY and ETORO_USER_KEY are required when MARKETS_DATA_PROVIDER=etoro"
        )

    interval, candle_span = resolve_candle_size(candle_size)
    interval_map = {
        "1h": "OneHour",
        "1d": "OneDay",
    }
    etoro_interval = interval_map.get(interval)
    if etoro_interval is None:
        logger.error(
            "eToro candle history invalid interval symbol=%s candle_size=%s",
            symbol.upper(),
            candle_size,
        )
        raise ValueError("eToro only supports candle_size values that map to 1h or 1d intervals.")

    requested_rows = max(int(lookback_candles * candle_span), 1)
    candles_count = min(requested_rows, 1000)
    if candles_count < requested_rows:
        logger.warning(
            "eToro candle history capped candles_count symbol=%s requested_rows=%s max_rows=%s",
            symbol.upper(),
            requested_rows,
            candles_count,
        )

    headers = {
        "Accept": "application/json",
        "X-Request-Id": os.getenv("ETORO_REQUEST_ID", str(uuid.uuid4())),
        "X-Api-Key": api_key,
        "X-User-Key": user_key,
        "User-Agent": "Mozilla/5.0",
    }
    base_url = "https://public-api.etoro.com/api/v1/market-data"
    search_query = urlencode(
        {
            "searchText": symbol.upper(),
            "fields": "instrumentId,symbol,displayname",
            "pageSize": 25,
            "pageNumber": 1,
        }
    )
    search_url = f"{base_url}/search?{search_query}"
    search_request = Request(search_url, headers=headers, method="GET")
    logger.info(
        "External request service=etoro endpoint=market_search symbol=%s url=%s",
        symbol.upper(),
        search_url,
    )

    retry_enabled = _is_env_flag_enabled("RETRY_BACKOFFF_ENABLED", default=True)
    max_attempts = 3 if retry_enabled else 1
    search_payload_raw = ""
    for attempt in range(1, max_attempts + 1):
        try:
            with urlopen(search_request, timeout=8) as response:
                search_payload_raw = response.read().decode("utf-8")
            break
        except HTTPError as exc:
            is_rate_limited = exc.code == 429
            should_retry = retry_enabled and is_rate_limited and attempt < max_attempts
            logger.warning(
                "External response service=etoro endpoint=market_search "
                "symbol=%s status=%s rate_limited=%s retry=%s attempt=%s/%s",
                symbol.upper(),
                exc.code,
                is_rate_limited,
                should_retry,
                attempt,
                max_attempts,
            )
            if should_retry:
                _sleep_with_exponential_backoff(attempt)
                continue
            logger.error(
                "External response service=etoro endpoint=market_search symbol=%s error=%s",
                symbol.upper(),
                exc,
            )
            raise RuntimeError("Unable to resolve instrumentId from eToro market search.") from exc
        except (URLError, TimeoutError) as exc:
            logger.error(
                "External response service=etoro endpoint=market_search symbol=%s error=%s",
                symbol.upper(),
                exc,
            )
            raise RuntimeError("Unable to resolve instrumentId from eToro market search.") from exc

    try:
        search_payload = json.loads(search_payload_raw)
    except json.JSONDecodeError as exc:
        logger.error(
            "External response service=etoro endpoint=market_search symbol=%s error=%s body=%s",
            symbol.upper(),
            exc,
            search_payload_raw[:500],
        )
        raise RuntimeError("Unable to parse eToro market search response.") from exc

    items = search_payload.get("items") if isinstance(search_payload, dict) else None
    instrument_id: int | None = None
    if isinstance(items, list):
        normalized_symbol = symbol.upper()
        for item in items:
            if not isinstance(item, dict):
                continue
            item_symbol = str(item.get("symbol") or "").strip().upper()
            candidate_id = item.get("instrumentId")
            if item_symbol == normalized_symbol and isinstance(candidate_id, int):
                instrument_id = candidate_id
                break
        if instrument_id is None:
            for item in items:
                if not isinstance(item, dict):
                    continue
                candidate_id = item.get("instrumentId")
                if isinstance(candidate_id, int):
                    instrument_id = candidate_id
                    break
    if instrument_id is None:
        logger.error(
            "External response service=etoro endpoint=market_search "
            "symbol=%s missing_instrument_id=true",
            symbol.upper(),
        )
        raise RuntimeError(
            f"Unable to resolve instrumentId from eToro market search for '{symbol}'."
        )

    candles_url = (
        f"{base_url}/instruments/{instrument_id}/history/candles/asc/{etoro_interval}/{candles_count}"
    )
    candles_request = Request(candles_url, headers=headers, method="GET")
    logger.info(
        "External request service=etoro endpoint=candles "
        "symbol=%s instrument_id=%s interval=%s rows=%s",
        symbol.upper(),
        instrument_id,
        etoro_interval,
        candles_count,
    )

    candles_raw_payload = ""
    for attempt in range(1, max_attempts + 1):
        try:
            with urlopen(candles_request, timeout=8) as response:
                candles_raw_payload = response.read().decode("utf-8")
            break
        except HTTPError as exc:
            is_rate_limited = exc.code == 429
            should_retry = retry_enabled and is_rate_limited and attempt < max_attempts
            logger.warning(
                "External response service=etoro endpoint=candles "
                "symbol=%s instrument_id=%s status=%s rate_limited=%s retry=%s attempt=%s/%s",
                symbol.upper(),
                instrument_id,
                exc.code,
                is_rate_limited,
                should_retry,
                attempt,
                max_attempts,
            )
            if should_retry:
                _sleep_with_exponential_backoff(attempt)
                continue
            logger.error(
                "External response service=etoro endpoint=candles "
                "symbol=%s instrument_id=%s error=%s",
                symbol.upper(),
                instrument_id,
                exc,
            )
            raise RuntimeError("Unable to fetch candles from eToro.") from exc
        except (URLError, TimeoutError) as exc:
            logger.error(
                "External response service=etoro endpoint=candles "
                "symbol=%s instrument_id=%s error=%s",
                symbol.upper(),
                instrument_id,
                exc,
            )
            raise RuntimeError("Unable to fetch candles from eToro.") from exc

    try:
        payload = json.loads(candles_raw_payload)
    except json.JSONDecodeError as exc:
        logger.error(
            "External response service=etoro endpoint=candles "
            "symbol=%s instrument_id=%s error=%s body=%s",
            symbol.upper(),
            instrument_id,
            exc,
            candles_raw_payload[:500],
        )
        raise RuntimeError("Unable to parse candles response from eToro.") from exc

    grouped_candles = payload.get("candles") if isinstance(payload, dict) else None
    rows: list[dict[str, object]] = []
    if isinstance(grouped_candles, list):
        for group in grouped_candles:
            if not isinstance(group, dict):
                continue
            candles = group.get("candles")
            if not isinstance(candles, list):
                continue
            for candle in candles:
                if not isinstance(candle, dict):
                    continue
                from_date = candle.get("fromDate")
                open_value = candle.get("open")
                high_value = candle.get("high")
                low_value = candle.get("low")
                close_value = candle.get("close")
                volume_value = candle.get("volume")
                if None in {
                    from_date,
                    open_value,
                    high_value,
                    low_value,
                    close_value,
                    volume_value,
                }:
                    continue
                normalized_date = pd.to_datetime(
                    str(from_date),
                    errors="coerce",
                    utc=True,
                ).tz_localize(None)
                rows.append(
                    {
                        "Date": normalized_date,
                        "Open": float(open_value),
                        "High": float(high_value),
                        "Low": float(low_value),
                        "Close": float(close_value),
                        "Volume": float(volume_value),
                    }
                )

    frame = pd.DataFrame(rows).dropna()
    if frame.empty:
        logger.error(
            "External response service=etoro endpoint=candles "
            "symbol=%s instrument_id=%s parsed_rows=0",
            symbol.upper(),
            instrument_id,
        )
        raise ValueError(f"Not enough data found for symbol '{symbol}'.")

    frame = frame.set_index("Date").sort_index()
    if requested_rows > 0:
        frame = frame.tail(requested_rows)
    if frame.empty:
        logger.error(
            "External response service=etoro endpoint=candles "
            "symbol=%s instrument_id=%s tail_rows=0",
            symbol.upper(),
            instrument_id,
        )
        raise ValueError(f"Not enough data found for symbol '{symbol}'.")

    logger.info(
        "External response service=etoro endpoint=candles symbol=%s instrument_id=%s rows=%s",
        symbol.upper(),
        instrument_id,
        len(frame),
    )
    return frame


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
    provider = os.getenv("MARKETS_DATA_PROVIDER", "yfinance").strip().lower()

    if provider == "stooq":
        logger.info(
            "Skipping fundamentals lookup because service=stooq "
            "does not provide fundamentals symbol=%s",
            symbol.upper(),
        )
        return FundamentalMetrics(
            pe_ratio=None,
            eps=None,
            debt_to_equity=None,
            market_cap=None,
        )

    if provider == "alpha_vantage":
        return _fetch_alpha_vantage_fundamentals(symbol)

    if provider == "massive":
        return _fetch_massive_fundamentals(symbol)

    if provider == "etoro":
        logger.info(
            "Skipping fundamentals lookup because service=etoro does not provide "
            "fundamentals endpoint in this application symbol=%s",
            symbol.upper(),
        )
        return FundamentalMetrics(
            pe_ratio=None,
            eps=None,
            debt_to_equity=None,
            market_cap=None,
        )

    if provider != "yfinance":
        logger.error(
            "Fundamentals invalid provider selection symbol=%s provider=%s",
            symbol.upper(),
            provider,
        )
        raise ValueError(
            "MARKETS_DATA_PROVIDER must be one of: yfinance, stooq, alpha_vantage, massive, etoro"
        )

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed. Install dependencies from requirements.txt"
        ) from exc

    logger.info("External request service=yfinance endpoint=fundamentals symbol=%s", symbol.upper())
    ticker = yf.Ticker(symbol)
    try:
        info = ticker.info or {}
    except Exception:
        logger.exception(
            "External response service=yfinance endpoint=fundamentals symbol=%s failed",
            symbol.upper(),
        )
        raise
    logger.info(
        "External response service=yfinance endpoint=fundamentals symbol=%s keys=%s",
        symbol.upper(),
        sorted(info.keys())[:12],
    )
    return FundamentalMetrics(
        pe_ratio=_safe_float(info.get("forwardPE") or info.get("trailingPE")),
        eps=_safe_float(info.get("trailingEps") or info.get("epsCurrentYear")),
        debt_to_equity=_safe_float(info.get("debtToEquity")),
        market_cap=_safe_float(info.get("marketCap")),
    )


def _fetch_alpha_vantage_fundamentals(symbol: str) -> FundamentalMetrics:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ALPHA_VANTAGE_API_KEY is required when MARKETS_DATA_PROVIDER=alpha_vantage"
        )

    api_url = (
        "https://www.alphavantage.co/query?"
        f"function=OVERVIEW&symbol={quote_plus(symbol.upper())}"
        f"&apikey={quote_plus(api_key)}"
    )
    request = Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    logger.info(
        "External request service=alpha_vantage endpoint=fundamentals symbol=%s",
        symbol.upper(),
    )

    try:
        with urlopen(request, timeout=8) as response:
            raw_payload = response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError) as exc:
        logger.warning(
            "External response service=alpha_vantage endpoint=fundamentals symbol=%s error=%s",
            symbol.upper(),
            exc,
        )
        raise RuntimeError("Unable to fetch fundamentals from Alpha Vantage.") from exc

    payload = json.loads(raw_payload)
    if "Error Message" in payload:
        raise RuntimeError(f"Alpha Vantage error for symbol '{symbol}'.")
    if "Note" in payload:
        raise RuntimeError("Alpha Vantage rate limit reached. Try again later.")

    return FundamentalMetrics(
        pe_ratio=_safe_float(payload.get("PERatio")),
        eps=_safe_float(payload.get("EPS")),
        debt_to_equity=_safe_float(payload.get("DebtToEquityRatio")),
        market_cap=_safe_float(payload.get("MarketCapitalization")),
    )


def _fetch_massive_fundamentals(symbol: str) -> FundamentalMetrics:
    api_key = os.getenv("MASSIVE_API_KEY", "").strip()
    if not api_key:
        logger.error(
            "Massive fundamentals missing API key symbol=%s provider=massive",
            symbol.upper(),
        )
        raise RuntimeError("MASSIVE_API_KEY is required when MARKETS_DATA_PROVIDER=massive")

    api_url = (
        "https://api.massive.com/v3/reference/tickers/"
        f"{quote_plus(symbol.upper())}?apiKey={quote_plus(api_key)}"
    )
    request = Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    logger.info("External request service=massive endpoint=fundamentals symbol=%s", symbol.upper())

    try:
        with urlopen(request, timeout=8) as response:
            raw_payload = response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError) as exc:
        logger.error(
            "External response service=massive endpoint=fundamentals symbol=%s error=%s",
            symbol.upper(),
            exc,
        )
        raise RuntimeError("Unable to fetch fundamentals from Massive.") from exc

    payload = json.loads(raw_payload)
    result = payload.get("results")
    if not isinstance(result, dict):
        logger.error(
            "External response service=massive endpoint=fundamentals "
            "symbol=%s missing_results=true",
            symbol.upper(),
        )
        raise RuntimeError(f"Massive fundamentals error for symbol '{symbol}'.")

    return FundamentalMetrics(
        pe_ratio=None,
        eps=None,
        debt_to_equity=None,
        market_cap=_safe_float(result.get("market_cap")),
    )


def fetch_macro_indicators() -> list[MacroIndicator]:
    series_codes = {
        "US_CPI": "CPIAUCSL",
        "US_UNEMPLOYMENT": "UNRATE",
        "US_10Y_TREASURY": "DGS10",
    }
    indicators: list[MacroIndicator] = []
    for series_name, fred_code in series_codes.items():
        try:
            values = _fetch_fred_series(fred_code)
        except RuntimeError as exc:
            logger.warning(
                "Skipping macro indicator series=%s code=%s error=%s",
                series_name,
                fred_code,
                exc,
            )
            continue
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
            logger.info("External request service=rss endpoint=market_news url=%s", feed)
            with urlopen(request, timeout=8) as response:
                payload = response.read().decode("utf-8")
            logger.info(
                "External response service=rss endpoint=market_news url=%s bytes=%s",
                feed,
                len(payload),
            )
        except (HTTPError, URLError, TimeoutError) as exc:
            logger.warning(
                "External response service=rss endpoint=market_news url=%s error=%s",
                feed,
                exc,
            )
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


def fetch_crypto_market_analysis(limit: int = 5) -> dict | None:
    api_key = os.getenv("BINANCE_CRYPTO_API_KEY", "").strip()
    if not api_key:
        return None

    provider = os.getenv("CRYPTO_MARKET_PROVIDER", "binance").strip().lower()
    if provider != "binance":
        raise ValueError("CRYPTO_MARKET_PROVIDER must be set to 'binance'.")

    raw_symbols = os.getenv("CRYPTO_MARKET_SYMBOLS", "BTCUSDT,ETHUSDT,BNBUSDT")
    symbols = [item.strip().upper() for item in raw_symbols.split(",") if item.strip()]
    symbols = symbols[: max(limit, 1)]
    if not symbols:
        raise ValueError("CRYPTO_MARKET_SYMBOLS must include at least one symbol.")

    tickers: list[CryptoTicker] = []
    headers = {"User-Agent": "Mozilla/5.0", "X-MBX-APIKEY": api_key}
    for symbol in symbols:
        api_url = (
            "https://api.binance.com/api/v3/ticker/24hr"
            f"?symbol={quote_plus(symbol)}"
        )
        request = Request(api_url, headers=headers)
        logger.info(
            "External request service=binance endpoint=ticker_24h symbol=%s",
            symbol,
        )
        try:
            with urlopen(request, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            logger.warning(
                "External response service=binance endpoint=ticker_24h symbol=%s error=%s",
                symbol,
                exc,
            )
            continue

        tickers.append(
            CryptoTicker(
                symbol=symbol,
                last_price=float(payload.get("lastPrice", 0.0)),
                change_pct_24h=float(payload.get("priceChangePercent", 0.0)),
                volume_base=float(payload.get("volume", 0.0)),
                volume_quote=float(payload.get("quoteVolume", 0.0)),
            )
        )

    if not tickers:
        raise RuntimeError("Unable to fetch crypto market data from Binance.")

    sorted_by_perf = sorted(tickers, key=lambda ticker: ticker.change_pct_24h, reverse=True)
    avg_change = mean(item.change_pct_24h for item in tickers)
    total_quote_volume = sum(item.volume_quote for item in tickers)
    return {
        "provider": provider,
        "experimental": True,
        "symbols": [item.symbol for item in tickers],
        "avg_change_pct_24h": round(avg_change, 4),
        "total_quote_volume_24h": round(total_quote_volume, 2),
        "top_gainer": sorted_by_perf[0].__dict__,
        "top_loser": sorted_by_perf[-1].__dict__,
        "tickers": [item.__dict__ for item in sorted_by_perf],
    }


def average_macro_delta(indicators: list[MacroIndicator]) -> float:
    if not indicators:
        return 0.0
    return float(mean(abs(item.delta) for item in indicators))


def _fetch_fred_series(series_code: str) -> list[float]:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_code}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    logger.info("External request service=fred endpoint=series code=%s url=%s", series_code, url)
    try:
        with urlopen(request, timeout=8) as response:
            lines = response.read().decode("utf-8").splitlines()
    except (HTTPError, URLError, TimeoutError) as exc:
        logger.warning(
            "External response service=fred endpoint=series code=%s error=%s",
            series_code,
            exc,
        )
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
    logger.info(
        "External response service=fred endpoint=series code=%s points=%s",
        series_code,
        len(values),
    )
    return values


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

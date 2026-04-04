import json
import types
import uuid
from datetime import datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

import pytest

from trading_bot.bot import data_sources


class _DummyResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_fetch_trending_symbols_retries_on_429_by_default(monkeypatch):
    calls = {"count": 0}
    sleeps: list[int] = []

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        if calls["count"] < 3:
            raise HTTPError(
                url="http://test",
                code=429,
                msg="Too Many Requests",
                hdrs=None,
                fp=None,
            )
        return _DummyResponse({"finance": {"result": [{"quotes": [{"symbol": "AAPL"}]}]}})

    monkeypatch.delenv("RETRY_BACKOFFF_ENABLED", raising=False)
    monkeypatch.setenv("TRENDING_CANDIDATES_SEARCH_NUMBER", "12")
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)
    monkeypatch.setattr(data_sources.time, "sleep", lambda seconds: sleeps.append(seconds))

    symbols = data_sources.fetch_trending_symbols(limit=1)

    assert symbols == ["AAPL"]
    assert calls["count"] == 3
    assert sleeps == [1, 2]


def test_fetch_trending_symbols_uses_env_count_in_url(monkeypatch):
    captured = {"url": ""}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return _DummyResponse({"finance": {"result": [{"quotes": [{"symbol": "AAPL"}]}]}})

    monkeypatch.setenv("TRENDING_CANDIDATES_SEARCH_NUMBER", "15")
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)

    data_sources.fetch_trending_symbols(region="us", limit=3)

    assert captured["url"].endswith("/US?count=15")


def test_fetch_trending_symbols_uses_yfinance_provider_by_default(monkeypatch):
    captured = {"url": ""}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return _DummyResponse({"finance": {"result": [{"quotes": [{"symbol": "AAPL"}]}]}})

    monkeypatch.delenv("MARKET_TRENDING_TICKERS_PROVIDER", raising=False)
    monkeypatch.setenv("TRENDING_CANDIDATES_SEARCH_NUMBER", "9")
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)

    symbols = data_sources.fetch_trending_symbols(region="us", limit=2)

    assert symbols == ["AAPL"]
    assert captured["url"] == "https://query1.finance.yahoo.com/v1/finance/trending/US?count=9"


def test_fetch_trending_symbols_uses_etoro_provider(monkeypatch):
    captured = {"url": "", "headers": {}}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        return _DummyResponse(
            {
                "watchlists": [
                    {
                        "items": [
                            {"itemType": "Instrument", "market": {"id": "1", "symbolName": "AAPL"}}
                        ]
                    },
                    {
                        "items": [
                            {"itemType": "Instrument", "market": {"id": "2", "symbolName": "MSFT"}}
                        ]
                    },
                ]
            }
        )

    monkeypatch.setenv("MARKET_TRENDING_TICKERS_PROVIDER", "ETORO")
    monkeypatch.setenv("ETORO_API_KEY", "api-key")
    monkeypatch.setenv("ETORO_USER_KEY", "user-key")
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)

    symbols = data_sources.fetch_trending_symbols(region="US", limit=10)

    assert symbols == ["AAPL", "MSFT"]
    assert captured["url"] == "https://public-api.etoro.com/api/v1/watchlists"
    assert captured["headers"]["X-api-key"] == "api-key"
    assert captured["headers"]["X-user-key"] == "user-key"
    assert "X-request-id" in captured["headers"]
    uuid.UUID(captured["headers"]["X-request-id"])


def test_fetch_trending_symbols_etoro_raises_on_422_without_fallback(monkeypatch):
    captured_urls: list[str] = []

    def fake_urlopen(request, timeout):
        captured_urls.append(request.full_url)
        raise HTTPError(
            url=request.full_url,
            code=422,
            msg="Unprocessable Entity",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setenv("MARKET_TRENDING_TICKERS_PROVIDER", "ETORO")
    monkeypatch.setenv("ETORO_API_KEY", "api-key")
    monkeypatch.setenv("ETORO_USER_KEY", "user-key")
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Unable to fetch trending symbols from eToro"):
        data_sources.fetch_trending_symbols(region="US", limit=10)

    assert captured_urls == ["https://public-api.etoro.com/api/v1/watchlists"]


def test_fetch_trending_symbols_etoro_requires_keys(monkeypatch):
    monkeypatch.setenv("MARKET_TRENDING_TICKERS_PROVIDER", "ETORO")
    monkeypatch.delenv("ETORO_API_KEY", raising=False)
    monkeypatch.delenv("ETORO_USER_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ETORO_API_KEY and ETORO_USER_KEY are required"):
        data_sources.fetch_trending_symbols(limit=2)


def test_fetch_trending_symbols_etoro_raises_when_symbol_name_missing(monkeypatch):
    def fake_urlopen(request, timeout):
        return _DummyResponse(
            {
                "watchlists": [
                    {
                        "items": [
                            {"itemType": "Instrument", "itemId": 1001, "market": {"id": "1001"}}
                        ]
                    }
                ]
            }
        )

    monkeypatch.setenv("MARKET_TRENDING_TICKERS_PROVIDER", "ETORO")
    monkeypatch.setenv("ETORO_API_KEY", "api-key")
    monkeypatch.setenv("ETORO_USER_KEY", "user-key")
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Unable to resolve eToro symbols from symbolName"):
        data_sources.fetch_trending_symbols(limit=2)


def test_fetch_trending_symbols_etoro_ignores_trending_candidates_count(monkeypatch):
    captured = {"url": ""}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return _DummyResponse(
            {
                "watchlists": [
                    {
                        "items": [
                            {"itemType": "Instrument", "market": {"id": "1", "symbolName": "AAPL"}}
                        ]
                    }
                ]
            }
        )

    monkeypatch.setenv("MARKET_TRENDING_TICKERS_PROVIDER", "ETORO")
    monkeypatch.setenv("ETORO_API_KEY", "api-key")
    monkeypatch.setenv("ETORO_USER_KEY", "user-key")
    monkeypatch.setenv("TRENDING_CANDIDATES_SEARCH_NUMBER", "999")
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)

    symbols = data_sources.fetch_trending_symbols(limit=1)

    assert symbols == ["AAPL"]
    assert captured["url"] == "https://public-api.etoro.com/api/v1/watchlists"


def test_fetch_trending_symbols_etoro_accepts_public_api_watchlist_items(monkeypatch):
    def fake_urlopen(request, timeout):
        return _DummyResponse(
            {
                "watchlists": [
                    {
                        "items": [
                            {"itemId": 1, "itemType": "Instrument"},
                            {
                                "itemType": "Instrument",
                                "market": {"id": "2", "symbolName": "NVDA"},
                            },
                            {"itemId": 0, "itemType": "Instrument"},
                            {
                                "itemId": 9999,
                                "itemType": "Portfolio",
                                "market": {"id": "99", "symbolName": "SHOULD_SKIP"},
                            },
                            {
                                "itemId": 3,
                                "itemType": "Instrument",
                                "market": {"id": "3", "symbolName": "AAPL"},
                            },
                            {
                                "itemId": 4,
                                "itemType": "Instrument",
                                "market": {"id": "4", "symbolName": "MSFT"},
                            },
                        ]
                    }
                ]
            }
        )

    monkeypatch.setenv("MARKET_TRENDING_TICKERS_PROVIDER", "ETORO")
    monkeypatch.setenv("ETORO_API_KEY", "api-key")
    monkeypatch.setenv("ETORO_USER_KEY", "user-key")
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)

    symbols = data_sources.fetch_trending_symbols(limit=10)

    assert symbols == ["NVDA", "AAPL", "MSFT"]


def test_fetch_trending_symbols_invalid_provider(monkeypatch):
    monkeypatch.setenv("MARKET_TRENDING_TICKERS_PROVIDER", "UNKNOWN")
    with pytest.raises(ValueError, match="MARKET_TRENDING_TICKERS_PROVIDER must be one of"):
        data_sources.fetch_trending_symbols(limit=1)


def test_fetch_trending_symbols_falls_back_to_limit_when_env_count_invalid(monkeypatch):
    captured = {"url": ""}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return _DummyResponse({"finance": {"result": [{"quotes": [{"symbol": "AAPL"}]}]}})

    monkeypatch.setenv("TRENDING_CANDIDATES_SEARCH_NUMBER", "invalid")
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)

    data_sources.fetch_trending_symbols(region="EU", limit=7)

    assert captured["url"].endswith("/EU?count=7")


def test_fetch_trending_symbols_handles_empty_result_list(monkeypatch):
    def fake_urlopen(request, timeout):
        return _DummyResponse({"finance": {"result": []}})

    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="returned no trending symbols"):
        data_sources.fetch_trending_symbols(limit=1)


def test_fetch_trending_symbols_does_not_retry_when_disabled(monkeypatch):
    def fake_urlopen(request, timeout):
        raise HTTPError(url="http://test", code=429, msg="Too Many Requests", hdrs=None, fp=None)

    monkeypatch.setenv("RETRY_BACKOFFF_ENABLED", "false")
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Unable to fetch trending symbols"):
        data_sources.fetch_trending_symbols(limit=1)


class _DummyCsvResponse:
    def __init__(self, payload: str):
        self._payload = payload

    def read(self):
        return self._payload.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_get_candle_history_stooq_provider(monkeypatch):
    csv_payload = """Date,Open,High,Low,Close,Volume
2024-01-02,10,11,9,10.5,100
2024-01-03,11,12,10,11.5,120
"""

    captured = {"url": ""}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return _DummyCsvResponse(csv_payload)

    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "stooq")
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)

    history = data_sources.get_candle_history(symbol="AAPL", candle_size="1d", lookback_candles=2)

    assert list(history.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(history) == 2
    assert str(history.index[0].date()) == "2024-01-02"
    assert "s=aapl.us" in captured["url"]






def test_get_candle_history_stooq_retries_on_transient_network_error(monkeypatch):
    csv_payload = """Date,Open,High,Low,Close,Volume
2024-01-02,10,11,9,10.5,100
"""

    calls = {"count": 0}
    sleeps: list[int] = []

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        if calls["count"] < 3:
            raise URLError(
                "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol"
            )
        return _DummyCsvResponse(csv_payload)

    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "stooq")
    monkeypatch.delenv("RETRY_BACKOFFF_ENABLED", raising=False)
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)
    monkeypatch.setattr(data_sources.time, "sleep", lambda seconds: sleeps.append(seconds))

    history = data_sources.get_candle_history(symbol="AAPL", candle_size="1d", lookback_candles=1)

    assert len(history) == 1
    assert calls["count"] == 3
    assert sleeps == [1, 2]


def test_get_candle_history_stooq_does_not_retry_when_disabled(monkeypatch):
    calls = {"count": 0}

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        raise URLError(
            "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol"
        )

    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "stooq")
    monkeypatch.setenv("RETRY_BACKOFFF_ENABLED", "false")
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Unable to fetch candles from Stooq"):
        data_sources.get_candle_history(symbol="AAPL", candle_size="1d", lookback_candles=1)

    assert calls["count"] == 1

def test_get_candle_history_stooq_uses_env_timeout(monkeypatch):
    csv_payload = """Date,Open,High,Low,Close,Volume
2024-01-02,10,11,9,10.5,100
"""

    captured = {"timeout": None}

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        return _DummyCsvResponse(csv_payload)

    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "stooq")
    monkeypatch.setenv("STOOQ_CR_TIMEOUT", "10800")
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)

    data_sources.get_candle_history(symbol="AAPL", candle_size="1d", lookback_candles=1)

    assert captured["timeout"] == 10800


def test_get_candle_history_alpha_vantage_provider(monkeypatch):
    payload = {
        "Time Series (Daily)": {
            "2024-01-03": {
                "1. open": "11",
                "2. high": "12",
                "3. low": "10",
                "4. close": "11.5",
                "6. volume": "120",
            },
            "2024-01-02": {
                "1. open": "10",
                "2. high": "11",
                "3. low": "9",
                "4. close": "10.5",
                "6. volume": "100",
            },
        }
    }

    def fake_urlopen(request, timeout):
        return _DummyResponse(payload)

    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "alpha_vantage")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "demo")
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)

    history = data_sources.get_candle_history(symbol="AAPL", candle_size="1d", lookback_candles=2)

    assert list(history.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(history) == 2
    assert str(history.index[0].date()) == "2024-01-02"


def test_get_candle_history_alpha_vantage_1h_uses_row_target_not_days(monkeypatch):
    lookback_candles = 240
    start = datetime(2025, 1, 1, 9, 0)
    series: dict[str, dict[str, str]] = {}
    for index in range(lookback_candles + 30):
        timestamp = (start + timedelta(hours=index)).strftime("%Y-%m-%d %H:%M:%S")
        price = 100 + index
        series[timestamp] = {
            "1. open": str(price),
            "2. high": str(price + 1),
            "3. low": str(price - 1),
            "4. close": str(price + 0.5),
            "5. volume": str(1000 + index),
        }
    payload = {"Time Series (60min)": series}

    def fake_urlopen(request, timeout):
        return _DummyResponse(payload)

    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "alpha_vantage")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "demo")
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)

    history = data_sources.get_candle_history(
        symbol="AAPL",
        candle_size="1h",
        lookback_candles=lookback_candles,
    )

    assert len(history) == lookback_candles


def test_get_candle_history_alpha_vantage_requires_api_key(monkeypatch):
    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "alpha_vantage")
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ALPHA_VANTAGE_API_KEY"):
        data_sources.get_candle_history(symbol="AAPL")

def test_get_candle_history_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="MARKETS_DATA_PROVIDER must be one of"):
        data_sources.get_candle_history(symbol="AAPL")


def test_get_candle_history_massive_provider(monkeypatch):
    payload = {
        "results": [
            {"t": 1704153600000, "o": 10, "h": 11, "l": 9, "c": 10.5, "v": 100},
            {"t": 1704240000000, "o": 11, "h": 12, "l": 10, "c": 11.5, "v": 120},
        ]
    }
    captured = {"url": ""}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return _DummyResponse(payload)

    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "massive")
    monkeypatch.setenv("MASSIVE_API_KEY", "demo")
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)

    history = data_sources.get_candle_history(symbol="AAPL", candle_size="1d", lookback_candles=2)

    assert list(history.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(history) == 2
    assert str(history.index[0].date()) == "2024-01-02"
    assert "api.massive.com/v2/aggs/ticker/AAPL/range/1/day/" in captured["url"]
    assert "apiKey=demo" in captured["url"]


def test_get_candle_history_massive_1h_uses_row_target_not_days(monkeypatch):
    lookback_candles = 240
    start = datetime(2025, 1, 1, 9, 0)
    payload = {
        "results": [
            {
                "t": int((start + timedelta(hours=index)).timestamp() * 1000),
                "o": 100 + index,
                "h": 101 + index,
                "l": 99 + index,
                "c": 100.5 + index,
                "v": 1000 + index,
            }
            for index in range(lookback_candles + 60)
        ]
    }
    captured = {"url": ""}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return _DummyResponse(payload)

    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "massive")
    monkeypatch.setenv("MASSIVE_API_KEY", "demo")
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)

    history = data_sources.get_candle_history(
        symbol="AAPL",
        candle_size="1h",
        lookback_candles=lookback_candles,
    )

    assert len(history) == lookback_candles
    assert "api.massive.com/v2/aggs/ticker/AAPL/range/1/hour/" in captured["url"]


def test_get_candle_history_massive_requires_api_key(monkeypatch):
    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "massive")
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="MASSIVE_API_KEY"):
        data_sources.get_candle_history(symbol="AAPL")


def test_get_candle_history_etoro_provider(monkeypatch):
    captured = {"search_url": "", "candles_url": ""}

    def fake_urlopen(request, timeout):
        if "/market-data/search" in request.full_url:
            captured["search_url"] = request.full_url
            return _DummyResponse(
                {
                    "items": [
                        {"instrumentId": 1001, "symbol": "MSFT"},
                        {"instrumentId": 42, "symbol": "AAPL"},
                    ]
                }
            )
        captured["candles_url"] = request.full_url
        return _DummyResponse(
            {
                "interval": "OneDay",
                "candles": [
                    {
                        "instrumentId": 42,
                        "candles": [
                            {
                                "instrumentID": 42,
                                "fromDate": "2024-01-02T00:00:00Z",
                                "open": 10,
                                "high": 11,
                                "low": 9,
                                "close": 10.5,
                                "volume": 100,
                            },
                            {
                                "instrumentID": 42,
                                "fromDate": "2024-01-03T00:00:00Z",
                                "open": 11,
                                "high": 12,
                                "low": 10,
                                "close": 11.5,
                                "volume": 120,
                            },
                        ],
                    }
                ],
            }
        )

    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "etoro")
    monkeypatch.setenv("ETORO_API_KEY", "api-key")
    monkeypatch.setenv("ETORO_USER_KEY", "user-key")
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)

    history = data_sources.get_candle_history(symbol="AAPL", candle_size="1d", lookback_candles=2)

    parsed = urlparse(captured["search_url"])
    query = parse_qs(parsed.query)
    assert parsed.path == "/api/v1/market-data/search"
    assert query["searchText"] == ["AAPL"]
    assert query["fields"] == ["instrumentId,symbol,displayname"]
    assert captured["candles_url"].endswith(
        "/api/v1/market-data/instruments/42/history/candles/asc/OneDay/2"
    )
    assert list(history.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(history) == 2
    assert str(history.index[0].date()) == "2024-01-02"


def test_get_candle_history_etoro_requires_credentials(monkeypatch):
    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "etoro")
    monkeypatch.delenv("ETORO_API_KEY", raising=False)
    monkeypatch.delenv("ETORO_USER_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ETORO_API_KEY and ETORO_USER_KEY are required"):
        data_sources.get_candle_history(symbol="AAPL")


def test_get_candle_history_etoro_raises_when_instrument_missing(monkeypatch):
    def fake_urlopen(request, timeout):
        return _DummyResponse({"items": []})

    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "etoro")
    monkeypatch.setenv("ETORO_API_KEY", "api-key")
    monkeypatch.setenv("ETORO_USER_KEY", "user-key")
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Unable to resolve instrumentId"):
        data_sources.get_candle_history(symbol="AAPL")


def test_get_candle_history_etoro_market_search_backoff_is_incremental_per_ticker(monkeypatch):
    sleeps: list[int] = []
    call_count = {"value": 0}

    def fake_urlopen(request, timeout):
        if "/market-data/search" in request.full_url:
            call_count["value"] += 1
            raise HTTPError(
                url=request.full_url,
                code=429,
                msg="Too Many Requests",
                hdrs=None,
                fp=None,
            )
        raise AssertionError(
            "Candles endpoint should not be called when market search keeps failing."
        )

    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "etoro")
    monkeypatch.setenv("ETORO_API_KEY", "api-key")
    monkeypatch.setenv("ETORO_USER_KEY", "user-key")
    monkeypatch.setenv("RETRY_BACKOFFF_ENABLED", "true")
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)
    monkeypatch.setattr(data_sources.time, "sleep", lambda seconds: sleeps.append(seconds))
    data_sources._ETORO_TICKER_BACKOFF_SECONDS.clear()
    data_sources._ETORO_TICKER_BACKOFF_SECONDS["AAPL"] = 299

    with pytest.raises(RuntimeError, match="Unable to resolve instrumentId"):
        data_sources.get_candle_history(symbol="AAPL")

    assert call_count["value"] == 3
    assert sleeps == [300, 300]


def test_get_candle_history_yfinance_falls_back_to_stooq(monkeypatch):
    csv_payload = """Date,Open,High,Low,Close,Volume
2024-01-02,10,11,9,10.5,100
2024-01-03,11,12,10,11.5,120
"""

    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "yfinance")
    monkeypatch.setenv("YFINANCE_FALLBACK_TO_STOOQ_ENABLED", "true")

    class _DummyTicker:
        def __init__(self, _symbol):
            pass

        def history(self, period, interval):
            raise RuntimeError("yfinance upstream timeout")

    monkeypatch.setitem(
        __import__("sys").modules,
        "yfinance",
        types.SimpleNamespace(Ticker=_DummyTicker),
    )
    monkeypatch.setattr(
        data_sources,
        "urlopen",
        lambda request, timeout: _DummyCsvResponse(csv_payload),
    )

    history = data_sources.get_candle_history(symbol="AAPL", candle_size="1d", lookback_candles=2)

    assert len(history) == 2
    assert float(history.iloc[-1]["Close"]) == 11.5


def test_get_candle_history_yfinance_fallback_disabled_raises_original_error(monkeypatch):
    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "yfinance")
    monkeypatch.setenv("YFINANCE_FALLBACK_TO_STOOQ_ENABLED", "false")

    class _DummyTicker:
        def __init__(self, _symbol):
            pass

        def history(self, period, interval):
            raise RuntimeError("yfinance upstream timeout")

    monkeypatch.setitem(
        __import__("sys").modules,
        "yfinance",
        types.SimpleNamespace(Ticker=_DummyTicker),
    )

    with pytest.raises(RuntimeError, match="yfinance upstream timeout"):
        data_sources.get_candle_history(symbol="AAPL", candle_size="1d", lookback_candles=2)


def test_get_candle_history_yfinance_intraday_does_not_fallback_to_stooq(monkeypatch):
    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "yfinance")
    monkeypatch.setenv("YFINANCE_FALLBACK_TO_STOOQ_ENABLED", "true")

    calls = {"stooq": 0}

    class _DummyTicker:
        def __init__(self, _symbol):
            pass

        def history(self, period, interval):
            raise RuntimeError("Too Many Requests. Rate limited. Try after a while.")

    def _unexpected_stooq(*args, **kwargs):
        calls["stooq"] += 1
        raise AssertionError("stooq fallback should not be used for intraday candle_size")

    monkeypatch.setitem(
        __import__("sys").modules,
        "yfinance",
        types.SimpleNamespace(Ticker=_DummyTicker),
    )
    monkeypatch.setattr(data_sources, "_get_stooq_candle_history", _unexpected_stooq)

    with pytest.raises(RuntimeError, match="Too Many Requests"):
        data_sources.get_candle_history(symbol="AAPL", candle_size="1h", lookback_candles=10)

    assert calls["stooq"] == 0


def test_fetch_fundamental_metrics_skips_for_stooq(monkeypatch):
    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "stooq")

    metrics = data_sources.fetch_fundamental_metrics("AAPL")

    assert metrics.pe_ratio is None
    assert metrics.eps is None
    assert metrics.debt_to_equity is None
    assert metrics.market_cap is None


def test_fetch_fundamental_metrics_skips_for_etoro(monkeypatch):
    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "etoro")

    metrics = data_sources.fetch_fundamental_metrics("AAPL")

    assert metrics.pe_ratio is None
    assert metrics.eps is None
    assert metrics.debt_to_equity is None
    assert metrics.market_cap is None


def test_fetch_fundamental_metrics_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="MARKETS_DATA_PROVIDER must be one of"):
        data_sources.fetch_fundamental_metrics("AAPL")


def test_fetch_fundamental_metrics_alpha_vantage_provider(monkeypatch):
    payload = {
        "PERatio": "23.4",
        "EPS": "4.2",
        "DebtToEquityRatio": "1.1",
        "MarketCapitalization": "1000000000",
    }

    def fake_urlopen(request, timeout):
        return _DummyResponse(payload)

    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "alpha_vantage")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "demo")
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)

    metrics = data_sources.fetch_fundamental_metrics("AAPL")

    assert metrics.pe_ratio == 23.4
    assert metrics.eps == 4.2
    assert metrics.debt_to_equity == 1.1
    assert metrics.market_cap == 1000000000.0


def test_fetch_fundamental_metrics_massive_provider(monkeypatch):
    payload = {"results": {"market_cap": 3000000000000}}
    captured = {"url": ""}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return _DummyResponse(payload)

    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "massive")
    monkeypatch.setenv("MASSIVE_API_KEY", "demo")
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)

    metrics = data_sources.fetch_fundamental_metrics("AAPL")

    assert metrics.pe_ratio is None
    assert metrics.eps is None
    assert metrics.debt_to_equity is None
    assert metrics.market_cap == 3000000000000.0
    assert "api.massive.com/v3/reference/tickers/AAPL?apiKey=demo" in captured["url"]


def test_fetch_fundamental_metrics_massive_requires_api_key(monkeypatch):
    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "massive")
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="MASSIVE_API_KEY"):
        data_sources.fetch_fundamental_metrics("AAPL")


def test_fetch_crypto_market_analysis_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("BINANCE_CRYPTO_API_KEY", raising=False)

    assert data_sources.fetch_crypto_market_analysis(limit=2) is None


def test_fetch_crypto_market_analysis_rejects_non_binance_provider(monkeypatch):
    monkeypatch.setenv("BINANCE_CRYPTO_API_KEY", "secret")
    monkeypatch.setenv("CRYPTO_MARKET_PROVIDER", "coinbase")

    with pytest.raises(ValueError, match="CRYPTO_MARKET_PROVIDER"):
        data_sources.fetch_crypto_market_analysis(limit=1)


def test_fetch_crypto_market_analysis_with_binance(monkeypatch):
    payloads = {
        "BTCUSDT": {
            "lastPrice": "68000",
            "priceChangePercent": "3.2",
            "volume": "100",
            "quoteVolume": "6800000",
        },
        "ETHUSDT": {
            "lastPrice": "3500",
            "priceChangePercent": "-1.5",
            "volume": "200",
            "quoteVolume": "700000",
        },
    }

    def fake_urlopen(request, timeout):
        symbol = request.full_url.split("symbol=", maxsplit=1)[1]
        return _DummyResponse(payloads[symbol])

    monkeypatch.setenv("BINANCE_CRYPTO_API_KEY", "secret")
    monkeypatch.setenv("CRYPTO_MARKET_PROVIDER", "binance")
    monkeypatch.setenv("CRYPTO_MARKET_SYMBOLS", "BTCUSDT,ETHUSDT")
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)

    result = data_sources.fetch_crypto_market_analysis(limit=2)

    assert result is not None
    assert result["provider"] == "binance"
    assert result["experimental"] is True
    assert result["top_gainer"]["symbol"] == "BTCUSDT"
    assert result["top_loser"]["symbol"] == "ETHUSDT"
    assert len(result["tickers"]) == 2

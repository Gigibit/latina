import json
from urllib.error import HTTPError

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


def test_get_candle_history_alpha_vantage_requires_api_key(monkeypatch):
    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "alpha_vantage")
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ALPHA_VANTAGE_API_KEY"):
        data_sources.get_candle_history(symbol="AAPL")

def test_get_candle_history_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="MARKETS_DATA_PROVIDER must be one of"):
        data_sources.get_candle_history(symbol="AAPL")


def test_fetch_fundamental_metrics_skips_for_stooq(monkeypatch):
    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "stooq")

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

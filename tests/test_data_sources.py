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
    monkeypatch.setattr(data_sources, "urlopen", fake_urlopen)
    monkeypatch.setattr(data_sources.time, "sleep", lambda seconds: sleeps.append(seconds))

    symbols = data_sources.fetch_trending_symbols(limit=1)

    assert symbols == ["AAPL"]
    assert calls["count"] == 3
    assert sleeps == [1, 2]


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


def test_get_candle_history_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="MARKETS_DATA_PROVIDER must be one of"):
        data_sources.get_candle_history(symbol="AAPL")

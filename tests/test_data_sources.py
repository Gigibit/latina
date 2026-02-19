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

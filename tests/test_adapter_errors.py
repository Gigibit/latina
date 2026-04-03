from io import BytesIO
from urllib.error import HTTPError

import pytest

from trading_bot.bot.binance_adapter import BinanceAdapter
from trading_bot.bot.etoro_adapter import EtoroAdapter


def test_etoro_adapter_includes_http_error_body(monkeypatch):
    def raise_http_error(request, timeout):
        raise HTTPError(
            url=request.full_url,
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=BytesIO(b'{"error":"invalid_key","detail":"token expired"}'),
        )

    monkeypatch.setattr("trading_bot.bot.etoro_adapter.urlopen", raise_http_error)
    adapter = EtoroAdapter(api_key="token", base_url="https://api.etoro.test")

    with pytest.raises(RuntimeError) as exc:
        adapter.getAccountSummary()

    message = str(exc.value)
    assert "status=403" in message
    assert "Forbidden" in message
    assert "token expired" in message


def test_binance_adapter_includes_http_error_body(monkeypatch):
    def raise_http_error(request, timeout):
        raise HTTPError(
            url=request.full_url,
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=BytesIO(b'{"code":-1003,"msg":"Too much request weight used"}'),
        )

    monkeypatch.setattr("trading_bot.bot.binance_adapter.urlopen", raise_http_error)
    adapter = BinanceAdapter(api_key="token", base_url="https://api.binance.test")

    with pytest.raises(RuntimeError) as exc:
        adapter.getPositions()

    message = str(exc.value)
    assert "status=429" in message
    assert "Too Many Requests" in message
    assert "Too much request weight used" in message

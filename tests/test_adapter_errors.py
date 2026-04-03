import json
import sys
import types
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


def test_etoro_adapter_uses_public_api_headers(monkeypatch):
    captured = {}

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = {k.lower(): v for k, v in request.header_items()}
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr("trading_bot.bot.etoro_adapter.urlopen", fake_urlopen)
    adapter = EtoroAdapter(
        api_key="token",
        base_url="https://public-api.etoro.com/api/v1",
        user_key="user-key",
    )

    adapter.getAccountSummary()

    assert captured["url"] == "https://public-api.etoro.com/api/v1/trading/info/real/portfolio"
    assert captured["headers"]["x-api-key"] == "token"
    assert captured["headers"]["x-user-key"] == "user-key"
    assert "x-request-id" in captured["headers"]
    assert captured["timeout"] == 15


def test_etoro_adapter_uses_env_specific_account_summary_path(monkeypatch):
    captured = {}

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return DummyResponse()

    monkeypatch.setenv("ETORO_ENV", "DEMO")
    monkeypatch.setattr("trading_bot.bot.etoro_adapter.urlopen", fake_urlopen)
    adapter = EtoroAdapter(
        api_key="token",
        base_url="https://public-api.etoro.com/api/v1",
        user_key="user-key",
    )

    adapter.getAccountSummary()

    assert captured["url"] == "https://public-api.etoro.com/api/v1/trading/info/demo/portfolio"


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


def test_etoro_adapter_websocket_auth_success(monkeypatch):
    class DummySocket:
        def __init__(self):
            self.messages = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def send(self, payload):
            self.messages.append(json.loads(payload))

        def recv(self):
            return json.dumps({"success": True, "operation": "Authenticate"})

    socket = DummySocket()
    websocket_module = types.SimpleNamespace(create_connection=lambda *_args, **_kwargs: socket)
    monkeypatch.setitem(sys.modules, "websocket", websocket_module)

    adapter = EtoroAdapter(
        api_key="token",
        base_url="https://api.etoro.test",
        ws_url="wss://api.etoro.test/ws",
        user_key="user",
    )
    response = adapter.authenticateWebsocket()

    assert response["success"] is True
    assert socket.messages[0]["operation"] == "Authenticate"
    assert socket.messages[0]["data"]["userKey"] == "user"
    assert socket.messages[0]["data"]["apiKey"] == "token"


def test_etoro_adapter_websocket_auth_failure(monkeypatch):
    class DummySocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def send(self, payload):
            assert payload

        def recv(self):
            return json.dumps(
                {
                    "success": False,
                    "operation": "Authenticate",
                    "errorCode": "Forbidden",
                    "errorMessage": "InvalidKey",
                }
            )

    websocket_module = types.SimpleNamespace(
        create_connection=lambda *_args, **_kwargs: DummySocket()
    )
    monkeypatch.setitem(sys.modules, "websocket", websocket_module)

    adapter = EtoroAdapter(
        api_key="token",
        base_url="https://api.etoro.test",
        ws_url="wss://api.etoro.test/ws",
        user_key="user",
    )
    with pytest.raises(RuntimeError) as exc:
        adapter.authenticateWebsocket()

    assert "errorCode=Forbidden" in str(exc.value)

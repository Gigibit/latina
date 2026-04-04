from trading_bot.bot.etoro import execute_etoro_action


def test_etoro_autotrade_disabled(monkeypatch):
    monkeypatch.delenv("ETORO_AUTOTRADE_ENABLED", raising=False)
    result = execute_etoro_action(symbol="AAPL", action="BUY")
    assert result["status"] == "disabled"


def test_etoro_sell_action_disabled(monkeypatch):
    monkeypatch.setenv("ETORO_AUTOTRADE_ENABLED", "true")
    monkeypatch.setenv("ETORO_ENABLE_SELL_ACTION", "false")

    result = execute_etoro_action(symbol="AAPL", action="SELL")

    assert result["status"] == "skipped"
    assert "disabled" in result["reason"].lower()


def test_etoro_wait_window_blocks_execution(monkeypatch, tmp_path):
    monkeypatch.setenv("ETORO_AUTOTRADE_ENABLED", "true")
    monkeypatch.setenv("ETORO_ENABLE_BUY_ACTION", "true")
    monkeypatch.setenv("ETORO_NEXT_SUGGESTION_WAIT_SECONDS", "999999")
    monkeypatch.setenv("ETORO_AUTOTRADE_STATE_FILE", str(tmp_path / "state.json"))

    first = execute_etoro_action(symbol="AAPL", action="HOLD")
    second = execute_etoro_action(symbol="AAPL", action="HOLD")

    assert first["status"] == "skipped"
    assert second["status"] == "cooldown"
    assert second["retry_after_seconds"] > 0


def test_etoro_logs_request_and_response_bodies(monkeypatch, caplog):
    monkeypatch.setenv("ETORO_AUTOTRADE_ENABLED", "true")
    monkeypatch.setenv("ETORO_ENABLE_BUY_ACTION", "true")
    monkeypatch.setenv("ETORO_API_BASE_URL", "https://api.etoro.test")
    monkeypatch.setenv("ETORO_API_KEY", "secret")
    monkeypatch.setenv("ETORO_USER_KEY", "user-secret")
    monkeypatch.setenv("ETORO_INSTRUMENT_ID", "1001")
    monkeypatch.setenv("ETORO_ENV", "DEMO")
    monkeypatch.setenv("ETORO_NEXT_SUGGESTION_WAIT_SECONDS", "0")

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"ok": true, "id": "abc123"}'

    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = {k.lower(): v for k, v in request.header_items()}
        return DummyResponse()

    monkeypatch.setattr("trading_bot.bot.etoro.urlopen", fake_urlopen)

    caplog.set_level("DEBUG", logger="trading_bot.bot.etoro")
    result = execute_etoro_action(symbol="AAPL", action="BUY", confidence=88)

    assert result["status"] == "executed"
    assert "eToro request body:" in caplog.text
    assert "AAPL" in caplog.text
    assert "eToro response body:" in caplog.text
    assert '"id": "abc123"' in caplog.text
    assert captured["url"] == "https://api.etoro.test/trading/execution/demo/market-open-orders/by-amount"
    assert captured["headers"]["x-api-key"] == "secret"
    assert captured["headers"]["x-user-key"] == "user-secret"
    assert "x-request-id" in captured["headers"]


def test_etoro_can_disable_http_logs(monkeypatch, caplog):
    monkeypatch.setenv("ETORO_AUTOTRADE_ENABLED", "true")
    monkeypatch.setenv("ETORO_ENABLE_BUY_ACTION", "true")
    monkeypatch.setenv("ETORO_API_BASE_URL", "https://api.etoro.test")
    monkeypatch.setenv("ETORO_API_KEY", "secret")
    monkeypatch.setenv("ETORO_USER_KEY", "user-secret")
    monkeypatch.setenv("ETORO_INSTRUMENT_ID", "1001")
    monkeypatch.setenv("ETORO_NEXT_SUGGESTION_WAIT_SECONDS", "0")
    monkeypatch.setenv("HTTP_LOG_ENABLED", "false")

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"ok": true, "id": "abc123"}'

    monkeypatch.setattr("trading_bot.bot.etoro.urlopen", lambda request, timeout: DummyResponse())

    caplog.set_level("DEBUG", logger="trading_bot.bot.etoro")
    result = execute_etoro_action(symbol="AAPL", action="BUY", confidence=88)

    assert result["status"] == "executed"
    assert "Evaluating this symbol for eToro execution:" in caplog.text
    assert "Calling this service for eToro order execution:" in caplog.text
    assert "eToro request body:" not in caplog.text
    assert "eToro response body:" not in caplog.text


def test_etoro_returns_error_when_env_is_invalid(monkeypatch):
    monkeypatch.setenv("ETORO_AUTOTRADE_ENABLED", "true")
    monkeypatch.setenv("ETORO_ENABLE_BUY_ACTION", "true")
    monkeypatch.setenv("ETORO_API_BASE_URL", "https://api.etoro.test")
    monkeypatch.setenv("ETORO_API_KEY", "secret")
    monkeypatch.setenv("ETORO_USER_KEY", "user-secret")
    monkeypatch.setenv("ETORO_INSTRUMENT_ID", "1001")
    monkeypatch.setenv("ETORO_ENV", "PAPER")

    result = execute_etoro_action(symbol="AAPL", action="BUY")

    assert result["status"] == "error"
    assert "ETORO_ENV" in result["reason"]


def test_etoro_order_price_evaluator_skips_buy_when_price_is_not_convenient(monkeypatch):
    monkeypatch.setenv("ETORO_AUTOTRADE_ENABLED", "true")
    monkeypatch.setenv("ETORO_ENABLE_BUY_ACTION", "true")
    monkeypatch.setenv("ETORO_ORDER_PRICE_EVALUATOR_ENABLED", "true")
    monkeypatch.setenv("ETORO_ORDER_PRICE_MAX_DEVIATION_PCT", "0.50")

    result = execute_etoro_action(
        symbol="AAPL",
        action="BUY",
        current_price=101.0,
        reference_price=100.0,
    )

    assert result["status"] == "skipped"
    assert "price evaluator" in result["reason"].lower()
    assert "exceeds max_buy_price" in result["price_evaluation"]


def test_etoro_order_price_evaluator_allows_buy_when_price_is_convenient(monkeypatch):
    monkeypatch.setenv("ETORO_AUTOTRADE_ENABLED", "true")
    monkeypatch.setenv("ETORO_ENABLE_BUY_ACTION", "true")
    monkeypatch.setenv("ETORO_ORDER_PRICE_EVALUATOR_ENABLED", "true")
    monkeypatch.setenv("ETORO_ORDER_PRICE_MAX_DEVIATION_PCT", "0.50")
    monkeypatch.setenv("ETORO_API_BASE_URL", "https://api.etoro.test")
    monkeypatch.setenv("ETORO_API_KEY", "secret")
    monkeypatch.setenv("ETORO_USER_KEY", "user-secret")
    monkeypatch.setenv("ETORO_INSTRUMENT_ID", "1001")
    monkeypatch.setenv("ETORO_NEXT_SUGGESTION_WAIT_SECONDS", "0")

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"ok": true}'

    monkeypatch.setattr("trading_bot.bot.etoro.urlopen", lambda request, timeout: DummyResponse())

    result = execute_etoro_action(
        symbol="AAPL",
        action="BUY",
        current_price=100.3,
        reference_price=100.0,
    )

    assert result["status"] == "executed"

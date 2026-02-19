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
    monkeypatch.setenv("ETORO_NEXT_SUGGESTION_WAIT_SECONDS", "0")

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
    assert "eToro request body:" in caplog.text
    assert "AAPL" in caplog.text
    assert "eToro response body:" in caplog.text
    assert '"id": "abc123"' in caplog.text

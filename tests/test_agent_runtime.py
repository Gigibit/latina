import pytest

from trading_bot.bot.agent_runtime import AgentRuntime


@pytest.mark.parametrize(
    "api_key,error_message",
    [
        ("", "ETORO_API_KEY is missing"),
        ("bad key", "ETORO_API_KEY format is not valid"),
        ("short", "ETORO_API_KEY format is not valid"),
    ],
)
def test_validate_broker_config_rejects_invalid_api_key(monkeypatch, api_key, error_message):
    runtime = AgentRuntime(market="trader")
    monkeypatch.setenv("ETORO_API_BASE_URL", "https://api.etoro.com")
    monkeypatch.setenv("ETORO_API_KEY", api_key)

    with pytest.raises(ValueError, match=error_message):
        runtime._validate_broker_config()


def test_validate_broker_config_rejects_invalid_base_url(monkeypatch):
    runtime = AgentRuntime(market="trader")
    monkeypatch.setenv("ETORO_API_KEY", "valid_key_token_123")
    monkeypatch.setenv("ETORO_API_BASE_URL", "http://api.etoro.com")

    with pytest.raises(ValueError, match="ETORO_API_BASE_URL"):
        runtime._validate_broker_config()


def test_validate_broker_config_logs_target_without_secret(monkeypatch, caplog):
    runtime = AgentRuntime(market="trader")
    monkeypatch.setenv("ETORO_API_KEY", "valid_key_token_123")
    monkeypatch.setenv("ETORO_API_BASE_URL", "https://broker.example.test/v1")
    monkeypatch.setenv("ETORO_ENV", "DEMO")

    caplog.set_level("INFO", logger="trading_bot.bot.agent_runtime")
    runtime._validate_broker_config()

    assert "broker preflight target" in caplog.text
    assert "host=broker.example.test" in caplog.text
    assert "target_path=/v1/trading/info/demo/portfolio" in caplog.text
    assert "env=DEMO" in caplog.text
    assert "valid_key_token_123" not in caplog.text


def test_validate_broker_config_rejects_invalid_etoro_env(monkeypatch):
    runtime = AgentRuntime(market="trader")
    monkeypatch.setenv("ETORO_API_KEY", "valid_key_token_123")
    monkeypatch.setenv("ETORO_API_BASE_URL", "https://broker.example.test/v1")
    monkeypatch.setenv("ETORO_ENV", "PAPER")

    with pytest.raises(ValueError, match="ETORO_ENV"):
        runtime._validate_broker_config()


def test_validate_broker_config_skips_etoro_for_crypto(monkeypatch):
    runtime = AgentRuntime(market="crypto")
    monkeypatch.delenv("ETORO_API_KEY", raising=False)
    monkeypatch.delenv("ETORO_API_BASE_URL", raising=False)

    runtime._validate_broker_config()


def test_error_classifier_marks_403_as_fatal_auth():
    runtime = AgentRuntime(market="trader")
    exc = RuntimeError("eToro API request failed status=403 method=GET url=x reason=Forbidden")

    assert runtime._is_auth_fatal_error(exc) is True
    assert runtime._is_recoverable_error(exc) is False


def test_error_classifier_marks_429_as_recoverable():
    runtime = AgentRuntime(market="trader")
    exc = RuntimeError("eToro API request failed status=429 method=GET url=x reason=Too Many")

    assert runtime._is_recoverable_error(exc) is True
    assert runtime._is_auth_fatal_error(exc) is False


@pytest.mark.parametrize(
    "proposal_action,expected",
    [
        ("BUY", "BUY"),
        ("sell", "SELL"),
        ("BUY_CANDIDATE", "BUY"),
        ("CLOSE_CANDIDATE", "CLOSE"),
        ("hold", None),
    ],
)
def test_resolve_execution_action_supports_direct_and_candidate_actions(
    proposal_action, expected
):
    runtime = AgentRuntime(market="trader")

    assert runtime._resolve_execution_action(proposal_action) == expected

from types import SimpleNamespace

from django.test import RequestFactory

from trading_bot.bot.research_session import ResearchJob, ResearchSessionStore
from trading_bot.bot.views import trading_suggestion_view


def test_research_job_applies_acceptance_threshold(monkeypatch):
    monkeypatch.setenv("SYMBOL_ACTION_ACCEPTANCE_THRESHOLD", "70")

    monkeypatch.setattr(
        "trading_bot.bot.research_session.fetch_trending_symbols",
        lambda limit: ["AAA"],
    )
    monkeypatch.setattr(
        "trading_bot.bot.research_session.get_market_snapshot",
        lambda symbol: SimpleNamespace(
            symbol=symbol,
            latest_volume=200.0,
            avg_volume_20d=100.0,
            pct_change_5d=2.0,
        ),
    )
    monkeypatch.setattr(
        "trading_bot.bot.research_session.fetch_x_sentiment_scores",
        lambda symbol, days: {day: 0.2 for day in days},
    )
    monkeypatch.setattr(
        "trading_bot.bot.research_session.generate_suggestion",
        lambda symbol, user_risk_profile: {
            "symbol": symbol,
            "decision": {
                "action": "BUY",
                "confidence": 55,
                "risk_notes": "original",
            },
        },
    )
    monkeypatch.setattr(
        "trading_bot.bot.research_session._summarize_result_with_llm",
        lambda payload: "Summary from LLM",
    )

    store = ResearchSessionStore()
    job = ResearchJob(session_id="s1", risk_profile="medium")

    store._run_job(job)

    assert job.status == "completed"
    assert job.result is not None
    assert job.result["decision"]["action"] == "HOLD"
    assert job.result["discovery"]["acceptance_threshold"] == 70.0
    assert job.result["discovery"]["accepted"] is False
    assert job.result["readable_summary"] == "Summary from LLM"
    assert any("Research session started" in line for line in job.stream_log)
    assert any("Research complete" in line for line in job.stream_log)


def test_research_job_skips_readable_summary_when_disabled(monkeypatch):
    monkeypatch.setenv("OUTPUTS_READABILITY_ENABLED", "false")

    monkeypatch.setattr(
        "trading_bot.bot.research_session.fetch_trending_symbols",
        lambda limit: ["AAA"],
    )
    monkeypatch.setattr(
        "trading_bot.bot.research_session.get_market_snapshot",
        lambda symbol: SimpleNamespace(
            symbol=symbol,
            latest_volume=200.0,
            avg_volume_20d=100.0,
            pct_change_5d=2.0,
        ),
    )
    monkeypatch.setattr(
        "trading_bot.bot.research_session.fetch_x_sentiment_scores",
        lambda symbol, days: {day: 0.2 for day in days},
    )
    monkeypatch.setattr(
        "trading_bot.bot.research_session.generate_suggestion",
        lambda symbol, user_risk_profile: {
            "symbol": symbol,
            "decision": {
                "action": "BUY",
                "confidence": 80,
                "risk_notes": "original",
            },
        },
    )

    store = ResearchSessionStore()
    job = ResearchJob(session_id="s-disable-summary", risk_profile="medium")

    store._run_job(job)

    assert job.status == "completed"
    assert job.result is not None
    assert "readable_summary" not in job.result


def test_research_job_uses_manual_symbols_when_auto_detection_disabled(monkeypatch):
    monkeypatch.setenv("AUTO_DETECTION_SYMBOL_NUMBER", "false")

    monkeypatch.setattr(
        "trading_bot.bot.research_session.fetch_trending_symbols",
        lambda limit: ["AUTO"],
    )
    monkeypatch.setattr(
        "trading_bot.bot.research_session.get_market_snapshot",
        lambda symbol: SimpleNamespace(
            symbol=symbol,
            latest_volume=100.0,
            avg_volume_20d=100.0,
            pct_change_5d=2.0 if symbol == "AAPL" else 1.0,
        ),
    )
    monkeypatch.setattr(
        "trading_bot.bot.research_session.fetch_x_sentiment_scores",
        lambda symbol, days: {day: 0.1 for day in days},
    )
    monkeypatch.setattr(
        "trading_bot.bot.research_session.generate_suggestion",
        lambda symbol, user_risk_profile: {
            "symbol": symbol,
            "decision": {
                "action": "BUY",
                "confidence": 85,
                "risk_notes": "ok",
            },
        },
    )

    store = ResearchSessionStore()
    job = ResearchJob(
        session_id="s2",
        risk_profile="medium",
        candidate_symbols=["AAPL", "MSFT"],
    )

    store._run_job(job)

    assert job.status == "completed"
    assert job.result is not None
    assert job.result["symbol"] == "AAPL"
    assert any("Auto symbol detection disabled" in line for line in job.stream_log)


def test_trading_suggestion_view_async_returns_session_status(monkeypatch):
    fake_job = SimpleNamespace(
        status="running",
        stream_log=["[10:00:00] Research session started."],
        result=None,
        error=None,
    )

    monkeypatch.setattr(
        "trading_bot.bot.views.research_sessions.get_or_create",
        lambda session_id, risk_profile, candidate_symbols: fake_job,
    )

    request = RequestFactory().get(
        "/api/suggestion/?async=true&session_id=test-session&risk=low"
    )
    response = trading_suggestion_view(request)

    assert response.status_code == 200
    assert b'"session_id": "test-session"' in response.content
    assert b'"status": "running"' in response.content


def test_discover_symbol_requires_manual_symbols_when_auto_detection_disabled(monkeypatch):
    monkeypatch.setenv("AUTO_DETECTION_SYMBOL_NUMBER", "0")

    store = ResearchSessionStore()
    job = ResearchJob(session_id="s3", risk_profile="low")

    try:
        store._discover_symbol(job)
        raise AssertionError("Expected ValueError when symbols are missing")
    except ValueError as exc:
        assert "Provide symbols separated by commas" in str(exc)


def test_summarize_result_with_llm_fallback_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    from trading_bot.bot import research_session as module

    payload = {
        "symbol": "AAPL",
        "decision": {"action": "HOLD", "confidence": 42, "risk_notes": "Mixed signals."},
    }

    text = module._summarize_result_with_llm(payload)

    assert "Symbol: AAPL" in text
    assert "Action: HOLD" in text

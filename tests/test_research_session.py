from types import SimpleNamespace

from django.test import RequestFactory

from trading_bot.bot.research_session import ResearchJob, ResearchSessionStore
from trading_bot.bot.views import (
    best_projection_view,
    candle_playground_view,
    trading_suggestion_view,
)


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




def test_research_job_non_model_confidence_skips_threshold(monkeypatch):
    monkeypatch.setenv("SYMBOL_ACTION_ACCEPTANCE_THRESHOLD", "95")

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
                "confidence": 10,
                "final_confidence": 20,
                "confidence_source": "fallback",
                "risk_notes": "original",
            },
        },
    )

    store = ResearchSessionStore()
    job = ResearchJob(session_id="s-fallback", risk_profile="medium")

    store._run_job(job)

    assert job.status == "completed"
    assert job.result is not None
    assert job.result["decision"]["action"] == "BUY"
    assert job.result["decision"]["decision_quality"] == "low_schema_reliability"
    assert job.result["discovery"]["accepted"] is True
    assert "warning" in job.result["discovery"]


def test_research_job_threshold_uses_calibrated_final_confidence(monkeypatch):
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
                "confidence": 92,
                "final_confidence": 65,
                "confidence_source": "model",
                "risk_notes": "original",
            },
        },
    )

    store = ResearchSessionStore()
    job = ResearchJob(session_id="s-calibrated", risk_profile="medium")

    store._run_job(job)

    assert job.status == "completed"
    assert job.result is not None
    assert job.result["decision"]["action"] == "HOLD"
    assert job.result["discovery"]["accepted"] is False
    assert (
        "Calibrated confidence 65.00 below threshold 70.00."
        == job.result["decision"]["risk_notes"]
    )

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


def test_best_projection_view_returns_chart_payload(monkeypatch):
    monkeypatch.setattr(
        "trading_bot.bot.views.get_best_candidates",
        lambda limit, user_risk_profile: {
            "candidates": [
                {"symbol": "AAA", "combined_score": 10.5, "score": 9.1},
            ]
        },
    )
    monkeypatch.setattr(
        "trading_bot.bot.views.generate_suggestion",
        lambda symbol, user_risk_profile: {
            "candle_size": "1d",
            "candle_rag_granularity_size": "1h",
            "buy_probability": 0.63,
            "sell_probability": 0.37,
            "decision": {"action": "BUY"},
        },
    )

    class _FakeSeries:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return self._values

        @property
        def iloc(self):
            class _FakeIloc:
                def __init__(self, values):
                    self._values = values

                def __getitem__(self, idx):
                    return self._values[idx]

            return _FakeIloc(self._values)

    class _FakeIndex(list):
        def __getitem__(self, item):
            return super().__getitem__(item)

    class _FakeFrame:
        def __init__(self, closes, timestamps):
            self._closes = closes
            self.index = _FakeIndex(timestamps)

        def tail(self, _n):
            return self

        def copy(self):
            return self

        def __getitem__(self, key):
            if key == "Close":
                return _FakeSeries(self._closes)
            raise KeyError(key)

    from datetime import datetime, timedelta

    timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(80)]
    closes = [100 + (i * 0.1) for i in range(80)]
    monkeypatch.setattr(
        "trading_bot.bot.views.get_candle_history",
        lambda symbol, candle_size, lookback_candles: _FakeFrame(closes, timestamps),
    )

    request = RequestFactory().get("/api/projections/?risk=medium&limit=5")
    response = best_projection_view(request)

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert '"candidate_name": "AAA"' in content
    assert '"candle_rag_granularity_size": "1h"' in content
    assert '"predicted_granularity_closes":' in content


def test_candle_playground_view_returns_sentiment(monkeypatch):
    from datetime import date

    class _FakeFrame:
        def __init__(self):
            self._rows = [
                (date(2024, 1, 1), {"Open": 10.0, "Close": 9.0}),
                (date(2024, 1, 2), {"Open": 8.0, "Close": 8.1}),
                (date(2024, 1, 3), {"Open": 9.2, "Close": 9.0}),
            ]

        def iterrows(self):
            for idx, row in self._rows:
                yield idx, row

        def __len__(self):
            return len(self._rows)

    monkeypatch.setattr(
        "trading_bot.bot.views.get_candle_history",
        lambda symbol, candle_size, lookback_candles: _FakeFrame(),
    )
    monkeypatch.setattr(
        "trading_bot.bot.views.LLMDecider.predict_next_candle_character",
        lambda self, sequence: "G",
    )

    request = RequestFactory().get(
        "/api/playground/?symbol=AAPL&date=2024-01-02&size=1d&history_limit=10000"
    )
    response = candle_playground_view(request)

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert '"sentiment": "positive"' in content
    assert '"sequence": "G( probability=0.0123 ) R( probability=0.0217 )"' in content
    assert '"prediction": "G"' in content
    assert '"candles_count": 2' in content


def test_candle_playground_view_rejects_invalid_size():
    request = RequestFactory().get("/api/playground/?symbol=AAPL&date=2024-01-02&size=2d")
    response = candle_playground_view(request)
    assert response.status_code == 400
    assert b"size must be one of" in response.content


def test_candle_playground_view_rejects_invalid_history_limit():
    request = RequestFactory().get(
        "/api/playground/?symbol=AAPL&date=2024-01-02&size=1d&history_limit=12000"
    )
    response = candle_playground_view(request)
    assert response.status_code == 400
    assert b"history_limit must be between 1 and 10000" in response.content


def test_candle_playground_view_1h_sequence_is_not_truncated(monkeypatch):
    import json
    from datetime import datetime, timedelta

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

    start = datetime(2025, 12, 1, 14, 0)
    candles = []
    for index in range(2500):
        timestamp = int((start + timedelta(hours=index)).timestamp() * 1000)
        open_price = 100 + index
        close_price = open_price + (1 if index % 2 == 0 else -1)
        candles.append(
            {
                "t": timestamp,
                "o": open_price,
                "h": open_price + 2,
                "l": open_price - 2,
                "c": close_price,
                "v": 1000 + index,
            }
        )

    monkeypatch.setenv("MARKETS_DATA_PROVIDER", "massive")
    monkeypatch.setenv("MASSIVE_API_KEY", "demo")
    monkeypatch.setattr(
        data_sources,
        "urlopen",
        lambda request, timeout: _DummyResponse({"results": candles}),
    )
    monkeypatch.setattr(
        "trading_bot.bot.views.LLMDecider.predict_next_candle_character",
        lambda self, sequence: "G",
    )

    request = RequestFactory().get("/api/playground/?symbol=AAPL&date=2025-12-25&size=1h")
    response = candle_playground_view(request)

    assert response.status_code == 200
    payload = json.loads(response.content.decode("utf-8"))
    assert payload["size"] == "1h"
    assert len(payload["sequence"]) > 95



def test_research_job_retries_on_rate_limit_and_completes(monkeypatch):
    monkeypatch.setenv("SUGGESTION_RATE_LIMIT_MAX_ATTEMPTS", "3")

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

    calls = {"count": 0}

    def _generate(symbol, user_risk_profile):
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("Too Many Requests. Rate limited. Try after a while.")
        return {
            "symbol": symbol,
            "decision": {
                "action": "BUY",
                "confidence": 80,
                "risk_notes": "ok",
            },
        }

    monkeypatch.setattr("trading_bot.bot.research_session.generate_suggestion", _generate)
    monkeypatch.setattr("trading_bot.bot.research_session.time.sleep", lambda _: None)

    store = ResearchSessionStore()
    job = ResearchJob(session_id="s-retry", risk_profile="medium")

    store._run_job(job)

    assert calls["count"] == 3
    assert job.status == "completed"
    assert job.result is not None
    assert any("Suggestion model rate limited" in line for line in job.stream_log)


def test_research_job_fails_when_rate_limit_retries_exhausted(monkeypatch):
    monkeypatch.setenv("SUGGESTION_RATE_LIMIT_MAX_ATTEMPTS", "2")

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
        lambda symbol, user_risk_profile: (_ for _ in ()).throw(
            RuntimeError("Too Many Requests. Rate limited. Try after a while.")
        ),
    )
    monkeypatch.setattr("trading_bot.bot.research_session.time.sleep", lambda _: None)

    store = ResearchSessionStore()
    job = ResearchJob(session_id="s-retry-fail", risk_profile="medium")

    store._run_job(job)

    assert job.status == "failed"
    assert job.error is not None
    assert "rate limiting" in job.error.lower()


def test_research_job_continues_with_next_symbol_after_rate_limit_anomaly(monkeypatch):
    monkeypatch.setenv("SUGGESTION_RATE_LIMIT_MAX_ATTEMPTS", "1")

    monkeypatch.setattr(
        "trading_bot.bot.research_session.fetch_trending_symbols",
        lambda limit: ["AAA", "BBB"],
    )

    snapshots = {
        "AAA": SimpleNamespace(
            symbol="AAA",
            latest_volume=200.0,
            avg_volume_20d=100.0,
            pct_change_5d=4.0,
        ),
        "BBB": SimpleNamespace(
            symbol="BBB",
            latest_volume=180.0,
            avg_volume_20d=100.0,
            pct_change_5d=3.0,
        ),
    }
    monkeypatch.setattr(
        "trading_bot.bot.research_session.get_market_snapshot",
        lambda symbol: snapshots[symbol],
    )
    monkeypatch.setattr(
        "trading_bot.bot.research_session.fetch_x_sentiment_scores",
        lambda symbol, days: {day: 0.5 for day in days},
    )

    def _generate(symbol, user_risk_profile):
        if symbol == "AAA":
            raise RuntimeError("Too Many Requests. Rate limited. Try after a while.")
        return {
            "symbol": symbol,
            "decision": {
                "action": "BUY",
                "confidence": 80,
                "risk_notes": "ok",
            },
        }

    monkeypatch.setattr("trading_bot.bot.research_session.generate_suggestion", _generate)
    monkeypatch.setattr("trading_bot.bot.research_session.time.sleep", lambda _: None)

    store = ResearchSessionStore()
    job = ResearchJob(session_id="s-rate-limit-fallback", risk_profile="medium")

    store._run_job(job)

    assert job.status == "completed"
    assert job.result is not None
    assert job.result["symbol"] == "BBB"
    assert job.result["discovery"]["selected_symbol"] == "BBB"
    assert job.result["discovery"]["anomalies"]
    assert any("Suggestion model anomaly on AAA" in line for line in job.stream_log)
